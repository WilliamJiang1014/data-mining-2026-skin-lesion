from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from skin_lesion_risk.evaluation.calibration import expected_calibration_error
from skin_lesion_risk.evaluation.metrics import binary_classification_metrics
from skin_lesion_risk.evaluation.thresholds import threshold_at_min_sensitivity, youden_threshold
from skin_lesion_risk.models.base import BaseModelAdapter, ModelBatch, PredictionResult

ID_COLUMNS = {"patient_id", "sample_id", "lesion_id", "image_id", "isic_id"}


def _log_epoch(name: str, epoch: int, total: int, row: dict, remain: int, best: float) -> None:
    pauc = row.get("val_pauc_tpr80", float("nan"))
    auroc = row.get("val_auroc", float("nan"))
    loss = row.get("train_loss", float("nan"))
    lr = row.get("lr", float("nan"))
    print(f"[{name}] epoch {epoch}/{total}  loss={loss:.4f}  pAUC={pauc:.4f}  AUROC={auroc:.4f}  lr={lr:.2e}  best={best:.4f}  patience_left={remain}", flush=True)


def _loader_kwargs(params: dict, device: torch.device) -> dict:
    num_workers = int(params.get("num_workers", 0))
    prefetch_factor = int(params.get("prefetch_factor", 2))
    pin_memory = device.type == "cuda"
    kwargs: dict[str, Any] = {"pin_memory": pin_memory}
    if num_workers > 0:
        kwargs["num_workers"] = num_workers
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


# ---------------------------------------------------------------------------
# Multimodal Dataset (image + metadata + label, shuffled together)
# ---------------------------------------------------------------------------

class MultimodalDataset(Dataset):
    """Returns (image, metadata_row, label) so shuffling keeps them aligned."""

    def __init__(self, image_paths: list[str], metadata_tensor: torch.Tensor, labels: np.ndarray, transform: Any | None = None) -> None:
        self.image_paths = image_paths
        self.metadata_tensor = metadata_tensor
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Any, torch.Tensor, float]:
        from PIL import Image

        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        meta_row = self.metadata_tensor[idx]
        label = float(self.labels[idx]) if self.labels is not None else 0.0
        return img, meta_row, label


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Gated Fusion Module
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """Learns element-wise gate weights to fuse image and metadata features."""

    def __init__(self, image_dim: int, metadata_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(image_dim + metadata_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, image_dim),
            nn.Sigmoid(),
        )

    def forward(self, image_features: torch.Tensor, metadata_features: torch.Tensor) -> torch.Tensor:
        gate_weights = self.gate(torch.cat([image_features, metadata_features], dim=1))
        gated_image = gate_weights * image_features
        return torch.cat([gated_image, metadata_features], dim=1)


# ---------------------------------------------------------------------------
# Metadata Encoder
# ---------------------------------------------------------------------------

class MetadataEncoder(nn.Module):
    """MLP encoder for tabular metadata features."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# M4: Multimodal Gated Fusion Model
# ---------------------------------------------------------------------------

class MultimodalFusionModel(BaseModelAdapter):
    """ConvNeXt-Tiny image encoder + MLP metadata encoder + Gated Fusion + classifier head."""

    model_type = "multimodal"

    def __init__(self, *, model_name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, params=params)
        self.image_encoder: nn.Module | None = None
        self.metadata_encoder: MetadataEncoder | None = None
        self.gate: GatedFusion | None = None
        self.head: nn.Module | None = None
        self.device = torch.device(str(self.params.get("device", "cpu")))
        self.image_feature_dim: int = 0
        self.metadata_input_dim: int = 0
        self.metadata_output_dim: int = 0
        self.feature_names: list[str] = []
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_thresholds: dict[str, float] = {}
        self.logs: list[dict[str, float]] = []

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "MultimodalFusionModel":
        if train.image_paths is None:
            raise ValueError("MultimodalFusionModel.fit requires ModelBatch.image_paths.")
        if train.metadata is None:
            raise ValueError("MultimodalFusionModel.fit requires ModelBatch.metadata.")
        if train.labels is None:
            raise ValueError("MultimodalFusionModel.fit requires labels.")

        encoder_name = str(self.params.get("image_encoder", "convnext_tiny"))
        pretrained = bool(self.params.get("pretrained", True))
        image_size = int(self.params.get("image_size", 224))
        epochs = int(self.params.get("epochs", 20))
        patience = int(self.params.get("patience", 5))
        lr = float(self.params.get("learning_rate", 1e-4))
        weight_decay = float(self.params.get("weight_decay", 1e-4))
        batch_size = int(self.params.get("batch_size", 32))
        dropout = float(self.params.get("dropout", 0.3))
        metadata_hidden_dim = int(self.params.get("metadata_hidden_dim", 128))
        metadata_output_dim = int(self.params.get("metadata_output_dim", 64))
        gate_hidden_dim = int(self.params.get("gate_hidden_dim", 128))
        out_dir = Path(str(self.params.get("out_dir", "data/artifacts/trained_models/m4_multimodal_fusion/fold0"))).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build image encoder
        import timm

        self.image_encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0)
        self.image_feature_dim = self.image_encoder.num_features

        # Build metadata encoder
        x_train_meta = self._metadata_frame(train.metadata, fit=True)
        self.metadata_input_dim = x_train_meta.shape[1]
        self.metadata_output_dim = metadata_output_dim
        self.metadata_encoder = MetadataEncoder(self.metadata_input_dim, metadata_hidden_dim, metadata_output_dim, dropout=dropout)

        # Build gated fusion
        self.gate = GatedFusion(self.image_feature_dim, metadata_output_dim, gate_hidden_dim)

        # Build classifier head
        fused_dim = self.image_feature_dim + metadata_output_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        # Move everything to device
        self.image_encoder.to(self.device)
        self.metadata_encoder.to(self.device)
        self.gate.to(self.device)
        self.head.to(self.device)

        y_train = np.asarray(train.labels).astype(int)
        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        all_params = list(self.image_encoder.parameters()) + list(self.metadata_encoder.parameters()) + list(self.gate.parameters()) + list(self.head.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # MultimodalDataset keeps image/metadata/label aligned when shuffling
        train_meta_tensor = torch.tensor(x_train_meta.values.astype(np.float32), dtype=torch.float32)
        train_ds = MultimodalDataset(train.image_paths, train_meta_tensor, y_train, transform=_train_transforms(image_size))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **_loader_kwargs(self.params, self.device))

        val_loader = None
        if valid is not None and valid.image_paths is not None and valid.labels is not None and valid.metadata is not None:
            y_val = np.asarray(valid.labels).astype(int)
            x_val_meta = self._metadata_frame(valid.metadata, fit=False)
            val_meta_tensor = torch.tensor(x_val_meta.values.astype(np.float32), dtype=torch.float32)
            val_ds = MultimodalDataset(valid.image_paths, val_meta_tensor, y_val, transform=_eval_transforms(image_size))
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(self.params, self.device))

        best_metric = -float("inf")
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            self.image_encoder.train()
            self.metadata_encoder.train()
            self.gate.train()
            self.head.train()
            epoch_loss = 0.0
            n_batches = 0
            for images, meta_batch, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                meta_batch = meta_batch.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                img_features = self.image_encoder(images)
                meta_features = self.metadata_encoder(meta_batch)
                fused = self.gate(img_features, meta_features)
                logits = self.head(fused).squeeze(1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1
            scheduler.step()

            row: dict[str, float] = {"epoch": float(epoch), "train_loss": epoch_loss / max(n_batches, 1), "lr": float(optimizer.param_groups[0]["lr"])}

            if val_loader is not None and valid is not None:
                val_result = self._predict_with_loader(val_loader, valid)
                val_labels = val_result.labels
                val_scores = val_result.scores
                val_threshold = youden_threshold(val_labels, val_scores) if len(np.unique(val_labels)) == 2 else 0.5
                high_threshold = threshold_at_min_sensitivity(val_labels, val_scores, min_sensitivity=0.90) if len(np.unique(val_labels)) == 2 else 0.5
                val_metrics = binary_classification_metrics(val_labels, val_scores, threshold=val_threshold)
                row.update({
                    "val_pauc_tpr80": float(val_metrics.get("pauc_tpr80", float("nan"))),
                    "val_auprc": float(val_metrics.get("auprc", float("nan"))),
                    "val_auroc": float(val_metrics.get("auroc", float("nan"))),
                    "val_brier": float(val_metrics.get("brier", float("nan"))),
                    "val_ece": float(expected_calibration_error(val_labels, val_scores)),
                    "threshold_youden": float(val_threshold),
                    "threshold_sensitivity_0_90": float(high_threshold),
                })
                metric = row["val_pauc_tpr80"] if np.isfinite(row["val_pauc_tpr80"]) else row.get("val_auprc", -float("inf"))
            else:
                metric = -row["train_loss"]

            self.logs.append(row)
            _append_log(out_dir / "train_log.csv", row)
            _append_log(out_dir / "loss_curve.csv", {"epoch": row["epoch"], "train_loss": row["train_loss"], "val_loss": row.get("val_loss", float("nan"))})

            if metric > best_metric:
                best_metric = float(metric)
                bad_epochs = 0
                self.best_state = {
                    "image_encoder": {k: v.detach().cpu() for k, v in self.image_encoder.state_dict().items()},
                    "metadata_encoder": {k: v.detach().cpu() for k, v in self.metadata_encoder.state_dict().items()},
                    "gate": {k: v.detach().cpu() for k, v in self.gate.state_dict().items()},
                    "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
                }
                self.best_thresholds = {
                    "youden": float(row.get("threshold_youden", 0.5)),
                    "sensitivity_at_least_0_90": float(row.get("threshold_sensitivity_0_90", 0.5)),
                }
                torch.save({
                    "state_dict": self.best_state,
                    "model_config": {
                        "image_encoder": encoder_name,
                        "image_feature_dim": self.image_feature_dim,
                        "metadata_input_dim": self.metadata_input_dim,
                        "metadata_hidden_dim": metadata_hidden_dim,
                        "metadata_output_dim": metadata_output_dim,
                        "gate_hidden_dim": gate_hidden_dim,
                        "dropout": dropout,
                        "feature_names": self.feature_names,
                    },
                }, out_dir / "best.ckpt")
            else:
                bad_epochs += 1

            _log_epoch(self.model_name, epoch, epochs, row, patience - bad_epochs, best_metric)

            if bad_epochs >= patience:
                break

        if self.best_state:
            self.image_encoder.load_state_dict(self.best_state["image_encoder"])
            self.metadata_encoder.load_state_dict(self.best_state["metadata_encoder"])
            self.gate.load_state_dict(self.best_state["gate"])
            self.head.load_state_dict(self.best_state["head"])
        self.is_fitted = True
        return self

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        if self.image_encoder is None or self.head is None:
            raise RuntimeError("Model is not fitted.")
        if batch.image_paths is None:
            raise ValueError("MultimodalFusionModel.predict_proba requires ModelBatch.image_paths.")
        if batch.metadata is None:
            raise ValueError("MultimodalFusionModel.predict_proba requires ModelBatch.metadata.")

        image_size = int(self.params.get("image_size", 224))
        batch_size = int(self.params.get("batch_size", 32))
        labels = np.asarray(batch.labels).astype(int) if batch.labels is not None else None
        x_meta = self._metadata_frame(batch.metadata, fit=False)
        meta_tensor = torch.tensor(x_meta.values.astype(np.float32), dtype=torch.float32)
        ds = MultimodalDataset(batch.image_paths, meta_tensor, np.zeros(len(batch.image_paths)), transform=_eval_transforms(image_size))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(self.params, self.device))

        self.image_encoder.eval()
        self.metadata_encoder.eval()
        self.gate.eval()
        self.head.eval()

        all_scores: list[np.ndarray] = []
        with torch.no_grad():
            for images, meta_batch, _ in loader:
                images = images.to(self.device, non_blocking=True)
                meta_batch = meta_batch.to(self.device, non_blocking=True)
                img_features = self.image_encoder(images)
                meta_features = self.metadata_encoder(meta_batch)
                fused = self.gate(img_features, meta_features)
                logits = self.head(fused).squeeze(1)
                probs = torch.sigmoid(logits)
                all_scores.append(probs.detach().cpu().numpy())
        scores = np.concatenate(all_scores)
        return PredictionResult(sample_ids=batch.sample_ids, scores=scores, labels=labels)

    def save(self, path: str | Path) -> None:
        if self.image_encoder is None or self.head is None:
            raise RuntimeError("Cannot save before model is fitted.")
        payload = {
            "model_name": self.model_name,
            "params": self.params,
            "state_dict": self.best_state or {
                "image_encoder": {k: v.detach().cpu() for k, v in self.image_encoder.state_dict().items()},
                "metadata_encoder": {k: v.detach().cpu() for k, v in self.metadata_encoder.state_dict().items()},
                "gate": {k: v.detach().cpu() for k, v in self.gate.state_dict().items()},
                "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
            },
            "model_config": {
                "image_encoder": str(self.params.get("image_encoder", "convnext_tiny")),
                "image_feature_dim": self.image_feature_dim,
                "metadata_input_dim": self.metadata_input_dim,
                "metadata_hidden_dim": int(self.params.get("metadata_hidden_dim", 128)),
                "metadata_output_dim": self.metadata_output_dim,
                "gate_hidden_dim": int(self.params.get("gate_hidden_dim", 128)),
                "dropout": float(self.params.get("dropout", 0.3)),
                "feature_names": self.feature_names,
            },
            "thresholds": self.best_thresholds,
            "logs": self.logs,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "MultimodalFusionModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        adapter = cls(model_name=payload.get("model_name", "m4_multimodal_fusion"), params=payload.get("params", {}))
        model_config = payload.get("model_config", {})
        import timm

        encoder_name = str(model_config.get("image_encoder", "convnext_tiny"))
        adapter.image_encoder = timm.create_model(encoder_name, pretrained=False, num_classes=0)
        adapter.image_feature_dim = int(model_config.get("image_feature_dim", adapter.image_encoder.num_features))
        adapter.metadata_input_dim = int(model_config.get("metadata_input_dim", 0))
        adapter.metadata_output_dim = int(model_config.get("metadata_output_dim", 64))
        metadata_hidden_dim = int(model_config.get("metadata_hidden_dim", 128))
        gate_hidden_dim = int(model_config.get("gate_hidden_dim", 128))
        dropout = float(model_config.get("dropout", adapter.params.get("dropout", 0.3)))
        adapter.feature_names = list(model_config.get("feature_names", []))

        adapter.metadata_encoder = MetadataEncoder(adapter.metadata_input_dim, metadata_hidden_dim, adapter.metadata_output_dim, dropout=dropout)
        adapter.gate = GatedFusion(adapter.image_feature_dim, adapter.metadata_output_dim, gate_hidden_dim)
        fused_dim = adapter.image_feature_dim + adapter.metadata_output_dim
        adapter.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        state_dict = payload["state_dict"]
        adapter.image_encoder.load_state_dict(state_dict["image_encoder"])
        adapter.metadata_encoder.load_state_dict(state_dict["metadata_encoder"])
        adapter.gate.load_state_dict(state_dict["gate"])
        adapter.head.load_state_dict(state_dict["head"])
        adapter.best_state = state_dict
        adapter.best_thresholds = payload.get("thresholds", {})
        adapter.logs = payload.get("logs", [])
        adapter.is_fitted = True
        return adapter

    def _metadata_frame(self, metadata: Any, *, fit: bool) -> pd.DataFrame:
        frame = metadata if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)
        frame = frame.copy()
        frame = frame.drop(columns=[c for c in frame.columns if c in ID_COLUMNS], errors="ignore")
        frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if fit:
            self.feature_names = frame.columns.astype(str).tolist()
            return frame.astype("float32")
        for name in self.feature_names:
            if name not in frame:
                frame[name] = 0.0
        extra = [c for c in frame.columns if c not in self.feature_names]
        frame = frame.drop(columns=extra, errors="ignore")
        return frame[self.feature_names].astype("float32")

    def _predict_with_loader(self, loader: DataLoader, batch: ModelBatch) -> PredictionResult:
        self.image_encoder.eval()
        self.metadata_encoder.eval()
        self.gate.eval()
        self.head.eval()

        all_scores: list[np.ndarray] = []
        with torch.no_grad():
            for images, meta_batch, _ in loader:
                images = images.to(self.device, non_blocking=True)
                meta_batch = meta_batch.to(self.device, non_blocking=True)
                img_features = self.image_encoder(images)
                meta_features = self.metadata_encoder(meta_batch)
                fused = self.gate(img_features, meta_features)
                logits = self.head(fused).squeeze(1)
                probs = torch.sigmoid(logits)
                all_scores.append(probs.detach().cpu().numpy())
        scores = np.concatenate(all_scores)
        labels = np.asarray(batch.labels).astype(int) if batch.labels is not None else None
        return PredictionResult(sample_ids=batch.sample_ids, scores=scores, labels=labels)


# ---------------------------------------------------------------------------
# Local transforms (avoids top-level torchvision import)
# ---------------------------------------------------------------------------

def _train_transforms(image_size: int = 224) -> Any:
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _eval_transforms(image_size: int = 224) -> Any:
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _append_log(path: Path, row: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header:
            fieldnames = existing_header
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
