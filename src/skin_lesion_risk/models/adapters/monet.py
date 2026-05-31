from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from skin_lesion_risk.evaluation.calibration import expected_calibration_error
from skin_lesion_risk.evaluation.metrics import binary_classification_metrics
from skin_lesion_risk.evaluation.thresholds import threshold_at_min_sensitivity, youden_threshold
from skin_lesion_risk.features.text_prompts import build_prompts
from skin_lesion_risk.models.base import BaseModelAdapter, ModelBatch, PredictionResult


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
# Dataset for frozen-encoder feature extraction
# ---------------------------------------------------------------------------

class ImageFeatureDataset(Dataset):
    """Loads images and returns them for feature extraction."""

    def __init__(self, image_paths: list[str], transform: Any | None = None) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Any:
        from PIL import Image

        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def _eval_transforms(image_size: int = 224) -> Any:
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Frozen encoder builder
# ---------------------------------------------------------------------------

def build_frozen_encoder(encoder_name: str = "efficientnet_b0", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build a frozen feature encoder via timm. Returns (encoder, feature_dim)."""
    import timm

    model = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0)
    for param in model.parameters():
        param.requires_grad = False
    return model, model.num_features


# ---------------------------------------------------------------------------
# M2: MONET Feature Model (frozen encoder + MLP head)
# ---------------------------------------------------------------------------

class MonetFeatureModel(BaseModelAdapter):
    """Frozen visual encoder + optional text prompt features + MLP classifier head.

    Uses a pretrained image encoder (e.g. EfficientNet-B0 or a VLM-compatible
    model) with all parameters frozen. An MLP head is trained on top of the
    concatenated [image_features, text_prompt_features] (if available) or
    image_features alone.
    """

    model_type = "monet_feature"

    def __init__(self, *, model_name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, params=params)
        self.encoder: nn.Module | None = None
        self.head: nn.Module | None = None
        self.device = torch.device(str(self.params.get("device", "cpu")))
        self.feature_dim: int = 0
        self.text_feature_dim: int = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_thresholds: dict[str, float] = {}
        self.logs: list[dict[str, float]] = []

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "MonetFeatureModel":
        if train.image_paths is None:
            raise ValueError("MonetFeatureModel.fit requires ModelBatch.image_paths.")
        if train.labels is None:
            raise ValueError("MonetFeatureModel.fit requires labels.")

        encoder_name = str(self.params.get("encoder", "efficientnet_b0"))
        pretrained = bool(self.params.get("pretrained", True))
        freeze_encoder = bool(self.params.get("freeze_encoder", True))
        image_size = int(self.params.get("image_size", 224))
        epochs = int(self.params.get("epochs", 20))
        patience = int(self.params.get("patience", 5))
        lr = float(self.params.get("learning_rate", 1e-3))
        weight_decay = float(self.params.get("weight_decay", 1e-4))
        batch_size = int(self.params.get("batch_size", 64))
        dropout = float(self.params.get("dropout", 0.3))
        use_text = bool(self.params.get("use_text_prompts", False))
        out_dir = Path(str(self.params.get("out_dir", "data/artifacts/trained_models/m2_monet_feature_baseline/fold0"))).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        self.encoder, self.feature_dim = build_frozen_encoder(encoder_name, pretrained=pretrained)
        self.encoder.to(self.device, non_blocking=True)
        if not freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = True

        # Text prompt features: simple TF-IDF-like hash encoding
        self.text_feature_dim = 0
        if use_text and train.metadata is not None:
            self.text_feature_dim = 64

        total_dim = self.feature_dim + self.text_feature_dim
        self.head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        ).to(self.device, non_blocking=True)

        # Extract frozen features for train
        train_img_features = self._extract_image_features(train.image_paths, image_size, batch_size)
        train_text_features = self._extract_text_features(train) if self.text_feature_dim > 0 else None
        train_x = np.concatenate([train_img_features] + ([train_text_features] if train_text_features is not None else []), axis=1)
        y_train = np.asarray(train.labels).astype(int)

        # Extract frozen features for val
        val_x = None
        if valid is not None and valid.image_paths is not None and valid.labels is not None:
            val_img_features = self._extract_image_features(valid.image_paths, image_size, batch_size)
            val_text_features = self._extract_text_features(valid) if self.text_feature_dim > 0 else None
            val_x = np.concatenate([val_img_features] + ([val_text_features] if val_text_features is not None else []), axis=1)

        # Train MLP head
        train_tensor = torch.tensor(train_x, dtype=torch.float32)
        train_labels_tensor = torch.tensor(y_train, dtype=torch.float32)
        val_tensor = torch.tensor(val_x, dtype=torch.float32) if val_x is not None else None
        val_labels = np.asarray(valid.labels).astype(int) if valid is not None and valid.labels is not None else None

        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_metric = -float("inf")
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            self.head.train()
            indices = torch.randperm(len(train_tensor))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(train_tensor), batch_size):
                idx = indices[start:start + batch_size]
                x_batch = train_tensor[idx].to(self.device, non_blocking=True)
                y_batch = train_labels_tensor[idx].to(self.device, non_blocking=True)
                optimizer.zero_grad()
                logits = self.head(x_batch).squeeze(1)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1
            scheduler.step()

            row: dict[str, float] = {"epoch": float(epoch), "train_loss": epoch_loss / max(n_batches, 1), "lr": float(optimizer.param_groups[0]["lr"])}

            if val_tensor is not None and val_labels is not None:
                self.head.eval()
                with torch.no_grad():
                    val_logits = self.head(val_tensor.to(self.device, non_blocking=True)).squeeze(1)
                    val_scores = torch.sigmoid(val_logits).detach().cpu().numpy()
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
                self.best_state = {k: v.detach().cpu() for k, v in self.head.state_dict().items()}
                self.best_thresholds = {
                    "youden": float(row.get("threshold_youden", 0.5)),
                    "sensitivity_at_least_0_90": float(row.get("threshold_sensitivity_0_90", 0.5)),
                }
                torch.save({
                    "state_dict": self.best_state,
                    "model_config": {
                        "encoder": encoder_name,
                        "feature_dim": self.feature_dim,
                        "text_feature_dim": self.text_feature_dim,
                        "dropout": dropout,
                    },
                }, out_dir / "best.ckpt")
            else:
                bad_epochs += 1

            _log_epoch(self.model_name, epoch, epochs, row, patience - bad_epochs, best_metric)

            if bad_epochs >= patience:
                break

        if self.best_state:
            self.head.load_state_dict(self.best_state)
        self.is_fitted = True
        return self

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        if self.encoder is None or self.head is None:
            raise RuntimeError("Model is not fitted.")
        if batch.image_paths is None:
            raise ValueError("MonetFeatureModel.predict_proba requires ModelBatch.image_paths.")

        image_size = int(self.params.get("image_size", 224))
        batch_size = int(self.params.get("batch_size", 64))
        img_features = self._extract_image_features(batch.image_paths, image_size, batch_size)
        text_features = self._extract_text_features(batch) if self.text_feature_dim > 0 else None
        x = np.concatenate([img_features] + ([text_features] if text_features is not None else []), axis=1)

        self.head.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device, non_blocking=True)
            logits = self.head(x_tensor).squeeze(1)
            scores = torch.sigmoid(logits).detach().cpu().numpy()
        labels = np.asarray(batch.labels).astype(int) if batch.labels is not None else None
        return PredictionResult(sample_ids=batch.sample_ids, scores=scores, labels=labels)

    def save(self, path: str | Path) -> None:
        if self.encoder is None or self.head is None:
            raise RuntimeError("Cannot save before model is fitted.")
        payload = {
            "model_name": self.model_name,
            "params": self.params,
            "state_dict": self.best_state or {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
            "encoder_state_dict": {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()},
            "model_config": {
                "encoder": str(self.params.get("encoder", "efficientnet_b0")),
                "feature_dim": self.feature_dim,
                "text_feature_dim": self.text_feature_dim,
                "dropout": float(self.params.get("dropout", 0.3)),
            },
            "thresholds": self.best_thresholds,
            "logs": self.logs,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "MonetFeatureModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        adapter = cls(model_name=payload.get("model_name", "m2_monet_feature_baseline"), params=payload.get("params", {}))
        model_config = payload.get("model_config", {})
        encoder_name = str(model_config.get("encoder", "efficientnet_b0"))
        # If checkpoint has saved encoder weights, skip pretrained download; otherwise fallback
        encoder_state = payload.get("encoder_state_dict")
        adapter.encoder, adapter.feature_dim = build_frozen_encoder(
            encoder_name, pretrained=(encoder_state is None),
        )
        if encoder_state is not None:
            adapter.encoder.load_state_dict(encoder_state)
        adapter.text_feature_dim = int(model_config.get("text_feature_dim", 0))
        dropout = float(model_config.get("dropout", adapter.params.get("dropout", 0.3)))
        total_dim = adapter.feature_dim + adapter.text_feature_dim
        adapter.head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )
        # Load head state dict
        adapter.head.load_state_dict(payload["state_dict"])
        adapter.best_state = payload["state_dict"]
        adapter.best_thresholds = payload.get("thresholds", {})
        adapter.logs = payload.get("logs", [])
        adapter.is_fitted = True
        # Keep on CPU; device migration is handled by the caller (evaluate.py _move_model_to_device)
        adapter.device = torch.device("cpu")
        return adapter

    def _extract_image_features(self, image_paths: list[str], image_size: int, batch_size: int) -> np.ndarray:
        ds = ImageFeatureDataset(image_paths, transform=_eval_transforms(image_size))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(self.params, self.device))
        self.encoder.eval()
        all_features: list[np.ndarray] = []
        with torch.no_grad():
            for images in loader:
                images = images.to(self.device, non_blocking=True)
                features = self.encoder(images)
                all_features.append(features.detach().cpu().numpy())
        return np.concatenate(all_features, axis=0)

    def _extract_text_features(self, batch: ModelBatch) -> np.ndarray | None:
        # Use raw_metadata (original clinical fields) for prompts, not encoded metadata
        raw = getattr(batch, "raw_metadata", None)
        if raw is None:
            raw = batch.metadata
        if raw is None:
            return None
        import pandas as pd

        df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        prompts = build_prompts(df)
        # Stable hash-based bag-of-words encoding for text prompts
        rng = np.random.RandomState(42)
        n_hashes = self.text_feature_dim
        projection = rng.randn(128, n_hashes).astype(np.float32)
        features = np.zeros((len(prompts), n_hashes), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            tokens = prompt.lower().split()
            for token in tokens:
                h = _stable_hash(token) % 128
                features[i] += projection[h]
        norms = np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-6)
        return (features / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _stable_hash(text: str) -> int:
    """Deterministic hash不受 PYTHONHASHSEED 影响。"""
    import hashlib
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF


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
