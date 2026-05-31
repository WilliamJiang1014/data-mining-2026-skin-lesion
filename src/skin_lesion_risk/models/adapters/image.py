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
# Dataset
# ---------------------------------------------------------------------------

class LesionImageDataset(Dataset):
    """Thin wrapper that loads images from paths on-the-fly."""

    def __init__(self, image_paths: list[str], labels: np.ndarray | None, transform: Any | None = None) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Any, float]:
        from PIL import Image

        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = float(self.labels[idx]) if self.labels is not None else 0.0
        return img, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def train_transforms(image_size: int = 224) -> Any:
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


def eval_transforms(image_size: int = 224) -> Any:
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Backbone builders
# ---------------------------------------------------------------------------

def build_cnn_backbone(encoder: str = "efficientnet_b0", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build CNN backbone via timm. Returns (feature_extractor, feature_dim)."""
    import timm

    model = timm.create_model(encoder, pretrained=pretrained, num_classes=0)
    feature_dim = model.num_features
    return model, feature_dim


def build_transformer_backbone(encoder: str = "swin_tiny_patch4_window7_224", pretrained: bool = True, image_size: int = 384) -> tuple[nn.Module, int]:
    """Build ViT/Swin backbone via timm. Returns (feature_extractor, feature_dim)."""
    import timm

    model = timm.create_model(encoder, pretrained=pretrained, num_classes=0, img_size=image_size)
    feature_dim = model.num_features
    return model, feature_dim


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, *, alpha: float = 0.25, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_factor * focal_factor * bce
        return loss.mean()


# ---------------------------------------------------------------------------
# M1: CNN Image Model
# ---------------------------------------------------------------------------

class CNNImageModel(BaseModelAdapter):
    """EfficientNet-B0 image baseline with weighted BCE loss."""

    model_type = "image"

    def __init__(self, *, model_name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, params=params)
        self.backbone: nn.Module | None = None
        self.head: nn.Module | None = None
        self.device = torch.device(str(self.params.get("device", "cpu")))
        self.feature_dim: int = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_thresholds: dict[str, float] = {}
        self.logs: list[dict[str, float]] = []

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "CNNImageModel":
        if train.image_paths is None:
            raise ValueError("CNNImageModel.fit requires ModelBatch.image_paths.")
        if train.labels is None:
            raise ValueError("CNNImageModel.fit requires labels.")

        encoder = str(self.params.get("encoder", "efficientnet_b0"))
        pretrained = bool(self.params.get("pretrained", True))
        image_size = int(self.params.get("image_size", 224))
        epochs = int(self.params.get("epochs", 20))
        patience = int(self.params.get("patience", 5))
        lr = float(self.params.get("learning_rate", 1e-4))
        weight_decay = float(self.params.get("weight_decay", 1e-4))
        batch_size = int(self.params.get("batch_size", 32))
        dropout = float(self.params.get("dropout", 0.3))
        out_dir = Path(str(self.params.get("out_dir", "data/artifacts/trained_models/m1_cnn_baseline/fold0"))).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        self.backbone, self.feature_dim = build_cnn_backbone(encoder, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
        )
        model = nn.Sequential(self.backbone, self.head).to(self.device, non_blocking=True)

        y_train = np.asarray(train.labels).astype(int)
        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_ds = LesionImageDataset(train.image_paths, y_train, transform=train_transforms(image_size))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **_loader_kwargs(self.params, self.device))

        val_loader = None
        if valid is not None and valid.image_paths is not None and valid.labels is not None:
            y_val = np.asarray(valid.labels).astype(int)
            val_ds = LesionImageDataset(valid.image_paths, y_val, transform=eval_transforms(image_size))
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(self.params, self.device))

        best_metric = -float("inf")
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                features = self.backbone(images)
                logits = self.head(features).squeeze(1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1
            scheduler.step()

            row: dict[str, float] = {"epoch": float(epoch), "train_loss": epoch_loss / max(n_batches, 1), "lr": float(optimizer.param_groups[0]["lr"])}

            if val_loader is not None:
                val_result = self._predict_loader(val_loader, valid)
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
                self.best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                self.best_thresholds = {
                    "youden": float(row.get("threshold_youden", 0.5)),
                    "sensitivity_at_least_0_90": float(row.get("threshold_sensitivity_0_90", 0.5)),
                }
                torch.save({"state_dict": self.best_state, "model_config": {"encoder": encoder, "feature_dim": self.feature_dim, "dropout": dropout}}, out_dir / "best.ckpt")
            else:
                bad_epochs += 1

            _log_epoch(self.model_name, epoch, epochs, row, patience - bad_epochs, best_metric)

            if bad_epochs >= patience:
                break

        if self.best_state:
            model.load_state_dict(self.best_state)
        self.is_fitted = True
        return self

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        if self.backbone is None or self.head is None:
            raise RuntimeError("Model is not fitted.")
        if batch.image_paths is None:
            raise ValueError("CNNImageModel.predict_proba requires ModelBatch.image_paths.")

        image_size = int(self.params.get("image_size", 224))
        batch_size = int(self.params.get("batch_size", 32))
        labels = np.asarray(batch.labels).astype(int) if batch.labels is not None else None
        ds = LesionImageDataset(batch.image_paths, labels, transform=eval_transforms(image_size))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(self.params, self.device))

        model = nn.Sequential(self.backbone, self.head).to(self.device, non_blocking=True)
        model.eval()
        all_scores: list[np.ndarray] = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device, non_blocking=True)
                features = self.backbone(images)
                logits = self.head(features).squeeze(1)
                probs = torch.sigmoid(logits)
                all_scores.append(probs.detach().cpu().numpy())
        scores = np.concatenate(all_scores)
        return PredictionResult(sample_ids=batch.sample_ids, scores=scores, labels=labels)

    def save(self, path: str | Path) -> None:
        if self.backbone is None or self.head is None:
            raise RuntimeError("Cannot save before model is fitted.")
        payload = {
            "model_name": self.model_name,
            "params": self.params,
            "state_dict": self.best_state or {k: v.detach().cpu() for k, v in nn.Sequential(self.backbone, self.head).state_dict().items()},
            "model_config": {
                "encoder": str(self.params.get("encoder", "efficientnet_b0")),
                "feature_dim": self.feature_dim,
                "dropout": float(self.params.get("dropout", 0.3)),
            },
            "thresholds": self.best_thresholds,
            "logs": self.logs,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "CNNImageModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        adapter = cls(model_name=payload.get("model_name", "m1_cnn_baseline"), params=payload.get("params", {}))
        model_config = payload.get("model_config", {})
        encoder = str(model_config.get("encoder", "efficientnet_b0"))
        pretrained = False
        adapter.backbone, adapter.feature_dim = build_cnn_backbone(encoder, pretrained=pretrained)
        dropout = float(model_config.get("dropout", adapter.params.get("dropout", 0.3)))
        adapter.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(adapter.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
        )
        model = nn.Sequential(adapter.backbone, adapter.head)
        model.load_state_dict(payload["state_dict"])
        adapter.best_state = payload["state_dict"]
        adapter.best_thresholds = payload.get("thresholds", {})
        adapter.logs = payload.get("logs", [])
        adapter.is_fitted = True
        return adapter

    def _predict_loader(self, loader: DataLoader, batch: ModelBatch) -> PredictionResult:
        model = nn.Sequential(self.backbone, self.head).to(self.device, non_blocking=True)
        model.eval()
        all_scores: list[np.ndarray] = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device, non_blocking=True)
                features = self.backbone(images)
                logits = self.head(features).squeeze(1)
                probs = torch.sigmoid(logits)
                all_scores.append(probs.detach().cpu().numpy())
        scores = np.concatenate(all_scores)
        labels = np.asarray(batch.labels).astype(int) if batch.labels is not None else None
        return PredictionResult(sample_ids=batch.sample_ids, scores=scores, labels=labels)


# ---------------------------------------------------------------------------
# M3: Transformer Image Model (ViT / Swin)
# ---------------------------------------------------------------------------

class TransformerImageModel(BaseModelAdapter):
    """ViT or Swin Transformer image baseline with focal loss."""

    model_type = "image_transformer"

    def __init__(self, *, model_name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, params=params)
        self.backbone: nn.Module | None = None
        self.head: nn.Module | None = None
        self.device = torch.device(str(self.params.get("device", "cpu")))
        self.feature_dim: int = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_thresholds: dict[str, float] = {}
        self.logs: list[dict[str, float]] = []

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "TransformerImageModel":
        if train.image_paths is None:
            raise ValueError("TransformerImageModel.fit requires ModelBatch.image_paths.")
        if train.labels is None:
            raise ValueError("TransformerImageModel.fit requires labels.")

        encoder = str(self.params.get("encoder", "swin_tiny_patch4_window7_224"))
        pretrained = bool(self.params.get("pretrained", True))
        image_size = int(self.params.get("image_size", 384))
        epochs = int(self.params.get("epochs", 15))
        patience = int(self.params.get("patience", 4))
        lr = float(self.params.get("learning_rate", 5e-5))
        weight_decay = float(self.params.get("weight_decay", 0.05))
        batch_size = int(self.params.get("batch_size", 16))
        dropout = float(self.params.get("dropout", 0.3))
        focal_alpha = float(self.params.get("focal_alpha", 0.25))
        focal_gamma = float(self.params.get("focal_gamma", 2.0))
        out_dir = Path(str(self.params.get("out_dir", "data/artifacts/trained_models/m3_transformer_baseline/fold0"))).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        self.backbone, self.feature_dim = build_transformer_backbone(encoder, pretrained=pretrained, image_size=image_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
        )
        model = nn.Sequential(self.backbone, self.head).to(self.device, non_blocking=True)

        y_train = np.asarray(train.labels).astype(int)
        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=self.device)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_ds = LesionImageDataset(train.image_paths, y_train, transform=train_transforms(image_size))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **_loader_kwargs(self.params, self.device))

        val_loader = None
        if valid is not None and valid.image_paths is not None and valid.labels is not None:
            y_val = np.asarray(valid.labels).astype(int)
            val_ds = LesionImageDataset(valid.image_paths, y_val, transform=eval_transforms(image_size))
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(self.params, self.device))

        best_metric = -float("inf")
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                features = self.backbone(images)
                logits = self.head(features).squeeze(1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1
            scheduler.step()

            row: dict[str, float] = {"epoch": float(epoch), "train_loss": epoch_loss / max(n_batches, 1), "lr": float(optimizer.param_groups[0]["lr"])}

            if val_loader is not None:
                val_result = self._predict_loader(val_loader, valid)
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
                self.best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                self.best_thresholds = {
                    "youden": float(row.get("threshold_youden", 0.5)),
                    "sensitivity_at_least_0_90": float(row.get("threshold_sensitivity_0_90", 0.5)),
                }
                torch.save({"state_dict": self.best_state, "model_config": {"encoder": encoder, "feature_dim": self.feature_dim, "dropout": dropout, "image_size": image_size}}, out_dir / "best.ckpt")
            else:
                bad_epochs += 1

            _log_epoch(self.model_name, epoch, epochs, row, patience - bad_epochs, best_metric)

            if bad_epochs >= patience:
                break

        if self.best_state:
            model.load_state_dict(self.best_state)
        self.is_fitted = True
        return self

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        if self.backbone is None or self.head is None:
            raise RuntimeError("Model is not fitted.")
        if batch.image_paths is None:
            raise ValueError("TransformerImageModel.predict_proba requires ModelBatch.image_paths.")

        image_size = int(self.params.get("image_size", 384))
        batch_size = int(self.params.get("batch_size", 16))
        labels = np.asarray(batch.labels).astype(int) if batch.labels is not None else None
        ds = LesionImageDataset(batch.image_paths, labels, transform=eval_transforms(image_size))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(self.params, self.device))

        model = nn.Sequential(self.backbone, self.head).to(self.device, non_blocking=True)
        model.eval()
        all_scores: list[np.ndarray] = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device, non_blocking=True)
                features = self.backbone(images)
                logits = self.head(features).squeeze(1)
                probs = torch.sigmoid(logits)
                all_scores.append(probs.detach().cpu().numpy())
        scores = np.concatenate(all_scores)
        return PredictionResult(sample_ids=batch.sample_ids, scores=scores, labels=labels)

    def save(self, path: str | Path) -> None:
        if self.backbone is None or self.head is None:
            raise RuntimeError("Cannot save before model is fitted.")
        payload = {
            "model_name": self.model_name,
            "params": self.params,
            "state_dict": self.best_state or {k: v.detach().cpu() for k, v in nn.Sequential(self.backbone, self.head).state_dict().items()},
            "model_config": {
                "encoder": str(self.params.get("encoder", "swin_tiny_patch4_window7_224")),
                "feature_dim": self.feature_dim,
                "dropout": float(self.params.get("dropout", 0.3)),
                "image_size": int(self.params.get("image_size", 384)),
            },
            "thresholds": self.best_thresholds,
            "logs": self.logs,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "TransformerImageModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        adapter = cls(model_name=payload.get("model_name", "m3_transformer_baseline"), params=payload.get("params", {}))
        model_config = payload.get("model_config", {})
        encoder = str(model_config.get("encoder", "swin_tiny_patch4_window7_224"))
        image_size = int(model_config.get("image_size", adapter.params.get("image_size", 384)))
        adapter.backbone, adapter.feature_dim = build_transformer_backbone(encoder, pretrained=False, image_size=image_size)
        dropout = float(model_config.get("dropout", adapter.params.get("dropout", 0.3)))
        adapter.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(adapter.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
        )
        model = nn.Sequential(adapter.backbone, adapter.head)
        model.load_state_dict(payload["state_dict"])
        adapter.best_state = payload["state_dict"]
        adapter.best_thresholds = payload.get("thresholds", {})
        adapter.logs = payload.get("logs", [])
        adapter.is_fitted = True
        return adapter

    def _predict_loader(self, loader: DataLoader, batch: ModelBatch) -> PredictionResult:
        model = nn.Sequential(self.backbone, self.head).to(self.device, non_blocking=True)
        model.eval()
        all_scores: list[np.ndarray] = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device, non_blocking=True)
                features = self.backbone(images)
                logits = self.head(features).squeeze(1)
                probs = torch.sigmoid(logits)
                all_scores.append(probs.detach().cpu().numpy())
        scores = np.concatenate(all_scores)
        labels = np.asarray(batch.labels).astype(int) if batch.labels is not None else None
        return PredictionResult(sample_ids=batch.sample_ids, scores=scores, labels=labels)


# ---------------------------------------------------------------------------
# Shared helpers
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
