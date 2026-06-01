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
    """Loads images and labels for end-to-end encoder training."""

    def __init__(self, image_paths: list[str], labels: np.ndarray | None = None, transform: Any | None = None) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Any, float, int]:
        from PIL import Image

        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = float(self.labels[idx]) if self.labels is not None else 0.0
        return img, label, idx


def _eval_transforms(image_size: int = 384) -> Any:
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _train_transforms(image_size: int = 384) -> Any:
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
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
        freeze_encoder = bool(self.params.get("freeze_encoder", False))
        unfreeze_after = int(self.params.get("unfreeze_after", 5))
        image_size = int(self.params.get("image_size", 384))
        epochs = int(self.params.get("epochs", 30))
        patience = int(self.params.get("patience", 10))
        lr = float(self.params.get("learning_rate", 2e-4))
        encoder_lr_scale = float(self.params.get("encoder_lr_scale", 0.1))
        weight_decay = float(self.params.get("weight_decay", 1e-4))
        batch_size = int(self.params.get("batch_size", 32))
        dropout = float(self.params.get("dropout", 0.3))
        use_text = bool(self.params.get("use_text_prompts", False))
        max_pos_weight = float(self.params.get("max_pos_weight", 100.0))
        warmup_epochs = int(self.params.get("warmup_epochs", 5))
        grad_clip_norm = float(self.params.get("grad_clip_norm", 1.0))
        out_dir = Path(str(self.params.get("out_dir", "data/artifacts/trained_models/m2_monet_feature_baseline/fold0"))).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build encoder (initially frozen if freeze_encoder=True, will unfreeze later)
        import timm
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.encoder.num_features
        self.encoder.to(self.device, non_blocking=True)

        # Freeze all encoder params initially
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Text prompt features
        self.text_feature_dim = 0
        if use_text and train.metadata is not None:
            self.text_feature_dim = 64

        total_dim = self.feature_dim + self.text_feature_dim
        self.head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        ).to(self.device, non_blocking=True)

        y_train = np.asarray(train.labels).astype(int)

        # Pre-extract text features (static, no augmentation)
        train_text_features = self._extract_text_features(train) if self.text_feature_dim > 0 else None
        train_text_tensor = torch.tensor(train_text_features, dtype=torch.float32) if train_text_features is not None else None

        val_text_features = None
        val_text_tensor = None
        if valid is not None and valid.metadata is not None and self.text_feature_dim > 0:
            val_text_features = self._extract_text_features(valid)
            val_text_tensor = torch.tensor(val_text_features, dtype=torch.float32) if val_text_features is not None else None

        # Build image datasets with augmentation
        train_img_ds = ImageFeatureDataset(train.image_paths, labels=y_train, transform=_train_transforms(image_size))
        train_img_loader = DataLoader(train_img_ds, batch_size=batch_size, shuffle=True, **_loader_kwargs(self.params, self.device))

        val_loader = None
        val_labels = None
        if valid is not None and valid.image_paths is not None and valid.labels is not None:
            val_labels = np.asarray(valid.labels).astype(int)
            val_img_ds = ImageFeatureDataset(valid.image_paths, labels=val_labels, transform=_eval_transforms(image_size))
            val_loader = DataLoader(val_img_ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(self.params, self.device))

        # Loss with capped pos_weight
        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
        raw_pw = negatives / max(positives, 1.0)
        capped_pw = min(raw_pw, max_pos_weight)
        pos_weight = torch.tensor([capped_pw], dtype=torch.float32, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer: head only initially, encoder added when unfrozen
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=weight_decay)
        # Initial warmup + cosine for head-only phase
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs],
        )

        encoder_unfrozen = False
        best_metric = -float("inf")
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            # Progressive unfreezing: unfreeze encoder after unfreeze_after epochs
            if not encoder_unfrozen and not freeze_encoder and epoch > unfreeze_after:
                for param in self.encoder.parameters():
                    param.requires_grad = True
                # Rebuild optimizer with encoder params at lower lr
                optimizer = torch.optim.AdamW([
                    {"params": self.encoder.parameters(), "lr": lr * encoder_lr_scale},
                    {"params": self.head.parameters(), "lr": lr},
                ], weight_decay=weight_decay)
                # Rebuild scheduler with warmup for unfrozen phase
                remaining = epochs - epoch + 1
                warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=min(3, remaining // 3))
                cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining - min(3, remaining // 3))
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[min(3, remaining // 3)],
                )
                encoder_unfrozen = True

            self.encoder.train()
            self.head.train()
            epoch_loss = 0.0
            n_batches = 0

            for images, y_batch, indices in train_img_loader:
                images = images.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                img_features = self.encoder(images)
                if train_text_tensor is not None:
                    text_batch = train_text_tensor[indices].to(self.device, non_blocking=True)
                    combined = torch.cat([img_features, text_batch], dim=1)
                else:
                    combined = img_features
                logits = self.head(combined).squeeze(1)
                loss = criterion(logits, y_batch)
                loss.backward()
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.head.parameters()), grad_clip_norm)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1

            scheduler.step()

            row: dict[str, float] = {"epoch": float(epoch), "train_loss": epoch_loss / max(n_batches, 1), "lr": float(optimizer.param_groups[0]["lr"])}

            if val_loader is not None and val_labels is not None:
                self.encoder.eval()
                self.head.eval()
                all_val_scores: list[np.ndarray] = []
                with torch.no_grad():
                    for val_images, _, val_indices in val_loader:
                        val_images = val_images.to(self.device, non_blocking=True)
                        val_img_feat = self.encoder(val_images)
                        if val_text_tensor is not None:
                            vt = val_text_tensor[val_indices].to(self.device, non_blocking=True)
                            val_combined = torch.cat([val_img_feat, vt], dim=1)
                        else:
                            val_combined = val_img_feat
                        val_logits = self.head(val_combined).squeeze(1)
                        val_scores = torch.sigmoid(val_logits).detach().cpu().numpy()
                        all_val_scores.append(val_scores)
                val_scores_all = np.concatenate(all_val_scores)
                val_threshold = youden_threshold(val_labels, val_scores_all) if len(np.unique(val_labels)) == 2 else 0.5
                high_threshold = threshold_at_min_sensitivity(val_labels, val_scores_all, min_sensitivity=0.90) if len(np.unique(val_labels)) == 2 else 0.5
                val_metrics = binary_classification_metrics(val_labels, val_scores_all, threshold=val_threshold)
                row.update({
                    "val_pauc_tpr80": float(val_metrics.get("pauc_tpr80", float("nan"))),
                    "val_auprc": float(val_metrics.get("auprc", float("nan"))),
                    "val_auroc": float(val_metrics.get("auroc", float("nan"))),
                    "val_brier": float(val_metrics.get("brier", float("nan"))),
                    "val_ece": float(expected_calibration_error(val_labels, val_scores_all)),
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
                    "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
                    "encoder": {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()},
                }
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
            self.head.load_state_dict(self.best_state["head"])
            self.encoder.load_state_dict(self.best_state["encoder"])
        self.is_fitted = True
        return self

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        if self.encoder is None or self.head is None:
            raise RuntimeError("Model is not fitted.")
        if batch.image_paths is None:
            raise ValueError("MonetFeatureModel.predict_proba requires ModelBatch.image_paths.")

        image_size = int(self.params.get("image_size", 384))
        batch_size = int(self.params.get("batch_size", 32))
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
        head_state = self.best_state.get("head", {k: v.detach().cpu() for k, v in self.head.state_dict().items()}) if self.best_state else {k: v.detach().cpu() for k, v in self.head.state_dict().items()}
        encoder_state = self.best_state.get("encoder", {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()}) if self.best_state else {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()}
        payload = {
            "model_name": self.model_name,
            "params": self.params,
            "state_dict": {"head": head_state, "encoder": encoder_state},
            "encoder_state_dict": encoder_state,
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
        import timm

        encoder_state = payload.get("encoder_state_dict")
        adapter.encoder = timm.create_model(encoder_name, pretrained=(encoder_state is None), num_classes=0)
        adapter.feature_dim = adapter.encoder.num_features
        if encoder_state is not None:
            adapter.encoder.load_state_dict(encoder_state)
        adapter.text_feature_dim = int(model_config.get("text_feature_dim", 0))
        dropout = float(model_config.get("dropout", adapter.params.get("dropout", 0.3)))
        total_dim = adapter.feature_dim + adapter.text_feature_dim
        adapter.head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )
        # Load state dict (handle both old and new format)
        state_dict = payload["state_dict"]
        if isinstance(state_dict, dict) and "head" in state_dict and "encoder" in state_dict:
            adapter.head.load_state_dict(state_dict["head"])
            adapter.encoder.load_state_dict(state_dict["encoder"])
            adapter.best_state = state_dict
        else:
            # Old format: state_dict is head-only
            adapter.head.load_state_dict(state_dict)
            adapter.best_state = {"head": state_dict, "encoder": {k: v.detach().cpu() for k, v in adapter.encoder.state_dict().items()}}
        adapter.best_thresholds = payload.get("thresholds", {})
        adapter.logs = payload.get("logs", [])
        adapter.is_fitted = True
        adapter.device = torch.device("cpu")
        return adapter

    def _extract_image_features(self, image_paths: list[str], image_size: int, batch_size: int) -> np.ndarray:
        ds = ImageFeatureDataset(image_paths, transform=_eval_transforms(image_size))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, **_loader_kwargs(self.params, self.device))
        self.encoder.eval()
        all_features: list[np.ndarray] = []
        with torch.no_grad():
            for images, _, _ in loader:
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
