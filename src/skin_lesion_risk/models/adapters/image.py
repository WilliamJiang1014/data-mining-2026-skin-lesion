from __future__ import annotations

from skin_lesion_risk.models.base import BaseModelAdapter, ModelBatch, PredictionResult


class PlaceholderImageModel(BaseModelAdapter):
    """Adapter slot for CNN and vision Transformer baselines."""

    model_type = "image"

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "PlaceholderImageModel":
        raise NotImplementedError(
            "Implement image training here. Use ModelBatch.image_paths or ModelBatch.images and keep "
            "validation transforms deterministic."
        )

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        raise NotImplementedError("Return PredictionResult with one malignant-risk score per image.")

