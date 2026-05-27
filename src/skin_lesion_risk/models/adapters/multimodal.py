from __future__ import annotations

from skin_lesion_risk.models.base import BaseModelAdapter, ModelBatch, PredictionResult


class PlaceholderMultimodalModel(BaseModelAdapter):
    """Adapter slot for image plus metadata fusion models."""

    model_type = "multimodal"

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "PlaceholderMultimodalModel":
        raise NotImplementedError(
            "Implement fusion model here. Consume ModelBatch.image_paths/images and ModelBatch.metadata "
            "through the same sample ordering."
        )

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        raise NotImplementedError("Return PredictionResult with fused malignant-risk scores.")

