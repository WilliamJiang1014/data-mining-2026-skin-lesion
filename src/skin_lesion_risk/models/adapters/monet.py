from __future__ import annotations

from skin_lesion_risk.models.base import BaseModelAdapter, ModelBatch, PredictionResult


class PlaceholderMonetFeatureModel(BaseModelAdapter):
    """Adapter slot for MONET or compatible visual-language feature baselines."""

    model_type = "monet_feature"

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "PlaceholderMonetFeatureModel":
        raise NotImplementedError(
            "Implement frozen feature extraction plus classifier head here. Store feature cache paths in "
            "the adapter metadata or experiment config."
        )

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        raise NotImplementedError("Return PredictionResult using cached or on-the-fly VLM features.")

