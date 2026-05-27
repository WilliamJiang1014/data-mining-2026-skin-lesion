from __future__ import annotations

from skin_lesion_risk.models.base import BaseModelAdapter, ModelBatch, PredictionResult


class PlaceholderGraphMultimodalModel(BaseModelAdapter):
    """Adapter slot for LGKE-GNN style lesion graph models."""

    model_type = "graph_multimodal"

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "PlaceholderGraphMultimodalModel":
        raise NotImplementedError(
            "Implement graph model here. Consume ModelBatch.graph plus aligned image and metadata "
            "features; construct graph edges outside the model using training-fold-only statistics."
        )

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        raise NotImplementedError("Return PredictionResult for graph inference nodes.")

