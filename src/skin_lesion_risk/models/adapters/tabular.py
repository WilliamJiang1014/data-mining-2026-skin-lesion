from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from skin_lesion_risk.models.base import BaseModelAdapter, ModelBatch, PredictionResult


class ConstantRiskModel(BaseModelAdapter):
    """Small baseline used for smoke tests and pipeline checks."""

    model_type = "constant"

    def __init__(self, *, model_name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, params=params)
        self.constant_score = float(self.params.get("score", 0.5))

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "ConstantRiskModel":
        if train.labels is not None and len(train.labels) > 0:
            self.constant_score = float(np.mean(np.asarray(train.labels).astype(float)))
        self.is_fitted = True
        return self

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        scores = np.full(len(batch), self.constant_score, dtype=float)
        labels = np.asarray(batch.labels).astype(int) if batch.labels is not None else None
        return PredictionResult(sample_ids=batch.sample_ids, scores=scores, labels=labels)

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Serialization will be added when real training artifacts are introduced.")


class PlaceholderTabularModel(BaseModelAdapter):
    """Adapter slot for LightGBM/logistic regression/MLP metadata baselines."""

    model_type = "tabular"

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "PlaceholderTabularModel":
        raise NotImplementedError(
            "Implement tabular preprocessing and classifier here. Input comes from ModelBatch.metadata; "
            "patient_id/sample_id/lesion_id must stay excluded from ordinary features."
        )

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        raise NotImplementedError("Return PredictionResult with one malignant-risk score per sample.")

