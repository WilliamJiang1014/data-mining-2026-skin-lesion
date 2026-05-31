from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from skin_lesion_risk.evaluation.metrics import MetricBundle, binary_classification_metrics


ArrayLike = Any
TableLike = Any


@dataclass
class ModelBatch:
    """Unified input contract for all candidate models.

    Each adapter consumes only the fields it needs. For example, tabular models
    use `metadata`, image models use `image_paths` or `images`, and graph models
    additionally use `graph`.
    """

    sample_ids: list[str]
    labels: ArrayLike | None = None
    image_paths: list[str | Path] | None = None
    images: ArrayLike | None = None
    metadata: TableLike | None = None
    raw_metadata: TableLike | None = None
    graph: Any | None = None
    groups: dict[str, list[Any]] = field(default_factory=dict)
    fold: int | None = None
    source: str | None = None

    def __len__(self) -> int:
        return len(self.sample_ids)


@dataclass
class PredictionResult:
    """Unified output contract for validation, test and external evaluation."""

    sample_ids: list[str]
    scores: np.ndarray
    labels: np.ndarray | None = None
    threshold: float | None = None
    metrics: MetricBundle | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_metrics(self, *, threshold: float = 0.5, threshold_rule: str = "fixed_0_5") -> "PredictionResult":
        if self.labels is None:
            return self
        values = binary_classification_metrics(self.labels, self.scores, threshold=threshold)
        self.threshold = threshold
        self.metrics = MetricBundle(values=values, threshold=threshold, threshold_rule=threshold_rule)
        return self

    def to_records(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        labels = self.labels.tolist() if self.labels is not None else [None] * len(self.sample_ids)
        for sample_id, score, label in zip(self.sample_ids, self.scores.tolist(), labels):
            rows.append({"sample_id": sample_id, "score": float(score), "label": label})
        return rows


@runtime_checkable
class LesionRiskModel(Protocol):
    """Protocol implemented by every model adapter."""

    model_name: str
    model_type: str

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "LesionRiskModel":
        ...

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        ...

    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    def load(cls, path: str | Path) -> "LesionRiskModel":
        ...


class BaseModelAdapter:
    """Base class for concrete adapters."""

    model_type = "base"

    def __init__(self, *, model_name: str, params: dict[str, Any] | None = None) -> None:
        self.model_name = model_name
        self.params = params or {}
        self.is_fitted = False

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "BaseModelAdapter":
        raise NotImplementedError(f"{self.__class__.__name__}.fit is not implemented yet.")

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        raise NotImplementedError(f"{self.__class__.__name__}.predict_proba is not implemented yet.")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}.save is not implemented yet.")

    @classmethod
    def load(cls, path: str | Path) -> "BaseModelAdapter":
        raise NotImplementedError(f"{cls.__name__}.load is not implemented yet.")

