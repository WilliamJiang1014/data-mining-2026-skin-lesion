from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd

from skin_lesion_risk.models.base import BaseModelAdapter, ModelBatch, PredictionResult

ID_COLUMNS = {"patient_id", "sample_id", "lesion_id", "image_id", "isic_id"}


def balanced_sample_weight(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    if len(labels) == 0:
        return np.asarray([], dtype=float)
    positives = max(float(np.sum(labels == 1)), 1.0)
    negatives = max(float(np.sum(labels == 0)), 1.0)
    return np.where(labels == 1, len(labels) / (2.0 * positives), len(labels) / (2.0 * negatives)).astype(float)


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
        payload = {
            "model_name": self.model_name,
            "params": self.params,
            "constant_score": self.constant_score,
            "is_fitted": self.is_fitted,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "ConstantRiskModel":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        model = cls(model_name=payload.get("model_name", "constant"), params=payload.get("params", {}))
        model.constant_score = float(payload.get("constant_score", model.constant_score))
        model.is_fitted = bool(payload.get("is_fitted", True))
        return model


class LightGBMTabularModel(BaseModelAdapter):
    """Fold-scoped metadata baseline with LightGBM and a sklearn fallback."""

    model_type = "tabular_lgbm"

    def __init__(self, *, model_name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, params=params)
        self.estimator: Any | None = None
        self.feature_names: list[str] = []
        self.backend_used: str | None = None

    def fit(self, train: ModelBatch, valid: ModelBatch | None = None) -> "LightGBMTabularModel":
        if train.metadata is None:
            raise ValueError("LightGBMTabularModel.fit requires ModelBatch.metadata.")
        if train.labels is None:
            raise ValueError("LightGBMTabularModel.fit requires labels.")
        x_train = self._metadata_frame(train.metadata, fit=True)
        y_train = np.asarray(train.labels).astype(int)
        self.estimator, self.backend_used = self._build_estimator()
        fit_kwargs: dict[str, Any] = {}
        if self.backend_used == "lightgbm" and valid is not None and valid.metadata is not None and valid.labels is not None:
            x_valid = self._metadata_frame(valid.metadata, fit=False)
            fit_kwargs["eval_set"] = [(x_valid, np.asarray(valid.labels).astype(int))]
            fit_kwargs["eval_metric"] = self.params.get("eval_metric", "auc")
        else:
            fit_kwargs["sample_weight"] = balanced_sample_weight(y_train)
        self.estimator.fit(x_train, y_train, **fit_kwargs)
        self.is_fitted = True
        return self

    def predict_proba(self, batch: ModelBatch) -> PredictionResult:
        if self.estimator is None:
            raise RuntimeError("Model is not fitted.")
        if batch.metadata is None:
            raise ValueError("LightGBMTabularModel.predict_proba requires ModelBatch.metadata.")
        x = self._metadata_frame(batch.metadata, fit=False)
        if hasattr(self.estimator, "predict_proba"):
            scores = np.asarray(self.estimator.predict_proba(x)[:, 1], dtype=float)
        else:
            scores = np.clip(np.asarray(self.estimator.predict(x), dtype=float), 0.0, 1.0)
        labels = np.asarray(batch.labels).astype(int) if batch.labels is not None else None
        return PredictionResult(sample_ids=batch.sample_ids, scores=scores, labels=labels)

    def save(self, path: str | Path) -> None:
        if self.estimator is None:
            raise RuntimeError("Cannot save before model is fitted.")
        payload = {
            "model_name": self.model_name,
            "params": self.params,
            "estimator": self.estimator,
            "feature_names": self.feature_names,
            "backend_used": self.backend_used,
            "is_fitted": self.is_fitted,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMTabularModel":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        model = cls(model_name=payload.get("model_name", "m0_lightgbm"), params=payload.get("params", {}))
        model.estimator = payload["estimator"]
        model.feature_names = list(payload.get("feature_names", []))
        model.backend_used = payload.get("backend_used")
        model.is_fitted = bool(payload.get("is_fitted", True))
        return model

    def _build_estimator(self) -> tuple[Any, str]:
        backend = str(self.params.get("backend", self.params.get("preferred_backend", "lightgbm"))).lower()
        if backend in {"lightgbm", "lgbm"}:
            try:
                from lightgbm import LGBMClassifier

                estimator = LGBMClassifier(
                    n_estimators=int(self.params.get("n_estimators", 300)),
                    learning_rate=float(self.params.get("learning_rate", 0.03)),
                    num_leaves=int(self.params.get("num_leaves", 31)),
                    max_depth=int(self.params.get("max_depth", -1)),
                    subsample=float(self.params.get("subsample", 0.9)),
                    colsample_bytree=float(self.params.get("colsample_bytree", 0.9)),
                    class_weight=self.params.get("class_weight", "balanced"),
                    random_state=int(self.params.get("seed", 2026)),
                    n_jobs=int(self.params.get("n_jobs", -1)),
                )
                return estimator, "lightgbm"
            except Exception:
                if not bool(self.params.get("allow_sklearn_fallback", True)):
                    raise

        from sklearn.ensemble import HistGradientBoostingClassifier

        estimator = HistGradientBoostingClassifier(
            learning_rate=float(self.params.get("learning_rate", 0.03)),
            max_iter=int(self.params.get("n_estimators", self.params.get("max_iter", 300))),
            max_leaf_nodes=int(self.params.get("num_leaves", 31)),
            l2_regularization=float(self.params.get("l2_regularization", 0.0)),
            random_state=int(self.params.get("seed", 2026)),
        )
        return estimator, "sklearn_hist_gradient_boosting"

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

