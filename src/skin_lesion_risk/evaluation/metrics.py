from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MetricBundle:
    """Standard metrics emitted by every model."""

    values: dict[str, float]
    threshold: float | None = None
    threshold_rule: str | None = None
    subgroup_values: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "values": dict(self.values),
            "threshold": self.threshold,
            "threshold_rule": self.threshold_rule,
            "subgroup_values": self.subgroup_values,
        }


def binary_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute framework-independent binary metrics.

    AUROC/AUPRC are included when scikit-learn is installed. Confusion-matrix
    metrics are implemented locally so smoke tests do not require heavy ML deps.
    """

    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    pred = (s >= threshold).astype(int)

    tp = float(np.sum((pred == 1) & (y == 1)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))

    sensitivity = tp / (tp + fn) if tp + fn > 0 else float("nan")
    specificity = tn / (tn + fp) if tn + fp > 0 else float("nan")
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity > 0 else float("nan")
    brier = float(np.mean((s - y) ** 2)) if len(y) else float("nan")

    metrics = {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "fnr": fn / (fn + tp) if fn + tp > 0 else float("nan"),
        "brier": brier,
    }

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(np.unique(y)) == 2:
            metrics["auroc"] = float(roc_auc_score(y, s))
            metrics["auprc"] = float(average_precision_score(y, s))
            metrics["pauc_tpr80"] = partial_auc_high_sensitivity(y, s, min_tpr=0.80)
    except Exception:
        pass

    return metrics


def partial_auc_high_sensitivity(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    min_tpr: float = 0.80,
) -> float:
    """Compute normalized ROC partial AUC in the high-sensitivity region.

    The implementation integrates the ROC curve area where TPR is at least
    `min_tpr`, then normalizes by the largest possible rectangle in that band.
    It is intentionally dependency-light and stable for report generation.
    """

    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    if len(y) == 0 or len(np.unique(y)) != 2:
        return float("nan")

    try:
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y, s)
    except Exception:
        return float("nan")

    points: list[tuple[float, float]] = []
    for idx in range(len(fpr) - 1):
        x0, x1 = float(fpr[idx]), float(fpr[idx + 1])
        y0, y1 = float(tpr[idx]), float(tpr[idx + 1])
        if y0 >= min_tpr:
            points.append((x0, y0))
        if (y0 < min_tpr <= y1) or (y1 < min_tpr <= y0):
            ratio = (min_tpr - y0) / (y1 - y0) if y1 != y0 else 0.0
            points.append((x0 + ratio * (x1 - x0), min_tpr))
        if y1 >= min_tpr:
            points.append((x1, y1))

    if len(points) < 2:
        return 0.0
    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    raw_area = float(np.trapz(ys - min_tpr, xs))
    max_area = 1.0 - min_tpr
    return max(0.0, min(1.0, raw_area / max_area if max_area > 0 else float("nan")))


def subgroup_metric_gaps(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    groups: Mapping[str, Sequence[object]],
    *,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Compute max-min gaps for protected or clinical subgroups."""

    y = np.asarray(y_true)
    s = np.asarray(y_score)
    output: dict[str, dict[str, float]] = {}
    for group_name, group_values in groups.items():
        g = np.asarray(group_values)
        sensitivity_values: list[float] = []
        specificity_values: list[float] = []
        for value in sorted(set(g.tolist())):
            mask = g == value
            if np.sum(mask) == 0:
                continue
            metrics = binary_classification_metrics(y[mask], s[mask], threshold=threshold)
            sensitivity_values.append(metrics["sensitivity"])
            specificity_values.append(metrics["specificity"])
        output[group_name] = {
            "sensitivity_gap": _gap(sensitivity_values),
            "specificity_gap": _gap(specificity_values),
        }
    return output


def subgroup_full_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    groups: Mapping[str, Sequence[object]],
    *,
    threshold: float = 0.5,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return per-value metrics for report fairness tables."""

    y = np.asarray(y_true)
    s = np.asarray(y_score)
    output: dict[str, dict[str, dict[str, float]]] = {}
    for group_name, group_values in groups.items():
        g = np.asarray(group_values)
        output[group_name] = {}
        for value in sorted(set(g.tolist())):
            mask = g == value
            if np.sum(mask) == 0:
                continue
            values = binary_classification_metrics(y[mask], s[mask], threshold=threshold)
            output[group_name][str(value)] = values | {"n": float(np.sum(mask)), "positive": float(np.sum(y[mask]))}
    return output


def _gap(values: Sequence[float]) -> float:
    finite = [v for v in values if not np.isnan(v)]
    return float(max(finite) - min(finite)) if finite else float("nan")
