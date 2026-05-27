from __future__ import annotations

import numpy as np


def threshold_at_min_sensitivity(y_true, y_score, *, min_sensitivity: float = 0.90) -> float:
    """Return the highest-specificity threshold satisfying a minimum sensitivity."""

    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    thresholds = np.unique(s)
    best_threshold = float(np.min(thresholds)) if len(thresholds) else 0.5
    best_specificity = -1.0
    for threshold in thresholds:
        pred = (s >= threshold).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        sensitivity = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        if sensitivity >= min_sensitivity and specificity > best_specificity:
            best_threshold = float(threshold)
            best_specificity = float(specificity)
    return best_threshold


def youden_threshold(y_true, y_score) -> float:
    """Return threshold maximizing sensitivity + specificity - 1."""

    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    thresholds = np.unique(s)
    best_threshold = float(np.median(s)) if len(s) else 0.5
    best_youden = -float("inf")
    for threshold in thresholds:
        pred = (s >= threshold).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        sensitivity = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        score = sensitivity + specificity - 1.0
        if score > best_youden:
            best_youden = float(score)
            best_threshold = float(threshold)
    return best_threshold

