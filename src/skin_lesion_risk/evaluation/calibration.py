from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true, y_score, *, n_bins: int = 10) -> float:
    y = np.asarray(y_true).astype(float)
    s = np.asarray(y_score).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (s >= left) & (s < right if right < 1.0 else s <= right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(s[mask]))
        accuracy = float(np.mean(y[mask]))
        ece += float(np.mean(mask)) * abs(confidence - accuracy)
    return ece

