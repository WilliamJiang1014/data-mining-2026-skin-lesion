from __future__ import annotations

import numpy as np
import pandas as pd

from skin_lesion_risk.data.preprocessing import default_preprocessor_for
from skin_lesion_risk.data.splits import make_patient_level_folds
from skin_lesion_risk.evaluation.metrics import partial_auc_high_sensitivity


def test_patient_folds_do_not_overlap_groups() -> None:
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(30)],
            "patient_id": [f"p{i // 2}" for i in range(30)],
            "target": [(i // 2) % 2 for i in range(30)],
        }
    )
    folds = make_patient_level_folds(df, n_splits=3, val_ratio=0.2, seed=7)
    for fold in sorted(folds["fold"].unique()):
        groups = {
            split: set(folds[(folds["fold"] == fold) & (folds["split"] == split)]["patient_id"])
            for split in ("train", "val", "test")
        }
        assert not groups["train"] & groups["val"]
        assert not groups["train"] & groups["test"]
        assert not groups["val"] & groups["test"]


def test_preprocessor_fits_train_schema_and_transforms_unknown_categories() -> None:
    train = pd.DataFrame(
        {
            "age": [40, 50, np.nan, 70],
            "size_mm": [1.0, 2.0, 3.0, np.nan],
            "sex": ["male", "female", "female", None],
            "anatom_site": ["torso", "head", "torso", "leg"],
        }
    )
    valid = pd.DataFrame(
        {
            "age": [60],
            "size_mm": [4.0],
            "sex": ["other"],
            "anatom_site": ["unknown_site"],
        }
    )
    preprocessor = default_preprocessor_for(train, rare_min_count=2).fit(train)
    transformed = preprocessor.transform(valid)
    assert list(transformed.columns) == preprocessor.feature_names
    assert transformed.shape[0] == 1
    assert np.isfinite(transformed.to_numpy()).all()


def test_partial_auc_high_sensitivity_is_bounded() -> None:
    value = partial_auc_high_sensitivity([0, 0, 1, 1], [0.1, 0.3, 0.8, 0.9], min_tpr=0.8)
    assert 0.0 <= value <= 1.0
