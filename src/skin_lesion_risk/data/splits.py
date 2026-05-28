"""Patient-level fold helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_patient_level_folds(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    val_ratio: float = 0.1,
    seed: int = 2026,
    group_col: str = "patient_id",
    label_col: str = "target",
) -> pd.DataFrame:
    """Create long-format train/val/test assignments for every outer fold."""

    required = {"sample_id", group_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot make folds, missing columns: {sorted(missing)}")

    groups = df[group_col].fillna("UNKNOWN_PATIENT").astype(str).to_numpy()
    labels = df[label_col].astype(int).to_numpy()
    indices = np.arange(len(df))

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_splits = list(outer.split(indices, labels, groups))
    except Exception:
        from sklearn.model_selection import GroupKFold

        outer = GroupKFold(n_splits=n_splits)
        outer_splits = list(outer.split(indices, labels, groups))

    rows: list[pd.DataFrame] = []
    for fold, (train_val_idx, test_idx) in enumerate(outer_splits):
        train_idx, val_idx = split_validation_patients(
            df.iloc[train_val_idx],
            val_ratio=val_ratio,
            seed=seed + fold,
            group_col=group_col,
            label_col=label_col,
        )
        train_idx = df.iloc[train_val_idx].index.to_numpy()[train_idx]
        val_idx = df.iloc[train_val_idx].index.to_numpy()[val_idx]
        test_idx = df.index.to_numpy()[test_idx]

        for split, split_idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            part = df.loc[split_idx, ["sample_id", group_col, label_col]].copy()
            part["fold"] = fold
            part["split"] = split
            rows.append(part)

        assert_no_group_overlap(df, train_idx, val_idx, test_idx, group_col=group_col)

    return pd.concat(rows, ignore_index=True)


def split_validation_patients(
    train_val_df: pd.DataFrame,
    *,
    val_ratio: float,
    seed: int,
    group_col: str,
    label_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    groups = train_val_df[group_col].fillna("UNKNOWN_PATIENT").astype(str)
    patient_labels = train_val_df.groupby(groups)[label_col].max().astype(int)
    patients = patient_labels.index.to_numpy()
    labels = patient_labels.to_numpy()
    if len(patients) < 3:
        return np.arange(len(train_val_df)), np.array([], dtype=int)

    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_patient_idx, val_patient_idx = next(splitter.split(patients, labels))
    except Exception:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(len(patients))
        n_val = max(1, int(round(len(patients) * val_ratio)))
        val_patient_idx = shuffled[:n_val]
        train_patient_idx = shuffled[n_val:]

    train_patients = set(patients[train_patient_idx].tolist())
    val_patients = set(patients[val_patient_idx].tolist())
    row_groups = groups.to_numpy()
    train_idx = np.where(np.isin(row_groups, list(train_patients)))[0]
    val_idx = np.where(np.isin(row_groups, list(val_patients)))[0]
    return train_idx, val_idx


def assert_no_group_overlap(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    group_col: str,
) -> None:
    split_groups = [
        set(df.loc[idx, group_col].fillna("UNKNOWN_PATIENT").astype(str).tolist())
        for idx in (train_idx, val_idx, test_idx)
    ]
    if split_groups[0] & split_groups[1] or split_groups[0] & split_groups[2] or split_groups[1] & split_groups[2]:
        raise AssertionError("Patient/group leakage detected across train/val/test splits.")
