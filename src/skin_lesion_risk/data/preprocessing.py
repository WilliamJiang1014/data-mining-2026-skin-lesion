"""Fold-scoped metadata preprocessing.

Preprocessors are fitted on training folds only, then applied to validation,
test and external validation data without refitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import pickle

import numpy as np
import pandas as pd


DEFAULT_NUMERIC_COLUMNS = (
    "age",
    "size_mm",
    "tbp_lv_A",
    "tbp_lv_Aext",
    "tbp_lv_B",
    "tbp_lv_Bext",
    "tbp_lv_C",
    "tbp_lv_Cext",
    "tbp_lv_H",
    "tbp_lv_Hext",
    "tbp_lv_L",
    "tbp_lv_Lext",
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaA",
    "tbp_lv_deltaB",
    "tbp_lv_deltaL",
    "tbp_lv_deltaLB",
    "tbp_lv_deltaLBnorm",
    "tbp_lv_eccentricity",
    "tbp_lv_minorAxisMM",
    "tbp_lv_nevi_confidence",
    "tbp_lv_norm_border",
    "tbp_lv_norm_color",
    "tbp_lv_perimeterMM",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_stdL",
    "tbp_lv_stdLExt",
    "tbp_lv_symm_2axis",
    "tbp_lv_symm_2axis_angle",
    "tbp_lv_x",
    "tbp_lv_y",
    "tbp_lv_z",
)

DEFAULT_CATEGORICAL_COLUMNS = ("sex", "anatom_site", "tbp_lv_location", "tbp_lv_location_simple")


@dataclass
class FoldTabularPreprocessor:
    """Simple serializable tabular preprocessor for metadata and graph features."""

    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    rare_min_count: int = 20
    medians: dict[str, float] = field(default_factory=dict)
    centers: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    category_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    age_bins: list[float] = field(default_factory=list)
    size_bins: list[float] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FoldTabularPreprocessor":
        self.numeric_columns = [c for c in self.numeric_columns if c in df.columns]
        self.categorical_columns = [c for c in self.categorical_columns if c in df.columns]

        for column in self.numeric_columns:
            values = pd.to_numeric(df[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            q1 = float(values.quantile(0.25)) if values.notna().any() else median
            q3 = float(values.quantile(0.75)) if values.notna().any() else median
            scale = q3 - q1
            self.medians[column] = median
            self.centers[column] = median
            self.scales[column] = scale if np.isfinite(scale) and scale > 1e-8 else 1.0

        for column in self.categorical_columns:
            values = self._clean_category(df[column])
            counts = values.value_counts(dropna=False)
            kept = sorted([str(v) for v, n in counts.items() if n >= self.rare_min_count and str(v) != "UNK"])
            tokens = ["UNK", "RARE"] + kept
            self.category_maps[column] = {token: idx for idx, token in enumerate(tokens)}

        self.age_bins = self._fit_quantile_bins(df.get("age"), fallback=[0, 30, 40, 50, 60, 70, 200])
        self.size_bins = self._fit_quantile_bins(df.get("size_mm"), fallback=[0, 2, 4, 8, 20, 500])
        self.feature_names = self._build_feature_names()
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FoldTabularPreprocessor must be fitted before transform().")

        columns: dict[str, np.ndarray] = {}
        for column in self.numeric_columns:
            raw = pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.Series(np.nan, index=df.index)
            missing = raw.isna().astype(float).to_numpy()
            filled = raw.fillna(self.medians[column]).astype(float)
            scaled = ((filled - self.centers[column]) / self.scales[column]).to_numpy(dtype=float)
            columns[column] = scaled
            columns[f"{column}__missing"] = missing

        for column in self.categorical_columns:
            raw = self._clean_category(df[column]) if column in df.columns else pd.Series("UNK", index=df.index)
            mapping = self.category_maps[column]
            normalized = raw.map(lambda x: x if x in mapping and x != "UNK" else ("UNK" if x == "UNK" else "RARE"))
            for token in mapping:
                columns[f"{column}={token}"] = (normalized == token).astype(float).to_numpy()

        output = pd.DataFrame(columns, index=df.index)
        for name in self.feature_names:
            if name not in output:
                output[name] = 0.0
        return output[self.feature_names].astype("float32")

    def add_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["age_bin"] = bin_series(out.get("age"), self.age_bins, prefix="age")
        out["size_bin"] = bin_series(out.get("size_mm"), self.size_bins, prefix="size")
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    def save_schema(self, path: str | Path) -> None:
        payload = {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "rare_min_count": self.rare_min_count,
            "age_bins": self.age_bins,
            "size_bins": self.size_bins,
            "feature_names": self.feature_names,
            "category_maps": self.category_maps,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FoldTabularPreprocessor":
        with Path(path).open("rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, cls):
            raise TypeError(f"Unexpected preprocessor object in {path}: {type(loaded)!r}")
        return loaded

    def _build_feature_names(self) -> list[str]:
        names: list[str] = []
        for column in self.numeric_columns:
            names.append(column)
            names.append(f"{column}__missing")
        for column in self.categorical_columns:
            names.extend([f"{column}={token}" for token in self.category_maps[column]])
        return names

    @staticmethod
    def _clean_category(series: pd.Series) -> pd.Series:
        return series.fillna("UNK").astype(str).str.strip().replace({"": "UNK", "nan": "UNK", "None": "UNK"})

    @staticmethod
    def _fit_quantile_bins(series: pd.Series | None, *, fallback: list[float]) -> list[float]:
        if series is None:
            return fallback
        values = pd.to_numeric(series, errors="coerce").dropna()
        if len(values) < 10:
            return fallback
        quantiles = np.quantile(values.to_numpy(dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
        bins = sorted(set(float(x) for x in quantiles if np.isfinite(x)))
        if len(bins) < 3:
            return fallback
        bins[0] = min(bins[0], fallback[0])
        bins[-1] = max(bins[-1], fallback[-1])
        return bins


def default_preprocessor_for(df: pd.DataFrame, *, rare_min_count: int = 20) -> FoldTabularPreprocessor:
    return FoldTabularPreprocessor(
        numeric_columns=[c for c in DEFAULT_NUMERIC_COLUMNS if c in df.columns],
        categorical_columns=[c for c in DEFAULT_CATEGORICAL_COLUMNS if c in df.columns],
        rare_min_count=rare_min_count,
    )


def bin_series(series: pd.Series | None, bins: list[float], *, prefix: str) -> pd.Series:
    if series is None:
        return pd.Series([f"{prefix}=UNK"])
    values = pd.to_numeric(series, errors="coerce")
    labels: list[str] = []
    for value in values:
        if pd.isna(value):
            labels.append(f"{prefix}=UNK")
            continue
        label = f"{prefix}=>={bins[-2]:g}"
        for left, right in zip(bins[:-1], bins[1:]):
            if left <= float(value) < right:
                label = f"{prefix}={left:g}-{right:g}"
                break
        labels.append(label)
    return pd.Series(labels, index=values.index)
