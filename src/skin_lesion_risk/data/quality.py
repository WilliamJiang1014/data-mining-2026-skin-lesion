"""Data quality audits for images and manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image


def validate_image_paths(
    df: pd.DataFrame,
    *,
    image_col: str = "image_path",
    mode: str = "sample",
    sample_size: int = 200,
    seed: int = 2026,
) -> list[dict[str, str]]:
    """Return image path failures without mutating the manifest."""

    if image_col not in df.columns or mode == "none":
        return []
    check_df = df
    if mode == "sample" and len(df) > sample_size:
        check_df = df.sample(n=sample_size, random_state=seed)
    failures: list[dict[str, str]] = []
    for _, row in check_df.iterrows():
        path = Path(str(row[image_col]))
        sample_id = str(row.get("sample_id", path.stem))
        if not path.exists():
            failures.append({"sample_id": sample_id, "image_path": str(path), "error": "missing"})
            continue
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            failures.append({"sample_id": sample_id, "image_path": str(path), "error": type(exc).__name__})
    return failures


def manifest_quality_summary(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    patient_col: str = "patient_id",
    extra_lines: Iterable[str] = (),
) -> str:
    n = len(df)
    positives = int(pd.to_numeric(df[target_col], errors="coerce").fillna(0).sum()) if target_col in df else 0
    patients = int(df[patient_col].nunique(dropna=True)) if patient_col in df else 0
    lines = [
        "# Data Quality Report",
        "",
        f"- n_samples: {n}",
        f"- n_patients: {patients}",
        f"- n_positive: {positives}",
        f"- positive_rate: {positives / n if n else 0:.6f}",
        "",
        "## Missing Rate By Column",
        "",
    ]
    for column, rate in df.isna().mean().sort_values(ascending=False).items():
        lines.append(f"- {column}: {rate:.6f}")
    lines.extend(["", "## Notes", ""])
    lines.extend([f"- {line}" for line in extra_lines])
    return "\n".join(lines) + "\n"
