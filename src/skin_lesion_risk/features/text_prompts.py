from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def metadata_to_prompt(row: Mapping[str, Any], *, include_tbp_summary: bool = True) -> str:
    """Convert lesion metadata into a short clinical description for VLM features."""

    age = format_value(row.get("age"))
    sex = format_value(row.get("sex"))
    site = format_value(row.get("anatom_site"))
    size = format_numeric(row.get("size_mm"), suffix=" mm")
    parts = ["Clinical skin lesion image"]
    details = []
    if age:
        details.append(f"patient age approximately {age}")
    if sex:
        details.append(f"sex {sex}")
    if site:
        details.append(f"anatomical site {site}")
    if size:
        details.append(f"long diameter {size}")
    if details:
        parts.append("with " + ", ".join(details))
    if include_tbp_summary:
        tbp = tbp_summary(row)
        if tbp:
            parts.append(tbp)
    return ". ".join(parts) + "."


def build_prompts(metadata: pd.DataFrame, *, include_tbp_summary: bool = True) -> list[str]:
    return [metadata_to_prompt(row, include_tbp_summary=include_tbp_summary) for row in metadata.to_dict("records")]


def tbp_summary(row: Mapping[str, Any]) -> str:
    values = {
        "color variation": format_numeric(row.get("tbp_lv_color_std_mean")),
        "border irregularity": format_numeric(row.get("tbp_lv_norm_border")),
        "color irregularity": format_numeric(row.get("tbp_lv_norm_color")),
        "asymmetry": format_numeric(row.get("tbp_lv_symm_2axis")),
    }
    present = [f"{name} {value}" for name, value in values.items() if value]
    return "TBP visual metadata: " + ", ".join(present) if present else ""


def format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none", "unk", "unknown"} else text.replace("_", " ")


def format_numeric(value: Any, *, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:g}{suffix}"

