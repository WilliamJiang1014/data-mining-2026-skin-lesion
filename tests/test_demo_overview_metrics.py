from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE4 = ROOT / "reports" / "tables" / "table4.csv"
M5_SNIPPET = ROOT / "demo" / "assets" / "m5_summary_snippet.json"


def _parse_pm_mean(value: str) -> float:
    text = str(value).strip()
    return float(text.split("±")[0].strip())


def _delta_m5_vs_m4_pp(table4_df: pd.DataFrame, m5_summary: dict[str, object]) -> float:
    m4 = table4_df[table4_df["Model"] == "Multimodal Fusion"]
    if m4.empty:
        raise AssertionError("table4.csv missing Multimodal Fusion")
    m4_pauc = _parse_pm_mean(str(m4.iloc[0]["pAUC"])) * 100
    m5_pauc = float(m5_summary["pauc_tpr80_pct"])
    return m5_pauc - m4_pauc


def test_overview_delta_m5_vs_m4_is_expected() -> None:
    table4_df = pd.read_csv(TABLE4)
    m5_summary = json.loads(M5_SNIPPET.read_text(encoding="utf-8"))
    delta_pp = _delta_m5_vs_m4_pp(table4_df, m5_summary)
    assert abs(delta_pp - 4.01) <= 0.05, f"unexpected delta: {delta_pp:.4f}pp"
