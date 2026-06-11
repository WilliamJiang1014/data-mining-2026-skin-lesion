from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "demo" / "assets"
TABLES = ROOT / "reports" / "tables"

REQUIRED_TABLES = ("table4.csv", "main_results.csv", "split_stats.csv")
REQUIRED_ASSET_FILES = (
    "sample_predictions_manifest.json",
    "m5_summary_snippet.json",
    "fairness_subgroup.csv",
    "graph_ablation_summary.csv",
    "pad_adaptation_summary.csv",
    "figures/ml_pipeline.png",
    "figures/main_results_forest.png",
    "figures/graph_ablation_metrics.png",
    "figures/pad_domain_shift.png",
)


def test_reports_tables_present() -> None:
    for name in REQUIRED_TABLES:
        path = TABLES / name
        assert path.exists(), f"missing {path.relative_to(ROOT)}"
        assert path.stat().st_size > 0, f"empty {path.relative_to(ROOT)}"


def test_main_results_includes_m5_folds() -> None:
    df = pd.read_csv(TABLES / "main_results.csv")
    m5 = df[df["model_name"] == "m5_lgke_gnn"]
    assert len(m5) == 5, "expected 5 M5 fold rows in main_results.csv"


def test_demo_asset_files_present() -> None:
    for rel in REQUIRED_ASSET_FILES:
        path = ASSETS / rel
        assert path.exists(), f"missing {path.relative_to(ROOT)}"


def test_demo_manifest_predictions_and_metrics_exist() -> None:
    manifest = json.loads((ASSETS / "sample_predictions_manifest.json").read_text(encoding="utf-8"))
    models = manifest.get("models", [])
    assert models, "manifest has no models"
    for item in models:
        pred_rel = str(item.get("predictions", "")).strip()
        metrics_rel = str(item.get("metrics", "")).strip()
        assert pred_rel and (ASSETS / pred_rel).exists(), f"missing predictions for {item.get('key')}"
        assert metrics_rel and (ASSETS / metrics_rel).exists(), f"missing metrics for {item.get('key')}"
