from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

TABLE4_METRICS = ["pAUC", "AUPRC", "AUROC", "Sens.", "Spec."]
PLUS_MINUS_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-)\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_plus_minus(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        return float(value), None

    text = str(value).strip()
    if not text:
        return None, None
    if text.upper() == "NA":
        return None, None

    try:
        return float(text), None
    except ValueError:
        pass

    match = PLUS_MINUS_PATTERN.match(text)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def load_table4(root: Path) -> pd.DataFrame:
    df = _safe_read_csv(root / "reports/tables/table4.csv")
    if df.empty:
        return df
    for metric in TABLE4_METRICS:
        means: list[float | None] = []
        stds: list[float | None] = []
        for value in df[metric]:
            mean, std = parse_plus_minus(value)
            means.append(mean)
            stds.append(std)
        df[f"{metric}_mean"] = means
        df[f"{metric}_std"] = stds
    return df


def load_main_results(root: Path) -> pd.DataFrame:
    return _safe_read_csv(root / "reports/tables/main_results.csv")


def load_split_stats(root: Path) -> pd.DataFrame:
    df = _safe_read_csv(root / "reports/tables/split_stats.csv")
    if df.empty:
        return df
    for col in ("n_samples", "n_positive", "positive_rate", "n_patients"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_m5_summary(root: Path) -> dict[str, Any]:
    return _safe_read_json(root / "demo/assets/m5_summary_snippet.json")


def load_sample_predictions(root: Path) -> pd.DataFrame:
    return _safe_read_csv(root / "demo/assets/sample_predictions_fold0.csv")


def load_metrics(root: Path) -> dict[str, Any]:
    return _safe_read_json(root / "demo/assets/metrics_m5_fold0.json")


def load_sample_manifest(root: Path) -> dict[str, Any]:
    return _safe_read_json(root / "demo/assets/sample_predictions_manifest.json")


def load_model_predictions(root: Path, relative_path: str) -> pd.DataFrame:
    path = root / "demo/assets" / relative_path
    return _safe_read_csv(path)


def load_model_metrics(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / "demo/assets" / relative_path
    return _safe_read_json(path)


def _normalize_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    required = {"sample_id", "score", "label"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    out = df[["sample_id", "score", "label"]].copy()
    out["sample_id"] = out["sample_id"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["score", "label"]).copy()
    out["label"] = out["label"].astype(int)
    return out.sort_values("sample_id").reset_index(drop=True)


def load_all_sample_models(root: Path) -> dict[str, dict[str, Any]]:
    manifest = load_sample_manifest(root)
    models: dict[str, dict[str, Any]] = {}

    if manifest:
        for item in manifest.get("models", []):
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if not key:
                continue
            predictions_rel = str(item.get("predictions", "")).strip()
            if not predictions_rel:
                continue
            predictions = _normalize_prediction_frame(load_model_predictions(root, predictions_rel))
            if predictions.empty:
                continue
            metrics_rel = str(item.get("metrics", "")).strip()
            metrics = load_model_metrics(root, metrics_rel) if metrics_rel else {}
            models[key] = {
                "label": str(item.get("label", key)),
                "predictions": predictions,
                "metrics": metrics,
            }

    if models:
        return models

    legacy_predictions = _normalize_prediction_frame(load_sample_predictions(root))
    if legacy_predictions.empty:
        return {}

    return {
        "m5_lgke_gnn": {
            "label": "M5 LGKE-GNN",
            "predictions": legacy_predictions,
            "metrics": load_metrics(root),
        }
    }


def load_fairness(root: Path) -> pd.DataFrame:
    df = _safe_read_csv(root / "demo/assets/fairness_subgroup.csv")
    if df.empty:
        return df
    for col in ("n_samples", "n_positive", "auroc", "sensitivity", "fnr"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_graph_ablation(root: Path) -> pd.DataFrame:
    df = _safe_read_csv(root / "demo/assets/graph_ablation_summary.csv")
    if df.empty:
        return df
    for metric in ("pauc_tpr80", "auprc", "fnr"):
        means: list[float | None] = []
        stds: list[float | None] = []
        for value in df[metric]:
            mean, std = parse_plus_minus(value)
            means.append(mean)
            stds.append(std)
        df[f"{metric}_mean"] = means
        df[f"{metric}_std"] = stds
    return df


def load_pad_adaptation(root: Path) -> pd.DataFrame:
    df = _safe_read_csv(root / "demo/assets/pad_adaptation_summary.csv")
    if df.empty:
        return df
    for col in ("mean", "std"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_pad_zeroshot(root: Path) -> pd.DataFrame:
    df = _safe_read_csv(root / "demo/assets/pad_zeroshot_summary.csv")
    if df.empty:
        return df
    for col in ("mean", "std"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_pad_finetune(root: Path) -> pd.DataFrame:
    return load_pad_adaptation(root)


def load_pad_domain_comparison(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "zeroshot": load_pad_zeroshot(root),
        "finetune": load_pad_finetune(root),
    }


def figure_paths(root: Path) -> dict[str, Path]:
    base = root / "demo/assets/figures"
    names = {
        "ml_pipeline_png": "ml_pipeline.png",
        "main_results_forest": "main_results_forest.png",
        "graph_ablation": "graph_ablation_metrics.png",
        "pad_domain_shift": "pad_domain_shift.png",
    }
    out: dict[str, Path] = {}
    for key, name in names.items():
        path = base / name
        if path.exists():
            out[key] = path
    return out
