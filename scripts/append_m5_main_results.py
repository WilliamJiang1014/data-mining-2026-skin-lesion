from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MAIN_RESULT_COLUMNS = [
    "model_name",
    "fold",
    "auroc",
    "auprc",
    "pauc_tpr80",
    "sensitivity",
    "specificity",
    "f1",
    "brier",
    "ece",
]


def find_m5_model_root(bundle: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    pattern = "data/artifacts/trained_models/hparam_sweeps/*/m5_lgke_gnn/fold0/test_predictions.csv"
    for path in bundle.glob(pattern):
        try:
            line_count = sum(1 for _ in path.open(encoding="utf-8")) - 1
        except OSError:
            continue
        candidates.append((line_count, path.parent.parent))

    smoke = bundle / "data/artifacts/trained_models/m5_lgke_gnn/fold0/test_predictions.csv"
    if smoke.exists():
        try:
            line_count = sum(1 for _ in smoke.open(encoding="utf-8")) - 1
            candidates.append((line_count, smoke.parent.parent))
        except OSError:
            pass

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def metrics_row(metrics_path: Path, *, threshold_rule: str) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_name = str(payload.get("model") or metrics_path.parents[1].name)
    fold = int(payload.get("fold", metrics_path.parent.name.replace("fold", "")))
    test_metrics = payload.get("test", {})
    rule_metrics = test_metrics.get(threshold_rule)
    if rule_metrics is None and isinstance(test_metrics, dict) and test_metrics:
        rule_metrics = next(iter(test_metrics.values()))
    values = rule_metrics.get("values", {}) if isinstance(rule_metrics, dict) else {}
    return {
        "model_name": model_name,
        "fold": fold,
        "auroc": values.get("auroc", float("nan")),
        "auprc": values.get("auprc", float("nan")),
        "pauc_tpr80": values.get("pauc_tpr80", float("nan")),
        "sensitivity": values.get("sensitivity", float("nan")),
        "specificity": values.get("specificity", float("nan")),
        "f1": values.get("f1", float("nan")),
        "brier": values.get("brier", float("nan")),
        "ece": values.get("ece", float("nan")),
    }


def collect_m5_rows(model_root: Path, *, threshold_rule: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(model_root.glob("fold*/metrics.json")):
        rows.append(metrics_row(metrics_path, threshold_rule=threshold_rule))
    return sorted(rows, key=lambda row: row["fold"])


def merge_m5_into_main_results(
    main_results_path: Path,
    m5_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    if main_results_path.exists():
        frame = pd.read_csv(main_results_path)
        frame = frame[frame["model_name"] != "m5_lgke_gnn"].copy()
    else:
        frame = pd.DataFrame(columns=MAIN_RESULT_COLUMNS)

    m5_frame = pd.DataFrame(m5_rows, columns=MAIN_RESULT_COLUMNS)
    merged = pd.concat([frame, m5_frame], ignore_index=True)
    merged = merged.sort_values(["fold", "model_name"]).reset_index(drop=True)
    main_results_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(main_results_path, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Append M5 per-fold metrics into main_results.csv from bundle artifacts.")
    parser.add_argument(
        "--bundle-root",
        default="../训练数据/training_results_bundle",
        help="Path to training_results_bundle (relative to project root).",
    )
    parser.add_argument(
        "--main-results",
        default="reports/tables/main_results.csv",
        help="Target main_results.csv path (relative to project root).",
    )
    parser.add_argument(
        "--threshold-rule",
        default="sensitivity_at_least_0_90",
        help="Threshold rule used when reading metrics.json test split.",
    )
    args = parser.parse_args()

    bundle = (ROOT / args.bundle_root).resolve()
    if not bundle.exists():
        alt = (ROOT / "../training_results_bundle").resolve()
        if alt.exists():
            bundle = alt
        else:
            raise SystemExit(f"Bundle not found: {bundle}")

    model_root = find_m5_model_root(bundle)
    if model_root is None:
        raise SystemExit(f"No M5 artifacts found under bundle: {bundle}")

    m5_rows = collect_m5_rows(model_root, threshold_rule=args.threshold_rule)
    if not m5_rows:
        raise SystemExit(f"No M5 fold metrics found under: {model_root}")

    out_path = (ROOT / args.main_results).resolve()
    merged = merge_m5_into_main_results(out_path, m5_rows)
    print(f"M5 source: {model_root}")
    print(f"wrote {out_path} ({len(merged)} rows, added {len(m5_rows)} M5 rows)")


if __name__ == "__main__":
    main()
