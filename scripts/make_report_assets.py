from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report tables from experiment artifacts.")
    parser.add_argument("--artifacts-dir", default="data/artifacts/trained_models")
    parser.add_argument("--out", default="reports/tables/main_results.csv")
    parser.add_argument("--threshold-rule", default="sensitivity_at_least_0_90")
    args = parser.parse_args()

    rows = collect_main_results(ROOT / args.artifacts_dir, threshold_rule=args.threshold_rule)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MAIN_RESULT_COLUMNS).to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(rows)} rows)")


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


def collect_main_results(artifacts_dir: Path, *, threshold_rule: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(artifacts_dir.glob("*/fold*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_name = str(payload.get("model") or metrics_path.parents[1].name)
        fold = int(payload.get("fold", metrics_path.parent.name.replace("fold", "")))
        test_metrics = payload.get("test", {})
        rule_metrics = test_metrics.get(threshold_rule)
        if rule_metrics is None and isinstance(test_metrics, dict) and test_metrics:
            rule_metrics = next(iter(test_metrics.values()))
        values = rule_metrics.get("values", {}) if isinstance(rule_metrics, dict) else {}
        rows.append(
            {
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
        )
    return sorted(rows, key=lambda row: (row["fold"], row["model_name"]))


if __name__ == "__main__":
    main()

