"""
Generate Table 4: Main experiment model comparison.
All values are 5-fold patient-level test set mean ± std or 95% CI.

Usage:
    python scripts/make_table4.py
    python scripts/make_table4.py --input reports/tables/main_results.csv
    python scripts/make_table4.py --output reports/tables/table4.csv
    python scripts/make_table4.py --format latex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# Model display names for the paper
MODEL_NAMES = {
    "m0_constant": "Constant",
    "m0_tabular_baseline": "Tabular",
    "m0_lightgbm": "LightGBM",
    "m1_cnn_baseline": "CNN (EfficientNet-B0)",
    "m2_monet_feature_baseline": "MONET Feature",
    "m3_transformer_baseline": "Transformer (Swin-Tiny)",
    "m4_multimodal_fusion": "Multimodal Fusion",
    "m5_lgke_gnn": "LGKE-GNN",
}

# Columns to include in Table 4
TABLE_COLUMNS = ["pauc_tpr80", "auprc", "auroc", "sensitivity", "specificity"]
TABLE_HEADERS = {
    "pauc_tpr80": "pAUC",
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "sensitivity": "Sens.",
    "specificity": "Spec.",
}

# Models to exclude from the table (e.g., constant baseline)
EXCLUDE_MODELS = {"m0_constant", "m0_tabular_baseline"}


def mean_std_ci(values: np.ndarray, *, use_ci: bool = False, ci_level: float = 0.95) -> tuple[float, float]:
    """Compute mean and std or 95% CI."""
    n = len(values)
    if n < 2:
        return float(values.mean()), 0.0

    mean = float(values.mean())
    std = float(values.std(ddof=1))

    if use_ci:
        # 95% CI using t-distribution
        se = std / np.sqrt(n)
        t_val = stats.t.ppf((1 + ci_level) / 2, df=n - 1)
        ci = t_val * se
        return mean, ci

    return mean, std


def format_value(mean: float, spread: float, *, use_ci: bool = False) -> str:
    """Format mean ± std or mean ± CI."""
    if spread == 0:
        return f"{mean:.4f}"

    if use_ci:
        return f"{mean:.4f} ± {spread:.4f}"
    else:
        return f"{mean:.4f} ± {spread:.4f}"


def aggregate_results(df: pd.DataFrame, *, use_ci: bool = False) -> pd.DataFrame:
    """Aggregate per-fold results to mean ± std/CI."""
    # Filter out excluded models
    df = df[~df["model_name"].isin(EXCLUDE_MODELS)]

    # Get unique models in order
    model_order = [
        "m0_lightgbm",
        "m1_cnn_baseline",
        "m2_monet_feature_baseline",
        "m3_transformer_baseline",
        "m4_multimodal_fusion",
        "m5_lgke_gnn",
    ]
    models = [m for m in model_order if m in df["model_name"].unique()]

    rows = []
    for model in models:
        model_df = df[df["model_name"] == model]
        row = {"Model": MODEL_NAMES.get(model, model)}

        for col in TABLE_COLUMNS:
            values = model_df[col].dropna().values
            if len(values) == 0:
                row[TABLE_HEADERS[col]] = "N/A"
            else:
                mean, spread = mean_std_ci(values, use_ci=use_ci)
                row[TABLE_HEADERS[col]] = format_value(mean, spread, use_ci=use_ci)

        rows.append(row)

    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame) -> str:
    """Convert DataFrame to LaTeX table format."""
    # Escape model names with underscores
    df = df.copy()
    df["Model"] = df["Model"].str.replace("_", "\\_")

    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(df.columns) - 1),
        caption="Main experiment model comparison. All values are 5-fold patient-level test set mean ± standard deviation.",
        label="tab:main_results",
    )
    return latex


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Table 4: Main experiment model comparison.")
    parser.add_argument("--input", default="reports/tables/main_results.csv")
    parser.add_argument("--output", default="reports/tables/table4.csv")
    parser.add_argument("--format", choices=["csv", "latex", "both"], default="csv")
    parser.add_argument("--use-ci", action="store_true", help="Use 95% CI instead of std")
    args = parser.parse_args()

    input_path = ROOT / args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Models found: {sorted(df['model_name'].unique())}")
    print(f"Folds per model: {df.groupby('model_name')['fold'].count().to_dict()}")

    # Aggregate results
    table4 = aggregate_results(df, use_ci=args.use_ci)

    # Save output
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format in ("csv", "both"):
        table4.to_csv(out_path, index=False)
        print(f"\nWrote CSV to {out_path}")

    if args.format in ("latex", "both"):
        latex_path = out_path.with_suffix(".tex")
        latex_content = to_latex(table4)
        latex_path.write_text(latex_content, encoding="utf-8")
        print(f"Wrote LaTeX to {latex_path}")

    # Print table
    print("\n" + "=" * 80)
    print("Table 4: Main Experiment Model Comparison")
    print("All values are 5-fold patient-level test set mean ± standard deviation")
    print("=" * 80)
    print(table4.to_string(index=False))


if __name__ == "__main__":
    main()
