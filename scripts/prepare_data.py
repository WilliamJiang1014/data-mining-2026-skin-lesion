from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_lesion_risk.data.preprocessing import default_preprocessor_for
from skin_lesion_risk.data.quality import manifest_quality_summary, validate_image_paths
from skin_lesion_risk.data.splits import make_patient_level_folds


LEAKAGE_COLUMNS = {
    "diagnosis",
    "diagnosis_confirm_type",
    "iddx_full",
    "iddx_1",
    "iddx_2",
    "iddx_3",
    "iddx_4",
    "iddx_5",
    "malignant",
    "mel_mitotic_index",
    "mel_thick_mm",
    "target",
    "biopsed",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifests, patient-level folds and fold-scoped preprocessors.")
    parser.add_argument("--isic-root", default="data/raw/isic2024_permissive")
    parser.add_argument("--pad-root", default="data/raw/pad_ufes_20")
    parser.add_argument("--skip-pad", action="store_true", help="Skip PAD manifest build if data is not available.")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rare-min-count", type=int, default=20)
    parser.add_argument("--image-check", choices=["none", "sample", "all"], default="sample")
    parser.add_argument("--image-check-sample-size", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional stratified ISIC subset for CPU smoke tests.")
    args = parser.parse_args()

    out_dir = (ROOT / args.out_dir).resolve()
    reports_dir = (ROOT / args.reports_dir).resolve()
    preprocessor_dir = out_dir / "preprocessors"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_dir.mkdir(parents=True, exist_ok=True)

    isic = build_isic_manifest(ROOT / args.isic_root)
    if args.max_samples:
        isic = stratified_sample(isic, args.max_samples, seed=args.seed)

    pad_root = ROOT / args.pad_root
    if args.skip_pad or not (pad_root / "metadata.csv").exists():
        pad = pd.DataFrame()
        print(f"[prepare_data] Skipping PAD manifest (data not found or --skip-pad).")
    else:
        pad = build_pad_manifest(pad_root)

    isic_failures = validate_image_paths(
        isic,
        mode=args.image_check,
        sample_size=args.image_check_sample_size,
        seed=args.seed,
    )
    if len(pad) > 0:
        pad_failures = validate_image_paths(
            pad,
            mode=args.image_check,
            sample_size=args.image_check_sample_size,
            seed=args.seed,
        )
    else:
        pad_failures = []

    folds = make_patient_level_folds(
        isic,
        n_splits=args.folds,
        val_ratio=args.val_ratio,
        seed=args.seed,
        group_col="patient_id",
        label_col="target",
    )
    fold_lookup = folds[folds["split"] == "test"][["sample_id", "fold"]].drop_duplicates("sample_id")
    isic = isic.drop(columns=["fold"], errors="ignore").merge(fold_lookup, on="sample_id", how="left")
    isic["fold"] = isic["fold"].fillna(-1).astype(int)
    if len(pad) > 0:
        pad["fold"] = -1

    isic.to_csv(out_dir / "manifest_isic.csv", index=False)
    if len(pad) > 0:
        pad.to_csv(out_dir / "manifest_pad.csv", index=False)
    folds.to_csv(out_dir / "folds_isic.csv", index=False)

    build_preprocessors(isic, folds, preprocessor_dir, args.rare_min_count)
    write_split_stats(isic, folds, pad, reports_dir / "tables" / "split_stats.csv")
    write_quality_reports(isic, pad, isic_failures, pad_failures, reports_dir, skip_pad=len(pad) == 0)

    manifest_summary = {
        "isic_manifest": str(out_dir / "manifest_isic.csv"),
        "pad_manifest": str(out_dir / "manifest_pad.csv") if len(pad) > 0 else None,
        "folds": str(out_dir / "folds_isic.csv"),
        "preprocessors": str(preprocessor_dir),
        "isic_samples": int(len(isic)),
        "pad_samples": int(len(pad)),
        "isic_positive": int(isic["target"].sum()),
        "pad_positive": int(pad["target"].sum()) if len(pad) > 0 else 0,
        "image_check": args.image_check,
        "isic_image_failures": len(isic_failures),
        "pad_image_failures": len(pad_failures),
    }
    (out_dir / "prepare_summary.json").write_text(json.dumps(manifest_summary, indent=2), encoding="utf-8")
    print(json.dumps(manifest_summary, indent=2))


def build_isic_manifest(root: Path) -> pd.DataFrame:
    metadata = pd.read_csv(root / "metadata.csv")
    supplemental = pd.read_csv(root / "supplemental_metadata.csv")
    labels = pd.read_csv(root / "labels.csv")
    df = metadata.merge(supplemental[["isic_id", "lesion_id"]], on="isic_id", how="left")
    df = df.merge(labels.rename(columns={"malignant": "target"}), on="isic_id", how="inner")
    image_dir = root / "images"

    out = pd.DataFrame(
        {
            "sample_id": df["isic_id"].astype(str),
            "image_path": df["isic_id"].map(lambda x: str((image_dir / f"{x}.jpg").resolve())),
            "target": pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int),
            "patient_id": df["patient_id"].fillna("UNKNOWN_PATIENT").astype(str),
            "lesion_id": df["lesion_id"].fillna("").astype(str),
            "age": pd.to_numeric(df.get("age_approx"), errors="coerce"),
            "sex": normalize_category(df.get("sex")),
            "anatom_site": normalize_category(df.get("anatom_site_general")),
            "size_mm": pd.to_numeric(df.get("clin_size_long_diam_mm"), errors="coerce"),
            "source": "isic2024_slice3d_permissive",
            "fold": -1,
        }
    )
    for column in df.columns:
        if column.startswith("tbp_lv_") and column not in LEAKAGE_COLUMNS:
            out[column] = df[column]
    return out.drop_duplicates("sample_id").reset_index(drop=True)


def build_pad_manifest(root: Path) -> pd.DataFrame:
    metadata = pd.read_csv(root / "metadata.csv")
    mapping_positive = {"BCC", "MEL", "SCC", "BOD"}
    mapping_negative = {"ACK", "NEV", "SEK"}
    labels = metadata["diagnostic"].fillna("UNK").astype(str).str.upper()
    keep = labels.isin(mapping_positive | mapping_negative)
    df = metadata.loc[keep].copy()
    labels = labels.loc[keep]
    image_dir = root / "images"

    diameter_1 = pd.to_numeric(df.get("diameter_1"), errors="coerce")
    diameter_2 = pd.to_numeric(df.get("diameter_2"), errors="coerce")
    size_mm = pd.concat([diameter_1, diameter_2], axis=1).max(axis=1)
    out = pd.DataFrame(
        {
            "sample_id": df["img_id"].astype(str),
            "image_path": df["img_id"].map(lambda x: str((image_dir / str(x)).resolve())),
            "target": labels.isin(mapping_positive).astype(int),
            "patient_id": df["patient_id"].fillna("UNKNOWN_PATIENT").astype(str),
            "lesion_id": df["lesion_id"].fillna("").astype(str),
            "age": pd.to_numeric(df.get("age"), errors="coerce"),
            "sex": normalize_category(df.get("gender")),
            "anatom_site": normalize_category(df.get("region")),
            "size_mm": size_mm,
            "diagnostic_original": labels,
            "fitzpatrick": normalize_category(df.get("fitspatrick")),
            "source": "pad_ufes_20",
            "fold": -1,
        }
    )
    return out.drop_duplicates("sample_id").reset_index(drop=True)


def normalize_category(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=object)
    return series.fillna("UNK").astype(str).str.strip().replace({"": "UNK", "nan": "UNK", "None": "UNK"})


def stratified_sample(df: pd.DataFrame, n: int, *, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    pos = df[df["target"] == 1]
    neg = df[df["target"] == 0]
    n_pos = min(len(pos), max(1, n // 2))
    n_neg = min(len(neg), n - n_pos)
    sampled_pos = pos.sample(n=n_pos, random_state=seed) if n_pos else pos.iloc[[]]
    sampled_neg = neg.sample(n=n_neg, random_state=seed + 1) if n_neg else neg.iloc[[]]
    sampled = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
    sampled = sampled.iloc[rng.permutation(len(sampled))].reset_index(drop=True)
    return sampled


def build_preprocessors(isic: pd.DataFrame, folds: pd.DataFrame, out_dir: Path, rare_min_count: int) -> None:
    for fold in sorted(folds["fold"].unique()):
        train_ids = set(folds[(folds["fold"] == fold) & (folds["split"] == "train")]["sample_id"])
        train_df = isic[isic["sample_id"].isin(train_ids)].copy()
        preprocessor = default_preprocessor_for(train_df, rare_min_count=rare_min_count).fit(train_df)
        preprocessor.save(out_dir / f"fold{fold}_tabular.pkl")
        preprocessor.save_schema(out_dir / f"fold{fold}_tabular_schema.json")


def write_split_stats(isic: pd.DataFrame, folds: pd.DataFrame, pad: pd.DataFrame, path: Path) -> None:
    rows: list[dict[str, object]] = []
    for fold in sorted(folds["fold"].unique()):
        for split in ("train", "val", "test"):
            ids = set(folds[(folds["fold"] == fold) & (folds["split"] == split)]["sample_id"])
            part = isic[isic["sample_id"].isin(ids)]
            rows.append(split_stat_row(f"ISIC fold{fold} {split}", part))
    rows.append(split_stat_row("PAD external", pad))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def split_stat_row(name: str, df: pd.DataFrame) -> dict[str, object]:
    return {
        "split": name,
        "n_samples": int(len(df)),
        "n_positive": int(df["target"].sum()) if "target" in df else 0,
        "positive_rate": float(df["target"].mean()) if len(df) and "target" in df else 0.0,
        "n_patients": int(df["patient_id"].nunique()) if "patient_id" in df else 0,
    }


def write_quality_reports(
    isic: pd.DataFrame,
    pad: pd.DataFrame,
    isic_failures: list[dict[str, str]],
    pad_failures: list[dict[str, str]],
    reports_dir: Path,
    *,
    skip_pad: bool = False,
) -> None:
    lines = [
        "columns_removed_as_leakage: " + ", ".join(sorted(LEAKAGE_COLUMNS)),
        f"invalid_images_checked_or_missing: {len(isic_failures)}",
    ]
    if isic_failures:
        lines.extend([f"invalid_image {x['sample_id']}: {x['error']} {x['image_path']}" for x in isic_failures[:100]])
    (reports_dir / "data_quality_isic.md").write_text(
        manifest_quality_summary(isic, extra_lines=lines),
        encoding="utf-8",
    )

    if skip_pad:
        (reports_dir / "data_quality_pad.md").write_text(
            "# PAD-UFES-20 Quality Report\n\nSkipped: PAD data not available.\n",
            encoding="utf-8",
        )
        return

    pad_lines = [f"invalid_images_checked_or_missing: {len(pad_failures)}"]
    if pad_failures:
        pad_lines.extend([f"invalid_image {x['sample_id']}: {x['error']} {x['image_path']}" for x in pad_failures[:100]])
    (reports_dir / "data_quality_pad.md").write_text(
        manifest_quality_summary(pad, extra_lines=pad_lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
