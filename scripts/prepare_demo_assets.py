#!/usr/bin/env python3
"""Curate small demo assets from local experiment outputs into demo/assets/."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_CANDIDATES = [
    ROOT.parent / "训练数据" / "training_results_bundle",
    ROOT.parent / "training_results_bundle",
]
DEFAULT_M0_M4_ROOT = ROOT.parent / "训练数据" / "artifacts"
OUT_DIR = ROOT / "demo" / "assets"
FIGURES = [
    ("ml_pipeline.png", "ml_pipeline.pdf"),
    "main_results_forest.png",
    "graph_ablation_metrics.png",
    "pad_domain_shift.png",
]
SAMPLE_SEED = 2026
SAMPLE_TARGET = 100
MODEL_SPECS = [
    ("m0_lightgbm", "M0 LightGBM"),
    ("m4_multimodal_fusion", "M4 多模态融合"),
    ("m5_lgke_gnn", "M5 LGKE-GNN"),
]
PM_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:±|\+/-)\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def _safe_relative(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def resolve_bundle_root(provided: Path | None) -> Path | None:
    if provided is not None:
        path = provided.expanduser().resolve()
        return path if path.is_dir() else None
    for candidate in DEFAULT_BUNDLE_CANDIDATES:
        path = candidate.expanduser().resolve()
        if path.is_dir():
            return path
    return None


def find_trained_models_root(base_root: Path) -> Path | None:
    if not base_root.is_dir():
        return None

    direct_candidates = [
        base_root / "data" / "artifacts" / "trained_models",
        base_root / "artifacts" / "trained_models",
        base_root / "trained_models",
    ]
    for candidate in direct_candidates:
        if candidate.is_dir():
            return candidate

    discovered = sorted(base_root.glob("**/trained_models/m0_lightgbm/fold0/test_predictions.csv"))
    if not discovered:
        return None
    return discovered[0].parents[2]


def find_m5_fold0_predictions(bundle: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in bundle.glob("data/artifacts/trained_models/hparam_sweeps/*/m5_lgke_gnn/fold0/test_predictions.csv"):
        try:
            line_count = sum(1 for _ in path.open(encoding="utf-8")) - 1
        except OSError:
            continue
        candidates.append((line_count, path))

    smoke = bundle / "data/artifacts/trained_models/m5_lgke_gnn/fold0/test_predictions.csv"
    if smoke.exists():
        try:
            line_count = sum(1 for _ in smoke.open(encoding="utf-8")) - 1
            candidates.append((line_count, smoke))
        except OSError:
            pass

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def find_m5_sweep_root(predictions_path: Path) -> Path | None:
    if "hparam_sweeps" not in predictions_path.parts:
        return None
    return predictions_path.parents[1]


def read_prediction_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "score", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")
    out = df[["sample_id", "score", "label"]].copy()
    out["sample_id"] = out["sample_id"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["label"] = pd.to_numeric(out["label"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["score", "label"]).copy()
    out["label"] = out["label"].astype(int)
    return out


def sample_reference_ids(frame: pd.DataFrame, *, target: int, seed: int) -> pd.DataFrame:
    pos = frame[frame["label"] == 1]
    neg = frame[frame["label"] != 1]
    n_neg = max(0, target - len(pos))
    if n_neg > 0 and len(neg) > n_neg:
        neg = neg.sample(n=n_neg, random_state=seed)
    sampled = pd.concat([pos, neg], ignore_index=True)
    if len(sampled) > target:
        sampled = sampled.sample(n=target, random_state=seed)
    sampled = sampled[["sample_id", "label"]].drop_duplicates(subset=["sample_id"]).sort_values("sample_id")
    return sampled.reset_index(drop=True)


def parse_fairness_table(md_path: Path) -> list[dict[str, str]]:
    text = md_path.read_text(encoding="utf-8")
    start = text.find("## 表 7")
    if start < 0:
        raise ValueError("Section '## 表 7' not found")
    chunk = text[start:]
    end = chunk.find("子群差异摘要")
    if end > 0:
        chunk = chunk[:end]
    rows: list[dict[str, str]] = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| 分组") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        rows.append(
            {
                "group_variable": cells[0],
                "subgroup": cells[1],
                "n_samples": cells[2].replace(",", ""),
                "n_positive": cells[3].replace(",", ""),
                "auroc": cells[4],
                "sensitivity": cells[5],
                "fnr": cells[6],
                "notes": cells[7] if len(cells) > 7 else "",
            }
        )
    return rows


def parse_pm_token(value: str) -> tuple[float, float] | None:
    match = PM_PATTERN.match(value.strip())
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _find_markdown_row(lines: list[str], prefix: str) -> list[str] | None:
    for raw in lines:
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and cells[0] == prefix:
            return cells
    return None


def parse_pad_zeroshot_metrics(md_path: Path) -> dict[str, tuple[float, float]]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    main_row = _find_markdown_row(lines, "ISIC 训练，PAD 测试")
    detail_row = _find_markdown_row(lines, "ISIC fold 模型 -> PAD")
    if main_row is None or detail_row is None:
        raise ValueError("PAD zero-shot rows not found in markdown table")

    mapping = {
        "auprc": parse_pm_token(main_row[1]),
        "auroc": parse_pm_token(main_row[2]),
        "bacc": parse_pm_token(main_row[3]),
        "pauc_tpr80": parse_pm_token(detail_row[1]),
        "sensitivity": parse_pm_token(detail_row[2]),
        "specificity": parse_pm_token(detail_row[3]),
        "fnr": parse_pm_token(detail_row[4]),
        "brier": parse_pm_token(detail_row[5]),
        "ece": parse_pm_token(detail_row[6]),
    }
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(f"PAD zero-shot metrics parse failed: missing {missing}")
    return {k: v for k, v in mapping.items() if v is not None}


def parse_pad_finetune_bacc(md_path: Path) -> tuple[float, float] | None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    row = _find_markdown_row(lines, "ISIC 预训练 + PAD-only fine-tune，PAD 测试")
    if row is None:
        return None
    return parse_pm_token(row[3])


def write_metric_summary_csv(metrics: dict[str, tuple[float, float]], dst: Path) -> None:
    order = [
        "pauc_tpr80",
        "auprc",
        "auroc",
        "sensitivity",
        "specificity",
        "fnr",
        "brier",
        "ece",
        "bacc",
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std"])
        for key in order:
            if key not in metrics:
                continue
            mean, std = metrics[key]
            writer.writerow([key, mean, std])


def find_pad_external_summary(bundle: Path) -> Path | None:
    direct = (
        bundle
        / "data/artifacts/trained_models/hparam_sweeps/hidden128_lr2e4_do02/external_pad/pad_external_summary.csv"
    )
    if direct.exists():
        return direct
    matches = sorted(bundle.glob("**/external_pad/pad_external_summary.csv"))
    return matches[0] if matches else None


def ensure_bacc_row(csv_path: Path, *, bacc_from_md: tuple[float, float] | None = None) -> bool:
    if not csv_path.exists():
        return False
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False
    if any(str(row.get("metric", "")).strip() == "bacc" for row in rows):
        return False

    if bacc_from_md is not None:
        mean_value, std_value = bacc_from_md
    else:
        sens_row = next((r for r in rows if str(r.get("metric", "")).strip() == "sensitivity"), None)
        spec_row = next((r for r in rows if str(r.get("metric", "")).strip() == "specificity"), None)
        if sens_row is None or spec_row is None:
            return False
        try:
            sens_mean = float(str(sens_row.get("mean", "")).strip())
            spec_mean = float(str(spec_row.get("mean", "")).strip())
            sens_std = float(str(sens_row.get("std", "")).strip())
            spec_std = float(str(spec_row.get("std", "")).strip())
        except ValueError:
            return False
        mean_value = (sens_mean + spec_mean) / 2.0
        std_value = (sens_std + spec_std) / 2.0

    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bacc", f"{mean_value:.10g}", f"{std_value:.10g}"])
    return True


def write_fairness_csv(rows: list[dict[str, str]], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_variable",
        "subgroup",
        "n_samples",
        "n_positive",
        "auroc",
        "sensitivity",
        "fnr",
        "notes",
    ]
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_m5_snippet(summary_path: Path, dst: Path) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    snippet = {
        "model": "m5_lgke_gnn",
        "variant": payload.get("variant"),
        "folds": payload.get("folds", 5),
        "pauc_tpr80_pct": round(metrics.get("pauc_tpr80", {}).get("mean", 0) * 100, 2),
        "pauc_tpr80_std_pct": round(metrics.get("pauc_tpr80", {}).get("std", 0) * 100, 2),
        "auprc_pct": round(metrics.get("auprc", {}).get("mean", 0) * 100, 2),
        "auprc_std_pct": round(metrics.get("auprc", {}).get("std", 0) * 100, 2),
        "auroc_pct": round(metrics.get("auroc", {}).get("mean", 0) * 100, 2),
        "auroc_std_pct": round(metrics.get("auroc", {}).get("std", 0) * 100, 2),
        "sensitivity_pct": round(metrics.get("sensitivity", {}).get("mean", 0) * 100, 2),
        "specificity_pct": round(metrics.get("specificity", {}).get("mean", 0) * 100, 2),
    }
    dst.write_text(json.dumps(snippet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_thresholds_snippet(metrics_path: Path, dst: Path, *, model_key: str) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    snippet = {
        "fold": int(payload.get("fold", 0)),
        "model": model_key,
        "thresholds": payload.get("thresholds", {}),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(snippet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_manifest(
    out_path: Path,
    *,
    sample_size: int,
    sample_seed: int,
    models: list[dict[str, Any]],
    featured_sample_id: str | None = None,
) -> None:
    payload = {
        "fold": 0,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "models": models,
    }
    if featured_sample_id:
        payload["featured_sample_id"] = featured_sample_id
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def remove_if_exists(path: Path, log: list[str]) -> None:
    if path.exists():
        path.unlink()
        log.append(f"OK: removed legacy file {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare demo/assets from local experiment outputs.")
    parser.add_argument("--bundle-root", type=Path, default=None)
    parser.add_argument("--m0-m4-root", type=Path, default=DEFAULT_M0_M4_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_TARGET)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args()

    bundle = resolve_bundle_root(args.bundle_root)
    m0_m4_root = args.m0_m4_root.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)
    log: list[str] = []

    trained_models_root = find_trained_models_root(m0_m4_root)
    if trained_models_root is None:
        log.append(f"WARN: trained_models root not found under {m0_m4_root}")
    else:
        log.append(f"OK: using trained_models root {_safe_relative(trained_models_root, m0_m4_root)}")

    if bundle is None:
        if args.bundle_root is None:
            candidate_text = ", ".join(str(p) for p in DEFAULT_BUNDLE_CANDIDATES)
            log.append(f"WARN: bundle not found in defaults: {candidate_text}")
        else:
            log.append(f"WARN: bundle not found: {args.bundle_root}")
    else:
        log.append(f"OK: using bundle root {bundle}")

    source_paths: dict[str, dict[str, Any]] = {
        key: {"key": key, "label": label, "predictions": None, "metrics": None}
        for key, label in MODEL_SPECS
    }

    if trained_models_root is not None:
        for key in ("m0_lightgbm", "m4_multimodal_fusion"):
            pred_path = trained_models_root / key / "fold0" / "test_predictions.csv"
            metrics_path = trained_models_root / key / "fold0" / "metrics.json"
            if pred_path.exists():
                source_paths[key]["predictions"] = pred_path
                log.append(f"OK: found {key} predictions at {_safe_relative(pred_path, m0_m4_root)}")
            else:
                log.append(f"WARN: {key} fold0 predictions not found")
            if metrics_path.exists():
                source_paths[key]["metrics"] = metrics_path

    m5_pred_path: Path | None = None
    if bundle is not None:
        m5_pred_path = find_m5_fold0_predictions(bundle)
        if m5_pred_path is None:
            log.append("WARN: M5 fold0 predictions not found in bundle")
        else:
            source_paths["m5_lgke_gnn"]["predictions"] = m5_pred_path
            source_paths["m5_lgke_gnn"]["metrics"] = m5_pred_path.parent / "metrics.json"
            log.append(f"OK: found M5 predictions at {_safe_relative(m5_pred_path, bundle)}")

    base_key: str | None = None
    for candidate_key in ("m0_lightgbm", "m5_lgke_gnn", "m4_multimodal_fusion"):
        pred_path = source_paths[candidate_key]["predictions"]
        if isinstance(pred_path, Path) and pred_path.exists():
            base_key = candidate_key
            break

    exported_models: list[dict[str, str]] = []
    featured_sample_id = ""
    if base_key is None:
        log.append("WARN: no available prediction source for sample extraction")
    else:
        base_pred_path = source_paths[base_key]["predictions"]
        assert isinstance(base_pred_path, Path)
        base_frame = read_prediction_frame(base_pred_path)
        sampled_ref = sample_reference_ids(base_frame, target=args.sample_size, seed=args.seed)
        log.append(f"OK: sampled {len(sampled_ref)} IDs from {base_key} fold0")

        model_frames: dict[str, pd.DataFrame] = {}
        for key, spec in source_paths.items():
            pred_path = spec["predictions"]
            if not isinstance(pred_path, Path) or not pred_path.exists():
                continue
            try:
                model_frames[key] = read_prediction_frame(pred_path)
            except ValueError as exc:
                log.append(f"WARN: skip {key}: {exc}")

        base_labels = sampled_ref.set_index("sample_id")["label"].astype(int)
        dropped_ids: set[str] = set()
        for key, frame in model_frames.items():
            index = frame.set_index("sample_id")
            missing_ids = sorted(set(base_labels.index) - set(index.index))
            if missing_ids:
                dropped_ids.update(missing_ids)
                log.append(f"WARN: {key} missing {len(missing_ids)} sampled IDs")

            shared_ids = sorted(set(base_labels.index) & set(index.index))
            if not shared_ids:
                continue
            model_labels = index.loc[shared_ids, "label"].astype(int)
            base_shared = base_labels.loc[shared_ids].astype(int)
            mismatch_ids = base_shared.index[model_labels != base_shared].tolist()
            if mismatch_ids:
                dropped_ids.update(mismatch_ids)
                log.append(f"WARN: {key} label mismatch on {len(mismatch_ids)} sampled IDs")

        final_ref = sampled_ref[~sampled_ref["sample_id"].isin(dropped_ids)].copy()
        final_ref = final_ref.sort_values("sample_id").reset_index(drop=True)
        if dropped_ids:
            log.append(f"WARN: dropped {len(dropped_ids)} IDs due to missing/mismatch labels")
        log.append(f"OK: final aligned sample size {len(final_ref)}")

        m5_frame = model_frames.get("m5_lgke_gnn")
        if m5_frame is not None:
            m5_join = final_ref.merge(
                m5_frame[["sample_id", "score"]],
                on="sample_id",
                how="inner",
            )
            m5_pos = m5_join[m5_join["label"] == 1]
            if not m5_pos.empty:
                m5_pos = m5_pos.sort_values("score", ascending=False)
                featured_sample_id = str(m5_pos.iloc[0]["sample_id"])
                log.append(f"OK: featured_sample_id={featured_sample_id}")

        for key, label in MODEL_SPECS:
            frame = model_frames.get(key)
            if frame is None:
                continue
            merged = final_ref.merge(frame[["sample_id", "score"]], on="sample_id", how="left")
            merged = merged.dropna(subset=["score"]).copy()
            merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
            merged = merged.dropna(subset=["score"]).copy()
            merged["label"] = merged["label"].astype(int)
            merged = merged.sort_values("sample_id")

            pred_rel = Path("predictions") / f"{key}_fold0.csv"
            merged[["sample_id", "score", "label"]].to_csv(out / pred_rel, index=False)
            log.append(f"OK: {pred_rel} ({len(merged)} rows)")

            metrics_rel: str | None = None
            metrics_path = source_paths[key]["metrics"]
            if isinstance(metrics_path, Path) and metrics_path.exists():
                metrics_rel_path = Path("metrics") / f"{key}_fold0.json"
                build_thresholds_snippet(metrics_path, out / metrics_rel_path, model_key=key)
                metrics_rel = str(metrics_rel_path)
                log.append(f"OK: {metrics_rel_path}")

            exported_models.append(
                {
                    "key": key,
                    "label": label,
                    "predictions": str(pred_rel),
                    "metrics": metrics_rel or "",
                }
            )

    if exported_models:
        write_manifest(
            out / "sample_predictions_manifest.json",
            sample_size=min(args.sample_size, len(pd.read_csv(out / exported_models[0]["predictions"]))),
            sample_seed=args.seed,
            models=exported_models,
            featured_sample_id=featured_sample_id or None,
        )
        log.append("OK: sample_predictions_manifest.json")
    else:
        log.append("WARN: manifest not written because no model predictions were exported")

    remove_if_exists(out / "sample_predictions_fold0.csv", log)
    remove_if_exists(out / "metrics_m5_fold0.json", log)

    if bundle is not None and m5_pred_path is not None:
        sweep_root = find_m5_sweep_root(m5_pred_path)
        if sweep_root is not None:
            summary_src = sweep_root / "summary_stats.json"
            if summary_src.exists():
                build_m5_snippet(summary_src, out / "m5_summary_snippet.json")
                log.append(f"OK: m5_summary_snippet.json from {_safe_relative(summary_src, bundle)}")

    if bundle is not None:
        ablation_src = bundle / "reports/tables/graph_ablation_partial_summary.csv"
        if copy_if_exists(ablation_src, out / "graph_ablation_summary.csv"):
            log.append("OK: graph_ablation_summary.csv")

        zero_shot_src = find_pad_external_summary(bundle)
        if zero_shot_src is not None and copy_if_exists(zero_shot_src, out / "pad_zeroshot_summary.csv"):
            log.append(f"OK: pad_zeroshot_summary.csv from {_safe_relative(zero_shot_src, bundle)}")

        pad_src = (
            bundle
            / "data/artifacts/trained_models/pad_domain_adaptation/pad_only_finetune_hd128_lr1e4/pad_adaptation_summary.csv"
        )
        if copy_if_exists(pad_src, out / "pad_adaptation_summary.csv"):
            log.append("OK: pad_adaptation_summary.csv")

        report_md = bundle / "reports/m5_experiment_tables_filled.md"
        if report_md.exists():
            finetune_bacc = parse_pad_finetune_bacc(report_md)
            if finetune_bacc is not None and ensure_bacc_row(
                out / "pad_adaptation_summary.csv", bacc_from_md=finetune_bacc
            ):
                log.append("OK: pad_adaptation_summary.csv appended bacc from 表6 markdown")
            if not (out / "pad_zeroshot_summary.csv").exists():
                try:
                    zero_shot_metrics = parse_pad_zeroshot_metrics(report_md)
                    write_metric_summary_csv(zero_shot_metrics, out / "pad_zeroshot_summary.csv")
                    log.append("OK: pad_zeroshot_summary.csv parsed from 表6 markdown")
                except ValueError as exc:
                    log.append(f"WARN: pad zero-shot parse failed: {exc}")
            try:
                rows = parse_fairness_table(report_md)
                write_fairness_csv(rows, out / "fairness_subgroup.csv")
                log.append(f"OK: fairness_subgroup.csv ({len(rows)} rows)")
            except ValueError as exc:
                log.append(f"WARN: fairness parse failed: {exc}")
        else:
            log.append("WARN: m5_experiment_tables_filled.md not found")
            if ensure_bacc_row(out / "pad_adaptation_summary.csv"):
                log.append("OK: pad_adaptation_summary.csv appended derived bacc")

        fig_root = bundle / "数据挖掘项目/fig"
        for item in FIGURES:
            if isinstance(item, tuple):
                copied = False
                for name in item:
                    if copy_if_exists(fig_root / name, out / "figures" / name):
                        log.append(f"OK: figures/{name}")
                        copied = True
                        break
                if not copied:
                    log.append(f"WARN: figures/{item[0]} not found")
            elif copy_if_exists(fig_root / item, out / "figures" / item):
                log.append(f"OK: figures/{item}")
            else:
                log.append(f"WARN: figures/{item} not found")

    readme = out / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Demo Assets",
                "",
                "由 `scripts/prepare_demo_assets.py` 从本地实验产物生成。",
                "",
                "重新生成：",
                "",
                "```bash",
                "python scripts/prepare_demo_assets.py \\",
                "  --m0-m4-root ../训练数据/artifacts \\",
                "  --bundle-root ../训练数据/training_results_bundle",
                "```",
                "",
                "说明：`--bundle-root` 可省略，脚本会依次尝试 `../训练数据/training_results_bundle` 和 `../training_results_bundle`。",
                "",
                "## 本次生成日志",
                "",
                *[f"- {line}" for line in log],
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote demo assets to {out}")
    for line in log:
        print(line)


if __name__ == "__main__":
    main()
