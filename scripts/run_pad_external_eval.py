from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_graph import (
    build_graph,
    image_feature,
    load_or_build_image_feature_cache,
    relation_counts,
    resolve_knn_device,
    write_nodes_edges_tables,
)
from skin_lesion_risk.data.preprocessing import FoldTabularPreprocessor
from skin_lesion_risk.evaluation.calibration import expected_calibration_error
from skin_lesion_risk.evaluation.metrics import binary_classification_metrics, subgroup_full_metrics
from skin_lesion_risk.models.adapters.graph import LGKEGNNModelAdapter, write_predictions
from skin_lesion_risk.models.base import ModelBatch, PredictionResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PAD-UFES-20 external validation for trained M5 fold checkpoints.")
    parser.add_argument("--isic-manifest", default="data/processed/manifest_isic.csv")
    parser.add_argument("--pad-manifest", default="data/processed/manifest_pad.csv")
    parser.add_argument("--folds", default="data/processed/folds_isic.csv")
    parser.add_argument("--preprocessor-dir", default="data/processed/preprocessors")
    parser.add_argument("--checkpoint-root", default="data/artifacts/trained_models/hparam_sweeps/hidden128_lr2e4_do02")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--folds-to-run", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k-visual", type=int, default=10)
    parser.add_argument("--k-metadata", type=int, default=10)
    parser.add_argument("--n-visual-prototypes", type=int, default=64)
    parser.add_argument("--n-metadata-prototypes", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-feature-cache", default="data/processed/embeddings/image_stats_features_with_pad.npz")
    parser.add_argument("--base-image-feature-cache", default="data/processed/embeddings/image_stats_features.npz")
    parser.add_argument("--refresh-image-cache", action="store_true")
    parser.add_argument("--knn-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--knn-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reuse-graphs", action="store_true")
    args = parser.parse_args()

    isic = pd.read_csv(ROOT / args.isic_manifest)
    pad = pd.read_csv(ROOT / args.pad_manifest)
    folds = pd.read_csv(ROOT / args.folds)
    checkpoint_root = ROOT / args.checkpoint_root
    out_dir = ROOT / (args.out_dir or str(checkpoint_root / "external_pad"))
    graph_dir = out_dir / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    folds_to_run = [int(x) for x in args.folds_to_run.split(",") if x.strip()]
    image_cache = load_external_image_feature_cache(
        pd.concat([isic, pad], ignore_index=True),
        ROOT / args.image_feature_cache,
        base_path=ROOT / args.base_image_feature_cache,
        image_size=args.image_size,
        refresh=args.refresh_image_cache,
    )
    knn_device = resolve_knn_device(args.knn_device)

    fold_rows: list[dict[str, Any]] = []
    fold_predictions: list[pd.DataFrame] = []
    fold_thresholds: list[float] = []
    for fold in folds_to_run:
        fold_out = out_dir / f"fold{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        graph_path = graph_dir / f"fold{fold}_pad_external_graph.pt"
        if graph_path.exists() and args.reuse_graphs:
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        else:
            graph = build_external_graph(
                isic=isic,
                pad=pad,
                folds=folds,
                fold=fold,
                preprocessor_dir=ROOT / args.preprocessor_dir,
                image_cache=image_cache,
                k_visual=args.k_visual,
                k_metadata=args.k_metadata,
                n_visual_prototypes=args.n_visual_prototypes,
                n_metadata_prototypes=args.n_metadata_prototypes,
                image_size=args.image_size,
                knn_device=knn_device,
                knn_chunk_size=args.knn_chunk_size,
                seed=args.seed,
            )
            torch.save(graph, graph_path)
            write_nodes_edges_tables(graph, graph_dir, fold, "pad_external")
        write_graph_report(graph_dir / f"fold{fold}_pad_external_graph_report.json", graph, fold)

        ckpt = checkpoint_root / f"fold{fold}" / "best.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        adapter = LGKEGNNModelAdapter.load(ckpt)
        adapter.device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        if adapter.model is None:
            raise RuntimeError(f"Checkpoint did not contain a loadable model: {ckpt}")
        adapter.model.to(adapter.device)

        result = adapter.predict_proba(ModelBatch(sample_ids=graph["sample_ids"], labels=graph["y"].numpy(), graph=graph, fold=fold))
        pred_path = fold_out / "pad_predictions.csv"
        write_predictions(pred_path, result)
        pred_df = pd.read_csv(pred_path)
        pred_df["fold"] = fold
        fold_predictions.append(pred_df)

        thresholds = adapter.best_thresholds or {}
        threshold = float(thresholds.get("sensitivity_at_least_0_90", thresholds.get("youden", 0.5)))
        fold_thresholds.append(threshold)
        metrics = external_metrics(result, threshold=threshold, pad_manifest=pad)
        metrics_payload = {
            "fold": fold,
            "model": "m5_lgke_gnn",
            "checkpoint": str(ckpt),
            "threshold_rule": "sensitivity_at_least_0_90",
            "threshold": threshold,
            "pad_external": metrics,
            "artifacts": {
                "graph": str(graph_path),
                "predictions": str(pred_path),
            },
        }
        write_json(fold_out / "pad_metrics.json", metrics_payload)
        fold_rows.append(flatten_metric_row(fold, threshold, metrics))
        print(json.dumps({"fold": fold, "pad_metrics": metrics["values"]}, ensure_ascii=False), flush=True)

    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(out_dir / "pad_external_fold_metrics.csv", index=False)
    summary = summarize_fold_metrics(fold_summary)
    ensemble = ensemble_metrics(fold_predictions, pad, threshold=float(np.mean(fold_thresholds)) if fold_thresholds else 0.5)
    write_json(out_dir / "pad_external_summary.json", {"fold_mean_sd": summary, "ensemble": ensemble})
    write_summary_csv(out_dir / "pad_external_summary.csv", summary, ensemble)
    write_subgroup_csv(out_dir / "pad_external_subgroups.csv", ensemble.get("subgroups", {}))
    print(json.dumps({"fold_mean_sd": summary, "ensemble_values": ensemble.get("values", {})}, indent=2, ensure_ascii=False))


def build_external_graph(
    *,
    isic: pd.DataFrame,
    pad: pd.DataFrame,
    folds: pd.DataFrame,
    fold: int,
    preprocessor_dir: Path,
    image_cache: dict[str, np.ndarray],
    k_visual: int,
    k_metadata: int,
    n_visual_prototypes: int,
    n_metadata_prototypes: int,
    image_size: int,
    knn_device: str,
    knn_chunk_size: int,
    seed: int,
) -> dict[str, object]:
    train_ids = set(folds[(folds["fold"] == fold) & (folds["split"] == "train")]["sample_id"].astype(str))
    train_df = isic[isic["sample_id"].astype(str).isin(train_ids)].copy()
    pad_df = pad.copy()
    graph_df = pd.concat([train_df, pad_df], ignore_index=True)
    preprocessor = FoldTabularPreprocessor.load(preprocessor_dir / f"fold{fold}_tabular.pkl")
    return build_graph(
        graph_df=graph_df,
        train_df=train_df,
        query_ids=set(pad_df["sample_id"].astype(str)),
        split="pad_external",
        preprocessor=preprocessor,
        image_cache=image_cache,
        k_visual=k_visual,
        k_metadata=k_metadata,
        n_visual_prototypes=n_visual_prototypes,
        n_metadata_prototypes=n_metadata_prototypes,
        image_size=image_size,
        knn_device=knn_device,
        knn_chunk_size=knn_chunk_size,
        seed=seed + fold,
    )


def load_external_image_feature_cache(
    manifest: pd.DataFrame,
    path: Path,
    *,
    base_path: Path,
    image_size: int,
    refresh: bool,
) -> dict[str, np.ndarray]:
    if path.exists() and not refresh:
        return load_or_build_image_feature_cache(manifest, path, image_size=image_size, refresh=False)
    if not base_path.exists() or refresh:
        return load_or_build_image_feature_cache(manifest, path, image_size=image_size, refresh=refresh)

    loaded = np.load(base_path, allow_pickle=False)
    base_ids = loaded["sample_ids"].astype(str).tolist()
    base_features = loaded["features"].astype(np.float32)
    cache = {sample_id: base_features[idx] for idx, sample_id in enumerate(base_ids)}
    requested_ids = manifest["sample_id"].astype(str).tolist()
    missing_df = manifest[~manifest["sample_id"].astype(str).isin(cache)].copy()
    print(
        f"[pad_external] loaded base image cache: {base_path} ({len(cache)} samples); "
        f"building {len(missing_df)} missing external features",
        flush=True,
    )
    for idx, (_, row) in enumerate(missing_df.iterrows(), start=1):
        cache[str(row["sample_id"])] = image_feature(row, cache, image_size=image_size)
        if idx % 500 == 0:
            print(f"[pad_external] cached external image stats {idx}/{len(missing_df)}", flush=True)

    features = np.vstack([cache[sample_id] for sample_id in requested_ids]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sample_ids=np.asarray(requested_ids, dtype=str), features=features)
    print(f"[pad_external] wrote combined image feature cache: {path} ({len(requested_ids)} samples)", flush=True)
    return {sample_id: features[idx] for idx, sample_id in enumerate(requested_ids)}


def external_metrics(result: PredictionResult, *, threshold: float, pad_manifest: pd.DataFrame) -> dict[str, Any]:
    if result.labels is None:
        raise ValueError("External PAD metrics require labels.")
    values = binary_classification_metrics(result.labels, result.scores, threshold=threshold)
    values["ece"] = expected_calibration_error(result.labels, result.scores)
    values["bacc"] = balanced_accuracy(values)
    manifest = pad_manifest.set_index("sample_id").loc[list(result.sample_ids)]
    groups = {
        "fitzpatrick": manifest["fitzpatrick"].fillna("UNK").astype(str).tolist(),
        "sex": manifest["sex"].fillna("UNK").astype(str).tolist(),
        "anatom_site": manifest["anatom_site"].fillna("UNK").astype(str).tolist(),
        "diagnostic_original": manifest["diagnostic_original"].fillna("UNK").astype(str).tolist(),
    }
    subgroups = subgroup_full_metrics(result.labels, result.scores, groups, threshold=threshold)
    return {"values": values, "subgroups": subgroups}


def ensemble_metrics(predictions: list[pd.DataFrame], pad_manifest: pd.DataFrame, *, threshold: float) -> dict[str, Any]:
    merged: pd.DataFrame | None = None
    for frame in predictions:
        part = frame[["sample_id", "score", "label", "fold"]].rename(columns={"score": f"score_fold{int(frame['fold'].iloc[0])}"})
        part = part.drop(columns=["fold"])
        merged = part if merged is None else merged.merge(part, on=["sample_id", "label"], how="inner")
    if merged is None or merged.empty:
        return {}
    score_cols = [c for c in merged.columns if c.startswith("score_fold")]
    scores = merged[score_cols].mean(axis=1).to_numpy(dtype=float)
    labels = merged["label"].to_numpy(dtype=int)
    result = PredictionResult(sample_ids=merged["sample_id"].astype(str).tolist(), scores=scores, labels=labels)
    metrics = external_metrics(result, threshold=threshold, pad_manifest=pad_manifest)
    metrics["threshold"] = threshold
    metrics["threshold_rule"] = "ensemble_score_mean_fold_high_sensitivity_threshold"
    return metrics


def flatten_metric_row(fold: int, threshold: float, metrics: dict[str, Any]) -> dict[str, Any]:
    values = metrics["values"]
    keys = ["pauc_tpr80", "auprc", "auroc", "sensitivity", "specificity", "precision", "f1", "fnr", "brier", "ece", "bacc"]
    return {"fold": fold, "threshold": threshold} | {key: values.get(key, float("nan")) for key in keys}


def summarize_fold_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for column in frame.columns:
        if column in {"fold"}:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        output[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0,
        }
    return output


def balanced_accuracy(values: dict[str, float]) -> float:
    sens = values.get("sensitivity", float("nan"))
    spec = values.get("specificity", float("nan"))
    return float((sens + spec) / 2.0) if np.isfinite(sens) and np.isfinite(spec) else float("nan")


def write_graph_report(path: Path, graph: dict[str, object], fold: int) -> None:
    edge_counts = relation_counts(graph["edge_type"].numpy())
    write_json(
        path,
        {
            "fold": fold,
            "split": "pad_external",
            "n_nodes": int(graph["x"].shape[0]),
            "n_lesions": int(len(graph["lesion_indices"])),
            "n_eval_lesions": int(len(graph["eval_indices"])),
            "n_edges": int(graph["edge_index"].shape[1]),
            "edge_counts": edge_counts,
        },
    )


def write_summary_csv(path: Path, summary: dict[str, dict[str, float]], ensemble: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "fold_mean", "fold_std", "ensemble"])
        writer.writeheader()
        ensemble_values = ensemble.get("values", {})
        for metric, values in summary.items():
            writer.writerow(
                {
                    "metric": metric,
                    "fold_mean": values["mean"],
                    "fold_std": values["std"],
                    "ensemble": ensemble_values.get(metric, ""),
                }
            )


def write_subgroup_csv(path: Path, subgroups: dict[str, Any]) -> None:
    rows = []
    for group_name, group_values in subgroups.items():
        for subgroup, values in group_values.items():
            rows.append({"group": group_name, "subgroup": subgroup} | values)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    main()
