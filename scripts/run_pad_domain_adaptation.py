from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_graph import build_graph, resolve_knn_device
from run_pad_external_eval import load_external_image_feature_cache
from skin_lesion_risk.data.preprocessing import FoldTabularPreprocessor, default_preprocessor_for
from skin_lesion_risk.data.splits import make_patient_level_folds
from skin_lesion_risk.models.adapters.graph import prediction_metrics, write_metrics, write_predictions
from skin_lesion_risk.models.base import ModelBatch
from skin_lesion_risk.models.factory import ModelFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Train M5 with PAD domain adaptation and held-out PAD test evaluation.")
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--isic-manifest", default="data/processed/manifest_isic.csv")
    parser.add_argument("--pad-manifest", default="data/processed/manifest_pad.csv")
    parser.add_argument("--isic-folds", default="data/processed/folds_isic.csv")
    parser.add_argument("--pad-folds", default="data/processed/folds_pad_adapt.csv")
    parser.add_argument("--preprocessor-dir", default="data/processed/preprocessors")
    parser.add_argument("--model-config", default="configs/models/m5_lgke_gnn.yaml")
    parser.add_argument("--out-dir", default="data/artifacts/trained_models/pad_domain_adaptation")
    parser.add_argument("--folds-to-run", default="0,1,2,3,4")
    parser.add_argument("--train-mode", choices=["mixed", "pad_only"], default="mixed")
    parser.add_argument("--preprocessor-mode", choices=["isic", "refit"], default="isic")
    parser.add_argument("--initial-checkpoint-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pad-loss-weight", type=float, default=8.0)
    parser.add_argument("--isic-loss-weight", type=float, default=1.0)
    parser.add_argument("--edge-chunk-size", type=int, default=250_000)
    parser.add_argument("--k-visual", type=int, default=10)
    parser.add_argument("--k-metadata", type=int, default=10)
    parser.add_argument("--n-visual-prototypes", type=int, default=64)
    parser.add_argument("--n-metadata-prototypes", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-feature-cache", default="data/processed/embeddings/image_stats_features_with_pad.npz")
    parser.add_argument("--base-image-feature-cache", default="data/processed/embeddings/image_stats_features.npz")
    parser.add_argument("--knn-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--knn-chunk-size", type=int, default=4096)
    parser.add_argument("--reuse-graphs", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    isic = pd.read_csv(ROOT / args.isic_manifest)
    pad = pd.read_csv(ROOT / args.pad_manifest)
    isic_folds = pd.read_csv(ROOT / args.isic_folds)
    pad_folds = load_or_create_pad_folds(pad, ROOT / args.pad_folds, seed=args.seed)
    out_root = ROOT / args.out_dir / args.variant_name
    graph_root = out_root / "graphs"
    preprocessor_root = out_root / "preprocessors"
    out_root.mkdir(parents=True, exist_ok=True)
    graph_root.mkdir(parents=True, exist_ok=True)
    preprocessor_root.mkdir(parents=True, exist_ok=True)

    image_cache = load_external_image_feature_cache(
        pd.concat([isic, pad], ignore_index=True),
        ROOT / args.image_feature_cache,
        base_path=ROOT / args.base_image_feature_cache,
        image_size=args.image_size,
        refresh=False,
    )
    knn_device = resolve_knn_device(args.knn_device)

    fold_rows = []
    for fold in [int(x) for x in args.folds_to_run.split(",") if x.strip()]:
        fold_out = out_root / f"fold{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        train_graph, val_graph, test_graph = load_or_build_graphs(
            args=args,
            fold=fold,
            isic=isic,
            pad=pad,
            isic_folds=isic_folds,
            pad_folds=pad_folds,
            image_cache=image_cache,
            graph_root=graph_root,
            preprocessor_root=preprocessor_root,
            knn_device=knn_device,
        )
        params = model_params(args, fold=fold, fold_out=fold_out)
        model = ModelFactory().create({"type": "graph_multimodal", "params": params}, model_name="m5_lgke_gnn_pad_adapt")
        model.fit(
            ModelBatch(sample_ids=train_graph["sample_ids"], labels=train_graph["y"].numpy(), graph=train_graph, fold=fold),
            ModelBatch(sample_ids=val_graph["sample_ids"], labels=val_graph["y"].numpy(), graph=val_graph, fold=fold),
        )
        val_result = model.predict_proba(ModelBatch(sample_ids=val_graph["sample_ids"], labels=val_graph["y"].numpy(), graph=val_graph, fold=fold))
        test_result = model.predict_proba(ModelBatch(sample_ids=test_graph["sample_ids"], labels=test_graph["y"].numpy(), graph=test_graph, fold=fold))
        write_predictions(fold_out / "pad_val_predictions.csv", val_result)
        write_predictions(fold_out / "pad_test_predictions.csv", test_result)

        thresholds = getattr(model, "best_thresholds", {}) or {"youden": 0.5, "sensitivity_at_least_0_90": 0.5}
        metrics = {
            "fold": fold,
            "variant": args.variant_name,
            "train_mode": args.train_mode,
            "preprocessor_mode": args.preprocessor_mode,
            "thresholds": thresholds,
            "validation": {name: prediction_metrics(val_result, threshold=value) for name, value in thresholds.items()},
            "pad_test": {name: prediction_metrics(test_result, threshold=value) for name, value in thresholds.items()},
            "artifacts": {
                "best_ckpt": str(fold_out / "best.ckpt"),
                "train_log": str(fold_out / "train_log.csv"),
                "loss_curve": str(fold_out / "loss_curve.csv"),
                "pad_val_predictions": str(fold_out / "pad_val_predictions.csv"),
                "pad_test_predictions": str(fold_out / "pad_test_predictions.csv"),
            },
        }
        write_metrics(fold_out / "metrics.json", metrics)
        best = fold_out / "best.ckpt"
        latest = fold_out / "last.ckpt"
        if best.exists():
            shutil.copyfile(best, latest)
        row = flatten_fold_metrics(fold, metrics, rule="sensitivity_at_least_0_90")
        fold_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(out_root / "pad_adaptation_fold_metrics.csv", index=False)
    write_metrics(out_root / "pad_adaptation_summary.json", summarize(fold_summary))
    write_summary_csv(out_root / "pad_adaptation_summary.csv", fold_summary)


def load_or_create_pad_folds(pad: pd.DataFrame, path: Path, *, seed: int) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    folds = make_patient_level_folds(pad, n_splits=5, val_ratio=0.1, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)
    return folds


def load_or_build_graphs(
    *,
    args: argparse.Namespace,
    fold: int,
    isic: pd.DataFrame,
    pad: pd.DataFrame,
    isic_folds: pd.DataFrame,
    pad_folds: pd.DataFrame,
    image_cache: dict[str, np.ndarray],
    graph_root: Path,
    preprocessor_root: Path,
    knn_device: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = {split: graph_root / f"fold{fold}_{split}_graph.pt" for split in ("train", "val", "test")}
    if args.reuse_graphs and all(path.exists() for path in paths.values()):
        return tuple(torch.load(paths[split], map_location="cpu", weights_only=False) for split in ("train", "val", "test"))  # type: ignore[return-value]

    pad_ids = {
        split: set(pad_folds[(pad_folds["fold"] == fold) & (pad_folds["split"] == split)]["sample_id"].astype(str))
        for split in ("train", "val", "test")
    }
    pad_parts = {split: pad[pad["sample_id"].astype(str).isin(ids)].copy() for split, ids in pad_ids.items()}
    if args.train_mode == "mixed":
        isic_train_ids = set(isic_folds[(isic_folds["fold"] == fold) & (isic_folds["split"] == "train")]["sample_id"].astype(str))
        isic_train = isic[isic["sample_id"].astype(str).isin(isic_train_ids)].copy()
        train_df = pd.concat([isic_train, pad_parts["train"]], ignore_index=True)
    else:
        train_df = pad_parts["train"].copy()

    preprocessor = fit_or_load_preprocessor(
        args=args,
        fold=fold,
        train_df=train_df,
        out_dir=preprocessor_root,
    )
    graphs = {}
    for split in ("train", "val", "test"):
        query_df = train_df if split == "train" else pad_parts[split]
        graph_df = train_df if split == "train" else pd.concat([train_df, query_df], ignore_index=True)
        graph = build_graph(
            graph_df=graph_df,
            train_df=train_df,
            query_ids=set(query_df["sample_id"].astype(str)),
            split=split,
            preprocessor=preprocessor,
            image_cache=image_cache,
            k_visual=args.k_visual,
            k_metadata=args.k_metadata,
            n_visual_prototypes=args.n_visual_prototypes,
            n_metadata_prototypes=args.n_metadata_prototypes,
            image_size=args.image_size,
            knn_device=knn_device,
            knn_chunk_size=args.knn_chunk_size,
            seed=args.seed + fold,
        )
        torch.save(graph, paths[split])
        graphs[split] = graph
    return graphs["train"], graphs["val"], graphs["test"]


def fit_or_load_preprocessor(
    *,
    args: argparse.Namespace,
    fold: int,
    train_df: pd.DataFrame,
    out_dir: Path,
) -> FoldTabularPreprocessor:
    if args.preprocessor_mode == "isic":
        return FoldTabularPreprocessor.load(ROOT / args.preprocessor_dir / f"fold{fold}_tabular.pkl")
    path = out_dir / f"fold{fold}_tabular.pkl"
    if args.reuse_graphs and path.exists():
        return FoldTabularPreprocessor.load(path)
    preprocessor = default_preprocessor_for(train_df).fit(train_df)
    preprocessor.save(path)
    preprocessor.save_schema(out_dir / f"fold{fold}_tabular_schema.json")
    return preprocessor


def model_params(args: argparse.Namespace, *, fold: int, fold_out: Path) -> dict[str, Any]:
    with (ROOT / args.model_config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params = dict(cfg.get("params", {}))
    params.update(
        {
            "device": args.device,
            "out_dir": str(fold_out.resolve()),
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "edge_chunk_size": args.edge_chunk_size,
            "source_loss_weights": {
                "isic2024_slice3d_permissive": args.isic_loss_weight,
                "pad_ufes_20": args.pad_loss_weight,
                "*": 1.0,
            },
        }
    )
    if args.initial_checkpoint_root:
        params["initial_checkpoint"] = str((ROOT / args.initial_checkpoint_root / f"fold{fold}" / "best.ckpt").resolve())
    return params


def flatten_fold_metrics(fold: int, metrics: dict[str, Any], *, rule: str) -> dict[str, Any]:
    values = metrics["pad_test"][rule]["values"]
    keys = ["pauc_tpr80", "auprc", "auroc", "sensitivity", "specificity", "precision", "f1", "fnr", "brier", "ece"]
    row = {"fold": fold}
    row.update({key: values.get(key, float("nan")) for key in keys})
    return row


def summarize(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for column in frame.columns:
        if column == "fold":
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        output[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0,
        }
    return output


def write_summary_csv(path: Path, frame: pd.DataFrame) -> None:
    summary = summarize(frame)
    rows = [{"metric": key, "mean": value["mean"], "std": value["std"]} for key, value in summary.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
