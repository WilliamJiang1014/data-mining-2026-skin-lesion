from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_lesion_risk.models.base import ModelBatch
from skin_lesion_risk.models.factory import ModelFactory
from skin_lesion_risk.pipelines.train import list_experiment_models
from skin_lesion_risk.data.preprocessing import FoldTabularPreprocessor
from skin_lesion_risk.evaluation.thresholds import threshold_at_min_sensitivity, youden_threshold


class _Tee(io.TextIOBase):
    """Write to both terminal and log file."""

    def __init__(self, path: Path) -> None:
        self.terminal = sys.__stdout__
        self.log = open(path, "a", encoding="utf-8")
        from datetime import datetime
        self.log.write(f"\n{'=' * 60}\n# {datetime.now().isoformat()}\n{'=' * 60}\n")
        self.log.flush()

    def write(self, data: str) -> int:
        self.terminal.write(data)
        n = self.log.write(data)
        self.log.flush()
        return n

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


def _setup_logging(model: str, fold: int, log_dir: str | None) -> _Tee | None:
    """Setup logging to both terminal and log file. Returns None on failure."""
    if log_dir is None:
        return None
    try:
        d = Path(log_dir)
        if not d.is_absolute():
            d = ROOT / d
        d.mkdir(parents=True, exist_ok=True)
        log_path = d / f"{model}_fold{fold}.log"
        tee = _Tee(log_path)
        sys.stdout = tee
        sys.stderr = tee
        return tee
    except OSError:
        print(f"[warn] cannot create log dir, continuing without file logging", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train configured skin lesion risk models.")
    parser.add_argument("--config", default="configs/experiments/baselines.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--list-models", action="store_true")
    fold_group = parser.add_mutually_exclusive_group(required=False)
    fold_group.add_argument("--fold", type=int, default=None)
    fold_group.add_argument("--all-folds", action="store_true")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--log-dir", default="log")
    parser.add_argument("--manifest", default="data/processed/manifest_isic.csv")
    parser.add_argument("--folds", default="data/processed/folds_isic.csv")
    parser.add_argument("--preprocessor-dir", default="data/processed/preprocessors")
    parser.add_argument("--graph-dir", default="data/processed/graph")
    parser.add_argument("--out-dir", default="data/artifacts/trained_models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--lambda-smooth", type=float, default=None)
    parser.add_argument("--edge-chunk-size", type=int, default=None)
    args = parser.parse_args()

    if args.list_models:
        for name in list_experiment_models(ROOT / args.config):
            print(name)
        return
    if not args.model:
        raise SystemExit("--model is required unless --list-models is used")

    if args.all_folds:
        folds_to_run = list(range(args.num_folds))
    elif args.fold is not None:
        folds_to_run = [args.fold]
    else:
        raise SystemExit("Either --fold N or --all-folds is required")

    cfg = load_yaml(ROOT / args.config)
    model_entry = next((m for m in cfg.get("models", []) if m["name"] == args.model), None)
    if model_entry is None:
        raise SystemExit(f"Model {args.model!r} is not present in {args.config}")

    for fold in folds_to_run:
        args.fold = fold
        tee = _setup_logging(args.model, fold, args.log_dir)
        try:
            print(f"=== {args.model} fold {fold} ===")
            if model_entry.get("type") != "graph_multimodal":
                train_non_graph_model(args, cfg, model_entry)
            else:
                train_graph_model(args, cfg, model_entry)
        finally:
            if tee:
                tee.close()
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return loaded


def load_graph(path: Path) -> dict:
    import torch

    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}. Run scripts/build_graph.py first.")
    return torch.load(path, map_location="cpu", weights_only=False)


def prediction_metrics(result, *, threshold: float, groups: dict | None = None) -> dict:
    from skin_lesion_risk.evaluation.calibration import expected_calibration_error
    from skin_lesion_risk.evaluation.metrics import binary_classification_metrics, subgroup_metric_gaps

    if result.labels is None:
        return {}
    values = binary_classification_metrics(result.labels, result.scores, threshold=threshold)
    values["ece"] = expected_calibration_error(result.labels, result.scores)
    output = {"values": values, "threshold": threshold}
    if groups:
        output["subgroup_gaps"] = subgroup_metric_gaps(result.labels, result.scores, groups, threshold=threshold)
    return output


def write_predictions(path: str | Path, result) -> None:
    rows = result.to_records()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "score", "label"])
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


IMAGE_MODEL_TYPES = {"image", "image_transformer", "monet_feature", "multimodal"}
ALL_NON_GRAPH_TYPES = {"constant", "tabular_lgbm"} | IMAGE_MODEL_TYPES


def train_graph_model(args: argparse.Namespace, experiment_cfg: dict, model_entry: dict) -> None:
    model_cfg = load_yaml(ROOT / model_entry["config"])
    params = dict(model_cfg.get("params", {}))
    params["device"] = args.device
    params["out_dir"] = str((ROOT / args.out_dir / args.model / f"fold{args.fold}").resolve())
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.patience is not None:
        params["patience"] = args.patience
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        params["weight_decay"] = args.weight_decay
    if args.dropout is not None:
        params["dropout"] = args.dropout
    if args.hidden_dim is not None:
        params["hidden_dim"] = args.hidden_dim
    if args.num_layers is not None:
        params["num_layers"] = args.num_layers
    if args.lambda_smooth is not None:
        params["lambda_smooth"] = args.lambda_smooth
    if args.edge_chunk_size is not None:
        params["edge_chunk_size"] = args.edge_chunk_size

    out_dir = Path(params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("train_log.csv", "loss_curve.csv", "val_predictions.csv", "test_predictions.csv", "metrics.json", "last.ckpt"):
        stale_path = out_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    resolved = {"experiment": experiment_cfg, "model": model_cfg | {"params": params}, "fold": args.fold}
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    graph_dir = ROOT / args.graph_dir
    train_graph = load_graph(graph_dir / f"fold{args.fold}_train_graph.pt")
    val_graph = load_graph(graph_dir / f"fold{args.fold}_val_graph.pt")
    test_graph = load_graph(graph_dir / f"fold{args.fold}_test_graph.pt")

    model = ModelFactory().create({"type": "graph_multimodal", "params": params}, model_name=args.model)
    model.fit(
        ModelBatch(sample_ids=train_graph["sample_ids"], labels=train_graph["y"].numpy(), graph=train_graph, fold=args.fold),
        ModelBatch(sample_ids=val_graph["sample_ids"], labels=val_graph["y"].numpy(), graph=val_graph, fold=args.fold),
    )

    val_result = model.predict_proba(ModelBatch(sample_ids=val_graph["sample_ids"], labels=val_graph["y"].numpy(), graph=val_graph, fold=args.fold))
    test_result = model.predict_proba(ModelBatch(sample_ids=test_graph["sample_ids"], labels=test_graph["y"].numpy(), graph=test_graph, fold=args.fold))
    write_predictions(out_dir / "val_predictions.csv", val_result)
    write_predictions(out_dir / "test_predictions.csv", test_result)

    thresholds = getattr(model, "best_thresholds", {}) or {"youden": 0.5, "sensitivity_at_least_0_90": 0.5}
    metrics = {
        "fold": args.fold,
        "model": args.model,
        "thresholds": thresholds,
        "validation": {name: prediction_metrics(val_result, threshold=value) for name, value in thresholds.items()},
        "test": {name: prediction_metrics(test_result, threshold=value) for name, value in thresholds.items()},
        "artifacts": {
            "best_ckpt": str(out_dir / "best.ckpt"),
            "train_log": str(out_dir / "train_log.csv"),
            "loss_curve": str(out_dir / "loss_curve.csv"),
            "val_predictions": str(out_dir / "val_predictions.csv"),
            "test_predictions": str(out_dir / "test_predictions.csv"),
        },
    }
    write_metrics(out_dir / "metrics.json", metrics)

    latest = out_dir / "last.ckpt"
    best = out_dir / "best.ckpt"
    if best.exists():
        shutil.copyfile(best, latest)
    print(json.dumps(metrics["artifacts"], indent=2))


def train_non_graph_model(args: argparse.Namespace, experiment_cfg: dict, model_entry: dict) -> None:
    model_type = str(model_entry.get("type"))
    if model_type not in ALL_NON_GRAPH_TYPES:
        raise SystemExit(f"Unknown non-graph model type: {model_type}")

    model_cfg = load_yaml(ROOT / model_entry["config"])
    params = dict(model_cfg.get("params", {}))
    params.setdefault("seed", experiment_cfg.get("seed", 2026))

    # Load training overrides from configs/training.yaml (batch_size, epochs, etc.)
    training_cfg_path = ROOT / args.training_config
    if training_cfg_path.exists():
        training_cfg = load_yaml(training_cfg_path)
        model_training = training_cfg.get(args.model, {})
        params.update(model_training)

    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.patience is not None:
        params["patience"] = args.patience
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        params["weight_decay"] = args.weight_decay
    if args.dropout is not None:
        params["dropout"] = args.dropout

    is_image_model = model_type in IMAGE_MODEL_TYPES
    if is_image_model:
        params["device"] = args.device
        params["out_dir"] = str((ROOT / args.out_dir / args.model / f"fold{args.fold}").resolve())

    out_dir = (ROOT / args.out_dir / args.model / f"fold{args.fold}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stale_files = ["train_log.csv", "train_summary.csv", "loss_curve.csv", "val_predictions.csv", "test_predictions.csv", "metrics.json", "model.pkl", "best.ckpt", "last.ckpt", "config_resolved.yaml"]
    for stale in stale_files:
        stale_path = out_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    resolved = {"experiment": experiment_cfg, "model": model_cfg | {"params": params}, "fold": args.fold}
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    manifest = pd.read_csv(ROOT / args.manifest)
    folds = pd.read_csv(ROOT / args.folds)
    preprocessor = FoldTabularPreprocessor.load(ROOT / args.preprocessor_dir / f"fold{args.fold}_tabular.pkl")

    include_metadata = model_type in {"tabular_lgbm", "monet_feature", "multimodal"}
    train_batch = build_manifest_batch(manifest, folds, preprocessor, args.fold, "train", include_metadata=include_metadata)
    val_batch = build_manifest_batch(manifest, folds, preprocessor, args.fold, "val", include_metadata=include_metadata)
    test_batch = build_manifest_batch(manifest, folds, preprocessor, args.fold, "test", include_metadata=include_metadata)

    model = ModelFactory().create({"type": model_type, "params": params}, model_name=args.model)
    model.fit(train_batch, val_batch)

    if is_image_model:
        model_path = out_dir / "best.ckpt"
    else:
        model_path = out_dir / "model.pkl"
    model.save(model_path)

    val_result = model.predict_proba(val_batch)
    test_result = model.predict_proba(test_batch)
    write_predictions(out_dir / "val_predictions.csv", val_result)
    write_predictions(out_dir / "test_predictions.csv", test_result)

    thresholds = validation_thresholds(val_result)
    metrics = {
        "fold": args.fold,
        "model": args.model,
        "model_type": model_type,
        "thresholds": thresholds,
        "validation": {name: prediction_metrics(val_result, threshold=value) for name, value in thresholds.items()},
        "test": {name: prediction_metrics(test_result, threshold=value) for name, value in thresholds.items()},
        "artifacts": {
            "model": str(model_path),
            "train_summary": str(out_dir / "train_summary.csv"),
            "val_predictions": str(out_dir / "val_predictions.csv"),
            "test_predictions": str(out_dir / "test_predictions.csv"),
            "metrics": str(out_dir / "metrics.json"),
        },
    }
    if is_image_model:
        metrics["artifacts"]["train_log"] = str(out_dir / "train_log.csv")
        metrics["artifacts"]["loss_curve"] = str(out_dir / "loss_curve.csv")
    backend_used = getattr(model, "backend_used", None)
    if backend_used:
        metrics["backend_used"] = backend_used
    write_metrics(out_dir / "metrics.json", metrics)
    write_train_log(out_dir / "train_summary.csv", args.model, args.fold, train_batch, val_batch, metrics, backend_used=backend_used)
    update_main_results(ROOT / args.reports_dir / "tables" / "main_results.csv", args.model, args.fold, metrics)
    print(json.dumps(metrics["artifacts"], indent=2))


def build_manifest_batch(
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    preprocessor: FoldTabularPreprocessor,
    fold: int,
    split: str,
    *,
    include_metadata: bool,
) -> ModelBatch:
    ids = folds[(folds["fold"] == fold) & (folds["split"] == split)]["sample_id"].astype(str).tolist()
    order = {sample_id: idx for idx, sample_id in enumerate(ids)}
    df = manifest[manifest["sample_id"].astype(str).isin(order)].copy()
    df["_order"] = df["sample_id"].astype(str).map(order)
    df = df.sort_values("_order").drop(columns=["_order"])
    metadata = preprocessor.transform(df) if include_metadata else None
    groups = {
        column: df[column].fillna("UNK").astype(str).tolist()
        for column in ("sex", "anatom_site")
        if column in df.columns
    }
    if include_metadata:
        binned = preprocessor.add_bins(df)
        for column in ("age_bin", "size_bin"):
            groups[column] = binned[column].fillna("UNK").astype(str).tolist()
    image_paths = df["image_path"].astype(str).tolist() if "image_path" in df.columns else None
    raw_metadata = df if include_metadata else None
    return ModelBatch(
        sample_ids=df["sample_id"].astype(str).tolist(),
        labels=df["target"].astype(int).to_numpy(),
        image_paths=image_paths,
        metadata=metadata,
        raw_metadata=raw_metadata,
        groups=groups,
        fold=fold,
        source="isic2024_slice3d_permissive",
    )


def validation_thresholds(result) -> dict[str, float]:
    if result.labels is None or len(np.unique(result.labels)) != 2:
        return {"youden": 0.5, "sensitivity_at_least_0_90": 0.5}
    return {
        "youden": float(youden_threshold(result.labels, result.scores)),
        "sensitivity_at_least_0_90": float(threshold_at_min_sensitivity(result.labels, result.scores, min_sensitivity=0.90)),
    }


def write_train_log(
    path: Path,
    model_name: str,
    fold: int,
    train_batch: ModelBatch,
    val_batch: ModelBatch,
    metrics: dict,
    *,
    backend_used: str | None,
) -> None:
    """Write a one-row summary to train_summary.csv (epoch-level logs are in train_log.csv)."""
    values = metrics["validation"]["youden"]["values"]
    row = {
        "model": model_name,
        "fold": fold,
        "backend_used": backend_used or "",
        "train_samples": len(train_batch),
        "val_samples": len(val_batch),
        "train_positive": int(np.sum(train_batch.labels)) if train_batch.labels is not None else "",
        "val_positive": int(np.sum(val_batch.labels)) if val_batch.labels is not None else "",
        "val_pauc_tpr80": values.get("pauc_tpr80", ""),
        "val_auprc": values.get("auprc", ""),
        "val_auroc": values.get("auroc", ""),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def update_main_results(path: Path, model_name: str, fold: int, metrics: dict) -> None:
    values = metrics["test"]["sensitivity_at_least_0_90"]["values"]
    row = {
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        frame = pd.read_csv(path)
        frame = frame[~((frame["model_name"] == model_name) & (frame["fold"] == fold))]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame = frame.sort_values(["fold", "model_name"]).reset_index(drop=True)
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    main()
