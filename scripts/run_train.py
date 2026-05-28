from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_lesion_risk.models.adapters.graph import prediction_metrics, write_metrics, write_predictions
from skin_lesion_risk.models.base import ModelBatch
from skin_lesion_risk.models.factory import ModelFactory
from skin_lesion_risk.pipelines.train import list_experiment_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train configured skin lesion risk models.")
    parser.add_argument("--config", default="configs/experiments/baselines.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--graph-dir", default="data/processed/graph")
    parser.add_argument("--out-dir", default="data/artifacts/trained_models")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
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

    cfg = load_yaml(ROOT / args.config)
    model_entry = next((m for m in cfg.get("models", []) if m["name"] == args.model), None)
    if model_entry is None:
        raise SystemExit(f"Model {args.model!r} is not present in {args.config}")
    if model_entry.get("type") != "graph_multimodal":
        model = ModelFactory().create(ROOT / model_entry["config"], model_name=args.model)
        print(f"Created model: {model.model_name} ({model.model_type})")
        print("Only m5_lgke_gnn graph training is implemented in this script.")
        return

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
    resolved = {"experiment": cfg, "model": model_cfg | {"params": params}, "fold": args.fold}
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


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return loaded


def load_graph(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}. Run scripts/build_graph.py first.")
    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    main()
