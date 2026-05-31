from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_train import (
    build_manifest_batch,
    load_yaml,
    prediction_metrics,
    update_main_results,
    validation_thresholds,
    write_metrics,
    write_predictions,
    IMAGE_MODEL_TYPES,
)
from skin_lesion_risk.data.preprocessing import FoldTabularPreprocessor
from skin_lesion_risk.models.adapters.tabular import ConstantRiskModel, LightGBMTabularModel
from skin_lesion_risk.models.base import ModelBatch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved model artifacts for an internal fold.")
    parser.add_argument("--config", default="configs/experiments/baselines.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--manifest", default="data/processed/manifest_isic.csv")
    parser.add_argument("--folds", default="data/processed/folds_isic.csv")
    parser.add_argument("--preprocessor-dir", default="data/processed/preprocessors")
    parser.add_argument("--graph-dir", default="data/processed/graph")
    parser.add_argument("--artifacts-dir", default="data/artifacts/trained_models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    experiment_cfg = load_yaml(ROOT / args.config)
    model_entry = next((m for m in experiment_cfg.get("models", []) if m["name"] == args.model), None)
    if model_entry is None:
        raise SystemExit(f"Model {args.model!r} is not present in {args.config}")

    out_dir = ROOT / args.artifacts_dir / args.model / f"fold{args.fold}"
    model_type = str(model_entry.get("type"))
    if model_type == "graph_multimodal":
        result = evaluate_graph(args, out_dir)
    elif model_type in {"constant", "tabular_lgbm"}:
        result = evaluate_manifest_model(args, out_dir, model_type)
    elif model_type in IMAGE_MODEL_TYPES:
        result = evaluate_image_model(args, out_dir, model_type)
    else:
        raise SystemExit(f"Evaluation for model type {model_type!r} is not supported.")

    thresholds = load_thresholds(out_dir / "metrics.json") or validation_thresholds(result)
    metrics = {
        "fold": args.fold,
        "model": args.model,
        "model_type": model_type,
        "thresholds": thresholds,
        "test": {name: prediction_metrics(result, threshold=value) for name, value in thresholds.items()},
        "artifacts": {
            "test_predictions": str(out_dir / "test_predictions.csv"),
            "metrics": str(out_dir / "metrics.json"),
        },
    }
    existing = load_existing_json(out_dir / "metrics.json")
    if existing:
        existing.update(metrics)
        metrics = existing
    write_predictions(out_dir / "test_predictions.csv", result)
    write_metrics(out_dir / "metrics.json", metrics)
    update_main_results(ROOT / args.reports_dir / "tables" / "main_results.csv", args.model, args.fold, metrics)
    print(json.dumps(metrics["artifacts"], indent=2))


def _move_model_to_device(model, device) -> None:
    """Move all sub-modules of a loaded adapter to the target device."""
    from torch import nn

    for attr in ("backbone", "head", "encoder", "image_encoder", "metadata_encoder", "gate"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            module.to(device)


def evaluate_manifest_model(args: argparse.Namespace, out_dir: Path, model_type: str):
    model_path = out_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    model = ConstantRiskModel.load(model_path) if model_type == "constant" else LightGBMTabularModel.load(model_path)
    manifest = pd.read_csv(ROOT / args.manifest)
    folds = pd.read_csv(ROOT / args.folds)
    preprocessor = FoldTabularPreprocessor.load(ROOT / args.preprocessor_dir / f"fold{args.fold}_tabular.pkl")
    batch = build_manifest_batch(manifest, folds, preprocessor, args.fold, "test", include_metadata=model_type != "constant")
    return model.predict_proba(batch)


def evaluate_image_model(args: argparse.Namespace, out_dir: Path, model_type: str):
    import torch
    from skin_lesion_risk.models.adapters.image import CNNImageModel, TransformerImageModel
    from skin_lesion_risk.models.adapters.monet import MonetFeatureModel
    from skin_lesion_risk.models.adapters.multimodal import MultimodalFusionModel

    ckpt = out_dir / "best.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if model_type == "image":
        model = CNNImageModel.load(ckpt)
    elif model_type == "image_transformer":
        model = TransformerImageModel.load(ckpt)
    elif model_type == "monet_feature":
        model = MonetFeatureModel.load(ckpt)
    elif model_type == "multimodal":
        model = MultimodalFusionModel.load(ckpt)
    else:
        raise SystemExit(f"Unknown image model type: {model_type}")

    model.device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    _move_model_to_device(model, model.device)
    include_metadata = model_type in {"monet_feature", "multimodal"}
    manifest = pd.read_csv(ROOT / args.manifest)
    folds = pd.read_csv(ROOT / args.folds)
    preprocessor = FoldTabularPreprocessor.load(ROOT / args.preprocessor_dir / f"fold{args.fold}_tabular.pkl")
    batch = build_manifest_batch(manifest, folds, preprocessor, args.fold, "test", include_metadata=include_metadata)
    return model.predict_proba(batch)


def evaluate_graph(args: argparse.Namespace, out_dir: Path):
    import torch
    from skin_lesion_risk.models.adapters.graph import LGKEGNNModelAdapter

    ckpt = out_dir / "best.ckpt"
    graph_path = ROOT / args.graph_dir / f"fold{args.fold}_test_graph.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    adapter = LGKEGNNModelAdapter.load(ckpt)
    adapter.device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if adapter.model is not None:
        adapter.model.to(adapter.device)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    return adapter.predict_proba(ModelBatch(sample_ids=graph["sample_ids"], labels=graph["y"].numpy(), graph=graph, fold=args.fold))


def load_thresholds(path: Path) -> dict[str, float] | None:
    payload = load_existing_json(path)
    if not payload:
        return None
    thresholds = payload.get("thresholds")
    if isinstance(thresholds, dict):
        return {str(k): float(v) for k, v in thresholds.items()}
    return None


def load_existing_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
