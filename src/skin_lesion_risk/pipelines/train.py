from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from skin_lesion_risk.models.factory import ModelFactory


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Experiment config must be a mapping: {path}")
    return cfg


def list_experiment_models(path: str | Path) -> list[str]:
    cfg = load_experiment_config(path)
    return [model["name"] for model in cfg.get("models", [])]


def build_model_from_experiment(path: str | Path, model_name: str):
    cfg = load_experiment_config(path)
    matches = [model for model in cfg.get("models", []) if model["name"] == model_name]
    if not matches:
        raise KeyError(f"Model not found in experiment config: {model_name}")
    model_cfg_path = matches[0]["config"]
    return ModelFactory().create(model_cfg_path, model_name=model_name)

