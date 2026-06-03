from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from skin_lesion_risk.models.adapters.tabular import ConstantRiskModel, LightGBMTabularModel, PlaceholderTabularModel
from skin_lesion_risk.models.base import BaseModelAdapter
from skin_lesion_risk.models.registry import ModelRegistry


class ModelFactory:
    """Create model adapters from dictionaries or YAML config files."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or default_registry()

    def create(self, config: str | Path | dict[str, Any], *, model_name: str | None = None) -> BaseModelAdapter:
        cfg = self._load_config(config)
        model_type = cfg["type"]
        name = model_name or cfg.get("name") or cfg.get("class_name") or model_type
        params = cfg.get("params", {})
        return self.registry.create(model_type, model_name=name, params=params)

    def available(self) -> tuple[str, ...]:
        return self.registry.available()

    @staticmethod
    def _load_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
        if isinstance(config, dict):
            return dict(config)
        with Path(config).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Model config must be a mapping: {config}")
        return loaded


def default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("constant", ConstantRiskModel)
    registry.register("tabular", PlaceholderTabularModel)
    registry.register("tabular_lgbm", LightGBMTabularModel)

    # Lazy imports for torch-dependent models — avoid forcing torch at import time
    try:
        from skin_lesion_risk.models.adapters.image import CNNImageModel, TransformerImageModel
        from skin_lesion_risk.models.adapters.monet import MonetFeatureModel
        from skin_lesion_risk.models.adapters.multimodal import MultimodalFusionModel

        registry.register("image", CNNImageModel)
        registry.register("image_transformer", TransformerImageModel)
        registry.register("monet_feature", MonetFeatureModel)
        registry.register("multimodal", MultimodalFusionModel)
    except ImportError:
        pass
    try:
        from skin_lesion_risk.models.adapters.graph import LGKEGNNModelAdapter

        registry.register("graph_multimodal", LGKEGNNModelAdapter)
    except ModuleNotFoundError:
        pass
    return registry
