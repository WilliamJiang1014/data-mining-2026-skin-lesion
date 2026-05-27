from __future__ import annotations

from typing import Any, Type

from skin_lesion_risk.models.base import BaseModelAdapter


class ModelRegistry:
    """Registry from config model type to adapter class."""

    def __init__(self) -> None:
        self._classes: dict[str, Type[BaseModelAdapter]] = {}

    def register(self, model_type: str, cls: Type[BaseModelAdapter]) -> None:
        if model_type in self._classes:
            raise ValueError(f"Model type already registered: {model_type}")
        self._classes[model_type] = cls

    def create(self, model_type: str, *, model_name: str, params: dict[str, Any] | None = None) -> BaseModelAdapter:
        if model_type not in self._classes:
            available = ", ".join(sorted(self._classes)) or "<empty>"
            raise KeyError(f"Unknown model type '{model_type}'. Available: {available}")
        return self._classes[model_type](model_name=model_name, params=params)

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._classes))

