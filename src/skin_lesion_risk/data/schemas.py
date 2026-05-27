from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_COLUMNS = (
    "sample_id",
    "image_path",
    "target",
    "patient_id",
    "lesion_id",
    "age",
    "sex",
    "anatom_site",
    "size_mm",
    "source",
    "fold",
)


@dataclass(frozen=True)
class ManifestSchema:
    """Column names used by all pipelines."""

    sample_id: str = "sample_id"
    image_path: str = "image_path"
    target: str = "target"
    patient_id: str = "patient_id"
    lesion_id: str = "lesion_id"
    fold: str = "fold"
    source: str = "source"
    protected_attributes: tuple[str, ...] = ("sex", "age_bin", "anatom_site", "fitzpatrick")


@dataclass(frozen=True)
class GraphFiles:
    """Per-fold graph inputs for LGKE-GNN style models."""

    node_table: Path | None = None
    edge_table: Path | None = None
    feature_table: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

