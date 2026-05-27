"""Utilities for building and validating unified manifest files."""

from __future__ import annotations

from collections.abc import Iterable

from skin_lesion_risk.data.schemas import REQUIRED_MANIFEST_COLUMNS


def missing_manifest_columns(columns: Iterable[str]) -> list[str]:
    present = set(columns)
    return [column for column in REQUIRED_MANIFEST_COLUMNS if column not in present]

