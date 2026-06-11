from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _assets_root() -> Path:
    return Path(__file__).resolve().parents[1] / "demo" / "assets"


def _load_manifest() -> dict:
    path = _assets_root() / "sample_predictions_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prediction(relative_path: str) -> pd.DataFrame:
    path = _assets_root() / relative_path
    df = pd.read_csv(path)
    assert {"sample_id", "score", "label"}.issubset(df.columns)
    return df[["sample_id", "score", "label"]].copy()


def test_manifest_has_expected_models() -> None:
    manifest = _load_manifest()
    models = manifest.get("models", [])
    keys = {str(item.get("key")) for item in models}
    assert {"m0_lightgbm", "m4_multimodal_fusion", "m5_lgke_gnn"}.issubset(keys)


def test_model_predictions_are_aligned_by_sample_id_and_label() -> None:
    manifest = _load_manifest()
    models = manifest.get("models", [])
    assert models

    frames: dict[str, pd.DataFrame] = {}
    for item in models:
        key = str(item["key"])
        frames[key] = _load_prediction(str(item["predictions"]))

    id_sets = [set(df["sample_id"].astype(str)) for df in frames.values()]
    assert len(id_sets) >= 1
    first_ids = id_sets[0]
    for ids in id_sets[1:]:
        assert ids == first_ids

    sample_size = int(manifest.get("sample_size", 0))
    assert sample_size == len(first_ids)

    label_maps = {
        key: df.assign(sample_id=df["sample_id"].astype(str)).set_index("sample_id")["label"].astype(int)
        for key, df in frames.items()
    }
    base_key = next(iter(label_maps))
    base_labels = label_maps[base_key]
    for key, labels in label_maps.items():
        assert labels.index.equals(base_labels.index)
        assert (labels == base_labels).all(), f"label mismatch in model: {key}"
