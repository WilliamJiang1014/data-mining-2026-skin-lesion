from __future__ import annotations

import numpy as np
import pandas as pd

from skin_lesion_risk.models.base import ModelBatch
from skin_lesion_risk.models.adapters.tabular import ConstantRiskModel
from skin_lesion_risk.models.factory import ModelFactory


def test_factory_registers_expected_model_types() -> None:
    factory = ModelFactory()
    assert "tabular" in factory.available()
    assert "tabular_lgbm" in factory.available()
    if "graph_multimodal" in factory.available():
        from skin_lesion_risk.models.adapters.graph import LGKEGNNModelAdapter

        assert isinstance(factory.create({"type": "graph_multimodal"}, model_name="m5"), LGKEGNNModelAdapter)


def test_constant_model_uses_training_positive_rate() -> None:
    model = ModelFactory().create({"type": "constant", "params": {"score": 0.25}}, model_name="smoke")
    train = ModelBatch(sample_ids=["a", "b", "c", "d"], labels=np.array([0, 0, 1, 1]))
    test = ModelBatch(sample_ids=["x", "y"], labels=np.array([0, 1]))

    model.fit(train)
    result = model.predict_proba(test).with_metrics(threshold=0.5)

    assert result.scores.tolist() == [0.5, 0.5]
    assert result.metrics is not None
    assert "sensitivity" in result.metrics.values


def test_constant_model_round_trips(tmp_path) -> None:
    model = ConstantRiskModel(model_name="constant", params={})
    train = ModelBatch(sample_ids=["a", "b"], labels=np.array([0, 1]))
    model.fit(train)
    path = tmp_path / "model.pkl"
    model.save(path)

    loaded = ConstantRiskModel.load(path)
    result = loaded.predict_proba(ModelBatch(sample_ids=["x"], labels=np.array([1])))

    assert result.scores.tolist() == [0.5]


def test_tabular_lgbm_adapter_fits_with_sklearn_fallback() -> None:
    model = ModelFactory().create(
        {
            "type": "tabular_lgbm",
            "params": {
                "preferred_backend": "sklearn",
                "n_estimators": 5,
                "learning_rate": 0.1,
                "num_leaves": 7,
            },
        },
        model_name="m0_lightgbm",
    )
    metadata = pd.DataFrame({"age": [30, 40, 70, 80, 35, 75], "size": [1, 2, 8, 9, 2, 10]})
    labels = np.array([0, 0, 1, 1, 0, 1])
    batch = ModelBatch(sample_ids=[f"s{i}" for i in range(len(labels))], labels=labels, metadata=metadata)

    model.fit(batch)
    result = model.predict_proba(batch)

    assert len(result.scores) == len(labels)
    assert np.all((0.0 <= result.scores) & (result.scores <= 1.0))
