from __future__ import annotations

import numpy as np

from skin_lesion_risk.models.base import ModelBatch
from skin_lesion_risk.models.adapters.graph import LGKEGNNModelAdapter
from skin_lesion_risk.models.factory import ModelFactory


def test_factory_registers_expected_model_types() -> None:
    factory = ModelFactory()
    assert "tabular" in factory.available()
    assert "graph_multimodal" in factory.available()
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
