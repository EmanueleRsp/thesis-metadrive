from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from thesis_rl.contracts.observation_schema import SemanticObservationBatchV12
from thesis_rl.envs.factory import _configure_agent_observation
from thesis_rl.envs.observations.semantic_state_v3 import SemanticStateObservationV3


def _valid_batch() -> SemanticObservationBatchV12:
    return SemanticObservationBatchV12(
        ego_history=np.zeros((5, 10), dtype=np.float32),
        ego_history_mask=np.asarray([0, 0, 0, 0, 1], dtype=np.float32),
        ego_current=np.zeros(3, dtype=np.float32),
        route=np.zeros((10, 7), dtype=np.float32),
        route_mask=np.zeros(10, dtype=np.float32),
        dynamic=np.zeros((16, 5, 22), dtype=np.float32),
        dynamic_mask=np.zeros((16, 5), dtype=np.float32),
        static=np.zeros((8, 13), dtype=np.float32),
        static_mask=np.zeros(8, dtype=np.float32),
        lane_road=np.zeros(14, dtype=np.float32),
        controls=np.zeros((8, 17), dtype=np.float32),
        controls_mask=np.zeros(8, dtype=np.float32),
        interactions=np.zeros((8, 35), dtype=np.float32),
        interactions_mask=np.zeros(8, dtype=np.float32),
        compliance_history=np.zeros((21, 24), dtype=np.float32),
        compliance_history_mask=np.asarray([0] * 20 + [1], dtype=np.float32),
        yellow_onset_memory=np.zeros(3, dtype=np.float32),
    )


def test_semantic_v3_requires_explicit_batch_builder() -> None:
    observation = SemanticStateObservationV3({})
    assert observation.observation_space.shape == (3064,)
    with pytest.raises(RuntimeError, match="perception-bounded batch builder"):
        observation.observe(object())


def test_factory_exposes_v3_as_an_explicit_observation_type() -> None:
    env_cfg: dict[str, object] = {}
    _configure_agent_observation(env_cfg, {"type": "semantic_v3"})
    assert env_cfg["agent_observation"] is SemanticStateObservationV3


def test_semantic_v3_flattens_only_schema_owned_batch() -> None:
    observation = SemanticStateObservationV3({})
    batch = _valid_batch()
    observation.set_batch_builder(lambda vehicle: batch)
    result = observation.observe(object())
    assert result.shape == (3064,)
    assert result.dtype == np.float32
    assert np.array_equal(result, observation.schema.flatten_numpy(batch))


def test_semantic_v3_rejects_nonzero_compliance_padding() -> None:
    observation = SemanticStateObservationV3({})
    batch = _valid_batch()
    compliance_history = batch.compliance_history.copy()
    compliance_history[0, 0] = 0.25
    invalid = replace(batch, compliance_history=compliance_history)
    observation.set_batch_builder(lambda vehicle: invalid)
    with pytest.raises(ValueError, match="Masked 'compliance_history' tokens"):
        observation.observe(object())


def test_semantic_v3_rejects_a_batch_with_all_variable_tokens_masked() -> None:
    observation = SemanticStateObservationV3({})
    batch = _valid_batch()
    batch = replace(
        batch,
        ego_history_mask=np.zeros(5, dtype=np.float32),
        compliance_history_mask=np.zeros(21, dtype=np.float32),
    )
    observation.set_batch_builder(lambda vehicle: batch)
    with pytest.raises(ValueError, match="mask all variable tokens"):
        observation.observe(object())
