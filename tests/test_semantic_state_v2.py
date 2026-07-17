from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from thesis_rl.contracts.observation_schema import SemanticObservationBatch
from thesis_rl.envs.factory import _configure_agent_observation
from thesis_rl.envs.observations.semantic_state_v2 import SemanticStateObservationV2


def _valid_batch() -> SemanticObservationBatch:
    return SemanticObservationBatch(
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
        temporal=np.zeros(5, dtype=np.float32),
    )


def test_semantic_v2_requires_explicit_batch_builder() -> None:
    observation = SemanticStateObservationV2({})
    assert observation.observation_space.shape == (2541,)
    with pytest.raises(RuntimeError, match="causal v1.1 batch builder"):
        observation.observe(object())


def test_factory_exposes_v2_as_an_explicit_observation_type() -> None:
    env_cfg: dict[str, object] = {}
    _configure_agent_observation(env_cfg, {"type": "semantic_v2"})
    assert env_cfg["agent_observation"] is SemanticStateObservationV2


def test_semantic_v2_flattens_only_schema_owned_batch() -> None:
    observation = SemanticStateObservationV2({})
    batch = _valid_batch()
    observation.set_batch_builder(lambda vehicle: batch)
    result = observation.observe(object())
    assert result.shape == (2541,)
    assert result.dtype == np.float32
    assert np.array_equal(result, observation.schema.flatten_numpy(batch))


def test_semantic_v2_rejects_nonzero_padding() -> None:
    observation = SemanticStateObservationV2({})
    batch = _valid_batch()
    route = batch.route.copy()
    route[0, 0] = 0.25
    invalid = replace(batch, route=route)
    observation.set_batch_builder(lambda vehicle: invalid)
    with pytest.raises(ValueError, match="Masked 'route' tokens"):
        observation.observe(object())


def test_semantic_v2_rejects_a_batch_with_all_variable_tokens_masked() -> None:
    observation = SemanticStateObservationV2({})
    batch = _valid_batch()
    batch = replace(batch, ego_history_mask=np.zeros(5, dtype=np.float32))
    observation.set_batch_builder(lambda vehicle: batch)
    with pytest.raises(ValueError, match="mask all variable tokens"):
        observation.observe(object())
