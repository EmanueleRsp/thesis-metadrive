from __future__ import annotations

import numpy as np
import pytest

from thesis_rl.adapters.policy_adapter import EnvInputPolicyBridge, PolicyAdapter


def test_env_input_policy_bridge_input_space() -> None:
    space = EnvInputPolicyBridge.get_input_space()
    assert tuple(space.shape) == (2,)
    assert space.dtype == np.float32


def test_policy_adapter_outputs_clipped_action_and_action_info() -> None:
    adapter = PolicyAdapter(low=-1.0, high=1.0, clip=True, expected_shape=(2,), policy_name="EnvInputPolicy")
    action = adapter(np.array([0.5, -0.25], dtype=np.float32))

    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)
    assert "action" in adapter.last_action_info


def test_policy_adapter_action_check_rejects_out_of_space() -> None:
    adapter = PolicyAdapter(
        low=-1.0,
        high=1.0,
        clip=True,
        expected_shape=(2,),
        policy_name="EnvInputPolicy",
        action_check=True,
    )

    with pytest.raises(ValueError):
        adapter(np.array([2.0, 0.0], dtype=np.float32))


def test_policy_adapter_rejects_unknown_policy_name() -> None:
    with pytest.raises(ValueError):
        PolicyAdapter(policy_name="LaneChangePolicy")
