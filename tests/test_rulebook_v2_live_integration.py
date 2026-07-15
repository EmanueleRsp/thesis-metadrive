from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thesis_rl.rulebook.v2.context.live_adapter import install_collision_callback_hook


@pytest.mark.integration
def test_collision_hook_preserves_waymo_dynamics_for_fixed_seed_and_actions() -> None:
    """The local callback observer must not perturb a real ScenarioEnv rollout."""

    from metadrive.engine.asset_loader import AssetLoader
    from metadrive.engine.core.collision_callback import collision_callback
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.env_input_policy import EnvInputPolicy

    data_directory = Path(AssetLoader.file_path("waymo", unix_style=False))
    config = {
        "data_directory": str(data_directory),
        "num_scenarios": 1,
        "start_scenario_index": 0,
        "sequential_seed": False,
        "agent_policy": EnvInputPolicy,
        "use_render": False,
        "log_level": 50,
        "store_data": False,
        "store_map": False,
        "reactive_traffic": True,
        "horizon": None,
        "allowed_more_steps": None,
        "truncate_as_terminate": False,
    }
    baseline = ScenarioEnv(config)
    baseline_trace: list[tuple[np.ndarray, float, bool, bool]] = []
    try:
        baseline_observation, _ = baseline.reset(seed=0)
        action = np.zeros(baseline.action_space.shape, dtype=baseline.action_space.dtype)
        baseline_trace.append((np.asarray(baseline_observation), 0.0, False, False))
        for _ in range(3):
            observation, reward, terminated, truncated, _ = baseline.step(action)
            baseline_trace.append((np.asarray(observation), float(reward), terminated, truncated))
            if terminated or truncated:
                break
    finally:
        baseline.close()

    hooked = ScenarioEnv(config)
    observed_contacts: list[object] = []
    try:
        hooked_observation, _ = hooked.reset(seed=0)
        np.testing.assert_allclose(hooked_observation, baseline_trace[0][0])
        install_collision_callback_hook(
            hooked.engine.physics_world.dynamic_world,
            original_callback=collision_callback,
            observer=lambda contact: observed_contacts.append(contact),
        )
        action = np.zeros(hooked.action_space.shape, dtype=hooked.action_space.dtype)
        for expected_observation, expected_reward, expected_terminated, expected_truncated in baseline_trace[1:]:
            observation, reward, terminated, truncated, _ = hooked.step(action)
            np.testing.assert_allclose(observation, expected_observation)
            assert reward == pytest.approx(expected_reward)
            assert (terminated, truncated) == (expected_terminated, expected_truncated)
            if terminated or truncated:
                break
        assert isinstance(observed_contacts, list)
    finally:
        hooked.close()
