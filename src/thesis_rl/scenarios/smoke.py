from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def smoke_test_scenario_env(
    data_directory: str | Path,
    *,
    scenario_index: int = 0,
    steps: int = 10,
    reactive_traffic: bool = True,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be at least 1")

    from metadrive.envs.scenario_env import ScenarioEnv  # type: ignore[import-not-found]
    from metadrive.policy.env_input_policy import EnvInputPolicy  # type: ignore[import-not-found]

    directory = Path(data_directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Scenario dataset directory does not exist: {directory}")

    env = ScenarioEnv(
        {
            "data_directory": str(directory),
            "num_scenarios": 1,
            "start_scenario_index": int(scenario_index),
            "sequential_seed": False,
            "agent_policy": EnvInputPolicy,
            "use_render": False,
            "log_level": 50,
            "store_data": False,
            "store_map": False,
            "reactive_traffic": bool(reactive_traffic),
            "horizon": None,
            "allowed_more_steps": None,
            "truncate_as_terminate": False,
        }
    )
    try:
        observation, _reset_info = env.reset(seed=int(scenario_index))
        observation_array = np.asarray(observation)
        if not np.isfinite(observation_array).all():
            raise ValueError("ScenarioEnv reset returned a non-finite observation")

        scenario_id = str(env.engine.data_manager.current_scenario_id)
        scenario_length = int(env.engine.data_manager.current_scenario_length)
        executed_steps = 0
        terminated = False
        truncated = False
        for _ in range(steps):
            action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            observation, reward, terminated, truncated, _step_info = env.step(action)
            if not np.isfinite(np.asarray(observation)).all():
                raise ValueError("ScenarioEnv step returned a non-finite observation")
            if not np.isfinite(float(reward)):
                raise ValueError("ScenarioEnv step returned a non-finite reward")
            executed_steps += 1
            if terminated or truncated:
                break

        return {
            "scenario_id": scenario_id,
            "scenario_index": int(scenario_index),
            "scenario_length": scenario_length,
            "observation_shape": list(observation_array.shape),
            "action_shape": list(env.action_space.shape),
            "requested_steps": int(steps),
            "executed_steps": executed_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reactive_traffic": bool(reactive_traffic),
        }
    finally:
        env.close()
