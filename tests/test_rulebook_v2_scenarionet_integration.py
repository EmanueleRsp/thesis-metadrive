from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from thesis_rl.runtime.wiring.builders import build_train_env


@pytest.mark.integration
@pytest.mark.parametrize("start_scenario_index", (0, 1358))
def test_rulebook_v2_deferred_adapter_completes_one_real_source_step(
    start_scenario_index: int,
) -> None:
    data_root = Path(os.environ.get("SCENARIONET_DATA_ROOT", "data/scenarionet"))
    catalog_path = data_root / "catalog" / "scenario_catalog.parquet"
    runtime_path = data_root / "runtime" / "train"
    if not catalog_path.is_file() or not (runtime_path / "dataset_summary.pkl").is_file():
        pytest.skip("requires the prepared canonical ScenarioNet runtime")
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "paths": {"logs_dir": str(data_root / "logs")},
            "reward": {"behavior": "monitor_only"},
            "rulebook": {"version": "4.7-final-implementation-complete"},
            "env": {
                "name": "scenarionet",
                "env_id": "ThesisScenarioEnv",
                "split": "train",
                "catalog_path": str(catalog_path.resolve()),
                "global_seed": 0,
                "config": {
                    "data_directory": str(runtime_path.resolve()),
                    "num_scenarios": 1,
                    "start_scenario_index": start_scenario_index,
                    "worker_index": 0,
                    "num_workers": 1,
                    "agent_policy": "env_input_policy",
                    "horizon": None,
                    "allowed_more_steps": None,
                    "truncate_as_terminate": False,
                    "reactive_traffic": True,
                    "store_data": False,
                    "store_map": False,
                },
                "episode_control": {"extra_steps_after_scenario": 0},
                "provider": {"kind": "fixed_sequence", "repeat": True},
            },
        }
    )
    env = build_train_env(cfg)
    try:
        observation, _reset_info = env.reset()
        assert np.isfinite(np.asarray(observation)).all()
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        for _ in range(10):
            next_observation, _reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(np.asarray(next_observation)).all()
            assert "rulebook" in info
            assert info["rulebook"]["complete_evaluation"] is True
            if terminated or truncated:
                break
    finally:
        env.close()
