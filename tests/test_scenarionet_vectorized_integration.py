from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from thesis_rl.runtime.wiring.builders import build_train_env
from thesis_rl.envs.factory import _validate_scenarionet_catalog_runtime
from thesis_rl.scenarios.catalog import read_scenario_catalog


def _require_matching_runtime(catalog_path: Path, runtime_path: Path) -> None:
    try:
        catalog = read_scenario_catalog(catalog_path)
        _validate_scenarionet_catalog_runtime(
            catalog, split="train", data_directory=str(runtime_path)
        )
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"requires matching real ScenarioNet fixture: {exc}")


def test_scenarionet_vectorized_spawn_smoke() -> None:
    data_root = Path(os.environ.get("SCENARIONET_DATA_ROOT", "data/scenarionet"))
    catalog_path = data_root / "catalog" / "scenario_catalog.parquet"
    runtime_path = data_root / "runtime" / "train"
    if not catalog_path.is_file() or not (runtime_path / "dataset_summary.pkl").is_file():
        pytest.skip("requires the prepared ScenarioNet catalog and runtime/train view")
    _require_matching_runtime(catalog_path, runtime_path)

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "paths": {"logs_dir": str(data_root / "logs")},
            "reward": {"behavior": "off"},
            "env": {
                "name": "scenarionet",
                "env_id": "ThesisScenarioEnv",
                "split": "train",
                "catalog_path": str(catalog_path.resolve()),
                "global_seed": 0,
                "config": {
                    "data_directory": str(runtime_path.resolve()),
                    "num_scenarios": -1,
                    "start_scenario_index": 0,
                    "worker_index": 0,
                    "num_workers": 1,
                    "sequential_seed": False,
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
                "vectorized": {"enabled": True, "num_envs": 2, "start_method": "spawn"},
            },
        }
    )

    vector_env = build_train_env(cfg)
    try:
        observations = vector_env.reset()
        assert observations.shape[0] == 2
        actions = np.zeros((2, *vector_env.action_space.shape), dtype=np.float32)
        next_observations, rewards, dones, infos = vector_env.step(actions)
        assert next_observations.shape[0] == 2
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert len(infos) == 2
        assert all(info.get("scenario_id") for info in infos)
        saw_auto_reset = False
        for _ in range(230):
            _observations, _rewards, dones, infos = vector_env.step(actions)
            if bool(np.any(dones)):
                saw_auto_reset = True
                terminal_infos = [
                    info
                    for info in infos
                    if info.get("TimeLimit.truncated")
                    or info.get("terminal_observation") is not None
                ]
                assert terminal_infos
                assert all(info.get("terminal_observation") is not None for info in terminal_infos)
                break
        assert saw_auto_reset
    finally:
        vector_env.close()


def test_scenarionet_mixed_uniform_provider_smoke() -> None:
    data_root = Path(os.environ.get("SCENARIONET_DATA_ROOT", "data/scenarionet"))
    catalog_path = data_root / "catalog" / "scenario_catalog.parquet"
    runtime_path = data_root / "runtime" / "train"
    if not catalog_path.is_file() or not (runtime_path / "dataset_summary.pkl").is_file():
        pytest.skip("requires the prepared mixed ScenarioNet catalog and runtime/train view")
    _require_matching_runtime(catalog_path, runtime_path)

    cfg = OmegaConf.create(
        {
            "seed": 17,
            "paths": {"logs_dir": str(data_root / "logs")},
            "reward": {"behavior": "off"},
            "env": {
                "name": "scenarionet",
                "env_id": "ThesisScenarioEnv",
                "split": "train",
                "catalog_path": str(catalog_path.resolve()),
                "global_seed": 17,
                "config": {
                    "data_directory": str(runtime_path.resolve()),
                    "num_scenarios": -1,
                    "start_scenario_index": 0,
                    "worker_index": 0,
                    "num_workers": 1,
                    "sequential_seed": False,
                    "agent_policy": "env_input_policy",
                    "horizon": None,
                    "allowed_more_steps": None,
                    "truncate_as_terminate": False,
                    "reactive_traffic": True,
                    "store_data": False,
                    "store_map": False,
                },
                "episode_control": {"extra_steps_after_scenario": 50},
                "provider": {
                    "kind": "uniform",
                    "strict": True,
                    "allow_fallback": False,
                    "source_probability": {"waymo": 0.5, "pg": 0.5},
                },
                "vectorized": {"enabled": True, "num_envs": 2, "start_method": "spawn"},
            },
        }
    )
    vector_env = build_train_env(cfg)
    try:
        observed_sources: set[str] = set()
        for _ in range(8):
            vector_env.reset()
            observed_sources.update(
                str(info.get("source"))
                for info in vector_env.reset_infos
                if info.get("source") is not None
            )
        assert observed_sources == {"waymo", "pg"}
        vector_env.reset()
        observations, rewards, _dones, infos = vector_env.step(
            np.zeros((2, *vector_env.action_space.shape), dtype=np.float32)
        )
        assert observations.shape[0] == 2
        assert observations.shape[1:] == vector_env.observation_space.shape
        assert rewards.shape == (2,)
        assert all(info.get("source") in {"waymo", "pg"} for info in infos)
    finally:
        vector_env.close()
