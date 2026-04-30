from __future__ import annotations

from omegaconf import OmegaConf

from thesis_rl.curriculum.config import CurriculumConfig
from thesis_rl.runtime.builders import merge_env_config_with_overrides
from thesis_rl.runtime.seeding import (
    apply_eval_scenario_seed_split,
    eval_base_seed_from_env_overrides,
    train_episode_seed_from_env_overrides,
    train_reset_seed_from_env_overrides,
)
from thesis_rl.train import (
    _missing_curriculum_metrics,
)


def test_merge_env_config_with_overrides_updates_only_config_block() -> None:
    cfg_env = OmegaConf.create(
        {
            "name": "metadrive",
            "env_id": "MetaDriveEnv",
            "config": {
                "traffic_density": 0.0,
                "start_seed": 5,
                "num_scenarios": 1,
            },
            "observation": {"type": "lidar_state"},
        }
    )

    merged = merge_env_config_with_overrides(
        cfg_env,
        {
            "traffic_density": 0.1,
            "start_seed": 100,
        },
    )

    assert float(merged.config.traffic_density) == 0.1
    assert int(merged.config.start_seed) == 100
    assert int(merged.config.num_scenarios) == 1
    assert str(merged.name) == "metadrive"
    assert str(merged.observation.type) == "lidar_state"


def test_missing_curriculum_metrics_lists_only_absent_keys() -> None:
    metrics = {
        "success_rate": 0.8,
        "success_rate_std": 0.02,
        "collision_rate": 0.01,
        "collision_rate_std": 0.01,
        "out_of_road_rate": 0.01,
        "top_rule_violation_rate": 0.01,
        "route_completion": 0.9,
    }
    curriculum_cfg = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "mode": "auto",
            "promotion": {
                "gates": {
                    "safety": {
                        "collision_rate_max": 0.05,
                        "top_rule_violation_rate_max": 0.02,
                        "out_of_road_rate_max": 0.03,
                    },
                    "task": {
                        "success_rate_min": 0.8,
                        "route_completion_min": 0.85,
                    },
                    "stability": {
                        "success_rate_std_max": 0.1,
                        "collision_rate_std_max": 0.03,
                    },
                }
            },
        }
    )

    missing = _missing_curriculum_metrics(metrics, curriculum_cfg)
    assert missing == []


def test_eval_scenario_seed_split_returns_valid_base_seed_and_episode_window() -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "config": {
                    "start_seed": 0,
                    "num_scenarios": 2,
                }
            },
            "scenario_splits": {
                "stride": 1000,
                "validation_offset": 10_000,
                "test_offset": 20_000,
            },
        }
    )

    overrides = apply_eval_scenario_seed_split(
        base_run_seed=42,
        eval_env_overrides=None,
        cfg=cfg,
        n_eval_episodes=5,
        split="validation",
    )

    assert int(overrides["start_seed"]) == 52_000
    assert int(overrides["num_scenarios"]) == 5
    assert eval_base_seed_from_env_overrides(overrides, cfg) == int(overrides["start_seed"])


def test_eval_scenario_seed_split_keeps_validation_and_test_disjoint() -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "config": {
                    "start_seed": 0,
                    "num_scenarios": 50,
                }
            },
            "scenario_splits": {
                "stride": 1000,
                "validation_offset": 10_000,
                "test_offset": 20_000,
            },
        }
    )

    validation = apply_eval_scenario_seed_split(
        base_run_seed=42,
        eval_env_overrides=None,
        cfg=cfg,
        n_eval_episodes=50,
        split="validation",
    )
    test = apply_eval_scenario_seed_split(
        base_run_seed=42,
        eval_env_overrides=None,
        cfg=cfg,
        n_eval_episodes=50,
        split="test",
    )

    validation_start = int(validation["start_seed"])
    validation_end = validation_start + int(validation["num_scenarios"]) - 1
    test_start = int(test["start_seed"])
    test_end = test_start + int(test["num_scenarios"]) - 1

    assert validation_end < test_start
    assert validation_start == 52_000
    assert test_start == 62_000
    assert test_end == 62_049


def test_train_reset_seed_stays_inside_baseline_train_pool() -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "config": {
                    "start_seed": 0,
                    "num_scenarios": 50,
                }
            }
        }
    )

    assert train_reset_seed_from_env_overrides(None, cfg, reset_offset=0) == 0
    assert train_reset_seed_from_env_overrides(None, cfg, reset_offset=1) == 1
    assert train_reset_seed_from_env_overrides(None, cfg, reset_offset=50) == 0


def test_train_reset_seed_stays_inside_curriculum_stage_pool() -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "config": {
                    "start_seed": 0,
                    "num_scenarios": 50,
                }
            }
        }
    )
    train_overrides = {
        "start_seed": 1000,
        "num_scenarios": 20,
    }

    assert train_reset_seed_from_env_overrides(train_overrides, cfg, reset_offset=0) == 1000
    assert train_reset_seed_from_env_overrides(train_overrides, cfg, reset_offset=19) == 1019
    assert train_reset_seed_from_env_overrides(train_overrides, cfg, reset_offset=20) == 1000


def test_train_episode_seed_sampler_is_deterministic_and_inside_pool() -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "config": {
                    "start_seed": 0,
                    "num_scenarios": 50,
                }
            }
        }
    )
    train_overrides = {
        "start_seed": 1000,
        "num_scenarios": 20,
    }

    first = train_episode_seed_from_env_overrides(
        train_overrides,
        cfg,
        run_seed=42,
        chunk_id=3,
        episode_index=7,
        stage_index=1,
    )
    second = train_episode_seed_from_env_overrides(
        train_overrides,
        cfg,
        run_seed=42,
        chunk_id=3,
        episode_index=7,
        stage_index=1,
    )
    other_episode = train_episode_seed_from_env_overrides(
        train_overrides,
        cfg,
        run_seed=42,
        chunk_id=3,
        episode_index=8,
        stage_index=1,
    )

    assert first == second
    assert 1000 <= first < 1020
    assert 1000 <= other_episode < 1020
