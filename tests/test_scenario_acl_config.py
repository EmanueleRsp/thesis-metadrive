from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from thesis_rl.curriculum import CurriculumConfig, CurriculumManager
from thesis_rl.curriculum.scenario_acl import validate_scenario_acl_runtime_support


def test_curriculum_config_parses_scenario_acl_block() -> None:
    curriculum = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {
                "mode": "mab_generate_only",
                "buffer_capacity": 32,
                "warmup_buffer_size": 8,
                "exploit_probability": 0.75,
                "generate_probability": 0.25,
                "mab": {
                    "num_arms": 4,
                    "eta": 0.1,
                    "alpha": 0.01,
                },
            },
        }
    )

    assert curriculum.enabled is True
    assert curriculum.kind == "scenario_acl"
    assert curriculum.is_scenario_acl is True
    assert curriculum.scenario_acl.mode == "mab_generate_only"
    assert curriculum.scenario_acl.buffer_capacity == 32
    assert curriculum.scenario_acl.mab.num_arms == 4


def test_curriculum_manager_supports_scenario_acl_placeholder_strategy() -> None:
    config = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {
                "mode": "mab_generate_only",
                "exploit_probability": 0.8,
                "generate_probability": 0.2,
            },
        }
    )

    manager = CurriculumManager(config)

    assert manager.get_current_stage().name == "scenario_acl"
    assert manager.get_env_config() == {}
    assert manager.should_promote() is False


def test_scenario_acl_runtime_validation_rejects_vectorized_envs() -> None:
    cfg = OmegaConf.create(
        {
            "env": {"vectorized": {"enabled": True}},
            "reward": {"behavior": "monitor_only"},
        }
    )
    curriculum = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {
                "mode": "mab_generate_only",
                "exploit_probability": 0.8,
                "generate_probability": 0.2,
            },
        }
    )

    with pytest.raises(ValueError, match="env.vectorized.enabled=false"):
        validate_scenario_acl_runtime_support(cfg, curriculum, context="training")


def test_scenario_acl_runtime_validation_is_explicit_until_driver_exists() -> None:
    cfg = OmegaConf.create(
        {
            "env": {"vectorized": {"enabled": False}},
            "reward": {"behavior": "monitor_only"},
        }
    )
    curriculum = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {
                "mode": "mab_generate_only",
                "exploit_probability": 0.8,
                "generate_probability": 0.2,
            },
        }
    )

    validate_scenario_acl_runtime_support(cfg, curriculum, context="training")


def test_scenario_acl_config_rejects_invalid_probability_sum() -> None:
    with pytest.raises(ValueError, match="must equal 1.0"):
        CurriculumConfig.from_mapping(
            {
                "enabled": True,
                "kind": "scenario_acl",
                "scenario_acl": {
                    "mode": "mab_generate_only",
                    "exploit_probability": 0.8,
                    "generate_probability": 0.3,
                },
            }
        )


def test_scenario_acl_config_parses_replay_mode_flags() -> None:
    curriculum = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {
                "mode": "mab_plus_replay",
                "use_replay": True,
                "use_mutation": False,
                "exploit_probability": 0.8,
                "generate_probability": 0.2,
            },
        }
    )

    assert curriculum.scenario_acl.mode == "mab_plus_replay"
    assert curriculum.scenario_acl.use_replay is True
    assert curriculum.scenario_acl.use_mutation is False


def test_scenario_acl_runtime_validation_rejects_mutation_until_supported() -> None:
    cfg = OmegaConf.create(
        {
            "env": {"vectorized": {"enabled": False}},
            "reward": {"behavior": "monitor_only"},
        }
    )
    curriculum = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {
                "mode": "full_curriculum",
                "use_mutation": True,
                "exploit_probability": 0.8,
                "generate_probability": 0.2,
            },
        }
    )

    with pytest.raises(ValueError, match="use_mutation=true"):
        validate_scenario_acl_runtime_support(cfg, curriculum, context="training")
