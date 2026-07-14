from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from thesis_rl.curriculum import CurriculumConfig
from thesis_rl.curriculum.scenario_acl import validate_scenario_acl_runtime_support


def test_curriculum_config_parses_scenario_acl_block() -> None:
    curriculum = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {
                "buffer_capacity": 32,
                "warmup_buffer_size": 8,
                "exploit_probability": 0.75,
                "mab": {
                    "num_arms": 6,
                    "eta": 0.1,
                    "alpha": 0.01,
                },
            },
        }
    )

    assert curriculum.enabled is True
    assert curriculum.kind == "scenario_acl"
    assert curriculum.is_scenario_acl is True
    assert curriculum.scenario_acl.buffer_capacity == 32
    assert curriculum.scenario_acl.mab.num_arms == 6


def test_curriculum_config_parses_scenarionet_semantic_acl_block() -> None:
    curriculum = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {
                "use_scenario_buffer": False,
                "use_replay": False,
                "mab": {"num_arms": 6},
            },
        }
    )

    assert curriculum.scenario_acl.mab.num_arms == 6


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
                "mab": {"num_arms": 6},
            },
        }
    )

    with pytest.raises(ValueError, match="env.vectorized.enabled=false"):
        validate_scenario_acl_runtime_support(cfg, curriculum, context="training")


def test_scenario_acl_runtime_validation_is_explicit_until_driver_exists() -> None:
    cfg = OmegaConf.create(
        {
            "env": {"name": "scenarionet", "vectorized": {"enabled": False}},
            "reward": {"behavior": "monitor_only"},
        }
    )
    curriculum = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {
                "mab": {"num_arms": 6},
            },
        }
    )

    validate_scenario_acl_runtime_support(cfg, curriculum, context="training")


def test_scenario_acl_requires_scenarionet_env() -> None:
    cfg = OmegaConf.create(
        {
            "env": {"name": "metadrive", "vectorized": {"enabled": False}},
            "reward": {"behavior": "monitor_only"},
        }
    )
    curriculum = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {
                "use_scenario_buffer": False,
                "use_replay": False,
                "mab": {"num_arms": 6},
            },
        }
    )

    with pytest.raises(ValueError, match="requires env=scenarionet"):
        validate_scenario_acl_runtime_support(cfg, curriculum, context="training")


def test_scenario_acl_config_parses_replay_flags() -> None:
    curriculum = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {
                "use_replay": True,
                "mab": {"num_arms": 6},
            },
        }
    )

    assert curriculum.scenario_acl.use_replay is True


def test_scenario_acl_config_rejects_mutation_scope() -> None:
    with pytest.raises(ValueError, match="mutation is out of scope"):
        CurriculumConfig.from_mapping(
            {
                "enabled": True,
                "kind": "scenario_acl",
                "scenario_acl": {
                    "use_mutation": True,
                    "mab": {"num_arms": 6},
                },
            }
        )
