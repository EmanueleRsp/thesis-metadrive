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
                "generate_probability": 0.25,
                "exploit_probability": 0.75,
                "mab": {
                    "num_arms": 6,
                    "eta": 0.1,
                    "alpha": 0.10,
                    "initial_score": 0.50,
                    "temperature": 0.50,
                },
            },
        }
    )

    assert curriculum.enabled is True
    assert curriculum.kind == "scenario_acl"
    assert curriculum.is_scenario_acl is True
    assert curriculum.scenario_acl.buffer_capacity == 32
    assert curriculum.scenario_acl.mab.num_arms == 6
    assert curriculum.scenario_acl.generate_probability == 0.25


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


@pytest.mark.parametrize(
    "mab_overrides",
    [
        {"initial_weight": 1.0},
        {"weight_clip_min": -5.0},
        {"use_importance_correction": True},
    ],
)
def test_acl_v11_rejects_legacy_or_removed_mab_options(mab_overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="ACL|Legacy"):
        CurriculumConfig.from_mapping(
            {
                "enabled": True,
                "kind": "scenario_acl",
                "scenario_acl": {"mab": {"num_arms": 6, **mab_overrides}},
            }
        )


def test_scenario_acl_runtime_validation_accepts_spawned_vectorized_envs() -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "name": "scenarionet",
                "vectorized": {"enabled": True, "num_envs": 2, "start_method": "spawn"},
            },
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


def test_scenario_acl_runtime_validation_rejects_non_spawn_vectorization() -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "name": "scenarionet",
                "vectorized": {"enabled": True, "num_envs": 2, "start_method": "fork"},
            },
            "reward": {"behavior": "monitor_only"},
        }
    )
    curriculum = CurriculumConfig.from_mapping(
        {"enabled": True, "kind": "scenario_acl", "scenario_acl": {"mab": {"num_arms": 6}}}
    )
    with pytest.raises(ValueError, match="start_method='spawn'"):
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


def test_scenario_acl_config_rejects_rulebook_usefulness_scope() -> None:
    with pytest.raises(ValueError, match="learning-potential-only"):
        CurriculumConfig.from_mapping(
            {
                "enabled": True,
                "kind": "scenario_acl",
                "scenario_acl": {
                    "use_rule_criticality": True,
                    "mab": {"num_arms": 6},
                },
            }
        )


def test_approved_v13_buffer_capacity_resolves_and_keeps_the_warmup_guard() -> None:
    """`TEST-CAT-010` / ACL v1.3 §9, `DEC-010` (ADR-032).

    `buffer_capacity` drops 1000 -> 250 so the scenario buffer is a selective
    active set (12.5% of the 2,000-record frozen training catalog) instead of
    an index over half of it (`RAT-011`). The `warmup <= capacity` guard must
    keep holding, both for the approved pair and for an invalid one.
    """

    curriculum = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {
                "buffer_capacity": 250,
                "warmup_buffer_size": 100,
                "mab": {"num_arms": 6},
            },
        }
    )

    assert curriculum.scenario_acl.buffer_capacity == 250
    assert curriculum.scenario_acl.warmup_buffer_size == 100

    with pytest.raises(ValueError, match="warmup_buffer_size must be <="):
        CurriculumConfig.from_mapping(
            {
                "enabled": True,
                "kind": "scenario_acl",
                "scenario_acl": {
                    "buffer_capacity": 250,
                    "warmup_buffer_size": 251,
                    "mab": {"num_arms": 6},
                },
            }
        )


def test_scenario_acl_buffer_capacity_default_is_the_approved_value() -> None:
    """`TEST-CAT-010`: the code default matches `conf/curriculum/scenario_acl.yaml`."""

    curriculum = CurriculumConfig.from_mapping(
        {
            "enabled": True,
            "kind": "scenario_acl",
            "scenario_acl": {"mab": {"num_arms": 6}},
        }
    )

    assert curriculum.scenario_acl.buffer_capacity == 250
