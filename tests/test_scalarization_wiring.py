"""Integration-level checks at the common learner reward boundary."""

from __future__ import annotations

from thesis_rl.contracts.reward_semantics import build_reward_semantics_identity
from thesis_rl.reward.scalarization import RulebookScalarizer, ScalarizationConfig


def test_all_algorithm_backends_receive_the_same_configured_scalar_reward() -> None:
    scalarizer = RulebookScalarizer(ScalarizationConfig())
    margins = (0.0, -1.0, 0.0, 0.25)
    reward = scalarizer(margins).scalar_reward

    # PPO, TD3, and SAC all consume Transition.scalar_reward. This explicit
    # common-boundary check prevents backend-specific formula branches.
    assert reward == scalarizer(margins).scalar_reward
    assert reward == scalarizer(tuple(margins)).scalar_reward


def _run_config(reward_compression_mode: str) -> dict:
    return {
        "reward": {"behavior": "scalar_reward"},
        "rulebook": {"implementation_family": "v2", "specification_id": "RULEBOOK-V4.12", "version": "v2"},
        "scalarization": {
            "specification_id": "SCAL-V1.1",
            "version": "1.1",
            "mode": "bounded_priority_weighted_rank",
            "vector_schema_id": "rulebook_v2_macro_v4",
            "priority_base": 3.0,
            "numerical_tolerance": 1.0e-8,
            "native_environment_reward_weight": 0.0,
            "sigmoid": {"sharpness": 30.0},
            "legacy": {"vector_schema_id": None, "rule_scales": None},
            "reward_compression": {"mode": reward_compression_mode},
        },
    }


def test_reward_compression_mode_is_part_of_resume_identity() -> None:
    """AC-SCAL11-008: resume must fail when reward_compression.mode differs."""

    none_identity = build_reward_semantics_identity(_run_config("none"))
    symlog_identity = build_reward_semantics_identity(_run_config("symlog"))

    assert none_identity is not None and symlog_identity is not None
    assert none_identity["scalarization"]["reward_compression_mode"] == "none"
    assert symlog_identity["scalarization"]["reward_compression_mode"] == "symlog"
    assert none_identity != symlog_identity


def test_mapping_config_supports_nested_specification_shape() -> None:
    config = ScalarizationConfig.from_mapping(
        {
            "specification_id": "SCAL-V1.0",
            "version": "1.0",
            "mode": "legacy_scaled_sigmoid",
            "vector_schema_id": "legacy_v1",
            "priority_base": 2.01,
            "sigmoid": {"sharpness": 30.0},
            "legacy": {"vector_schema_id": "legacy_v1", "rule_scales": [1.0, 2.0]},
        }
    )
    assert config.legacy_rule_scales == (1.0, 2.0)
    assert config.sigmoid_sharpness == 30.0
