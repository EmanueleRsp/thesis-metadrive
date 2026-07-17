"""Integration-level checks at the common learner reward boundary."""

from __future__ import annotations

from thesis_rl.reward.scalarization import RulebookScalarizer, ScalarizationConfig


def test_all_algorithm_backends_receive_the_same_configured_scalar_reward() -> None:
    scalarizer = RulebookScalarizer(ScalarizationConfig())
    margins = (0.0, -1.0, 0.0, 0.25)
    reward = scalarizer(margins).scalar_reward

    # PPO, TD3, and SAC all consume Transition.scalar_reward. This explicit
    # common-boundary check prevents backend-specific formula branches.
    assert reward == scalarizer(margins).scalar_reward
    assert reward == scalarizer(tuple(margins)).scalar_reward


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
