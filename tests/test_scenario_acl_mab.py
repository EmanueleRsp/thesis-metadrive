from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf
from types import SimpleNamespace

from thesis_rl.curriculum import (
    CurriculumConfig,
    ScenarioArmBandit,
    SCENARIO_ARM_NAMES,
    ScenarioAclMabConfig,
    build_default_scenario_arms,
)
from thesis_rl.curriculum.scenario_acl.buffer import ScenarioBuffer
from thesis_rl.curriculum.scenario_acl.driver import (
    EpisodeAclOutcome,
    IterationSpec,
    _build_record_from_catalog_entry,
    _choose_iteration_spec,
    _select_waymo_eval_arm,
    _selection_source_override,
    _summarize_episode_acl_outcomes,
    _normalize_learning_potential,
)


def test_generator_arm_bandit_probabilities_sum_to_one() -> None:
    bandit = ScenarioArmBandit(
        ScenarioAclMabConfig(
            num_arms=3,
            eta=0.2,
            initial_score=0.5,
        )
    )

    probs = bandit.probabilities()

    assert probs.shape == (3,)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs > 0.0)


def test_generator_arm_bandit_initializes_uniformly_with_ema_scores() -> None:
    bandit = ScenarioArmBandit(
        ScenarioAclMabConfig(
            num_arms=4,
            eta=0.0,
            initial_score=0.5,
        )
    )

    assert np.allclose(bandit.probabilities(), np.full(4, 0.25))
    assert np.allclose(bandit.scores, np.full(4, 0.5))


def test_generator_arm_bandit_ema_updates_only_selected_score() -> None:
    bandit = ScenarioArmBandit(
        ScenarioAclMabConfig(
            num_arms=2,
            alpha=0.5,
            initial_score=0.5,
        )
    )

    bandit.update(arm_index=0, normalized_usefulness=1.0, selection_probability=0.01)
    assert bandit.scores[0] == 0.75
    assert np.allclose(bandit.scores[1:], 0.5)


def test_generator_arm_bandit_probability_contract_and_checkpoint_schema() -> None:
    bandit = ScenarioArmBandit(ScenarioAclMabConfig(num_arms=6, eta=0.2, temperature=0.5))
    bandit.update(arm_index=0, normalized_usefulness=1.0)
    probabilities = bandit.probabilities()
    assert probabilities[0] > 1.0 / 6.0
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0.2 / 6.0)
    restored = ScenarioArmBandit.from_state_dict(bandit.config, bandit.state_dict())
    assert np.allclose(restored.scores, bandit.scores)
    with pytest.raises(ValueError, match="Incompatible"):
        ScenarioArmBandit.from_state_dict(bandit.config, {"weights": [1.0] * 6})


def test_reward_scale_estimator_initializes_uniformly_and_updates_with_ema() -> None:
    # DEC-006: shared initial value across arms means normalization is a
    # no-op until an arm's estimate has actually diverged from the prior.
    bandit = ScenarioArmBandit(ScenarioAclMabConfig(num_arms=3, alpha=0.5))
    assert bandit.reward_scale_estimate(0) == pytest.approx(1.0)
    assert bandit.normalize_learning_potential_by_reward_scale(0, 5.0) == pytest.approx(5.0)

    bandit.update_reward_scale(0, episode_reward=9.0)
    # (1 - 0.5) * 1.0 + 0.5 * 9.0 = 5.0
    assert bandit.reward_scale_estimate(0) == pytest.approx(5.0)
    assert bandit.reward_scale_estimate(1) == pytest.approx(1.0)
    assert bandit.normalize_learning_potential_by_reward_scale(0, 5.0) == pytest.approx(1.0)

    bandit.update_reward_scale(0, episode_reward=-9.0)
    # Negative rewards contribute their magnitude: (1-0.5)*5.0 + 0.5*9.0 = 7.0
    assert bandit.reward_scale_estimate(0) == pytest.approx(7.0)


def test_reward_scale_estimate_is_clamped_away_from_zero() -> None:
    bandit = ScenarioArmBandit(ScenarioAclMabConfig(num_arms=2, alpha=1.0))
    bandit.update_reward_scale(0, episode_reward=0.0)
    assert bandit.reward_scale_estimate(0) == pytest.approx(1e-3)
    assert bandit.normalize_learning_potential_by_reward_scale(0, 2.0) == pytest.approx(2000.0)


def test_reward_scale_state_round_trips_through_checkpoint() -> None:
    bandit = ScenarioArmBandit(ScenarioAclMabConfig(num_arms=3, alpha=0.5))
    bandit.update_reward_scale(1, episode_reward=12.0)
    restored = ScenarioArmBandit.from_state_dict(bandit.config, bandit.state_dict())
    assert np.allclose(restored.reward_scale, bandit.reward_scale)
    assert restored.state_dict()["schema"] == "acl_ema_v2"


def test_legacy_v1_checkpoint_schema_is_rejected() -> None:
    bandit = ScenarioArmBandit(ScenarioAclMabConfig(num_arms=6))
    legacy_state = {
        "schema": "acl_ema_v1",
        "scores": bandit.scores.tolist(),
        "target_scores": bandit.target_scores.tolist(),
        "update_count": 0,
    }
    with pytest.raises(ValueError, match="acl_ema_v2"):
        ScenarioArmBandit.from_state_dict(bandit.config, legacy_state)


def test_target_mab_is_available_but_disabled_by_default() -> None:
    bandit = ScenarioArmBandit(
        ScenarioAclMabConfig(num_arms=2, alpha=1.0, use_target_mab=True, target_sync_interval=2)
    )
    bandit.update(arm_index=0, normalized_usefulness=1.0)
    assert np.allclose(bandit.scores, [1.0, 0.5])
    assert np.allclose(bandit.target_scores, [0.5, 0.5])
    bandit.update(arm_index=0, normalized_usefulness=0.0)
    assert np.allclose(bandit.target_scores, bandit.scores)


def test_default_scenario_arms_match_canonical_scenarionet_taxonomy() -> None:
    arms = build_default_scenario_arms()

    assert tuple(arm.name for arm in arms) == SCENARIO_ARM_NAMES
    assert arms[0].sample_env_overrides() == {
        "provider": {
            "arm": "A0_simple_low_traffic",
            "source_probability": {"waymo": 0.0, "pg": 1.0},
        }
    }


def test_scenario_acl_driver_selects_catalog_arm_without_forcing_runtime_seed() -> None:
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
    arms = list(build_default_scenario_arms())
    from thesis_rl.curriculum import ScenarioArmBandit

    bandit = ScenarioArmBandit(curriculum.scenario_acl.mab)
    spec = _choose_iteration_spec(
        cfg=OmegaConf.create({"env": {"config": {"start_seed": 0}}}),
        curriculum_cfg=curriculum,
        arms=arms,
        bandit=bandit,
        buffer=ScenarioBuffer(capacity=4),
        rng=np.random.default_rng(0),
        chunk_id=1,
    )

    assert spec.mode == "sample"
    assert spec.arm_name in SCENARIO_ARM_NAMES
    assert spec.train_env_overrides is not None
    assert spec.train_env_overrides["provider"]["arm"] == spec.arm_name


def test_one_sided_semantic_arm_forces_its_available_source() -> None:
    arms = list(build_default_scenario_arms())
    a0 = next(arm for arm in arms if arm.name == "A0_simple_low_traffic")
    a1 = next(arm for arm in arms if arm.name == "A1_traffic")

    a0_overrides = a0.sample_env_overrides(
        rng=np.random.default_rng(0), scenario_seed=0, base_env_config={}
    )
    a1_overrides = a1.sample_env_overrides(
        rng=np.random.default_rng(0), scenario_seed=0, base_env_config={}
    )

    assert _selection_source_override(SimpleNamespace(train_env_overrides=a0_overrides)) == "pg"
    assert _selection_source_override(SimpleNamespace(train_env_overrides=a1_overrides)) is None


def test_waymo_evaluation_schedule_is_uniform_over_supported_arms() -> None:
    assert [_select_waymo_eval_arm(index) for index in range(10)] == [
        "A1_traffic",
        "A2_junction",
        "A3_complex_junction",
        "A4_vru",
        "A5_critical_mixed",
    ] * 2


def test_catalog_record_adapter_preserves_semantic_arm_and_runtime_index() -> None:
    catalog_record = SimpleNamespace(
        split="train",
        scenario_uid="pg:v1:42",
        source="pg",
        primary_arm="A2_junction",
        relative_path="pg/database/42.pkl",
        runtime_index=42,
    )
    record = _build_record_from_catalog_entry(
        catalog_record=catalog_record,
        cfg=OmegaConf.create({"paths": {"scenarionet_data_root": "/data/scenarionet"}}),
        chunk_id=1,
        learning_potential=0.5,
        normalized_usefulness=0.5,
        metrics={},
    )

    assert record.scenario_arm == "A2_junction"
    assert record.scenario_index == 42
    assert record.dataset_directory == "/data/scenarionet/runtime/train"


def test_semantic_acl_statistics_count_mixed_modes_per_completed_episode() -> None:
    sampled = IterationSpec(
        mode="sample",
        arm_index=1,
        arm_name="A1_traffic",
        arm_probabilities=[0.5, 0.5],
        scenario_seed=0,
        train_env_overrides={},
        replay_record=None,
        replay_probabilities=None,
    )
    replayed = IterationSpec(
        mode="exploit_replay",
        arm_index=2,
        arm_name="A2_junction",
        arm_probabilities=[0.5, 0.5],
        scenario_seed=0,
        train_env_overrides=None,
        replay_record=None,
        replay_probabilities=[1.0],
    )

    summary = _summarize_episode_acl_outcomes(
        [
            EpisodeAclOutcome(
                spec=sampled,
                record=SimpleNamespace(primary_arm="A1_traffic"),
                metrics={},
            ),
            EpisodeAclOutcome(
                spec=replayed,
                record=SimpleNamespace(primary_arm="A3_complex_junction"),
                metrics={},
            ),
        ]
    )

    assert summary == {
        "generate_count": 1,
        "replay_count": 1,
        "modes": ["exploit_replay", "sample"],
        "arms": ["A1_traffic", "A3_complex_junction"],
        "mode": "mixed",
    }


def test_learning_potential_normalization_assigns_average_rank_to_ties() -> None:
    # With one stronger value and two tied values, both tied observations get
    # rank 2.5 out of 3 rather than incorrectly receiving rank 2 (or rank 1).
    assert _normalize_learning_potential(1.0, [2.0, 1.0]) == 0.25
