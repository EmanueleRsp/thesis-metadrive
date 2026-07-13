from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf
from types import SimpleNamespace

from thesis_rl.curriculum import (
    CurriculumConfig,
    GeneratorArmBandit,
    SCENARIO_ARM_NAMES,
    ScenarioAclMabConfig,
    build_default_generator_arms,
    build_default_scenario_arms,
)
from thesis_rl.curriculum.scenario_acl.buffer import ScenarioBuffer
from thesis_rl.curriculum.scenario_acl.driver import (
    _build_record_from_catalog_entry,
    _choose_iteration_spec,
    _select_waymo_eval_arm,
    _selection_source_override,
)


def test_generator_arm_bandit_probabilities_sum_to_one() -> None:
    bandit = GeneratorArmBandit(
        ScenarioAclMabConfig(
            num_arms=3,
            eta=0.2,
            initial_weight=1.0,
        )
    )

    probs = bandit.probabilities()

    assert probs.shape == (3,)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs > 0.0)


def test_generator_arm_bandit_initializes_exponentially_by_arm_index() -> None:
    bandit = GeneratorArmBandit(
        ScenarioAclMabConfig(
            num_arms=4,
            eta=0.0,
            initial_weight_decay=0.5,
        )
    )

    assert np.allclose(bandit.probabilities(), np.asarray([8, 4, 2, 1]) / 15)
    assert np.allclose(bandit.target_weights, bandit.weights)


def test_generator_arm_bandit_target_sync_updates_live_weights_on_interval() -> None:
    bandit = GeneratorArmBandit(
        ScenarioAclMabConfig(
            num_arms=2,
            alpha=0.5,
            target_sync_interval=2,
            use_target_mab=True,
        )
    )

    initial_weights = bandit.weights.copy()
    bandit.update(arm_index=0, normalized_usefulness=1.0, selection_probability=0.5)
    assert np.allclose(bandit.weights, initial_weights)
    bandit.update(arm_index=0, normalized_usefulness=1.0, selection_probability=0.5)
    assert not np.allclose(bandit.weights, initial_weights)


def test_default_generator_arms_preserve_single_scenario_sampling_and_agent_model_flag() -> None:
    rng = np.random.default_rng(42)
    arms = build_default_generator_arms()
    sample = arms[0].sample_env_overrides(
        rng=rng,
        scenario_seed=123,
        base_env_config={"random_agent_model": True, "log_level": 50},
    )

    assert len(arms) == 7
    assert sample["num_scenarios"] == 1
    assert sample["start_seed"] == 123
    assert sample["random_agent_model"] is True


def test_default_scenario_arms_match_canonical_scenarionet_taxonomy() -> None:
    arms = build_default_scenario_arms()

    assert tuple(arm.name for arm in arms) == SCENARIO_ARM_NAMES
    assert arms[0].sample_env_overrides(
        rng=np.random.default_rng(42),
        scenario_seed=123,
        base_env_config={},
    ) == {
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
                "arm_space": "scenario",
                "use_scenario_buffer": False,
                "use_replay": False,
                "mab": {"num_arms": 6},
            },
        }
    )
    arms = list(build_default_scenario_arms())
    from thesis_rl.curriculum import GeneratorArmBandit

    bandit = GeneratorArmBandit(curriculum.scenario_acl.mab)
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
