from __future__ import annotations

import numpy as np

from thesis_rl.curriculum import (
    GeneratorArmBandit,
    ScenarioAclMabConfig,
    build_default_generator_arms,
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
