from __future__ import annotations

import pytest

from thesis_rl.curriculum.scenario_acl.usefulness import (
    compute_learning_potential,
    compute_ppo_learning_potential,
    compute_sac_learning_potential,
    compute_scenario_usefulness,
    compute_td3_learning_potential,
    compute_sac_td_residuals,
    compute_td3_td_residuals,
)


def test_rule_criticality_uses_rulebook_v1_priority_groups() -> None:
    usefulness = compute_scenario_usefulness(
        {
            "per_rule": [
                {"rule_name": "local_route_progress", "min_margin": -2.0},
                {"rule_name": "lane_marking_compliance", "min_margin": -0.1},
                {"rule_name": "collision_severity", "min_margin": -1.0},
            ]
        },
        learning_potential=4.0,
    )

    assert usefulness.rule_criticality == 3
    assert usefulness.dominant_rule == "collision_severity"


def test_rule_criticality_accepts_rulebook_v2_macro_names_without_changing_value_signal() -> None:
    usefulness = compute_scenario_usefulness(
        {
            "per_rule": [
                {"rule_name": "mission_progress", "min_margin": -0.2},
                {"rule_name": "interaction_risk", "min_margin": -0.1},
            ]
        },
        learning_potential=7.0,
    )
    assert usefulness.rule_criticality == 2
    assert usefulness.dominant_rule == "interaction_risk"
    assert usefulness.value == 7.0


def test_learning_potential_alone_determines_buffer_usefulness() -> None:
    critical = compute_scenario_usefulness(
        {"per_rule": [{"rule_name": "lane_marking_compliance", "min_margin": -0.01}]},
        learning_potential=0.0,
    )
    ordinary = compute_scenario_usefulness({"per_rule": []}, learning_potential=999_999.0)

    assert critical.value == 0.0
    assert ordinary.value == 999_999.0


def test_ppo_learning_potential_uses_positive_gae_style_advantages() -> None:
    assert compute_ppo_learning_potential(advantages=[-2.0, 1.0, 3.0]) == pytest.approx(4.0 / 3.0)


def test_ppo_learning_potential_builds_gae_from_transition_inputs() -> None:
    assert compute_ppo_learning_potential(
        rewards=[1.0, 1.0],
        values=[0.0, 0.0],
        next_values=[0.0, 0.0],
        dones=[False, True],
        gamma=1.0,
        gae_lambda=1.0,
    ) == pytest.approx(1.5)


def test_ppo_learning_potential_bootstraps_truncation_final_value() -> None:
    assert compute_ppo_learning_potential(
        rewards=[0.0],
        values=[0.0],
        next_values=[2.0],
        dones=[False],
        gamma=1.0,
        gae_lambda=1.0,
    ) == pytest.approx(2.0)


def test_td3_and_sac_learning_potential_use_positive_part_residuals() -> None:
    # DEC-006: max(delta, 0), not |delta| -- negative residuals (worse-than-
    # expected outcomes) contribute zero, matching PPO's hopelessness filter.
    assert compute_td3_learning_potential([-2.0, 1.0, 3.0]) == pytest.approx(4.0 / 3.0)
    assert compute_sac_learning_potential([-2.0, 1.0, 3.0]) == pytest.approx(4.0 / 3.0)
    assert compute_td3_learning_potential([-2.0, -1.0]) == pytest.approx(0.0)


def test_td3_and_sac_residuals_apply_terminal_and_entropy_terms() -> None:
    td3 = compute_td3_td_residuals(
        rewards=[1.0, 1.0], dones=[False, True], target_q=[2.0, 4.0], current_q=[0.0, 2.0]
    )
    sac = compute_sac_td_residuals(
        rewards=[1.0],
        dones=[False],
        target_q=[2.0],
        current_q=[0.0],
        log_pi=[-0.5],
        entropy_temperature=0.2,
    )
    assert td3.tolist() == pytest.approx([2.98, -1.0])
    assert sac.tolist() == pytest.approx([3.079])


def test_learning_potential_rejects_loss_only_proxy_and_accepts_backend_value() -> None:
    assert (
        compute_learning_potential(
            {"learning_potential": 2.5, "update_calls": 1}, planner_name="td3_sb3"
        )
        == 2.5
    )
    with pytest.raises(ValueError, match="planner update"):
        compute_learning_potential(
            {"learning_potential": 2.5, "update_calls": 0}, planner_name="td3_sb3"
        )
    with pytest.raises(ValueError, match="TD3 residual"):
        compute_learning_potential(
            {"critic_loss_ema": 2.5, "update_calls": 1}, planner_name="td3_sb3"
        )
