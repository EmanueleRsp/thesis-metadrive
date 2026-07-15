from __future__ import annotations

import pytest

from thesis_rl.curriculum.scenario_acl.usefulness import (
    compute_learning_potential,
    compute_scenario_usefulness,
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
                {"rule_name": "route_progress", "min_margin": -0.2},
                {"rule_name": "dynamic_interaction_safety", "min_margin": -0.1},
            ]
        },
        learning_potential=7.0,
    )
    assert usefulness.rule_criticality == 2
    assert usefulness.dominant_rule == "dynamic_interaction_safety"
    assert usefulness.value == 7.0


def test_learning_potential_alone_determines_buffer_usefulness() -> None:
    critical = compute_scenario_usefulness(
        {"per_rule": [{"rule_name": "lane_marking_compliance", "min_margin": -0.01}]},
        learning_potential=0.0,
    )
    ordinary = compute_scenario_usefulness({"per_rule": []}, learning_potential=999_999.0)

    assert critical.value == 0.0
    assert ordinary.value == 999_999.0


def test_learning_potential_uses_backend_critic_loss_and_rejects_missing_updates() -> None:
    assert compute_learning_potential(
        {"critic_loss_ema": 2.5, "update_calls": 1}, planner_name="td3_sb3"
    ) == 2.5
    with pytest.raises(ValueError, match="planner update"):
        compute_learning_potential(
            {"critic_loss_ema": 2.5, "update_calls": 0}, planner_name="td3_sb3"
        )
