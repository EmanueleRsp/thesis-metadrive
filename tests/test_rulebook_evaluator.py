from __future__ import annotations

import pytest

from thesis_rl.rulebook.evaluator import RulebookEvaluationError, ScenicRulesEvaluator
from thesis_rl.rulebook.registry import load_rulebook_from_config
from thesis_rl.rulebook.types import RuleEvalInput


V1_RULES = {
    "rules": [
        {"name": "collision_severity", "priority": 0},
        {"name": "allowed_driving_area", "priority": 1},
        {"name": "lane_marking_compliance", "priority": 2},
        {"name": "local_route_progress", "priority": 3},
    ]
}


def test_rulebook_loader_orders_by_priority_then_yaml_order() -> None:
    cfg = {
        "rules": [
            {"name": "speed_limit", "priority": 2},
            {"name": "drivable_area", "priority": 1},
            {"name": "wrong_way", "priority": 2},
        ]
    }

    specs = load_rulebook_from_config(cfg)
    names = [spec.name for spec in specs]
    assert names == ["drivable_area", "speed_limit", "wrong_way"]


def test_rulebook_evaluator_fallbacks_to_zero_on_rule_exception() -> None:
    cfg = {
        "rules": [
            {"name": "speed_limit", "priority": 0},
            {"name": "wrong_way", "priority": 1},
        ]
    }
    evaluator = ScenicRulesEvaluator.from_config(cfg)

    # Missing speed_limit and opposite_carriageway -> neutral margins.
    inputs = RuleEvalInput(ego_state={"speed": 10.0}, neighbors=[])
    result = evaluator.evaluate(inputs)

    assert result.names == ["speed_limit", "wrong_way"]
    assert result.values.tolist() == [0.0, 0.0]
    assert result.metadata["failed_rules"] == []


def test_speed_limit_rule_with_valid_input_returns_signed_margin() -> None:
    cfg = {"rules": [{"name": "speed_limit", "priority": 0}]}
    evaluator = ScenicRulesEvaluator.from_config(cfg)

    inputs = RuleEvalInput(
        ego_state={"speed": 12.5},
        neighbors=[],
        speed_limit=10.0,
    )
    result = evaluator.evaluate(inputs)

    assert result.names == ["speed_limit"]
    assert result.values.shape == (1,)
    assert float(result.values[0]) < 0.0


def test_rulebook_evaluator_marks_failed_rule_when_exception_raised() -> None:
    cfg = {
        "rules": [
            {"name": "speed_limit", "priority": 0, "params": {"unexpected": 1}},
        ]
    }
    evaluator = ScenicRulesEvaluator.from_config(cfg)
    inputs = RuleEvalInput(ego_state={"speed": 10.0}, neighbors=[], speed_limit=8.0)

    result = evaluator.evaluate(inputs)

    assert result.names == ["speed_limit"]
    assert result.values.tolist() == [0.0]
    assert result.metadata["failed_rules"] == ["speed_limit"]


def test_lane_centering_uses_ego_position_when_polygon_missing() -> None:
    cfg = {"rules": [{"name": "lane_centering", "priority": 0}]}
    evaluator = ScenicRulesEvaluator.from_config(cfg)

    inputs = RuleEvalInput(
        ego_state={"position": [1.0, 2.0]},
        neighbors=[],
        lane_centerline=[[0.0, 0.0], [0.0, 10.0]],
    )
    result = evaluator.evaluate(inputs)

    assert result.names == ["lane_centering"]
    assert result.metadata["failed_rules"] == []
    assert float(result.values[0]) == pytest.approx(-1.0)


def test_goal_progress_accepts_polygon_vertices_input() -> None:
    cfg = {"rules": [{"name": "goal_progress", "priority": 0}]}
    evaluator = ScenicRulesEvaluator.from_config(cfg)

    # MetaDrive-style polygon payloads are plain vertex lists, not shapely objects.
    # The rule should convert them and produce a meaningful (non-neutral) margin.
    inputs = RuleEvalInput(
        ego_state={
            "polygon": [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        },
        neighbors=[],
        target_region=[
            [10.0, 10.0],
            [11.0, 10.0],
            [11.0, 11.0],
            [10.0, 11.0],
        ],
    )
    result = evaluator.evaluate(inputs)

    assert result.names == ["goal_progress"]
    assert result.metadata["failed_rules"] == []
    assert float(result.values[0]) < 0.0


def test_rulebook_v1_returns_structured_rule_schema() -> None:
    evaluator = ScenicRulesEvaluator.from_config(V1_RULES)

    result = evaluator.evaluate(
        RuleEvalInput(
            ego_state={"position": [0.0, 0.0], "velocity": [2.0, 0.0]},
            neighbors=[],
            route_progress=3.5,
            prev_route_progress=3.0,
        )
    )

    assert result.names == [
        "collision_severity",
        "allowed_driving_area",
        "lane_marking_compliance",
        "local_route_progress",
    ]
    payload = result.metadata["rules"]
    assert set(payload["collision_severity"]) == {
        "name", "margin", "violated", "severity", "available", "fallback_used", "raw"
    }
    assert payload["collision_severity"]["margin"] == 0.0
    assert payload["local_route_progress"]["margin"] == pytest.approx(0.5)
    assert payload["local_route_progress"]["available"] is True


def test_rulebook_v1_collision_severity_distinguishes_speed_and_static_objects() -> None:
    evaluator = ScenicRulesEvaluator.from_config(
        {"rules": [{"name": "collision_severity", "priority": 0}]}
    )
    low_speed = evaluator.evaluate(
        RuleEvalInput(
            ego_state={"position": [0.0, 0.0], "velocity": [1.0, 0.0], "radius": 1.0},
            neighbors=[{"position": [0.5, 0.0], "radius": 1.0, "type": "static_obstacle"}],
        )
    )
    high_speed = evaluator.evaluate(
        RuleEvalInput(
            ego_state={"position": [0.0, 0.0], "velocity": [5.0, 0.0], "radius": 1.0},
            neighbors=[{"position": [0.5, 0.0], "radius": 1.0, "type": "static_obstacle"}],
        )
    )

    assert float(low_speed.values[0]) < 0.0
    assert abs(float(high_speed.values[0])) > abs(float(low_speed.values[0]))
    assert high_speed.metadata["rules"]["collision_severity"]["raw"]["collision_object_type"] == "static_obstacle"


def test_rulebook_v1_progress_checkpoint_fallback_is_transition_based() -> None:
    evaluator = ScenicRulesEvaluator.from_config(
        {"rules": [{"name": "local_route_progress", "priority": 0}]}
    )
    result = evaluator.evaluate(
        RuleEvalInput(
            ego_state={"position": [1.0, 0.0]},
            prev_ego_state={"position": [0.0, 0.0]},
            neighbors=[],
            route_checkpoints=[[10.0, 0.0], [20.0, 0.0]],
        )
    )

    assert float(result.values[0]) > 0.0
    assert result.metadata["rules"]["local_route_progress"]["fallback_used"] is True


def test_strict_rulebook_rejects_unavailable_rule_inputs() -> None:
    evaluator = ScenicRulesEvaluator.from_config(
        {"strict": True, "rules": [{"name": "allowed_driving_area", "priority": 0}]}
    )

    with pytest.raises(RulebookEvaluationError, match="unavailable"):
        evaluator.evaluate(RuleEvalInput(ego_state={"position": [0.0, 0.0]}, neighbors=[]))


def test_strict_rulebook_rejects_fallback_paths() -> None:
    evaluator = ScenicRulesEvaluator.from_config(
        {"strict": True, "rules": [{"name": "local_route_progress", "priority": 0}]}
    )

    with pytest.raises(RulebookEvaluationError, match="fallback"):
        evaluator.evaluate(
            RuleEvalInput(
                ego_state={"position": [1.0, 0.0]},
                prev_ego_state={"position": [0.0, 0.0]},
                neighbors=[],
                route_checkpoints=[[10.0, 0.0]],
            )
        )


def test_strict_rulebook_allows_route_progress_initialization_without_fallback() -> None:
    evaluator = ScenicRulesEvaluator.from_config(
        {"strict": True, "rules": [{"name": "local_route_progress", "priority": 0}]}
    )

    result = evaluator.evaluate(
        RuleEvalInput(ego_state={"position": [0.0, 0.0]}, neighbors=[], route_progress=0.0)
    )

    assert result.metadata["rules"]["local_route_progress"]["raw"]["initialization"] is True
