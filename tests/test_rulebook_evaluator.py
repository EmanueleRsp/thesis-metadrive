from __future__ import annotations

import pytest

from thesis_rl.rulebook.evaluator import ScenicRulesEvaluator
from thesis_rl.rulebook.rulebook_config import load_rulebook_from_config
from thesis_rl.rulebook.types import RuleEvalInput


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
