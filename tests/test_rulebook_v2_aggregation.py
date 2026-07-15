from __future__ import annotations

import pytest

from thesis_rl.rulebook.v2.aggregation import aggregate_max_component
from thesis_rl.rulebook.v2.types import ComponentStatus, RuleComponentResult


def _component(name: str, cost: float, applicable: bool = True) -> RuleComponentResult:
    status = ComponentStatus.VIOLATED if cost else ComponentStatus.SATISFIED
    return RuleComponentResult(name, cost, {"raw": cost}, applicable, True, status, {})


def test_max_aggregation_preserves_all_subcomponents_and_worst_name() -> None:
    result = aggregate_max_component(
        name="interaction",
        components=(_component("rss", 0.2), _component("ttc", 0.7), _component("clearance", 0.3)),
    )
    assert result.cost == pytest.approx(0.7)
    assert result.raw["worst_component"] == "ttc"
    assert len(result.raw["subcomponents"]) == 3


def test_max_aggregation_is_not_applicable_when_all_components_are_na() -> None:
    result = aggregate_max_component(
        name="interaction", components=(_component("rss", 0.0, False),)
    )
    assert result.status == ComponentStatus.NOT_APPLICABLE
    assert result.cost == 0.0
