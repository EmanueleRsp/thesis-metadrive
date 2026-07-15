"""Pure macro-component aggregation for Rulebook v2."""

from __future__ import annotations

from thesis_rl.rulebook.v2.types import ComponentStatus, RuleComponentResult
from thesis_rl.rulebook.v2.types import RulebookResult


def aggregate_max_component(
    *,
    name: str,
    components: tuple[RuleComponentResult, ...],
) -> RuleComponentResult:
    """Aggregate applicable components by max while retaining all diagnostics."""

    applicable = [component for component in components if component.applicable]
    if any(not component.evaluable for component in applicable):
        raise ValueError(f"Applicable {name} component is NOT_EVALUABLE")
    if not applicable:
        return RuleComponentResult(
            name=name,
            cost=0.0,
            raw={"subcomponents": tuple(component.to_dict() for component in components)},
            applicable=False,
            evaluable=True,
            status=ComponentStatus.NOT_APPLICABLE,
            diagnostics={"subcomponent_count": len(components)},
        )
    worst = max(applicable, key=lambda component: (component.cost, component.name))
    return RuleComponentResult(
        name=name,
        cost=worst.cost,
        raw={
            "worst_component": worst.name,
            "subcomponents": tuple(component.to_dict() for component in components),
        },
        applicable=True,
        evaluable=True,
        status=ComponentStatus.VIOLATED if worst.cost > 0.0 else ComponentStatus.SATISFIED,
        diagnostics={"worst_component": worst.name, "subcomponent_count": len(components)},
    )


def aggregate_rulebook_result(*, components: tuple[RuleComponentResult, ...], raw_progress_m: float,
                              progress_margin: float) -> RulebookResult:
    """Build the ordered four-margin monitor output without scalarization."""
    groups = {
        "collision_impact": tuple(c for c in components if c.name in {"collision", "collision_impact"}),
        "dynamic_interaction_safety": tuple(c for c in components if c.name in {"rss", "ttc", "clearance"}),
        "road_traffic_compliance": tuple(c for c in components if c.name in {"offroad", "wrongway", "solid_line", "dashed_line", "signal", "stop", "crosswalk", "vehicle_yield"}),
    }
    macro = tuple(aggregate_max_component(name=name, components=group) for name, group in groups.items())
    costs = tuple(max(0.0, min(1.0, result.cost)) for result in macro)
    if not -1.0 <= progress_margin <= 1.0:
        raise ValueError("Progress margin must be in [-1, 1]")
    all_components = {component.name: component for component in components}
    all_components.update({result.name: result for result in macro})
    return RulebookResult(
        margins=(-costs[0], -costs[1], -costs[2], progress_margin),
        costs=costs,
        raw_progress_m=raw_progress_m,
        components=all_components,
        complete_evaluation=all(component.evaluable for component in components),
    )
