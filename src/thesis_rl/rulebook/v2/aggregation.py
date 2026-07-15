"""Pure macro-component aggregation for Rulebook v2."""

from __future__ import annotations

from thesis_rl.rulebook.v2.types import ComponentStatus, RuleComponentResult


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
