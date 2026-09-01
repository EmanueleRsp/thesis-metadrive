"""Pure macro-component aggregation for Rulebook v2."""

from __future__ import annotations

from math import isfinite

from thesis_rl.rulebook.v2.types import (
    COST_MACRO_RULES,
    MACRO_RULE_ORDER,
    ComponentStatus,
    MacroRule,
    RuleComponentResult,
    RulebookResult,
)


def aggregate_max_component(
    *,
    name: str,
    components: tuple[RuleComponentResult, ...],
) -> RuleComponentResult:
    """Aggregate applicable components by max while retaining all diagnostics."""

    for component in components:
        if not isfinite(component.cost):
            raise ValueError(f"Component {component.name!r} cost must be finite")
        if component.cost < -1e-8 or component.cost > 1.0 + 1e-8:
            raise ValueError(f"Component {component.name!r} cost must be in [0, 1]")
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


def aggregate_sum_component(
    *,
    name: str,
    components: tuple[RuleComponentResult, ...],
    denominator: float,
) -> RuleComponentResult:
    """Aggregate by normalized sum over a **declared** denominator.

    RULEBOOK-V5.1 §3.3 uses this for L5, where the quantity of interest is the
    *total amount of relaxation*: a detour that crosses a solid line and enters
    the opposing carriageway must cost more than one that only straddles, and
    ``max`` would make the second violation free — which is what would break O6.

    The denominator is declared by the caller and never counted from how many
    sub-rules happened to apply. Counting would make the same physical violation
    cost three times more on a road with no dashed marking, which is a property
    of the map rather than of the driving; `TEST-RB5.1-11` guards exactly that.

    It is also validated: a denominator smaller than the number of sub-rules
    would let the level leave ``[0, 1]`` and silently outrank a higher one, so
    this fails closed instead of emitting such a cost.
    """

    if not isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"Component {name!r} denominator must be finite and positive")
    if denominator < len(components) - 1e-9:
        raise ValueError(
            f"Component {name!r} denominator {denominator} cannot bound "
            f"{len(components)} sub-rules within [0, 1]"
        )
    for component in components:
        if not isfinite(component.cost):
            raise ValueError(f"Component {component.name!r} cost must be finite")
        if component.cost < -1e-8 or component.cost > 1.0 + 1e-8:
            raise ValueError(f"Component {component.name!r} cost must be in [0, 1]")
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
            diagnostics={"subcomponent_count": len(components), "denominator": denominator},
        )
    total = sum(component.cost for component in applicable) / denominator
    cost = max(0.0, min(1.0, total))
    return RuleComponentResult(
        name=name,
        cost=cost,
        raw={
            "summed_components": tuple(component.name for component in applicable),
            "subcomponents": tuple(component.to_dict() for component in components),
        },
        applicable=True,
        evaluable=True,
        status=ComponentStatus.VIOLATED if cost > 0.0 else ComponentStatus.SATISFIED,
        diagnostics={
            "denominator": denominator,
            "applicable_count": len(applicable),
            "subcomponent_count": len(components),
        },
    )


# RULEBOOK-V5.1 §3.3. Declared, not counted: L5 divides by three whether or not
# all three lane sub-rules apply, and L6 divides by one because it has a single
# sub-rule and the denominator is kept explicit so the atomic vector stays
# uniform across levels.
LEVEL_DENOMINATORS: dict[MacroRule, float] = {
    MacroRule.RELAXABLE_LANE_COMPLIANCE: 3.0,
    MacroRule.PROGRESS_RATE: 1.0,
}


def aggregate_rulebook_result(
    *,
    components: tuple[RuleComponentResult, ...],
    raw_progress_m: float,
    progress_margin: float,
    registry: "RulebookV2Registry | None" = None,
) -> RulebookResult:
    """Build the ordered six-margin monitor output without scalarization.

    Level membership comes from the **registry**, not from a name list kept
    here. v4.7 kept both, and a sub-rule could then sit at one level for
    evaluation and another for aggregation; after ADR-072 moved three rules
    across a level boundary that would have been a silent reordering rather than
    an error.

    Non-normative components — `rss` longitudinal since ADR-063, and the
    infrastructure-only ones — are excluded here rather than filtered by every
    caller, so "reported but never in the reward" is enforced in one place.
    """

    from thesis_rl.rulebook.v2.registry import DEFAULT_RULEBOOK_V2_REGISTRY

    registry = registry or DEFAULT_RULEBOOK_V2_REGISTRY

    if not isfinite(raw_progress_m):
        raise ValueError("Raw route progress must be finite")
    if not isfinite(progress_margin) or not -1.0 <= progress_margin <= 1.0:
        raise ValueError("Progress margin must be finite and in [-1, 1]")
    names = [component.name for component in components]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate Rulebook component result")
    if any(component.applicable and not component.evaluable for component in components):
        raise ValueError("Applicable component is NOT_EVALUABLE")

    normative: dict[MacroRule, list[RuleComponentResult]] = {
        level: [] for level in COST_MACRO_RULES
    }
    for component in components:
        try:
            definition = registry.definition(component.name)
        except ValueError:
            # An aggregated macro result fed back in, or a component the
            # registry does not own: neither is a sub-rule and neither may
            # reach a channel.
            continue
        if not definition.normative_output or not definition.contributes_to_channel:
            continue
        if definition.macro_rule is MacroRule.MISSION_PROGRESS:
            continue
        normative[definition.macro_rule].append(component)

    macro: list[RuleComponentResult] = []
    for level in COST_MACRO_RULES:
        group = tuple(normative[level])
        denominator = LEVEL_DENOMINATORS.get(level)
        if denominator is None:
            macro.append(aggregate_max_component(name=level.value, components=group))
        else:
            macro.append(
                aggregate_sum_component(
                    name=level.value, components=group, denominator=denominator
                )
            )

    costs = tuple(max(0.0, min(1.0, result.cost)) for result in macro)
    cost_by_level = dict(zip(COST_MACRO_RULES, costs))
    # `-0.0` is equal to `0.0` but reads as a different value in recorded
    # artifacts and in test diffs, so a satisfied level is normalized to `+0.0`.
    margins = tuple(
        progress_margin
        if level is MacroRule.MISSION_PROGRESS
        else (-cost_by_level[level] if cost_by_level[level] else 0.0)
        for level in MACRO_RULE_ORDER
    )

    # A level and one of its sub-rules sharing a name would make the aggregated
    # result silently overwrite the atomic one, losing §3.4's contract that
    # every sub-rule cost is exposed. It went unnoticed once because L6's single
    # sub-rule was named after its level and the two values coincide, so the
    # collision is now an error rather than a coincidence.
    all_components = {component.name: component for component in components}
    collisions = sorted(set(all_components) & {result.name for result in macro})
    if collisions:
        raise ValueError(
            "Level names must not collide with sub-rule names: " + ", ".join(collisions)
        )
    all_components.update({result.name: result for result in macro})
    return RulebookResult(
        margins=margins,
        costs=costs,
        raw_progress_m=raw_progress_m,
        components=all_components,
        complete_evaluation=all(component.evaluable for component in components),
    )
