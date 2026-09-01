"""`RB51` M0: the frozen mandatory matrix for the six-level hierarchy.

These tests are written **before** the production change and are expected to
fail against the three-macro-rule rulebook. They encode `RULEBOOK-V5.1` §3 and
§3.3, and they are the contract `M1` has to meet — not a description of whatever
`M1` happens to build.

They deliberately assert *semantics* rather than internal structure: level
membership, the aggregation used at each level, and the ranges. Where a level's
recorded name matters it is asserted against §3's name, which `DEC-RB51-005`
approved on 2026-08-20 together with the artifact-schema break that follows.
"""

from __future__ import annotations

import pytest

from thesis_rl.rulebook.v2.aggregation import (
    aggregate_max_component,
    aggregate_sum_component,
)
from thesis_rl.rulebook.v2.registry import RulebookV2Registry
from thesis_rl.rulebook.v2.types import (
    MACRO_RULE_ORDER,
    ComponentStatus,
    MacroRule,
    RuleComponentResult,
)


def component(name: str, cost: float, *, applicable: bool = True) -> RuleComponentResult:
    """A minimal sub-rule result, so fixtures state only what they are about."""

    if not applicable:
        status = ComponentStatus.NOT_APPLICABLE
    elif cost > 0.0:
        status = ComponentStatus.VIOLATED
    else:
        status = ComponentStatus.SATISFIED
    return RuleComponentResult(
        name=name,
        cost=cost,
        raw={},
        applicable=applicable,
        evaluable=True,
        status=status,
        diagnostics={},
    )


# ---------------------------------------------------------------------------
# T-RB51-01 -- six levels, in the declared order, named once
# ---------------------------------------------------------------------------


def test_six_levels_in_the_specified_order() -> None:
    """`T-RB51-01` / `REQ-RB51-01`. RULEBOOK-V5.1 §3.

    `MACRO_RULE_ORDER` is the single source of truth for the ordering: any second
    place that hard-codes it can drift, and a hierarchy whose order is stated
    twice is a hierarchy with two orders.
    """

    assert MACRO_RULE_ORDER == (
        MacroRule.COLLISION_SAFETY,
        MacroRule.INTERACTION_RISK,
        MacroRule.NON_RELAXABLE_COMPLIANCE,
        MacroRule.MISSION_PROGRESS,
        MacroRule.RELAXABLE_LANE_COMPLIANCE,
        MacroRule.PROGRESS_RATE,
    )
    assert len(MACRO_RULE_ORDER) == len(set(MACRO_RULE_ORDER)) == 6
    assert tuple(MacroRule) == MACRO_RULE_ORDER


def test_recorded_level_names_are_the_specification_names() -> None:
    """`T-RB51-01`. The enum's *values* reach recorded artifacts.

    `DEC-RB51-005`: leaving L3 recorded as `road_traffic_compliance` after the
    relaxable road rules have left it is the stale naming that causes later
    misreadings, so the values are §3's.
    """

    assert [rule.value for rule in MACRO_RULE_ORDER] == [
        "collision_safety",
        "interaction_risk",
        "non_relaxable_compliance",
        "mission_progress",
        "relaxable_lane_compliance",
        "progress_rate",
    ]


# ---------------------------------------------------------------------------
# T-RB51-02 -- level membership, and `max` within L2 and L3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sub_rule", "level"),
    [
        ("collision", MacroRule.COLLISION_SAFETY),
        ("ttc", MacroRule.INTERACTION_RISK),
        ("clearance", MacroRule.INTERACTION_RISK),
        ("rss_lateral", MacroRule.INTERACTION_RISK),
        ("offroad", MacroRule.NON_RELAXABLE_COMPLIANCE),
        ("signal", MacroRule.NON_RELAXABLE_COMPLIANCE),
        ("stop", MacroRule.NON_RELAXABLE_COMPLIANCE),
        ("crosswalk", MacroRule.NON_RELAXABLE_COMPLIANCE),
        ("vehicle_yield", MacroRule.NON_RELAXABLE_COMPLIANCE),
        ("solid_line", MacroRule.RELAXABLE_LANE_COMPLIANCE),
        ("wrong_carriageway", MacroRule.RELAXABLE_LANE_COMPLIANCE),
        ("dashed_line", MacroRule.RELAXABLE_LANE_COMPLIANCE),
        ("advance_shortfall", MacroRule.PROGRESS_RATE),
    ],
)
def test_sub_rule_sits_at_its_specified_level(sub_rule: str, level: MacroRule) -> None:
    """`T-RB51-02` / `REQ-RB51-01`. §3's table, one row at a time.

    The three lane rules moving out of L3 and below progress is the entire
    restructure (ADR-072); asserting membership per sub-rule is what stops one
    of them silently staying behind.
    """

    registry = RulebookV2Registry()
    assert registry.definition(sub_rule).macro_rule is level


def test_l2_is_zero_exactly_when_all_three_sub_rules_are_zero() -> None:
    """`T-RB51-02` / `REQ-RB51-02`. §3.3.

    `max` is defensible inside L2 *because* the three are complementary
    detectors of one property, so the level is zero iff every detector is. That
    biconditional is the justification, so it is what gets asserted.
    """

    zero = tuple(component(name, 0.0) for name in ("ttc", "clearance", "rss_lateral"))
    assert aggregate_max_component(name="interaction_risk", components=zero).cost == 0.0

    for index in range(3):
        costs = [0.0, 0.0, 0.0]
        costs[index] = 0.3
        mixed = tuple(
            component(name, cost)
            for name, cost in zip(("ttc", "clearance", "rss_lateral"), costs)
        )
        aggregated = aggregate_max_component(name="interaction_risk", components=mixed)
        assert aggregated.cost == pytest.approx(0.3)


def test_l3_reports_the_worst_non_relaxable_violation() -> None:
    """`T-RB51-02` / `REQ-RB51-02`. Minimax semantics, §3.3."""

    components = (
        component("offroad", 0.2),
        component("signal", 0.7),
        component("stop", 0.0),
        component("crosswalk", 0.1),
    )
    aggregated = aggregate_max_component(
        name="non_relaxable_compliance", components=components
    )
    assert aggregated.cost == pytest.approx(0.7)
    assert aggregated.diagnostics["worst_component"] == "signal"


# ---------------------------------------------------------------------------
# T-RB51-03 -- L5 sums over a FIXED denominator
# ---------------------------------------------------------------------------


def test_l5_sums_rather_than_taking_the_max() -> None:
    """`T-RB51-03` / `REQ-RB51-03`. §3.3, and it is why O6 holds.

    Under `max` a second concurrent lane violation would be free, so a detour
    that crosses a solid line *and* enters the opposing carriageway would cost
    no more than one that only straddles.
    """

    both = (
        component("solid_line", 0.6),
        component("wrong_carriageway", 0.6),
        component("dashed_line", 0.0),
    )
    one = (
        component("solid_line", 0.6),
        component("wrong_carriageway", 0.0),
        component("dashed_line", 0.0),
    )
    cost_both = aggregate_sum_component(
        name="relaxable_lane_compliance", components=both, denominator=3.0
    ).cost
    cost_one = aggregate_sum_component(
        name="relaxable_lane_compliance", components=one, denominator=3.0
    ).cost

    assert cost_both == pytest.approx(0.4)
    assert cost_one == pytest.approx(0.2)
    assert cost_both > cost_one


def test_l5_denominator_stays_three_when_a_sub_rule_is_inapplicable() -> None:
    """`T-RB51-03` / `REQ-RB51-03`, and `TEST-RB5.1-11`.

    The denominator is **declared**, not counted from how many sub-rules
    happened to apply. Dividing by the applicable count would make the same
    physical violation cost three times more on a road with no dashed marking,
    which is a property of the map rather than of the driving.
    """

    components = (
        component("solid_line", 0.6),
        component("wrong_carriageway", 0.0, applicable=False),
        component("dashed_line", 0.0, applicable=False),
    )
    aggregated = aggregate_sum_component(
        name="relaxable_lane_compliance", components=components, denominator=3.0
    )
    assert aggregated.cost == pytest.approx(0.2)


def test_l5_is_bounded_above_by_one() -> None:
    """`T-RB51-03`. Three maximal violations reach exactly 1, never more."""

    saturated = tuple(
        component(name, 1.0)
        for name in ("solid_line", "wrong_carriageway", "dashed_line")
    )
    aggregated = aggregate_sum_component(
        name="relaxable_lane_compliance", components=saturated, denominator=3.0
    )
    assert aggregated.cost == pytest.approx(1.0)


def test_l6_denominator_is_explicit_at_one() -> None:
    """`T-RB51-03` / `REQ-RB51-04`. §3.3.

    L6 has a single sub-rule, so no aggregation arises; the denominator is kept
    explicit so the atomic vector stays uniform across levels and a second L6
    sub-rule cannot later be added without someone choosing a denominator.
    """

    aggregated = aggregate_sum_component(
        name="progress_rate", components=(component("advance_shortfall", 0.8),), denominator=1.0
    )
    assert aggregated.cost == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# T-RB51-08 / T-RB51-09 -- what the specification removes
# ---------------------------------------------------------------------------


def test_rss_longitudinal_is_not_normative() -> None:
    """`T-RB51-08` / `REQ-RB51-11`. ADR-063.

    `rss` longitudinal stays as a reported diagnostic — it fires on 18.29 % of
    applicable expert steps and is not controlled-invariant — so it must reach
    no channel.
    """

    registry = RulebookV2Registry()
    rss = registry.definition("rss")
    # Still evaluated and still published -- ADR-063 demoted it to a reported
    # diagnostic, not to silence.
    assert rss.normative_output is True
    assert rss.contributes_to_channel is False
    assert registry.definition("ttc").contributes_to_channel is True


def test_wrongway_is_deleted() -> None:
    """`T-RB51-09` / `REQ-RB51-12`. ADR-066.

    One violated step in 217,189 of expert replay. Deleted rather than kept as a
    diagnostic, because `wrong_carriageway` covers the observable subject.
    """

    registry = RulebookV2Registry()
    for absent in ("wrongway", "wrong_way"):
        with pytest.raises(ValueError):
            registry.definition(absent)


# ---------------------------------------------------------------------------
# T-RB51-11 -- ranges, on every level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cost", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_every_aggregated_level_stays_within_the_unit_interval(cost: float) -> None:
    """`T-RB51-11` / `REQ-RB51-01`.

    Every channel except L4 is a cost in `[0, 1]`; the priority weights of §5
    are only meaningful on that shared scale.
    """

    triple = tuple(component(f"sub{i}", cost) for i in range(3))
    assert 0.0 <= aggregate_max_component(name="l", components=triple).cost <= 1.0
    assert (
        0.0
        <= aggregate_sum_component(name="l", components=triple, denominator=3.0).cost
        <= 1.0
    )


def test_sum_aggregation_rejects_a_denominator_that_cannot_bound_the_level() -> None:
    """`T-RB51-11`. Fail closed rather than emit a cost above 1.

    A denominator smaller than the number of sub-rules would let the level leave
    `[0, 1]` and silently outrank a higher one.
    """

    triple = tuple(component(f"sub{i}", 1.0) for i in range(3))
    with pytest.raises(ValueError):
        aggregate_sum_component(name="l", components=triple, denominator=2.0)


def test_a_level_name_may_not_collide_with_a_sub_rule_name() -> None:
    """`T-RB51-11`. Regression: the aggregated result must not shadow an atomic one.

    L6's sub-rule was first named `progress_rate`, the same as its level, so the
    aggregated result overwrote the atomic one in the component map. The two
    values coincide there, so nothing failed — §3.4's contract that every atomic
    cost stays exposed was quietly broken. The sub-rule is now
    `advance_shortfall` and the collision is an error.
    """

    from thesis_rl.rulebook.v2.aggregation import aggregate_rulebook_result

    with pytest.raises(ValueError, match="collide"):
        aggregate_rulebook_result(
            components=(component("progress_rate", 0.0),),
            raw_progress_m=0.0,
            progress_margin=0.0,
        )
