"""The at-fault speed gate shared by the L2 interaction sub-rules (ADR-070).

`clearance`, `ttc` and `rss_lateral` all have their cost set by another agent's
state. From a stationary ego no action avoids what another agent brings to it, so
the region they define is **not controlled-invariant** -- the same test that
rejected `rss` longitudinal. Measured on 1100 Waymo `train` records, the three
charged 68.2 % of the penalty accrued while the ego was already stopped, with
`clearance` alone owing 79.9 % of its own cost to those steps.

The gate makes them **inapplicable** below a published stopped-speed threshold,
rather than satisfied at zero cost, which would be a different and wrong
statement: the rule has nothing to say about that state, it is not being met.

The threshold, the state it reads, and the sub-rules it covers are written here
once. Three components applying one rule from three private constants is how a
single rule becomes three slightly different rules.
"""

from __future__ import annotations

from math import hypot

from thesis_rl.rulebook.v2.types import (
    CacheDelta,
    ComponentStatus,
    MemoryDelta,
    RuleComponentResult,
)


# nuPlan's published at-fault value: `no_ego_at_fault_collisions`'s
# `stopped_speed_threshold = 5e-02`, "Threshold for 0 speed due to noise", under
# which `STOPPED_EGO_COLLISION` is excluded from the at-fault set.
#
# It is deliberately tiny. 0.05 m/s is 0.18 km/h, at which L4's margin is under
# 0.5 % of its maximum, so crawling under the gate buys immunity at no progress.
# That is the whole argument for transplanting an *evaluation* threshold into a
# *reward*, where an agent optimises against any hole: a gate at any of the
# larger swept thresholds (0.5, 1.0, 2.0 m/s) would be exploitable and must not
# be used.
AT_FAULT_GATE_SPEED_MPS = 5e-02

# The position sub-rules and the traffic-control sub-rules are deliberately not
# gated: from a state stopped astride a lane marking an action that leaves it
# exists, so the charge is a normative disagreement, not an unavoidable cost.
# nuPlan draws the same line -- `drivable_area_compliance` and
# `driving_direction_compliance` carry no speed gate -- and the expert panel
# measured it independently: `offroad` charges 0.0 % of its cost to a stopped
# ego.
AT_FAULT_GATED_SUB_RULES = frozenset({"clearance", "ttc", "rss_lateral"})


def ego_is_stopped(ego_velocity_xy: tuple[float, float]) -> bool:
    """Whether the ego is at or below the gate speed.

    Reads the **post-transition** ego velocity for all three sub-rules, even
    though `ttc` and `rss_lateral` compute their costs from the pre-transition
    state. The gate asks whether the ego is stopped *now*, at the state the cost
    is charged against; splitting it by sub-rule would make one gate into three.
    """

    return hypot(*ego_velocity_xy) <= AT_FAULT_GATE_SPEED_MPS


def at_fault_gated_result(name: str) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """The inapplicable result a gated sub-rule returns on a stopped ego."""

    if name not in AT_FAULT_GATED_SUB_RULES:
        raise ValueError(
            f"{name!r} is not an at-fault gated sub-rule; "
            f"expected one of {sorted(AT_FAULT_GATED_SUB_RULES)}."
        )
    result = RuleComponentResult(
        name=name,
        cost=0.0,
        raw={"actors": ()},
        applicable=False,
        evaluable=True,
        status=ComponentStatus.NOT_APPLICABLE,
        diagnostics={
            "at_fault_gated": True,
            "gate_speed_mps": AT_FAULT_GATE_SPEED_MPS,
        },
    )
    return result, MemoryDelta(), CacheDelta()
