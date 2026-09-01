"""L3 posted-speed-limit sub-rule (RULEBOOK-V5.0 §5.7, ADR-068).

**Why this exists rather than being merely complete.** L4's margin is
``clip(delta_s / (v_ref * dt), -1, 1)``, the ratio of route speed to an 80 km/h
reference, so the reward directly rewards driving at up to 80 km/h **including on
a 25 mph street**, and nothing else in the rulebook opposes it short of leaving
the road or colliding. This sub-rule closes a degeneracy L4's own definition
creates.

**Why it is admissible** where `rss` longitudinal was not: it is
controlled-invariant -- the ego can always decelerate, and no other agent can
push it above the limit -- memoryless, observable and physically plausible (the
posted limit is an HD-map attribute every production stack carries), anchored in
a published tolerance rather than a fitted one, and violated on **0.000 %** of
217,189 expert steps.
"""

from __future__ import annotations

from math import hypot, isfinite

from thesis_rl.rulebook.v2.types import (
    CacheDelta,
    ComponentStatus,
    MemoryDelta,
    RuleComponentResult,
)


# nuPlan's `speed_limit_compliance` tolerance: 5 mph. Published, not fitted.
SPEED_LIMIT_TOLERANCE_MPS = 2.23


def evaluate_speed_limit(
    *,
    ego_velocity_xy: tuple[float, float],
    posted_speed_limit_mps: float | None,
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Graded overspeed against the posted limit of the ego's associated lane.

    ``cost = clip((v_ego - (v_limit + tau)) / v_limit, 0, 1)``. The excess is
    normalized **by the limit** so a given cost means the same relative overspeed
    on a 15 mph street and a 45 mph road.

    With no posted limit the rule is **inapplicable**, never satisfied-at-zero
    and never evaluated against a substitute. The v1 extractor's fallback to
    ``ego_vehicle.max_speed_km_h`` is specifically prohibited: that constant is
    the vehicle's own cap, not a legal limit, and reusing it would make this a
    vehicle cap disguised as a norm.
    """

    if not all(isfinite(value) for value in ego_velocity_xy):
        raise ValueError("Speed-limit evaluation requires a finite ego velocity")
    if posted_speed_limit_mps is None:
        result = RuleComponentResult(
            name="speed_limit",
            cost=0.0,
            raw={"posted_limit_mps": None},
            applicable=False,
            evaluable=True,
            status=ComponentStatus.NOT_APPLICABLE,
            diagnostics={"tolerance_mps": SPEED_LIMIT_TOLERANCE_MPS},
        )
        return result, MemoryDelta(), CacheDelta()

    limit = float(posted_speed_limit_mps)
    if not isfinite(limit) or limit <= 0.0:
        raise ValueError(f"Posted speed limit must be finite and positive, got {limit!r}")
    ego_speed = hypot(*ego_velocity_xy)
    excess = ego_speed - (limit + SPEED_LIMIT_TOLERANCE_MPS)
    cost = min(max(excess / limit, 0.0), 1.0) if excess > 0.0 else 0.0
    result = RuleComponentResult(
        name="speed_limit",
        cost=cost,
        raw={
            "posted_limit_mps": limit,
            "ego_speed_mps": ego_speed,
            "excess_mps": max(excess, 0.0),
        },
        applicable=True,
        evaluable=True,
        status=ComponentStatus.VIOLATED if cost > 0.0 else ComponentStatus.SATISFIED,
        diagnostics={"tolerance_mps": SPEED_LIMIT_TOLERANCE_MPS},
    )
    return result, MemoryDelta(), CacheDelta()
