"""L6 `progress_rate`, RULEBOOK-V5.1 §4.6 (ADR-076).

The sixth level, below relaxable lane compliance. It exists because the mission
channel telescopes when summed undiscounted: two trajectories that reach the
same place score identically at L4 however long they take, and nothing else in
the rulebook prefers the faster one — `speed_limit` is an upper bound only.

*The premise is weaker than it was.* ADR-075 chose `gamma = 1`, which makes that
tie hold in the agent's return as well; ADR-081 then set `gamma = 0.996`, under
which a discounted sum weights the same increments by recency and the sooner
arrival scores strictly more. This level's *reason to exist* survives — an
undiscounted L4 still cannot prefer the faster completion, and the crawl
pathology below is measured, not derived from the discount — but its companion
claim, that placing L6 below L5 keeps time preference from paying for a lane
violation, does **not** survive: at the shipped discount the comparison resolves
at L4 before L5 is consulted. See `RULEBOOK-V5.1` §11.12.

Arrival is not a sufficient bound either. Measured over the frozen Waymo `train`
panel, the median mission needs only 3.40 m/s to arrive inside its horizon while
the vehicle is capped at 22.22 m/s, so the band in which L4 is indifferent is
6.5x wide, and inside it slower driving strictly reduces exposure to the L2
interaction sub-rules. Without L6 the optimum is to crawl at 12 km/h.

**Placed below L5, which is the whole point.** An illegal shortcut carries
`c_L5 > 0` and therefore loses before this level is ever consulted, so how
strongly the reward prefers arriving sooner is decoupled from whether arriving
sooner can pay for a lane violation. Folding the same quantity into L4 was
implemented, measured and rejected for exactly that reason.

That decoupling is a property of the **undiscounted** comparison. At
`gamma = 0.996` the shortcut's discounted L4 total exceeds the legal route's, so
the ordering is settled at L4 and this placement never gets to do its job for
that pair. The remedy is a decision, not a code change; it is tracked as
`REQ-RB5.1-O3-DISCOUNT`.
"""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from thesis_rl.rulebook.v2.components.progress import MISSION_PROGRESS_REFERENCE_SPEED_MPS
from thesis_rl.rulebook.v2.types import (
    CacheDelta,
    ComponentStatus,
    MemoryDelta,
    RuleComponentResult,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from thesis_rl.mission.types import MissionSnapshot


def progress_rate_cost(delta_q: float) -> float:
    """``c_L6 = 1 - clip(Delta q, 0, 1)``.

    Summed over a completing trajectory this is ``T - Q``, so the level ranks by
    **duration** exactly and the mission-length constant cancels from every
    comparison. A flat per-step time counter would give the same ranking with no
    local signal; this form additionally makes each step's advance lower the
    cost now, which is what a bootstrapped critic can learn from.

    Reverse motion clips to zero advance and therefore costs the maximum, which
    is consistent with L4's own signed treatment of it. Standing still likewise
    costs 1.
    """

    if not isfinite(delta_q):
        raise ValueError("Progress rate requires a finite advance")
    return 1.0 - max(0.0, min(delta_q, 1.0))


def evaluate_progress_rate(
    *,
    pre_mission: "MissionSnapshot",
    post_mission: "MissionSnapshot",
    delta_t_s: float,
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate L6 from the same station delta L4 reads.

    Nothing new is perceived: this is a function of ``Delta q``, which §4.1
    already computes, so the level adds no observation requirement.
    """

    from thesis_rl.mission.types import MissionSnapshot

    if not isinstance(pre_mission, MissionSnapshot) or not isinstance(
        post_mission, MissionSnapshot
    ):
        raise ValueError("Progress rate requires pre/post mission context")
    if pre_mission.mission_hash != post_mission.mission_hash:
        raise ValueError("Progress rate mission snapshot identity must match")
    if not isfinite(delta_t_s) or delta_t_s <= 0.0:
        raise ValueError("Progress rate requires a positive finite timestep")
    if pre_mission.s_m is None or post_mission.s_m is None:
        raise ValueError("L6 requires exact canonical route stations")

    reference_advance_m = MISSION_PROGRESS_REFERENCE_SPEED_MPS * delta_t_s
    raw_delta = post_mission.s_m - pre_mission.s_m
    delta_q = min(max(raw_delta / reference_advance_m, -1.0), 1.0)
    cost = progress_rate_cost(delta_q)

    result = RuleComponentResult(
        "advance_shortfall",
        cost,
        {"delta_s_m": raw_delta, "delta_q": delta_q},
        True,
        True,
        ComponentStatus.VIOLATED if cost > 0.0 else ComponentStatus.SATISFIED,
        {
            "reference_speed_mps": MISSION_PROGRESS_REFERENCE_SPEED_MPS,
            "reference_advance_m": reference_advance_m,
            "delta_t_s": delta_t_s,
        },
    )
    return result, MemoryDelta(writer="advance_shortfall"), CacheDelta()
