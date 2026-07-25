"""R2 scoped lateral-RSS evaluator (rulebook v4.8 specification, §6-9).

Replaces ``q_clear,vehicle`` (v4.7 §6.4) for the ``VEHICLE`` class only.
Static and VRU clearance are unaffected by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from thesis_rl.rulebook.v2.types import (
    CacheDelta,
    ComponentStatus,
    MemoryDelta,
    RuleComponentResult,
)

RHO_LAT_S = 0.5
LATERAL_ACC_MAX_MPS2 = 0.2
LATERAL_BRAKE_MIN_MPS2 = 0.8
LATERAL_MARGIN_MU_M = 0.10


@dataclass(frozen=True, slots=True)
class LateralRSSCandidate:
    actor_id: str
    lateral_gap_m: float
    ego_inward_speed_mps: float
    actor_inward_speed_mps: float
    longitudinal_unsafe: bool


def _lateral_displacement_m(inward_speed_mps: float) -> float:
    """Worst-case lateral displacement during ``RHO_LAT_S`` plus braking (§7)."""

    response_speed = inward_speed_mps + RHO_LAT_S * LATERAL_ACC_MAX_MPS2
    response_distance = (inward_speed_mps + response_speed) / 2.0 * RHO_LAT_S
    braking_distance = max(0.0, response_speed) ** 2 / (2.0 * LATERAL_BRAKE_MIN_MPS2)
    return response_distance + braking_distance


def lateral_safe_distance_m(*, ego_inward_speed_mps: float, actor_inward_speed_mps: float) -> float:
    """``d_safe,i^lat`` per rulebook v4.8 specification §7."""

    if not isfinite(ego_inward_speed_mps) or not isfinite(actor_inward_speed_mps):
        raise ValueError("Lateral RSS inward speeds must be finite")
    delta_ego = _lateral_displacement_m(ego_inward_speed_mps)
    delta_actor = _lateral_displacement_m(actor_inward_speed_mps)
    return max(0.0, LATERAL_MARGIN_MU_M + delta_ego + delta_actor)


def evaluate_rss_lateral(
    *,
    candidates: tuple[LateralRSSCandidate, ...],
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate the scoped lateral-RSS candidates; worst-of, NOT_APPLICABLE if empty."""

    if not candidates:
        result = RuleComponentResult(
            name="rss_lateral",
            cost=0.0,
            raw={"actors": ()},
            applicable=False,
            evaluable=True,
            status=ComponentStatus.NOT_APPLICABLE,
            diagnostics={"candidate_count": 0},
        )
        return result, MemoryDelta(), CacheDelta()
    values: list[tuple[str, float, float, float]] = []
    for candidate in candidates:
        if not isfinite(candidate.lateral_gap_m) or candidate.lateral_gap_m < 0.0:
            raise ValueError(f"Lateral RSS gap for actor {candidate.actor_id!r} is invalid")
        if not candidate.longitudinal_unsafe:
            values.append((candidate.actor_id, 0.0, 0.0, 0.0))
            continue
        safe = lateral_safe_distance_m(
            ego_inward_speed_mps=candidate.ego_inward_speed_mps,
            actor_inward_speed_mps=candidate.actor_inward_speed_mps,
        )
        cost = 0.0 if safe == 0.0 else max(0.0, min(1.0, 1.0 - candidate.lateral_gap_m / safe))
        values.append((candidate.actor_id, safe, candidate.lateral_gap_m, cost))
    worst_actor, safe, gap, cost = max(values, key=lambda value: (value[3], value[0]))
    result = RuleComponentResult(
        name="rss_lateral",
        cost=cost,
        raw={
            "worst_actor_id": worst_actor,
            "worst_lateral_safe_distance_m": safe,
            "worst_lateral_gap_m": gap,
            "actors": tuple(
                {
                    "actor_id": actor_id,
                    "lateral_safe_distance_m": actor_safe,
                    "lateral_gap_m": actor_gap,
                    "cost": actor_cost,
                }
                for actor_id, actor_safe, actor_gap, actor_cost in values
            ),
        },
        applicable=True,
        evaluable=True,
        status=ComponentStatus.VIOLATED if cost > 0.0 else ComponentStatus.SATISFIED,
        diagnostics={"candidate_count": len(values)},
    )
    return result, MemoryDelta(), CacheDelta()
