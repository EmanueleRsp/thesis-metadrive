"""R2 geometric clearance evaluator, scoped to VRU by rulebook v4.8 §6.4.

Vehicle clearance is replaced by the scoped lateral-RSS metric
(``components/rss_lateral.py``, REQ-R2-01). Static-obstacle clearance is
diagnostic-only (REQ-R2-02, DEC-R2-04): the worst static polygon distance is
still computed and exposed, but only in ``diagnostics``, never in ``raw`` or
``cost``.
"""

from __future__ import annotations

from math import isfinite

from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    CacheDelta,
    ComponentStatus,
    MemoryDelta,
    RuleComponentResult,
)


CLEARANCE_THRESHOLDS_M = {
    ActorClass.PEDESTRIAN: 1.0,
    ActorClass.CYCLIST: 1.0,
}

STATIC_DIAGNOSTIC_CLASS = ActorClass.STATIC_COLLIDABLE


def evaluate_clearance(
    *,
    ego_footprint,
    actors: tuple[ActorSnapshot, ...],
    vertically_compatible_actor_ids: frozenset[str],
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate VRU clearance cost; log worst static polygon distance only."""

    if ego_footprint.is_empty or not ego_footprint.is_valid:
        raise ValueError("Clearance requires a valid ego footprint")
    candidates: list[tuple[str, float, float]] = []
    static_polygon_distance_m: float | None = None
    for actor in actors:
        if actor.actor_id not in vertically_compatible_actor_ids:
            continue
        if actor.actor_class is STATIC_DIAGNOSTIC_CLASS:
            distance = ego_footprint.distance(actor.footprint)
            if not isfinite(distance):
                raise ValueError(f"Clearance distance is not finite for actor {actor.actor_id!r}")
            if static_polygon_distance_m is None or distance < static_polygon_distance_m:
                static_polygon_distance_m = distance
            continue
        threshold = CLEARANCE_THRESHOLDS_M.get(actor.actor_class)
        if threshold is None:
            continue
        distance = ego_footprint.distance(actor.footprint)
        if not isfinite(distance):
            raise ValueError(f"Clearance distance is not finite for actor {actor.actor_id!r}")
        cost = max(0.0, 1.0 - distance / threshold)
        candidates.append((actor.actor_id, distance, cost))
    diagnostics: dict[str, object] = {"candidate_count": len(candidates)}
    if static_polygon_distance_m is not None:
        diagnostics["static_polygon_distance_m"] = static_polygon_distance_m
    if not candidates:
        result = RuleComponentResult(
            name="clearance",
            cost=0.0,
            raw={"actors": ()},
            applicable=False,
            evaluable=True,
            status=ComponentStatus.NOT_APPLICABLE,
            diagnostics=diagnostics,
        )
        return result, MemoryDelta(), CacheDelta()
    worst_actor, distance, cost = max(candidates, key=lambda item: (item[2], item[0]))
    result = RuleComponentResult(
        name="clearance",
        cost=cost,
        raw={
            "worst_actor_id": worst_actor,
            "worst_distance_m": distance,
            "actors": tuple(
                {"actor_id": actor_id, "distance_m": actor_distance, "cost": actor_cost}
                for actor_id, actor_distance, actor_cost in candidates
            ),
        },
        applicable=True,
        evaluable=True,
        status=ComponentStatus.VIOLATED if cost > 0.0 else ComponentStatus.SATISFIED,
        diagnostics=diagnostics,
    )
    return result, MemoryDelta(), CacheDelta()
