"""R2 geometric clearance evaluator over the exhaustive live-actor set."""

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
    ActorClass.VEHICLE: 0.8,
    ActorClass.PEDESTRIAN: 1.0,
    ActorClass.CYCLIST: 1.0,
    ActorClass.STATIC_COLLIDABLE: 0.5,
}


def evaluate_clearance(
    *,
    ego_footprint,
    actors: tuple[ActorSnapshot, ...],
    vertically_compatible_actor_ids: frozenset[str],
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate all live, collidable, vertically compatible actors without broad phase."""

    if ego_footprint.is_empty or not ego_footprint.is_valid:
        raise ValueError("Clearance requires a valid ego footprint")
    candidates: list[tuple[str, float, float]] = []
    for actor in actors:
        threshold = CLEARANCE_THRESHOLDS_M.get(actor.actor_class)
        if threshold is None or actor.actor_id not in vertically_compatible_actor_ids:
            continue
        distance = ego_footprint.distance(actor.footprint)
        if not isfinite(distance):
            raise ValueError(f"Clearance distance is not finite for actor {actor.actor_id!r}")
        cost = max(0.0, 1.0 - distance / threshold)
        candidates.append((actor.actor_id, distance, cost))
    if not candidates:
        result = RuleComponentResult(
            name="clearance",
            cost=0.0,
            raw={"actors": ()},
            applicable=False,
            evaluable=True,
            status=ComponentStatus.NOT_APPLICABLE,
            diagnostics={"candidate_count": 0},
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
        diagnostics={"candidate_count": len(candidates)},
    )
    return result, MemoryDelta(), CacheDelta()
