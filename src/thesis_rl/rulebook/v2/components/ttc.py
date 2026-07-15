"""R2 generalized TTC evaluator backed by the canonical continuous SAT."""

from __future__ import annotations

from math import isfinite

from thesis_rl.rulebook.v2.geometry.continuous_sat import predict_occupancy_interval
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    CacheDelta,
    ComponentStatus,
    MemoryDelta,
    RuleComponentResult,
)


TTC_HORIZON_S = 3.0
TTC_THRESHOLDS_S = {
    ActorClass.VEHICLE: 0.8,
    ActorClass.STATIC_COLLIDABLE: 0.8,
    ActorClass.PEDESTRIAN: 1.0,
    ActorClass.CYCLIST: 1.0,
}


def evaluate_ttc(
    *,
    ego_footprint,
    ego_velocity_xy: tuple[float, float],
    actors: tuple[ActorSnapshot, ...],
    vertically_compatible_actor_ids: frozenset[str],
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate all compatible live collidable actors under constant velocity."""

    if not ego_footprint.is_valid or ego_footprint.is_empty:
        raise ValueError("TTC requires a valid ego footprint")
    if not all(isfinite(value) for value in ego_velocity_xy):
        raise ValueError("TTC ego velocity must be finite")
    candidates: list[tuple[str, float | None, float, float]] = []
    for actor in actors:
        threshold = TTC_THRESHOLDS_S.get(actor.actor_class)
        if threshold is None or actor.actor_id not in vertically_compatible_actor_ids:
            continue
        actor_velocity = (0.0, 0.0) if actor.actor_class == ActorClass.STATIC_COLLIDABLE else actor.velocity_xy
        relative_velocity = (
            actor_velocity[0] - ego_velocity_xy[0], actor_velocity[1] - ego_velocity_xy[1]
        )
        interval = predict_occupancy_interval(
            actor_footprint=actor.footprint,
            actor_velocity_xy=relative_velocity,
            zone=ego_footprint,
            horizon_s=TTC_HORIZON_S,
        )
        ttc = None if interval is None else interval.start_s
        cost = 0.0 if ttc is None else max(0.0, 1.0 - ttc / threshold)
        delta = 0.0 if ttc is None else max(0.0, threshold - ttc)
        candidates.append((actor.actor_id, ttc, cost, delta))
    if not candidates:
        return (
            RuleComponentResult(
                "ttc", 0.0, {"actors": ()}, False, True,
                ComponentStatus.NOT_APPLICABLE, {"candidate_count": 0}
            ),
            MemoryDelta(), CacheDelta(),
        )
    worst_actor, ttc, cost, delta = max(candidates, key=lambda item: (item[2], item[0]))
    result = RuleComponentResult(
        "ttc",
        cost,
        {
            "worst_actor_id": worst_actor,
            "worst_ttc_s": ttc if ttc is not None else -1.0,
            "delta_ttc_s": delta,
            "actors": tuple(
                {"actor_id": actor_id, "ttc_s": actor_ttc if actor_ttc is not None else -1.0, "cost": actor_cost}
                for actor_id, actor_ttc, actor_cost, _ in candidates
            ),
        },
        True,
        True,
        ComponentStatus.VIOLATED if cost > 0.0 else ComponentStatus.SATISFIED,
        {"candidate_count": len(candidates)},
    )
    return result, MemoryDelta(), CacheDelta()
