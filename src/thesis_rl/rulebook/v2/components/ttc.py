"""R2 generalized TTC evaluator backed by the canonical continuous SAT."""

from __future__ import annotations

from math import isfinite

from thesis_rl.rulebook.v2.components.at_fault_gate import (
    at_fault_gated_result,
    ego_is_stopped,
)
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
# ADR-067: nuPlan's published `least_min_ttc` bound, applied uniformly. It
# replaces a repository-chosen pair (0.8 s for vehicles, 1.0 s for VRU) whose
# split had no source; the uniform value costs 0.04 percentage points of expert
# violation, and at every swept threshold `ttc` stays an order of magnitude
# below the other R2 sub-rules, so it needs no tolerance of its own.
TTC_THRESHOLD_S = 0.95
# Which actor classes are candidates at all. This used to be carried implicitly
# by membership in the per-class threshold table, so "eligible" and "how urgent"
# were one object and could not be changed independently. They are two
# decisions; ADR-067 changes only the second.
TTC_CANDIDATE_CLASSES = frozenset(
    {
        ActorClass.VEHICLE,
        ActorClass.STATIC_COLLIDABLE,
        ActorClass.PEDESTRIAN,
        ActorClass.CYCLIST,
    }
)


def evaluate_ttc(
    *,
    ego_footprint,
    ego_velocity_xy: tuple[float, float],
    actors: tuple[ActorSnapshot, ...],
    vertically_compatible_actor_ids: frozenset[str],
    post_ego_velocity_xy: tuple[float, float],
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate all compatible live collidable actors under constant velocity.

    ADR-070: inapplicable when the post-transition ego is at or below the
    at-fault gate speed. nuPlan applies the same idea to its own TTC metric,
    which is not computed at all below `stopped_speed_threshold`.
    """

    if not ego_footprint.is_valid or ego_footprint.is_empty:
        raise ValueError("TTC requires a valid ego footprint")
    if not all(isfinite(value) for value in ego_velocity_xy):
        raise ValueError("TTC ego velocity must be finite")
    if not all(isfinite(value) for value in post_ego_velocity_xy):
        raise ValueError("TTC post-state ego velocity must be finite")
    if ego_is_stopped(post_ego_velocity_xy):
        return at_fault_gated_result("ttc")
    candidates: list[tuple[str, float | None, float, float]] = []
    for actor in actors:
        if actor.actor_class not in TTC_CANDIDATE_CLASSES:
            continue
        if actor.actor_id not in vertically_compatible_actor_ids:
            continue
        threshold = TTC_THRESHOLD_S
        actor_velocity = (
            (0.0, 0.0) if actor.actor_class == ActorClass.STATIC_COLLIDABLE else actor.velocity_xy
        )
        relative_velocity = (
            actor_velocity[0] - ego_velocity_xy[0],
            actor_velocity[1] - ego_velocity_xy[1],
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
                "ttc",
                0.0,
                {"actors": ()},
                False,
                True,
                ComponentStatus.NOT_APPLICABLE,
                {"candidate_count": 0},
            ),
            MemoryDelta(),
            CacheDelta(),
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
                {
                    "actor_id": actor_id,
                    "ttc_s": actor_ttc if actor_ttc is not None else -1.0,
                    "cost": actor_cost,
                }
                for actor_id, actor_ttc, actor_cost, _ in candidates
            ),
        },
        True,
        True,
        ComponentStatus.VIOLATED if cost > 0.0 else ComponentStatus.SATISFIED,
        {"candidate_count": len(candidates)},
    )
    return result, MemoryDelta(), CacheDelta()
