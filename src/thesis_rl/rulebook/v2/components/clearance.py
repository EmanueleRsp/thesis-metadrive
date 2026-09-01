"""R2 geometric clearance evaluator, scoped to VRU on the roadway.

Scoped to VRU by rulebook v4.8 §6.4; scoped further, to VRU **standing on the
drivable surface**, by ADR-067. Vehicle clearance is replaced by the scoped
lateral-RSS metric (``components/rss_lateral.py``, REQ-R2-01). Static-obstacle
clearance is diagnostic-only (REQ-R2-02, DEC-R2-04): the worst static polygon
distance is still computed and exposed, but only in ``diagnostics``, never in
``raw`` or ``cost``.
"""

from __future__ import annotations

from math import isfinite

from thesis_rl.rulebook.v2.components.at_fault_gate import (
    at_fault_gated_result,
    ego_is_stopped,
)
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
    drivable_surface,
    post_ego_velocity_xy: tuple[float, float],
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate VRU-on-roadway clearance cost; log worst static polygon distance only.

    ADR-067: only pedestrians and cyclists whose footprint **centroid** lies on
    the drivable surface are candidates. A person waiting on the kerb of a narrow
    street is inside 1 m of every passing vehicle and is in conflict with none of
    them, so the unscoped rule priced normal urban driving (0.327 % -> 0.199 % of
    expert steps). The centre-entry criterion is the same one
    ``evaluate_wrong_carriageway`` uses, so the rulebook applies one geometric
    convention rather than two.

    ADR-070: inapplicable when the post-transition ego is at or below the
    at-fault gate speed. A stopped ego cannot open a gap a pedestrian is closing,
    and `clearance` owed 79.9 % of its cost to exactly those steps.

    The surface is validated rather than defaulted: an unusable surface would
    silently empty the candidate set and report cost 0, which is exactly the
    failure mode a scoping change must not introduce. ``evaluate_offroad``
    validates the same object identically.
    """

    if ego_footprint.is_empty or not ego_footprint.is_valid:
        raise ValueError("Clearance requires a valid ego footprint")
    if ego_is_stopped(post_ego_velocity_xy):
        return at_fault_gated_result("clearance")
    if drivable_surface is None or drivable_surface.is_empty or not drivable_surface.is_valid:
        raise ValueError("Clearance requires a valid drivable surface")
    candidates: list[tuple[str, float, float]] = []
    static_polygon_distance_m: float | None = None
    # Reported so the scoping is auditable: how many VRU were within range but
    # off the roadway is the difference between this rule and its unscoped form.
    off_roadway_count = 0
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
        if not drivable_surface.contains(actor.footprint.centroid):
            off_roadway_count += 1
            continue
        distance = ego_footprint.distance(actor.footprint)
        if not isfinite(distance):
            raise ValueError(f"Clearance distance is not finite for actor {actor.actor_id!r}")
        cost = max(0.0, 1.0 - distance / threshold)
        candidates.append((actor.actor_id, distance, cost))
    diagnostics: dict[str, object] = {
        "candidate_count": len(candidates),
        "off_roadway_vru_count": off_roadway_count,
    }
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
