"""Source-neutral composition of one causal Rulebook v2 transition.

The live MetaDrive adapter is responsible only for constructing ``EnvSnapshot``
objects.  This module derives all evaluator inputs from those snapshots and the
immutable episode cache, then invokes the fixed registry exactly once.  Missing
topology or calibration is represented by an explicit NOT_APPLICABLE domain or
by a fail-fast validation error; no source-specific fallback is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, isfinite, pi, sin
import time

import shapely
from shapely.geometry.base import BaseGeometry

from thesis_rl.rulebook.v2.components.controls import CROSSWALK_GAP_S, signal_group_state
from thesis_rl.rulebook.v2.components.rss import (
    RSSCalibrationArtifact,
    RSSCandidate,
)
from thesis_rl.rulebook.v2.components.rss_lateral import LateralRSSCandidate
from thesis_rl.rulebook.v2.geometry.conflict_zones import (
    ConflictZoneCandidate,
    MovementCorridor,
    RouteConflictZoneCandidate,
    attach_route_intervals,
    build_vehicle_conflict_zone_candidates,
    select_first_ahead_or_occupied_zone,
    worst_case_temporal_gap_violation,
)
from thesis_rl.rulebook.v2.geometry.continuous_sat import OccupancyInterval
from thesis_rl.rulebook.v2.geometry.ctrv import predict_conflict_zone_occupancy_intervals
from thesis_rl.rulebook.v2.geometry.drivable import (
    DrivableLaneRecord,
    carriageway_surfaces_for_ego,
    drivable_surface_for_ego,
)
from thesis_rl.rulebook.v2.geometry.elevation import PolylineElevation
from thesis_rl.rulebook.v2.geometry.footprint import swept_front_bumper
from thesis_rl.rulebook.v2.geometry.lanes import (
    LaneAssociation,
    RouteLaneRecord,
    anchored_frame_extent,
    anchored_lateral_gap,
    associate_route_lane,
    bumper_to_bumper_gap,
    derive_lane_movement_key,
    footprint_route_coordinates,
    same_traffic_stream,
    tangent_intervals_overlap,
)
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.geometry.vertical import VERTICAL_COMPATIBILITY_TOLERANCE_M
from thesis_rl.rulebook.v2.memory import build_motion_history_preview
from thesis_rl.rulebook.v2.monitor import evaluate_registered_transition
from thesis_rl.rulebook.v2.types import (
    ApproachControl,
    ActorClass,
    ActorSnapshot,
    EpisodeCache,
    EnvSnapshot,
    MapFeatureClass,
    MovementPriority,
    RulebookMemory,
    CacheDelta,
    ConflictZoneRecord,
    VehicleConflictPairRecord,
)


# ADR-035: derived implementation constant, not a scientific parameter.  It
# must exclude perpendicular (90 deg) and opposing (180 deg) lanes while
# tolerating the 20-30 deg divergence between the tangents of an ego and an
# abreast vehicle on a tight curve.
LATERAL_RSS_MAX_TANGENT_MISALIGNMENT_RAD = pi / 4.0


@dataclass(frozen=True, slots=True)
class RulebookTransitionConfig:
    """Runtime-only constants already frozen by Rulebook v4.7."""

    rss_calibration: RSSCalibrationArtifact | None = None
    expected_config_hash: str = ""
    prediction_horizon_s: float = 3.0
    history_window_s: float = 0.5
    minimum_history_samples: int = 3
    disable_vehicle_yield_for_benchmark: bool = False


def build_episode_cache(static_result) -> EpisodeCache:
    """Build an immutable cache from a validated PG/Waymo static result."""

    if getattr(static_result, "validation_errors", ()):
        raise ValueError(
            "Static adapter result is not eligible: " + ", ".join(static_result.validation_errors)
        )
    route = static_result.assigned_route_polyline
    return EpisodeCache(
        scenario_id=static_result.scenario_uid,
        task_route=static_result.task_route,
        map_feature_catalog=static_result.map_features,
        traffic_control_catalog=static_result.traffic_controls,
        movement_priority_records=static_result.movement_priority_records,
        roundabout_priority_records=getattr(static_result, "roundabout_priority_records", ()),
        route_lanes=static_result.route_lanes,
        route_polyline=route,
        control_line_off_route_drop_count=getattr(
            static_result, "dropped_control_line_off_route_count", 0
        ),
    )


def initial_memory_for_snapshot(snapshot: EnvSnapshot, cache: EpisodeCache) -> RulebookMemory:
    """Initialize route and time ownership from the causal reset snapshot."""

    if cache.route_polyline is None:
        raise ValueError("Episode cache has no assigned route polyline")
    return RulebookMemory(
        previous_sim_time_s=snapshot.sim_time_s,
        previous_contact_ids=snapshot.active_contact_ids,
    )


def align_episode_cache_to_live_elevation(
    cache: EpisodeCache, snapshot: EnvSnapshot
) -> EpisodeCache:
    """Align one source map's absolute elevation datum to MetaDrive's live datum.

    MetaDrive flattens the initial ego spawn to ``z=0`` while some Waymo
    descriptions retain an absolute elevation offset.  XY topology and all
    relative elevation differences are preserved; only the common datum is
    translated, and no source file is changed.
    """

    if cache.route_polyline is None:
        raise ValueError("Episode cache has no assigned route polyline")
    reference = cache.route_polyline.project(snapshot.ego.position_xy)
    offset = snapshot.ego.position_z - reference.z_m
    if not isfinite(offset):
        raise ValueError("Live elevation datum offset must be finite")
    if abs(offset) <= 1.0e-9:
        return cache
    # step_timing_instrumentation_v1 investigation, 2026-08-01: preserve
    # ``lane_start_points_xyz`` (video overlay v1 REQ-001, checkpoint
    # markers) through the elevation shift. A plain-constructor rebuild here
    # previously reset it to its `()` default, silently dropping checkpoint
    # markers for every scenario with a nonzero elevation offset -- common
    # for Waymo sources per this function's docstring.
    shifted_route = RoutePolyline(
        tuple((x, y, z + offset) for x, y, z in cache.route_polyline.points_xyz),
        lane_start_points_xyz=tuple(
            (x, y, z + offset) for x, y, z in cache.route_polyline.lane_start_points_xyz
        ),
    )
    shifted_lanes = tuple(
        replace(
            lane,
            centerline=RoutePolyline(
                tuple((x, y, z + offset) for x, y, z in lane.centerline.points_xyz)
            ),
        )
        for lane in cache.route_lanes
    )
    shifted_features = {
        feature_id: replace(
            feature,
            elevation_m=None if feature.elevation_m is None else feature.elevation_m + offset,
        )
        for feature_id, feature in cache.map_feature_catalog.items()
    }
    shifted_controls = tuple(
        replace(control, elevation_m=control.elevation_m + offset)
        for control in cache.traffic_control_catalog
    )
    shifted_zones = {
        zone_id: replace(zone, elevation_m=zone.elevation_m + offset)
        for zone_id, zone in cache.conflict_zones.items()
    }
    return replace(
        cache,
        route_polyline=shifted_route,
        route_lanes=shifted_lanes,
        map_feature_catalog=shifted_features,
        traffic_control_catalog=shifted_controls,
        conflict_zones=shifted_zones,
    )


def _task_corridor(cache: EpisodeCache):
    """Union the polygons of the assigned-route lanes only (REQ-EF-15).

    ``cache.route_lanes`` holds every lane of the map, not just the assigned
    ones, so the corridor must be filtered by ``task_route.lane_ids``.
    """

    assigned = frozenset(cache.task_route.lane_ids)
    polygons = [lane.polygon_xy for lane in cache.route_lanes if lane.lane_id in assigned]
    if not polygons:
        return None
    return shapely.union_all(polygons)


def _delta_t(pre: EnvSnapshot, post: EnvSnapshot) -> float:
    delta_t = post.sim_time_s - pre.sim_time_s
    if not isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("Rulebook transition requires a positive finite simulation timestep")
    return delta_t


def _front_s(snapshot: EnvSnapshot, route) -> float:
    return footprint_route_coordinates(
        snapshot.ego.footprint, route, position_z=snapshot.ego.position_z
    ).front_s_m


def _vertical_actor_ids(ego: ActorSnapshot, actors: tuple[ActorSnapshot, ...]) -> frozenset[str]:
    return frozenset(
        actor.actor_id
        for actor in actors
        if abs(actor.position_z - ego.position_z) <= VERTICAL_COMPATIBILITY_TOLERANCE_M
    )


def _route_lane(cache: EpisodeCache, lane_id: str | None) -> RouteLaneRecord | None:
    return next((lane for lane in cache.route_lanes if lane.lane_id == lane_id), None)


def _snapshot_lane_associations(
    snapshot: EnvSnapshot,
    route_lanes: tuple[RouteLaneRecord, ...],
) -> tuple[LaneAssociation | None, dict[str, LaneAssociation | None]]:
    """Associate every live vehicle once for all Rulebook components of a snapshot."""

    ego_association = associate_route_lane(
        position_xy=snapshot.ego.position_xy,
        position_z=snapshot.ego.position_z,
        heading_rad=snapshot.ego.heading_rad,
        route_lanes=route_lanes,
    )
    actor_associations = {
        actor.actor_id: associate_route_lane(
            position_xy=actor.position_xy,
            position_z=actor.position_z,
            heading_rad=actor.heading_rad,
            route_lanes=route_lanes,
        )
        for actor in snapshot.actors
        if actor.actor_class is ActorClass.VEHICLE
    }
    return ego_association, actor_associations


def _rss_candidates(
    *,
    ego: ActorSnapshot,
    actors: tuple[ActorSnapshot, ...],
    route,
    route_lanes: tuple[RouteLaneRecord, ...],
    ego_association: LaneAssociation | None = None,
    actor_associations: dict[str, LaneAssociation | None] | None = None,
) -> tuple[RSSCandidate, ...]:
    ego_lane = ego_association
    if ego_lane is None:
        ego_lane = associate_route_lane(
            position_xy=ego.position_xy,
            position_z=ego.position_z,
            heading_rad=ego.heading_rad,
            route_lanes=route_lanes,
        )
    if ego_lane is None:
        return ()
    ego_heading = (cos(ego.heading_rad), sin(ego.heading_rad))
    if ego_heading[0] * ego_lane.tangent_xy[0] + ego_heading[1] * ego_lane.tangent_xy[1] <= 0.0:
        return ()
    try:
        ego_coords = footprint_route_coordinates(ego.footprint, route, position_z=ego.position_z)
    except ValueError:
        # Ego's footprint has no vertically compatible route segment nearby
        # (e.g. a grade-separated overpass/underpass); no RSS-longitudinal
        # candidate can be scoped this step.
        return ()
    candidates: list[RSSCandidate] = []
    for actor in actors:
        if actor.actor_class is not ActorClass.VEHICLE:
            continue
        actor_lane = (
            actor_associations.get(actor.actor_id)
            if actor_associations is not None
            else associate_route_lane(
                position_xy=actor.position_xy,
                position_z=actor.position_z,
                heading_rad=actor.heading_rad,
                route_lanes=route_lanes,
            )
        )
        if actor_lane is None or not same_traffic_stream(
            ego_lane.lane_id, actor_lane.lane_id, route_lanes
        ):
            continue
        actor_heading = (cos(actor.heading_rad), sin(actor.heading_rad))
        if (
            actor_heading[0] * actor_lane.tangent_xy[0]
            + actor_heading[1] * actor_lane.tangent_xy[1]
            <= 0.0
        ):
            continue
        try:
            other_coords = footprint_route_coordinates(
                actor.footprint, route, position_z=actor.position_z
            )
        except ValueError:
            # Actor's footprint has no vertically compatible route segment
            # nearby (e.g. it is on a grade-separated overpass/underpass
            # relative to ego's route); exclude it from this step's
            # candidates rather than aborting the whole evaluation.
            continue
        gap, is_front = bumper_to_bumper_gap(ego_coords, other_coords)
        if is_front:
            candidates.append(
                RSSCandidate(
                    actor_id=actor.actor_id,
                    gap_m=gap,
                    ego_speed_mps=max(
                        0.0,
                        ego.velocity_xy[0] * ego_lane.tangent_xy[0]
                        + ego.velocity_xy[1] * ego_lane.tangent_xy[1],
                    ),
                    front_speed_mps=max(
                        0.0,
                        actor.velocity_xy[0] * actor_lane.tangent_xy[0]
                        + actor.velocity_xy[1] * actor_lane.tangent_xy[1],
                    ),
                )
            )
    return tuple(sorted(candidates, key=lambda candidate: candidate.actor_id))


def _lane_tangent_misalignment_rad(ego_lane: LaneAssociation, actor_lane: LaneAssociation) -> float:
    """Return the unsigned angle between the two lanes' local tangents."""

    cross = (
        ego_lane.tangent_xy[0] * actor_lane.tangent_xy[1]
        - ego_lane.tangent_xy[1] * actor_lane.tangent_xy[0]
    )
    dot = (
        ego_lane.tangent_xy[0] * actor_lane.tangent_xy[0]
        + ego_lane.tangent_xy[1] * actor_lane.tangent_xy[1]
    )
    return abs(atan2(cross, dot))


def _rss_lateral_candidates(
    *,
    ego: ActorSnapshot,
    actors: tuple[ActorSnapshot, ...],
    route,
    route_lanes: tuple[RouteLaneRecord, ...],
    ego_brake_mps2: float | None,
    ego_association: LaneAssociation | None = None,
    actor_associations: dict[str, LaneAssociation | None] | None = None,
) -> tuple[LateralRSSCandidate, ...]:
    """Build R2 scoped lateral-RSS candidates (v4.8 §7-8, amended by ADR-035).

    A pair is admitted only when the two footprints are **abreast**: their
    tangent-axis intervals, measured on one frame anchored to the ego, must
    overlap, and the two lanes' local tangents must be within
    ``LATERAL_RSS_MAX_TANGENT_MISALIGNMENT_RAD``.

    That single predicate replaces three separate defects of the literal v4.8
    reading, all instances of applying an abreast-pair model to pairs that are
    not abreast: a receding rear vehicle scored ``1.0`` while a closing one
    scored ``0.0`` (the gate placed the ego unconditionally in the rear role);
    any same-lane leader scored ``1.0`` because two vehicles in one lane have a
    lateral gap of exactly zero against ``d_safe^lat ~= 0.1625 m``, masking the
    graded ``q_RSS,long``; and a vehicle on a perpendicular intersection branch
    scored ``0.937``.

    For an admitted pair v4.8 §7 step 2 already makes ``I_long,unsafe`` true by
    construction, so steps 3-5 (rear/front identification, reuse of the
    RSS-longitudinal formula) are unreachable and no braking calibration for a
    non-ego vehicle is needed.  Rear vehicles, same-lane leaders and crossing
    branches remain covered by TTC, collision, crosswalk/conflict-zone and
    vehicle-yield, per v4.8 §8.
    """

    if ego_brake_mps2 is None or ego_brake_mps2 <= 0.0:
        return ()
    ego_lane = ego_association
    if ego_lane is None:
        ego_lane = associate_route_lane(
            position_xy=ego.position_xy,
            position_z=ego.position_z,
            heading_rad=ego.heading_rad,
            route_lanes=route_lanes,
        )
    if ego_lane is None:
        return ()
    ego_heading = (cos(ego.heading_rad), sin(ego.heading_rad))
    if ego_heading[0] * ego_lane.tangent_xy[0] + ego_heading[1] * ego_lane.tangent_xy[1] <= 0.0:
        return ()
    try:
        ego_coords = footprint_route_coordinates(ego.footprint, route, position_z=ego.position_z)
    except ValueError:
        # Ego's footprint has no vertically compatible route segment nearby
        # (e.g. a grade-separated overpass/underpass); no lateral-RSS
        # candidate can be scoped this step.
        return ()
    # One frame for the whole pair set, anchored to the ego centroid (v4.8 §3):
    # comparing extents that each vertex projected onto its own nearest route
    # segment is only valid on a straight route.
    tangent = ego_coords.center_tangent_xy
    normal = (-tangent[1], tangent[0])
    ego_centroid = ego.footprint.centroid
    frame_origin = (float(ego_centroid.x), float(ego_centroid.y))
    ego_extent = anchored_frame_extent(ego.footprint, origin_xy=frame_origin, tangent_xy=tangent)
    ego_normal_speed = ego.velocity_xy[0] * normal[0] + ego.velocity_xy[1] * normal[1]
    candidates: list[LateralRSSCandidate] = []
    for actor in actors:
        if actor.actor_class is not ActorClass.VEHICLE:
            continue
        actor_lane = (
            actor_associations.get(actor.actor_id)
            if actor_associations is not None
            else associate_route_lane(
                position_xy=actor.position_xy,
                position_z=actor.position_z,
                heading_rad=actor.heading_rad,
                route_lanes=route_lanes,
            )
        )
        if actor_lane is None:
            continue
        actor_heading = (cos(actor.heading_rad), sin(actor.heading_rad))
        if (
            actor_heading[0] * actor_lane.tangent_xy[0]
            + actor_heading[1] * actor_lane.tangent_xy[1]
            <= 0.0
        ):
            continue
        # ADR-035: the pair must travel along compatible local directions for
        # the shared-tangent model to hold at all.  This excludes crossing and
        # opposing branches that the per-actor heading check alone admits.
        if (
            _lane_tangent_misalignment_rad(ego_lane, actor_lane)
            > LATERAL_RSS_MAX_TANGENT_MISALIGNMENT_RAD
        ):
            continue
        actor_extent = anchored_frame_extent(
            actor.footprint, origin_xy=frame_origin, tangent_xy=tangent
        )
        # ADR-035: only abreast pairs are in the domain of the lateral metric.
        if not tangent_intervals_overlap(ego_extent, actor_extent):
            continue
        gap, direction = anchored_lateral_gap(ego_extent, actor_extent)
        actor_normal_speed = actor.velocity_xy[0] * normal[0] + actor.velocity_xy[1] * normal[1]
        # Inward convention: positive when moving toward the other vehicle
        # along the shared normal (rulebook v4.8 specification §3, §7).
        ego_inward_speed = direction * ego_normal_speed
        actor_inward_speed = -direction * actor_normal_speed
        candidates.append(
            LateralRSSCandidate(
                actor_id=actor.actor_id,
                lateral_gap_m=gap,
                ego_inward_speed_mps=ego_inward_speed,
                actor_inward_speed_mps=actor_inward_speed,
                # True by construction for an abreast pair (v4.8 §7 step 2).
                longitudinal_unsafe=True,
            )
        )
    return tuple(sorted(candidates, key=lambda candidate: candidate.actor_id))


def _control_distances(
    control,
    *,
    pre_front_s: float,
    post_front_s: float,
    pre_footprint,
    post_footprint,
    pre_heading_rad: float,
    post_heading_rad: float,
) -> tuple[float, float, bool]:
    """Return signed distances and the normative swept-bumper crossing event."""

    pre_delta = control.route_s_m - pre_front_s
    post_delta = control.route_s_m - post_front_s
    swept_bumper = swept_front_bumper(
        pre_footprint,
        post_footprint,
        pre_heading_rad=pre_heading_rad,
        post_heading_rad=post_heading_rad,
    )
    crossing = (
        pre_delta >= -0.05 and post_delta < -0.05 and swept_bumper.intersects(control.control_line)
    )
    return pre_delta, post_delta, crossing


def route_reachable_control_lane_ids(cache: EpisodeCache) -> frozenset[str]:
    """Route lanes plus the route's unambiguous single successor (`ADR-051`).

    `assigned_route_lane_ids` for a fixed-window recording (e.g. Waymo's
    `training_20s`) is built from the human driver's actually-recorded path,
    which can legitimately end one lane short of a junction the driver had
    not yet crossed within the window. The RL policy driving the ego during
    training is not bound to that recorded path and can reach and cross that
    lane within the same episode, so a control governing it is physically
    relevant to the ego. Mirrors the same single-successor disambiguation
    `derive_lane_movement_key` already applies to the ego's own
    `exit_lane_id`: an unambiguous next lane counts as reachable, an
    ambiguous branch (more than one successor) is not guessed.
    """

    route_lane_ids = cache.task_route.lane_ids
    allowed = set(route_lane_ids)
    if route_lane_ids:
        terminal_lane = next(
            (lane for lane in cache.route_lanes if lane.lane_id == route_lane_ids[-1]), None
        )
        if terminal_lane is not None:
            successors = tuple(getattr(terminal_lane, "successor_lane_ids", ()) or ())
            if len(successors) == 1:
                allowed.add(successors[0])
    return frozenset(allowed)


def _selected_control(controls, control_type, front_s: float, resolved, route_lane_ids=()):
    """Select the first unresolved control ahead that governs the ego (§2.9.5).

    ``route_lane_ids`` implements the specification's "movimento ego
    pertinente" filter.  The relevant movement is *not* the ego's current
    ``MovementKey``: a signal at the end of the next route lane must remain
    selectable while the ego is still on the current one.  The operational
    predicate is therefore that the control's approach lane belongs to the
    assigned route, which admits every control the ego will actually meet and
    excludes controls governing other approaches.  Without it, a control on a
    foreign road could govern the ego, mask the correct one, or abort the
    scenario with an ``UNKNOWN`` state.
    """

    allowed = frozenset(route_lane_ids)
    candidates = tuple(
        control
        for control in controls
        if control.control_type is control_type
        and control.control_group_id not in resolved
        and control.route_s_m >= front_s
        and (not allowed or control.movement_key.approach_lane_id in allowed)
    )
    return min(candidates, key=lambda item: (item.route_s_m, item.control_group_id), default=None)


def control_line_diagnostics(cache: EpisodeCache) -> dict[str, int]:
    """Count controls silently unavailable to `_selected_control` (F4b).

    Static per scenario (depends only on `cache.traffic_control_catalog` and
    `cache.task_route.lane_ids`, neither of which changes over an episode),
    so this is cheap to compute once and does not need per-step memory.
    Reproduces `_selected_control`'s `approach_lane_id in allowed` filter
    (including the `ADR-051` route-successor extension) without recomputing
    selection, to answer the audit's open question: of the controls that
    survived adapter construction, how many are then excluded from ever being
    selectable because their approach lane is not reachable from the assigned
    route.
    """

    allowed = route_reachable_control_lane_ids(cache)
    signal_total = stop_total = 0
    signal_route_scoped = stop_route_scoped = 0
    for control in cache.traffic_control_catalog:
        on_route = not allowed or control.movement_key.approach_lane_id in allowed
        if control.control_type is ApproachControl.SIGNAL:
            signal_total += 1
            signal_route_scoped += int(on_route)
        elif control.control_type is ApproachControl.STOP:
            stop_total += 1
            stop_route_scoped += int(on_route)
    return {
        "control_line_off_route_drop_count": cache.control_line_off_route_drop_count,
        "signal_controls_total": signal_total,
        "signal_controls_approach_filter_dropped": signal_total - signal_route_scoped,
        "stop_controls_total": stop_total,
        "stop_controls_approach_filter_dropped": stop_total - stop_route_scoped,
    }


def _empty_interval() -> OccupancyInterval:
    # A zero-duration interval is used only when the corresponding rule domain
    # is empty; the evaluator remains NOT_APPLICABLE because no opposing actor
    # or zone is supplied.
    return OccupancyInterval(0.0, 0.0)


def _cleared_crosswalk_illegal_entries(
    *, cache: EpisodeCache, memory: RulebookMemory, ego_footprint: BaseGeometry
) -> frozenset[tuple[str, str]]:
    """Drop crosswalk illegal-entry latches for zones ego has already left.

    Same rationale as ``_cleared_vehicle_yield_illegal_entries`` (ADR-025):
    ``select_first_ahead_or_occupied_zone``'s epsilon-scoped "ahead" filter
    can exclude a crosswalk zone from candidacy before the ego footprint
    (vehicle-length scoped) actually stops intersecting it, so the zone_id
    selected for this step's cost evaluation is not a reliable signal for
    releasing latches on zones ego has fully exited.
    """
    active = set(memory.crosswalk_illegal_entries)
    for actor_id, zone_id in memory.crosswalk_illegal_entries:
        record = cache.conflict_zones.get(zone_id)
        if record is not None and not ego_footprint.intersects(record.polygon):
            active.discard((actor_id, zone_id))
    return frozenset(active)


def _crosswalk_inputs(
    *,
    pre: EnvSnapshot,
    post: EnvSnapshot,
    cache: EpisodeCache,
    memory: RulebookMemory,
    route,
    delta_t_s: float,
    post_front_s: float,
    prediction_horizon_s: float,
    minimum_history_samples: int,
    ego_association: LaneAssociation | None = None,
    post_state_histories: tuple = (),
) -> tuple[dict, CacheDelta]:
    from thesis_rl.rulebook.v2.geometry.conflict_zones import (
        build_crosswalk_conflict_zone_candidates,
        attach_route_intervals,
    )

    cleared_illegal_entries = _cleared_crosswalk_illegal_entries(
        cache=cache, memory=memory, ego_footprint=post.ego.footprint
    )
    crosswalks = tuple(
        feature
        for feature in cache.map_feature_catalog.values()
        if feature.feature_class is MapFeatureClass.CROSSWALK
    )
    association = ego_association
    if association is None:
        association = associate_route_lane(
            position_xy=post.ego.position_xy,
            position_z=post.ego.position_z,
            heading_rad=post.ego.heading_rad,
            route_lanes=cache.route_lanes,
        )
    lane = None if association is None else _route_lane(cache, association.lane_id)
    if not crosswalks or lane is None:
        return {
            "zone_id": "__no_crosswalk__",
            "ego_interval": _empty_interval(),
            "vru_intervals": (),
            "distance_to_entry_m": 0.0,
            "approach_speed_mps": 0.0,
            "delta_t_s": delta_t_s,
            "ego_occupied": False,
            "ego_entered": False,
            "preexisting_zone_ids": memory.preexisting_ego_occupancy_zone_ids,
            "previous_illegal_entries": cleared_illegal_entries,
            "vertical_applicable": False,
        }, CacheDelta()
    elevation = PolylineElevation(lane.centerline.points_xyz)
    movement_key = derive_lane_movement_key(lane, assigned_route_lane_ids=cache.task_route.lane_ids)
    if movement_key is None:
        return {
            "zone_id": "__ambiguous_crosswalk__",
            "ego_interval": _empty_interval(),
            "vru_intervals": (),
            "distance_to_entry_m": 0.0,
            "approach_speed_mps": 0.0,
            "delta_t_s": delta_t_s,
            "ego_occupied": False,
            "ego_entered": False,
            "preexisting_zone_ids": memory.preexisting_ego_occupancy_zone_ids,
            "previous_illegal_entries": cleared_illegal_entries,
            "vertical_applicable": False,
        }, CacheDelta()
    corridor = MovementCorridor(
        movement_key=movement_key,
        polygon=lane.polygon_xy,
        elevation_at_xy=elevation,
    )
    candidates = []
    for feature in crosswalks:
        if feature.elevation_m is None:
            continue
        candidates.extend(
            attach_route_intervals(
                route=route,
                candidates=build_crosswalk_conflict_zone_candidates(
                    scenario_id=cache.scenario_id,
                    ego_corridor=corridor,
                    crosswalk_id=feature.feature_id,
                    crosswalk_polygon=feature.geometry,
                    crosswalk_elevation_at_xy=lambda _x, _y, z=float(feature.elevation_m): z,
                ),
            )
        )
    if not candidates:
        return {
            "zone_id": "__no_crosswalk__",
            "ego_interval": _empty_interval(),
            "vru_intervals": (),
            "distance_to_entry_m": 0.0,
            "approach_speed_mps": 0.0,
            "delta_t_s": delta_t_s,
            "ego_occupied": False,
            "ego_entered": False,
            "preexisting_zone_ids": memory.preexisting_ego_occupancy_zone_ids,
            "previous_illegal_entries": cleared_illegal_entries,
            "vertical_applicable": False,
        }, CacheDelta()
    selected = select_first_ahead_or_occupied_zone(
        candidates=tuple(candidates),
        ego_footprint=post.ego.footprint,
        ego_front_s_m=post_front_s,
    )
    if selected is None:
        return {
            "zone_id": "__no_relevant_crosswalk__",
            "ego_interval": _empty_interval(),
            "vru_intervals": (),
            "distance_to_entry_m": 0.0,
            "approach_speed_mps": 0.0,
            "delta_t_s": delta_t_s,
            "ego_occupied": False,
            "ego_entered": False,
            "preexisting_zone_ids": memory.preexisting_ego_occupancy_zone_ids,
            "previous_illegal_entries": cleared_illegal_entries,
            "vertical_applicable": False,
        }, CacheDelta()
    zone = selected.candidate.polygon
    ego_interval, vru_intervals, _ = predict_conflict_zone_occupancy_intervals(
        ego=post.ego,
        actors=tuple(
            actor
            for actor in post.actors
            if actor.actor_class in {ActorClass.PEDESTRIAN, ActorClass.CYCLIST}
            and abs(actor.position_z - post.ego.position_z) <= VERTICAL_COMPATIBILITY_TOLERANCE_M
        ),
        # REQ-EF-12: a post-state prediction must see the post-state sample.
        # ``memory.actor_motion_histories`` is only updated by the monitor
        # after the components have run, so it lagged one step behind.
        histories=post_state_histories or memory.actor_motion_histories,
        sim_time_s=post.sim_time_s,
        zone=zone,
        horizon_s=prediction_horizon_s,
        minimum_history_samples=minimum_history_samples,
    )
    record = ConflictZoneRecord(
        zone_id=selected.candidate.zone_id,
        polygon=zone,
        ego_movement_key=selected.candidate.ego_movement_key,
        other_movement_key=None,
        route_entry_s_m=selected.route_entry_s_m,
        route_exit_s_m=selected.route_exit_s_m,
        elevation_m=post.ego.position_z,
        component_index=selected.candidate.component_index,
    )
    cache_delta = CacheDelta(new_conflict_zones=(record,))
    if ego_interval is None:
        return {
            "zone_id": selected.candidate.zone_id,
            "ego_interval": _empty_interval(),
            "vru_intervals": vru_intervals,
            "distance_to_entry_m": max(0.0, selected.route_entry_s_m - post_front_s),
            "approach_speed_mps": max(
                0.0,
                post.ego.velocity_xy[0] * association.tangent_xy[0]
                + post.ego.velocity_xy[1] * association.tangent_xy[1],
            ),
            "delta_t_s": delta_t_s,
            "ego_occupied": False,
            "ego_entered": False,
            "preexisting_zone_ids": memory.preexisting_ego_occupancy_zone_ids,
            "previous_illegal_entries": cleared_illegal_entries,
            "vertical_applicable": False,
        }, cache_delta
    return {
        "zone_id": selected.candidate.zone_id,
        "ego_interval": ego_interval,
        "vru_intervals": vru_intervals,
        "distance_to_entry_m": max(0.0, selected.route_entry_s_m - post_front_s),
        "approach_speed_mps": max(
            0.0,
            post.ego.velocity_xy[0] * association.tangent_xy[0]
            + post.ego.velocity_xy[1] * association.tangent_xy[1],
        ),
        "delta_t_s": delta_t_s,
        "ego_occupied": bool(post.ego.footprint.intersects(zone)),
        # REQ-VY-02-equivalent: the entry event uses the swept front bumper
        # pre->post, not the plain post-state footprint, so a fast crossing
        # that only overlaps the zone mid-step is still detected.
        "ego_entered": bool(
            not pre.ego.footprint.intersects(zone)
            and swept_front_bumper(
                pre.ego.footprint,
                post.ego.footprint,
                pre_heading_rad=pre.ego.heading_rad,
                post_heading_rad=post.ego.heading_rad,
            ).intersects(zone)
        ),
        "preexisting_zone_ids": memory.preexisting_ego_occupancy_zone_ids,
        "previous_illegal_entries": cleared_illegal_entries,
        "vertical_applicable": True,
    }, cache_delta


def _approach_control_for_movement(cache: EpisodeCache, movement_key) -> ApproachControl:
    """Return an explicit control for one unambiguous movement.

    Absence of a control record means ``NONE``; an adapter must expose an
    unknown source state as ``UNKNOWN`` rather than silently omitting it.
    Signal operationally suppresses STOP for the same movement.
    """

    controls = tuple(
        control for control in cache.traffic_control_catalog if control.movement_key == movement_key
    )
    if any(control.control_type is ApproachControl.UNKNOWN for control in controls):
        return ApproachControl.UNKNOWN
    if any(control.control_type is ApproachControl.SIGNAL for control in controls):
        return ApproachControl.SIGNAL
    if any(control.control_type is ApproachControl.STOP for control in controls):
        return ApproachControl.STOP
    return ApproachControl.NONE


def _cleared_vehicle_yield_illegal_entries(
    *, cache: EpisodeCache, memory: RulebookMemory, ego_footprint: BaseGeometry
) -> frozenset[tuple[str, str]]:
    """Drop illegal-entry latches for zones the ego footprint has already left.

    Section 2.8.2 selection can exclude a zone from "ahead" candidacy
    (epsilon-scoped) long before the ego footprint stops intersecting it
    (vehicle-length scoped), so the currently selected zone_id is not a
    reliable signal for releasing latches on zones ego has fully exited
    (ADR-025).
    """
    active = set(memory.vehicle_yield_illegal_entries)
    for actor_id, zone_id in memory.vehicle_yield_illegal_entries:
        record = cache.conflict_zones.get(zone_id)
        if record is not None and not ego_footprint.intersects(record.polygon):
            active.discard((actor_id, zone_id))
    return frozenset(active)


def _pre_state_priority_and_gap(
    *,
    pre: EnvSnapshot,
    zone,
    other_movement_key,
    ego_key,
    ego_lane: RouteLaneRecord,
    ego_control: ApproachControl,
    priority_records: dict[tuple[object, object], MovementPriority],
    cache: EpisodeCache,
    memory: RulebookMemory,
    actor_keys: dict[str, object],
    actor_lanes: dict[str, RouteLaneRecord],
    prediction_horizon_s: float,
    minimum_history_samples: int,
) -> tuple[frozenset[str], float]:
    """DEC-005 Fase A: re-evaluate the four scoped priority predicates for the
    already-selected zone against ``pre_state`` actor footprints, and derive
    the pre-state worst-case temporal gap r_gap^- that gates illegal-entry
    latch creation (spec Section 7.9). Movement-key identity (``actor_keys``,
    ``actor_lanes``) and the static/topological predicates (STOP-vs-NONE,
    pairwise, roundabout) are frozen-for-the-transition context, unchanged
    from the post-state pass; only the dynamic ``occupied`` predicate and the
    occupancy-interval prediction read pre-state data here.
    """

    pre_by_id = {actor.actor_id: actor for actor in pre.actors}
    prioritized_ids: set[str] = set()
    for actor_id, movement_key in actor_keys.items():
        if movement_key != other_movement_key:
            continue
        pre_actor = pre_by_id.get(actor_id)
        occupied = (
            pre_actor is not None
            and pre_actor.footprint.intersects(zone)
            and abs(pre_actor.position_z - pre.ego.position_z) <= VERTICAL_COMPATIBILITY_TOLERANCE_M
        )
        other_control = _approach_control_for_movement(cache, movement_key)
        stop_priority = (
            ego_control is ApproachControl.STOP and other_control is ApproachControl.NONE
        )
        pairwise_priority = (
            priority_records.get((ego_key, movement_key)) is MovementPriority.OTHER_HAS_PRIORITY
        )
        other_lane = actor_lanes.get(actor_id)
        roundabout_priority = other_lane is not None and any(
            record.entry_lane_id == ego_lane.lane_id
            and record.circulating_lane_id == other_lane.lane_id
            for record in cache.roundabout_priority_records
        )
        if occupied or stop_priority or pairwise_priority or roundabout_priority:
            prioritized_ids.add(actor_id)
    if not prioritized_ids:
        return frozenset(), 0.0
    pre_ego_interval, pre_intervals, _ = predict_conflict_zone_occupancy_intervals(
        ego=pre.ego,
        actors=tuple(actor for actor_id, actor in pre_by_id.items() if actor_id in prioritized_ids),
        histories=memory.actor_motion_histories,
        sim_time_s=pre.sim_time_s,
        zone=zone,
        horizon_s=prediction_horizon_s,
        minimum_history_samples=minimum_history_samples,
    )
    if pre_ego_interval is None:
        return frozenset(prioritized_ids), 0.0
    gap = worst_case_temporal_gap_violation(
        ego_interval=pre_ego_interval,
        other_intervals=pre_intervals,
        gap_scale_s=CROSSWALK_GAP_S,
    )
    return frozenset(prioritized_ids), gap


def _vehicle_yield_inputs(
    *,
    pre: EnvSnapshot,
    post: EnvSnapshot,
    cache: EpisodeCache,
    memory: RulebookMemory,
    route: RoutePolyline,
    delta_t_s: float,
    post_front_s: float,
    approach_speed_mps: float,
    prediction_horizon_s: float,
    minimum_history_samples: int,
    ego_brake_mps2: float | None,
    ego_association: LaneAssociation | None = None,
    actor_associations: dict[str, LaneAssociation | None] | None = None,
    diagnostic_timing_seconds: dict[str, float] | None = None,
    post_state_histories: tuple = (),
) -> tuple[dict[str, object], CacheDelta]:
    """Construct live vehicle-yield inputs without inferring priority from geometry."""

    cleared_illegal_entries = _cleared_vehicle_yield_illegal_entries(
        cache=cache, memory=memory, ego_footprint=post.ego.footprint
    )
    association = ego_association
    if association is None:
        association = associate_route_lane(
            position_xy=post.ego.position_xy,
            position_z=post.ego.position_z,
            heading_rad=post.ego.heading_rad,
            route_lanes=cache.route_lanes,
        )
    ego_lane = None if association is None else _route_lane(cache, association.lane_id)
    ego_key = (
        None
        if ego_lane is None
        else derive_lane_movement_key(ego_lane, assigned_route_lane_ids=cache.task_route.lane_ids)
    )
    empty = {
        "zone_id": "__no_vehicle_priority__",
        "ego_interval": _empty_interval(),
        "prioritized_intervals": (),
        "distance_to_entry_m": 0.0,
        "approach_speed_mps": approach_speed_mps,
        "delta_t_s": delta_t_s,
        "ego_occupied": False,
        "entered_actor_ids": frozenset(),
        "previous_illegal_entries": cleared_illegal_entries,
        "actor_movement_keys": (),
        "previous_frozen_movement_keys": memory.frozen_actor_movement_keys,
        "exited_actor_ids": frozenset(),
        "ego_brake_mps2": ego_brake_mps2,
    }
    # The component requires the shared calibrated braking value.  Keeping the
    # scoped domain empty when the runtime has no calibration preserves the
    # independent evaluability of the other Rulebook components.
    if ego_brake_mps2 is None or ego_brake_mps2 <= 0.0:
        return empty, CacheDelta()
    if ego_lane is None or ego_key is None:
        return empty, CacheDelta()
    ego_corridor = MovementCorridor(
        movement_key=ego_key,
        polygon=ego_lane.polygon_xy,
        elevation_at_xy=PolylineElevation(ego_lane.centerline.points_xyz),
    )
    candidates = []
    new_pair_records: list[VehicleConflictPairRecord] = []
    pending_pair_records: dict[tuple[object, object], VehicleConflictPairRecord] = {}
    actor_lanes: dict[str, RouteLaneRecord] = {}
    actor_keys: dict[str, object] = {}
    seen_pair_keys: set[tuple[object, object]] = set()
    actor_scan_started = time.perf_counter()
    pair_geometry_seconds = 0.0
    for actor in post.actors:
        if actor.actor_class is not ActorClass.VEHICLE:
            continue
        actor_association = (
            actor_associations.get(actor.actor_id)
            if actor_associations is not None
            else associate_route_lane(
                position_xy=actor.position_xy,
                position_z=actor.position_z,
                heading_rad=actor.heading_rad,
                route_lanes=cache.route_lanes,
            )
        )
        lane = None if actor_association is None else _route_lane(cache, actor_association.lane_id)
        movement_key = None if lane is None else derive_lane_movement_key(lane)
        if lane is None or movement_key is None:
            continue
        actor_lanes[actor.actor_id] = lane
        actor_keys[actor.actor_id] = movement_key
        pair_key = (ego_key, movement_key)
        if pair_key in seen_pair_keys:
            continue
        seen_pair_keys.add(pair_key)
        cached_pair = cache.vehicle_conflict_pairs.get(pair_key) or pending_pair_records.get(
            pair_key
        )
        if cached_pair is not None:
            candidates.extend(
                RouteConflictZoneCandidate(
                    candidate=ConflictZoneCandidate(
                        zone_id=record.zone_id,
                        component_index=record.component_index,
                        polygon=record.polygon,
                        ego_movement_key=record.ego_movement_key,
                        other_movement_key=record.other_movement_key,
                    ),
                    route_entry_s_m=record.route_entry_s_m,
                    route_exit_s_m=record.route_exit_s_m,
                )
                for record in cached_pair.candidates
            )
            continue
        other_corridor = MovementCorridor(
            movement_key=movement_key,
            polygon=lane.polygon_xy,
            elevation_at_xy=PolylineElevation(lane.centerline.points_xyz),
        )
        pair_geometry_started = time.perf_counter()
        pair_candidates = attach_route_intervals(
            route=route,
            candidates=build_vehicle_conflict_zone_candidates(
                scenario_id=cache.scenario_id,
                ego_corridor=ego_corridor,
                other_corridor=other_corridor,
            ),
        )
        pair_geometry_seconds += time.perf_counter() - pair_geometry_started
        candidates.extend(pair_candidates)
        pair_record = VehicleConflictPairRecord(
            ego_movement_key=ego_key,
            other_movement_key=movement_key,
            candidates=tuple(
                ConflictZoneRecord(
                    zone_id=item.candidate.zone_id,
                    polygon=item.candidate.polygon,
                    ego_movement_key=item.candidate.ego_movement_key,
                    other_movement_key=item.candidate.other_movement_key,
                    route_entry_s_m=item.route_entry_s_m,
                    route_exit_s_m=item.route_exit_s_m,
                    elevation_m=post.ego.position_z,
                    component_index=item.candidate.component_index,
                )
                for item in pair_candidates
            ),
        )
        pending_pair_records[pair_key] = pair_record
        new_pair_records.append(pair_record)
    if diagnostic_timing_seconds is not None:
        diagnostic_timing_seconds["vehicle_yield_actor_scan"] = (
            time.perf_counter() - actor_scan_started
        )
        diagnostic_timing_seconds["vehicle_yield_pair_geometry"] = pair_geometry_seconds
    selection_started = time.perf_counter()
    selected = select_first_ahead_or_occupied_zone(
        candidates=tuple(candidates), ego_footprint=post.ego.footprint, ego_front_s_m=post_front_s
    )
    if diagnostic_timing_seconds is not None:
        diagnostic_timing_seconds["vehicle_yield_zone_selection"] = (
            time.perf_counter() - selection_started
        )
    if selected is None:
        return empty, CacheDelta(new_vehicle_conflict_pairs=tuple(new_pair_records))
    candidate = selected.candidate
    zone = candidate.polygon
    ego_control = _approach_control_for_movement(cache, ego_key)
    priority_records = {
        (record.ego_movement_key, record.other_movement_key): record.relation
        for record in cache.movement_priority_records
    }
    priority_started = time.perf_counter()
    prioritized_ids: set[str] = set()
    for actor in post.actors:
        movement_key = actor_keys.get(actor.actor_id)
        if movement_key != candidate.other_movement_key:
            continue
        occupied = (
            actor.footprint.intersects(zone)
            and abs(actor.position_z - post.ego.position_z) <= VERTICAL_COMPATIBILITY_TOLERANCE_M
        )
        other_control = _approach_control_for_movement(cache, movement_key)
        stop_priority = (
            ego_control is ApproachControl.STOP and other_control is ApproachControl.NONE
        )
        pairwise_priority = (
            priority_records.get((ego_key, movement_key)) is MovementPriority.OTHER_HAS_PRIORITY
        )
        other_lane = actor_lanes[actor.actor_id]
        roundabout_priority = any(
            record.entry_lane_id == ego_lane.lane_id
            and record.circulating_lane_id == other_lane.lane_id
            for record in cache.roundabout_priority_records
        )
        if occupied or stop_priority or pairwise_priority or roundabout_priority:
            prioritized_ids.add(actor.actor_id)
    if diagnostic_timing_seconds is not None:
        diagnostic_timing_seconds["vehicle_yield_priority_selection"] = (
            time.perf_counter() - priority_started
        )
    occupancy_started = time.perf_counter()
    ego_interval, intervals, _ = predict_conflict_zone_occupancy_intervals(
        ego=post.ego,
        actors=tuple(actor for actor in post.actors if actor.actor_id in prioritized_ids),
        # REQ-EF-12: post-state prediction uses the post-state history.  The
        # pre-state pass in ``_pre_state_priority_and_gap`` deliberately keeps
        # ``memory.actor_motion_histories``: feeding it the post sample would
        # be future information relative to the snapshot it evaluates.
        histories=post_state_histories or memory.actor_motion_histories,
        sim_time_s=post.sim_time_s,
        zone=zone,
        horizon_s=prediction_horizon_s,
        minimum_history_samples=minimum_history_samples,
    )
    if diagnostic_timing_seconds is not None:
        diagnostic_timing_seconds["vehicle_yield_occupancy_prediction"] = (
            time.perf_counter() - occupancy_started
        )
    if ego_interval is None:
        return empty, CacheDelta(new_vehicle_conflict_pairs=tuple(new_pair_records))
    pre_by_id = {actor.actor_id: actor for actor in pre.actors}
    post_by_id = {actor.actor_id: actor for actor in post.actors}
    ego_occupied = post.ego.footprint.intersects(zone)
    # DEC-005 / REQ-VY-02: the entry event uses the swept front bumper
    # pre->post, not the plain post-state footprint, so a fast crossing
    # that only overlaps the zone mid-step is still detected.
    entry_swept_bumper = swept_front_bumper(
        pre.ego.footprint,
        post.ego.footprint,
        pre_heading_rad=pre.ego.heading_rad,
        post_heading_rad=post.ego.heading_rad,
    )
    ego_entered = not pre.ego.footprint.intersects(zone) and entry_swept_bumper.intersects(zone)
    pre_prioritized_ids: frozenset[str] = frozenset()
    pre_gap_violation = 0.0
    if ego_entered:
        # DEC-005 Fase A/B: judge entry legality from the pre-state view of
        # the priority actors, independent of whether they are still
        # present/prioritized by the time the post-state is observed
        # (REQ-VY-01).
        pre_prioritized_ids, pre_gap_violation = _pre_state_priority_and_gap(
            pre=pre,
            zone=zone,
            other_movement_key=candidate.other_movement_key,
            ego_key=ego_key,
            ego_lane=ego_lane,
            ego_control=ego_control,
            priority_records=priority_records,
            cache=cache,
            memory=memory,
            actor_keys=actor_keys,
            actor_lanes=actor_lanes,
            prediction_horizon_s=prediction_horizon_s,
            minimum_history_samples=minimum_history_samples,
        )
    exited = frozenset(
        actor_id
        for actor_id in prioritized_ids | pre_prioritized_ids
        if actor_id in pre_by_id
        and pre_by_id[actor_id].footprint.intersects(zone)
        and not (
            post_by_id.get(actor_id) is not None and post_by_id[actor_id].footprint.intersects(zone)
        )
    )
    record = ConflictZoneRecord(
        zone_id=candidate.zone_id,
        polygon=zone,
        ego_movement_key=candidate.ego_movement_key,
        other_movement_key=candidate.other_movement_key,
        route_entry_s_m=selected.route_entry_s_m,
        route_exit_s_m=selected.route_exit_s_m,
        elevation_m=post.ego.position_z,
        component_index=candidate.component_index,
    )
    return (
        {
            "zone_id": candidate.zone_id,
            "ego_interval": ego_interval,
            "prioritized_intervals": tuple(
                item for item in intervals if item[0] in prioritized_ids
            ),
            "distance_to_entry_m": max(0.0, selected.route_entry_s_m - post_front_s),
            "approach_speed_mps": approach_speed_mps,
            "delta_t_s": delta_t_s,
            "ego_occupied": ego_occupied,
            # DEC-005: the entry-event actor set is pre-state only, supplied
            # via ``pre_state_entered_actor_ids`` below; ``entered_actor_ids``
            # is left empty here so the two-set consistency guard in
            # ``evaluate_vehicle_yield`` never sees a spurious disagreement
            # between the (unused) post-state view and the pre-state one.
            "entered_actor_ids": frozenset(),
            "pre_state_entered_actor_ids": pre_prioritized_ids if ego_entered else frozenset(),
            "pre_state_gap_violation": pre_gap_violation if ego_entered else None,
            # ADR-025: cleared_illegal_entries already drops latches for zones
            # the ego footprint has fully left, independent of this step's
            # zone selection.
            "previous_illegal_entries": cleared_illegal_entries,
            "preexisting": candidate.zone_id in memory.preexisting_ego_occupancy_zone_ids,
            "actor_movement_keys": tuple(sorted(actor_keys.items())),
            "previous_frozen_movement_keys": memory.frozen_actor_movement_keys,
            "exited_actor_ids": exited,
            "ego_brake_mps2": ego_brake_mps2,
        },
        CacheDelta(
            new_conflict_zones=(record,),
            new_vehicle_conflict_pairs=tuple(new_pair_records),
        ),
    )


def evaluate_transition(
    *,
    pre_state: EnvSnapshot,
    post_state: EnvSnapshot,
    memory: RulebookMemory,
    cache: EpisodeCache,
    config: RulebookTransitionConfig,
):
    """Evaluate one complete pre/post transition through the fixed registry."""

    if pre_state.scenario_id != cache.scenario_id or post_state.scenario_id != cache.scenario_id:
        raise ValueError("Snapshot and episode cache scenario IDs must agree")
    if cache.route_polyline is None:
        raise ValueError("Episode cache has no assigned route polyline")
    route = cache.route_polyline
    delta_t_s = _delta_t(pre_state, post_state)
    pre_front_s = _front_s(pre_state, route)
    post_front_s = _front_s(post_state, route)
    post_route_tangent_xy = route.project(
        post_state.ego.position_xy,
        position_z=post_state.ego.position_z,
        previous_s_m=None,
    ).tangent_xy
    post_approach_speed_mps = max(
        0.0,
        post_state.ego.velocity_xy[0] * post_route_tangent_xy[0]
        + post_state.ego.velocity_xy[1] * post_route_tangent_xy[1],
    )
    pre_by_id = {actor.actor_id: actor for actor in pre_state.actors}
    actors = tuple(post_state.actors)
    post_ego_association, post_actor_associations = _snapshot_lane_associations(
        post_state, cache.route_lanes
    )
    pre_ego_association, pre_actor_associations = _snapshot_lane_associations(
        pre_state, cache.route_lanes
    )

    reachable_control_lane_ids = route_reachable_control_lane_ids(cache)
    signal_pre = _selected_control(
        cache.traffic_control_catalog,
        ApproachControl.SIGNAL,
        pre_front_s,
        memory.resolved_signal_group_ids,
        route_lane_ids=reachable_control_lane_ids,
    )
    signal_post = _selected_control(
        cache.traffic_control_catalog,
        ApproachControl.SIGNAL,
        post_front_s,
        memory.resolved_signal_group_ids,
        route_lane_ids=reachable_control_lane_ids,
    )
    signal = signal_pre or signal_post
    stop_pre = _selected_control(
        cache.traffic_control_catalog,
        ApproachControl.STOP,
        pre_front_s,
        memory.resolved_stop_group_ids,
        route_lane_ids=reachable_control_lane_ids,
    )
    stop_post = _selected_control(
        cache.traffic_control_catalog,
        ApproachControl.STOP,
        post_front_s,
        memory.resolved_stop_group_ids,
        route_lane_ids=reachable_control_lane_ids,
    )
    stop = stop_pre or stop_post
    if signal is not None and stop is not None and stop.movement_key == signal.movement_key:
        stop = None
    calibrated_brake_mps2 = (
        None if config.rss_calibration is None else config.rss_calibration.ego_min_brake_mps2
    )
    signal_distances = (
        (None, None, False)
        if signal is None
        else _control_distances(
            signal,
            pre_front_s=pre_front_s,
            post_front_s=post_front_s,
            pre_footprint=pre_state.ego.footprint,
            post_footprint=post_state.ego.footprint,
            pre_heading_rad=pre_state.ego.heading_rad,
            post_heading_rad=post_state.ego.heading_rad,
        )
    )
    stop_distances = (
        (None, None, False)
        if stop is None
        else _control_distances(
            stop,
            pre_front_s=pre_front_s,
            post_front_s=post_front_s,
            pre_footprint=pre_state.ego.footprint,
            post_footprint=post_state.ego.footprint,
            pre_heading_rad=pre_state.ego.heading_rad,
            post_heading_rad=post_state.ego.heading_rad,
        )
    )
    signal_input = {
        "control": signal,
        "pre_state": None
        if signal is None
        else signal_group_state(
            control=signal, signal_states_by_physical_id=pre_state.signal_states_by_physical_id
        ),
        "post_state": None
        if signal is None
        else signal_group_state(
            control=signal, signal_states_by_physical_id=post_state.signal_states_by_physical_id
        ),
        "pre_delta_m": signal_distances[0],
        "post_delta_m": signal_distances[1],
        "speed_mps": post_approach_speed_mps,
        "route_tangent_xy": post_route_tangent_xy,
        "delta_t_s": delta_t_s,
        "previous_yellow_must_stop": memory.yellow_must_stop,
        "previous_signal_delta_m": memory.previous_signal_delta_m,
        "previous_group_id": memory.active_signal_group_id,
        "resolved_group_ids": memory.resolved_signal_group_ids,
        "crossing": signal_distances[2],
        "ego_brake_mps2": calibrated_brake_mps2,
    }
    stop_input = {
        "control": stop,
        "pre_delta_m": stop_distances[0],
        "post_delta_m": stop_distances[1],
        "speed_mps": post_approach_speed_mps,
        "previous_continuous_s": memory.stop_continuous_timer_s,
        "previous_best_s": memory.stop_best_timer_s,
        "delta_t_s": delta_t_s,
        "previous_group_id": memory.active_stop_group_id,
        "resolved_group_ids": memory.resolved_stop_group_ids,
        "crossing": stop_distances[2],
    }
    # REQ-EF-12: build the causal motion-history preview once, before the
    # components run.  ``monitor.py`` remains the single writer of the
    # committed delta; this is a read-only view for post-state predictions.
    post_state_histories, _ = build_motion_history_preview(
        memory=memory,
        post_state=post_state,
        history_window_s=config.history_window_s,
    )
    phase_started = time.perf_counter()
    crosswalk_input, crosswalk_cache_delta = _crosswalk_inputs(
        pre=pre_state,
        post=post_state,
        cache=cache,
        memory=memory,
        route=route,
        delta_t_s=delta_t_s,
        post_front_s=post_front_s,
        prediction_horizon_s=config.prediction_horizon_s,
        minimum_history_samples=config.minimum_history_samples,
        ego_association=post_ego_association,
        post_state_histories=post_state_histories,
    )
    crosswalk_seconds = time.perf_counter() - phase_started
    crosswalk_input["ego_brake_mps2"] = calibrated_brake_mps2
    solid_boundaries = tuple(
        feature
        for feature in cache.map_feature_catalog.values()
        if feature.feature_class is MapFeatureClass.LANE_MARKING_SOLID
        and feature.elevation_m is not None
        and abs(feature.elevation_m - post_state.ego.position_z)
        <= VERTICAL_COMPATIBILITY_TOLERANCE_M
    )
    dashed_boundaries = tuple(
        feature
        for feature in cache.map_feature_catalog.values()
        if feature.feature_class is MapFeatureClass.LANE_MARKING_DASHED
        and feature.elevation_m is not None
        and abs(feature.elevation_m - post_state.ego.position_z)
        <= VERTICAL_COMPATIBILITY_TOLERANCE_M
    )
    phase_started = time.perf_counter()
    drivable = drivable_surface_for_ego(
        ego_footprint=post_state.ego.footprint,
        ego_position_xy=post_state.ego.position_xy,
        ego_position_z=post_state.ego.position_z,
        lanes=tuple(
            DrivableLaneRecord(lane.lane_id, lane.centerline, lane.polygon_xy, None)
            for lane in cache.route_lanes
        ),
    )
    drivable_seconds = time.perf_counter() - phase_started
    carriageway_surfaces = carriageway_surfaces_for_ego(
        ego_position_xy=post_state.ego.position_xy,
        ego_position_z=post_state.ego.position_z,
        route_tangent_xy=post_route_tangent_xy,
        lanes=tuple(
            DrivableLaneRecord(lane.lane_id, lane.centerline, lane.polygon_xy, None)
            for lane in cache.route_lanes
        ),
    )
    vehicle_input = None
    vehicle_cache_delta = CacheDelta()
    vehicle_yield_seconds = 0.0
    vehicle_yield_diagnostic_timing: dict[str, float] = {}
    if not config.disable_vehicle_yield_for_benchmark:
        phase_started = time.perf_counter()
        vehicle_input, vehicle_cache_delta = _vehicle_yield_inputs(
            pre=pre_state,
            post=post_state,
            cache=cache,
            memory=memory,
            route=route,
            delta_t_s=delta_t_s,
            post_front_s=post_front_s,
            approach_speed_mps=post_approach_speed_mps,
            prediction_horizon_s=config.prediction_horizon_s,
            minimum_history_samples=config.minimum_history_samples,
            ego_brake_mps2=calibrated_brake_mps2,
            ego_association=post_ego_association,
            actor_associations=post_actor_associations,
            diagnostic_timing_seconds=vehicle_yield_diagnostic_timing,
            post_state_histories=post_state_histories,
        )
        vehicle_yield_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    rss_candidates = _rss_candidates(
        ego=pre_state.ego,
        actors=pre_state.actors,
        route=route,
        route_lanes=cache.route_lanes,
        ego_association=pre_ego_association,
        actor_associations=pre_actor_associations,
    )
    rss_candidates_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    rss_lateral_candidates = _rss_lateral_candidates(
        ego=pre_state.ego,
        actors=pre_state.actors,
        route=route,
        route_lanes=cache.route_lanes,
        ego_brake_mps2=calibrated_brake_mps2,
        ego_association=pre_ego_association,
        actor_associations=pre_actor_associations,
    )
    rss_lateral_candidates_seconds = time.perf_counter() - phase_started
    component_inputs = {
        "collision": {
            "scenario_id": post_state.scenario_id,
            "step_index": post_state.step_index,
            "ego_configured_speed_cap_mps": pre_state.ego.configured_speed_cap_mps,
            "pre_ego": pre_state.ego,
            "pre_actors_by_id": pre_by_id,
            "onset_records": post_state.contact_onset_records,
            "previous_contact_ids": memory.previous_contact_ids,
            "post_active_contact_ids": post_state.active_contact_ids,
            # REQ-EF-13: lets the component tell "appeared this step" (not the
            # ego's fault) from "never observed by the snapshot pipeline"
            # (instrumentation gap).
            "post_actor_ids": frozenset(actor.actor_id for actor in post_state.actors),
        },
        "rss": {
            "scenario_id": post_state.scenario_id,
            "step_index": post_state.step_index,
            "candidates": rss_candidates,
            "calibration": config.rss_calibration,
            "expected_config_hash": config.expected_config_hash,
        },
        "ttc": {
            "ego_footprint": pre_state.ego.footprint,
            "ego_velocity_xy": pre_state.ego.velocity_xy,
            "actors": pre_state.actors,
            "vertically_compatible_actor_ids": _vertical_actor_ids(pre_state.ego, pre_state.actors),
        },
        "clearance": {
            "ego_footprint": post_state.ego.footprint,
            "actors": actors,
            "vertically_compatible_actor_ids": _vertical_actor_ids(post_state.ego, actors),
        },
        "rss_lateral": {
            "candidates": rss_lateral_candidates,
        },
        "offroad": {"ego_footprint": post_state.ego.footprint, "drivable_surface": drivable},
        "wrong_carriageway": {
            "ego_footprint": post_state.ego.footprint,
            "aligned_surface": carriageway_surfaces.aligned,
            "opposing_surface": carriageway_surfaces.opposing,
        },
        "wrong_way": {
            "ego": post_state.ego,
            "route": route,
            "previous_s_m": None,
        },
        "solid_line": {
            "ego_footprint": post_state.ego.footprint,
            "solid_boundaries": solid_boundaries,
            "swept_front_bumper": swept_front_bumper(
                pre_state.ego.footprint,
                post_state.ego.footprint,
                pre_heading_rad=pre_state.ego.heading_rad,
                post_heading_rad=post_state.ego.heading_rad,
            ),
        },
        "dashed_line": {
            "ego_footprint": post_state.ego.footprint,
            "dashed_boundaries": dashed_boundaries,
            "previous_boundary_id": memory.active_dashed_boundary_id,
            "previous_timer_s": memory.dashed_line_timer_s,
            "delta_t_s": delta_t_s,
        },
        "signal": signal_input,
        "stop": stop_input,
        "crosswalk": crosswalk_input,
        "progress": {
            "pre_mission": pre_state.mission_snapshot,
            "post_mission": post_state.mission_snapshot,
            "delta_t_s": delta_t_s,
        },
    }
    excluded_components = frozenset()
    if config.disable_vehicle_yield_for_benchmark:
        excluded_components = frozenset({"vehicle_yield"})
    else:
        component_inputs["vehicle_yield"] = vehicle_input
    combined_cache_delta = replace(
        vehicle_cache_delta,
        new_conflict_zones=vehicle_cache_delta.new_conflict_zones
        + crosswalk_cache_delta.new_conflict_zones,
    )
    phase_started = time.perf_counter()
    result, next_memory, cache_delta = evaluate_registered_transition(
        memory=memory,
        component_inputs=component_inputs,
        raw_progress_m=0.0,
        progress_margin=0.0,
        post_state=post_state,
        history_window_s=config.history_window_s,
        pending_cache_delta=combined_cache_delta,
        excluded_normative_components=excluded_components,
    )
    registry_seconds = time.perf_counter() - phase_started
    return (
        result,
        next_memory,
        replace(
            cache_delta,
            diagnostic_timing_seconds={
                "crosswalk": crosswalk_seconds,
                "drivable": drivable_seconds,
                "vehicle_yield": vehicle_yield_seconds,
                **vehicle_yield_diagnostic_timing,
                "rss_candidates": rss_candidates_seconds,
                "rss_lateral_candidates": rss_lateral_candidates_seconds,
                "registry": registry_seconds,
            },
        ),
    )


def transition_evaluator_factory(config: RulebookTransitionConfig):
    """Return the callable shape consumed by ``RulebookV2MonitorWrapper``."""

    def evaluate(*, pre_state, post_state, memory, cache):
        return evaluate_transition(
            pre_state=pre_state,
            post_state=post_state,
            memory=memory,
            cache=cache,
            config=config,
        )

    return evaluate
