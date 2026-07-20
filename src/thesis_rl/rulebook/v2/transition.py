"""Source-neutral composition of one causal Rulebook v2 transition.

The live MetaDrive adapter is responsible only for constructing ``EnvSnapshot``
objects.  This module derives all evaluator inputs from those snapshots and the
immutable episode cache, then invokes the fixed registry exactly once.  Missing
topology or calibration is represented by an explicit NOT_APPLICABLE domain or
by a fail-fast validation error; no source-specific fallback is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, isfinite, sin

from thesis_rl.rulebook.v2.components.controls import signal_group_state
from thesis_rl.rulebook.v2.components.rss import RSSCalibrationArtifact, RSSCandidate
from thesis_rl.rulebook.v2.geometry.conflict_zones import (
    MovementCorridor,
    select_first_ahead_or_occupied_zone,
)
from thesis_rl.rulebook.v2.geometry.continuous_sat import OccupancyInterval
from thesis_rl.rulebook.v2.geometry.ctrv import predict_conflict_zone_occupancy_intervals
from thesis_rl.rulebook.v2.geometry.drivable import DrivableLaneRecord, drivable_surface_for_ego
from thesis_rl.rulebook.v2.geometry.elevation import PolylineElevation
from thesis_rl.rulebook.v2.geometry.footprint import swept_front_bumper
from thesis_rl.rulebook.v2.geometry.lanes import (
    RouteLaneRecord,
    associate_route_lane,
    bumper_to_bumper_gap,
    derive_lane_movement_key,
    footprint_route_coordinates,
)
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.geometry.vertical import VERTICAL_COMPATIBILITY_TOLERANCE_M
from thesis_rl.rulebook.v2.monitor import evaluate_registered_transition
from thesis_rl.rulebook.v2.types import (
    ApproachControl,
    ActorClass,
    ActorSnapshot,
    EpisodeCache,
    EnvSnapshot,
    MapFeatureClass,
    RulebookMemory,
)


@dataclass(frozen=True, slots=True)
class RulebookTransitionConfig:
    """Runtime-only constants already frozen by Rulebook v4.7."""

    rss_calibration: RSSCalibrationArtifact | None = None
    expected_config_hash: str = ""
    prediction_horizon_s: float = 3.0
    history_window_s: float = 0.5
    minimum_history_samples: int = 3


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
        route_lanes=static_result.route_lanes,
        route_polyline=route,
    )


def initial_memory_for_snapshot(snapshot: EnvSnapshot, cache: EpisodeCache) -> RulebookMemory:
    """Initialize route and time ownership from the causal reset snapshot."""

    if cache.route_polyline is None:
        raise ValueError("Episode cache has no assigned route polyline")
    projection = cache.route_polyline.project(
        snapshot.ego.position_xy, position_z=snapshot.ego.position_z
    )
    return RulebookMemory(
        previous_route_s_m=projection.s_m,
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
    shifted_route = RoutePolyline(
        tuple((x, y, z + offset) for x, y, z in cache.route_polyline.points_xyz)
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


def _rss_candidates(
    *,
    ego: ActorSnapshot,
    actors: tuple[ActorSnapshot, ...],
    route,
    route_lanes: tuple[RouteLaneRecord, ...],
) -> tuple[RSSCandidate, ...]:
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
    ego_coords = footprint_route_coordinates(ego.footprint, route, position_z=ego.position_z)
    candidates: list[RSSCandidate] = []
    for actor in actors:
        if actor.actor_class is not ActorClass.VEHICLE:
            continue
        actor_lane = associate_route_lane(
            position_xy=actor.position_xy,
            position_z=actor.position_z,
            heading_rad=actor.heading_rad,
            route_lanes=route_lanes,
        )
        if actor_lane is None or actor_lane.lane_id != ego_lane.lane_id:
            continue
        actor_heading = (cos(actor.heading_rad), sin(actor.heading_rad))
        if (
            actor_heading[0] * actor_lane.tangent_xy[0]
            + actor_heading[1] * actor_lane.tangent_xy[1]
            <= 0.0
        ):
            continue
        other_coords = footprint_route_coordinates(
            actor.footprint, route, position_z=actor.position_z
        )
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


def _selected_control(controls, control_type, front_s: float, resolved):
    candidates = tuple(
        control
        for control in controls
        if control.control_type is control_type
        and control.control_group_id not in resolved
        and control.route_s_m >= front_s
    )
    return min(candidates, key=lambda item: (item.route_s_m, item.control_group_id), default=None)


def _empty_interval() -> OccupancyInterval:
    # A zero-duration interval is used only when the corresponding rule domain
    # is empty; the evaluator remains NOT_APPLICABLE because no opposing actor
    # or zone is supplied.
    return OccupancyInterval(0.0, 0.0)


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
):
    from thesis_rl.rulebook.v2.geometry.conflict_zones import (
        build_crosswalk_conflict_zone_candidates,
        attach_route_intervals,
    )

    crosswalks = tuple(
        feature
        for feature in cache.map_feature_catalog.values()
        if feature.feature_class is MapFeatureClass.CROSSWALK
    )
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
            "previous_illegal_entries": memory.crosswalk_illegal_entries,
            "vertical_applicable": False,
        }
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
            "previous_illegal_entries": memory.crosswalk_illegal_entries,
            "vertical_applicable": False,
        }
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
            "previous_illegal_entries": memory.crosswalk_illegal_entries,
            "vertical_applicable": False,
        }
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
            "previous_illegal_entries": memory.crosswalk_illegal_entries,
            "vertical_applicable": False,
        }
    zone = selected.candidate.polygon
    ego_interval, vru_intervals, _ = predict_conflict_zone_occupancy_intervals(
        ego=post.ego,
        actors=tuple(
            actor
            for actor in post.actors
            if actor.actor_class in {ActorClass.PEDESTRIAN, ActorClass.CYCLIST}
            and abs(actor.position_z - post.ego.position_z) <= VERTICAL_COMPATIBILITY_TOLERANCE_M
        ),
        histories=memory.actor_motion_histories,
        sim_time_s=post.sim_time_s,
        zone=zone,
        horizon_s=prediction_horizon_s,
        minimum_history_samples=minimum_history_samples,
    )
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
            "previous_illegal_entries": memory.crosswalk_illegal_entries,
            "vertical_applicable": False,
        }
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
        "ego_entered": bool(
            not pre.ego.footprint.intersects(zone) and post.ego.footprint.intersects(zone)
        ),
        "preexisting_zone_ids": memory.preexisting_ego_occupancy_zone_ids,
        "previous_illegal_entries": memory.crosswalk_illegal_entries,
        "vertical_applicable": True,
    }


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
        previous_s_m=memory.previous_route_s_m,
    ).tangent_xy
    post_approach_speed_mps = max(
        0.0,
        post_state.ego.velocity_xy[0] * post_route_tangent_xy[0]
        + post_state.ego.velocity_xy[1] * post_route_tangent_xy[1],
    )
    pre_by_id = {actor.actor_id: actor for actor in pre_state.actors}
    actors = tuple(post_state.actors)

    signal_pre = _selected_control(
        cache.traffic_control_catalog,
        ApproachControl.SIGNAL,
        pre_front_s,
        memory.resolved_signal_group_ids,
    )
    signal_post = _selected_control(
        cache.traffic_control_catalog,
        ApproachControl.SIGNAL,
        post_front_s,
        memory.resolved_signal_group_ids,
    )
    signal = signal_pre or signal_post
    stop_pre = _selected_control(
        cache.traffic_control_catalog,
        ApproachControl.STOP,
        pre_front_s,
        memory.resolved_stop_group_ids,
    )
    stop_post = _selected_control(
        cache.traffic_control_catalog,
        ApproachControl.STOP,
        post_front_s,
        memory.resolved_stop_group_ids,
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
    crosswalk_input = _crosswalk_inputs(
        pre=pre_state,
        post=post_state,
        cache=cache,
        memory=memory,
        route=route,
        delta_t_s=delta_t_s,
        post_front_s=post_front_s,
        prediction_horizon_s=config.prediction_horizon_s,
        minimum_history_samples=config.minimum_history_samples,
    )
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
    drivable = drivable_surface_for_ego(
        ego_footprint=post_state.ego.footprint,
        ego_position_xy=post_state.ego.position_xy,
        ego_position_z=post_state.ego.position_z,
        lanes=tuple(
            DrivableLaneRecord(lane.lane_id, lane.centerline, lane.polygon_xy, None)
            for lane in cache.route_lanes
        ),
    )
    vehicle_input = {
        "zone_id": "__no_vehicle_priority__",
        "ego_interval": _empty_interval(),
        "prioritized_intervals": (),
        "distance_to_entry_m": 0.0,
        "approach_speed_mps": post_approach_speed_mps,
        "delta_t_s": delta_t_s,
        "ego_occupied": False,
        "entered_actor_ids": frozenset(),
        "previous_illegal_entries": memory.vehicle_yield_illegal_entries,
        "actor_movement_keys": (),
        "previous_frozen_movement_keys": memory.frozen_actor_movement_keys,
        "exited_actor_ids": frozenset(),
        "ego_brake_mps2": calibrated_brake_mps2,
    }
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
        },
        "rss": {
            "scenario_id": post_state.scenario_id,
            "step_index": post_state.step_index,
            "candidates": _rss_candidates(
                ego=pre_state.ego,
                actors=pre_state.actors,
                route=route,
                route_lanes=cache.route_lanes,
            ),
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
        "offroad": {"ego_footprint": post_state.ego.footprint, "drivable_surface": drivable},
        "wrong_way": {
            "ego": post_state.ego,
            "route": route,
            "previous_s_m": memory.previous_route_s_m,
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
        "vehicle_yield": vehicle_input,
        "progress": {
            "pre_ego": pre_state.ego,
            "post_ego": post_state.ego,
            "route": route,
            "previous_route_s_m": memory.previous_route_s_m,
            "delta_t_s": delta_t_s,
        },
    }
    return evaluate_registered_transition(
        memory=memory,
        component_inputs=component_inputs,
        raw_progress_m=0.0,
        progress_margin=0.0,
        post_state=post_state,
        history_window_s=config.history_window_s,
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
