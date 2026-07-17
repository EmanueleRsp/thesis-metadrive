"""Source-neutral normalization of static map records for F3 adapters."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping
from types import MappingProxyType

from thesis_rl.rulebook.v2.geometry.canonical import canonicalize_geometry, stable_geometry_id
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline, build_assigned_route_polyline
from thesis_rl.rulebook.v2.types import (
    MapFeatureRecord,
    MovementPriorityRecord,
    TaskRouteRecord,
    TrafficControlRecord,
)
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot


@dataclass(frozen=True, slots=True)
class StaticAdapterResult:
    scenario_uid: str
    task_route: TaskRouteRecord
    route_lanes: tuple[RouteLaneRecord, ...]
    map_features: dict[str, MapFeatureRecord]
    traffic_controls: tuple[TrafficControlRecord, ...]
    movement_priority_records: tuple[MovementPriorityRecord, ...] = ()
    validation_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "map_features", MappingProxyType(dict(self.map_features)))

    @property
    def assigned_route_polyline(self) -> RoutePolyline:
        """Build the route exclusively from frozen task lane IDs and map lanes."""

        return build_assigned_route_polyline(
            self.task_route.lane_ids,
            {lane.lane_id: lane for lane in self.route_lanes},
        )


def validate_reset_contract(
    *,
    ego: ActorSnapshot,
    actors: tuple[ActorSnapshot, ...],
    signal_states_by_physical_id: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Validate offline reset invariants required before rulebook training."""
    errors: list[str] = []
    if ego.configured_speed_cap_mps is None or ego.configured_speed_cap_mps <= 0.0:
        errors.append("ego_speed_cap_invalid")
    for actor in actors:
        if actor.actor_class == ActorClass.VEHICLE and (
            actor.configured_speed_cap_mps is None or actor.configured_speed_cap_mps <= 0.0
        ):
            errors.append(f"vehicle_speed_cap_invalid:{actor.actor_id}")
        if ego.footprint.intersection(actor.footprint).area > 1.0e-6:
            errors.append(f"spawn_overlap:{actor.actor_id}")
    unknown = sorted(
        physical_id
        for physical_id, state in (signal_states_by_physical_id or {}).items()
        if state in {"UNKNOWN", "LANE_STATE_UNKNOWN"}
    )
    errors.extend(f"signal_state_unknown:{physical_id}" for physical_id in unknown)
    return tuple(errors)


def normalize_static_records(
    *,
    scenario_uid: str,
    task_route: TaskRouteRecord,
    route_lanes: tuple[RouteLaneRecord, ...],
    map_features: tuple[MapFeatureRecord, ...],
    traffic_controls: tuple[TrafficControlRecord, ...],
    movement_priority_records: tuple[MovementPriorityRecord, ...] = (),
) -> StaticAdapterResult:
    """Canonicalize adapter output and return typed validation errors, never fallbacks."""

    if not scenario_uid or task_route.scenario_uid != scenario_uid:
        raise ValueError("Static adapter scenario UID does not match task route")
    errors: list[str] = []
    route_lane_ids = {lane.lane_id for lane in route_lanes}
    if len(route_lane_ids) != len(route_lanes):
        errors.append("duplicate_route_lane_id")
    missing = [lane_id for lane_id in task_route.lane_ids if lane_id not in route_lane_ids]
    if missing:
        errors.append("task_route_lane_missing:" + ",".join(missing))
    if not missing:
        try:
            build_assigned_route_polyline(
                task_route.lane_ids,
                {lane.lane_id: lane for lane in route_lanes},
            )
        except ValueError as error:
            errors.append(f"assigned_route_invalid:{error}")
    normalized_features: dict[str, MapFeatureRecord] = {}
    for feature in map_features:
        if feature.elevation_m is not None and not isfinite(feature.elevation_m):
            errors.append(f"invalid_map_feature_elevation:{feature.feature_id}")
            continue
        try:
            geometry = canonicalize_geometry(feature.geometry)
        except ValueError as error:
            errors.append(f"invalid_map_feature:{feature.feature_id}:{error}")
            continue
        feature_id = feature.feature_id or stable_geometry_id(
            scenario_id=scenario_uid,
            namespace="map",
            feature_type=feature.feature_class.value,
            geometry=geometry,
        )
        if feature_id in normalized_features:
            errors.append(f"duplicate_map_feature_id:{feature_id}")
            continue
        normalized_features[feature_id] = MapFeatureRecord(
            feature_id=feature_id,
            feature_class=feature.feature_class,
            geometry=geometry,
            elevation_m=feature.elevation_m,
            logical_boundary_id=feature.logical_boundary_id,
        )
    control_ids: set[str] = set()
    for control in traffic_controls:
        if not control.control_group_id or control.control_group_id in control_ids:
            errors.append(f"duplicate_control_group_id:{control.control_group_id}")
        control_ids.add(control.control_group_id)
        if control.control_line.is_empty or not control.control_line.is_valid:
            errors.append(f"invalid_control_line:{control.control_group_id}")
        if not isfinite(control.route_s_m) or not isfinite(control.elevation_m):
            errors.append(f"invalid_control_coordinate:{control.control_group_id}")
        if (
            not control.controlled_lane_ids
            or not control.physical_control_ids
            and control.control_type.value == "signal"
        ):
            errors.append(f"incomplete_control_record:{control.control_group_id}")
    return StaticAdapterResult(
        scenario_uid=scenario_uid,
        task_route=task_route,
        route_lanes=route_lanes,
        map_features=normalized_features,
        traffic_controls=traffic_controls,
        movement_priority_records=movement_priority_records,
        validation_errors=tuple(errors),
    )
