"""Source-neutral normalization of static map records for F3 adapters."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from thesis_rl.rulebook.v2.geometry.canonical import canonicalize_geometry, stable_geometry_id
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline, build_assigned_route_polyline
from thesis_rl.rulebook.v2.types import (
    ApproachControl,
    ActorClass,
    ActorSnapshot,
    MapFeatureClass,
    MapFeatureRecord,
    MovementPriorityRecord,
    MovementPriority,
    MovementKey,
    RoundaboutPriorityRecord,
    TaskRouteRecord,
    TrafficControlRecord,
)


@dataclass(frozen=True, slots=True)
class StaticAdapterResult:
    scenario_uid: str
    task_route: TaskRouteRecord
    route_lanes: tuple[RouteLaneRecord, ...]
    map_features: dict[str, MapFeatureRecord]
    traffic_controls: tuple[TrafficControlRecord, ...]
    movement_priority_records: tuple[MovementPriorityRecord, ...] = ()
    roundabout_priority_records: tuple[RoundaboutPriorityRecord, ...] = ()
    validation_errors: tuple[str, ...] = ()
    # Additive diagnostic only (REQ-RBCOST-011): raw map-feature type strings
    # that had no MapFeatureClass mapping and were therefore dropped from
    # map_features. Never affects rulebook_eligible -- an unmapped type is a
    # coverage gap to fix in the adapter, not a data defect in the scenario.
    unmapped_feature_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "map_features", MappingProxyType(dict(self.map_features)))

    @property
    def assigned_route_polyline(self) -> RoutePolyline:
        """Build the route exclusively from frozen task lane IDs and map lanes."""

        return build_assigned_route_polyline(
            self.task_route.lane_ids,
            {lane.lane_id: lane for lane in self.route_lanes},
        )


def vehicle_yield_records_from_metadata(
    metadata: Mapping[str, object],
) -> tuple[
    tuple[MovementPriorityRecord, ...], tuple[RoundaboutPriorityRecord, ...], tuple[str, ...]
]:
    """Decode explicit source annotations; malformed records make a scenario ineligible."""

    raw = metadata.get("rulebook_vehicle_yield")
    if raw is None:
        return (), (), ()
    if not isinstance(raw, Mapping):
        return (), (), ("vehicle_yield_metadata_invalid",)

    def key(value: object) -> MovementKey | None:
        if not isinstance(value, Mapping):
            return None
        fields = ("approach_lane_id", "conflict_node_id", "exit_lane_id")
        if any(not isinstance(value.get(field), str) or not value[field] for field in fields):
            return None
        return MovementKey(*(value[field] for field in fields))

    priorities: list[MovementPriorityRecord] = []
    roundabouts: list[RoundaboutPriorityRecord] = []
    errors: list[str] = []
    raw_priorities = raw.get("movement_priorities", ())
    if not isinstance(raw_priorities, (list, tuple)):
        errors.append("vehicle_yield_movement_priorities_invalid")
    else:
        for index, item in enumerate(raw_priorities):
            if not isinstance(item, Mapping):
                errors.append(f"vehicle_yield_priority_invalid:{index}")
                continue
            ego_key = key(item.get("ego_movement_key"))
            other_key = key(item.get("other_movement_key"))
            try:
                relation = MovementPriority(str(item.get("relation")))
            except ValueError:
                relation = None
            if ego_key is None or other_key is None or relation is None:
                errors.append(f"vehicle_yield_priority_invalid:{index}")
                continue
            priorities.append(MovementPriorityRecord(ego_key, other_key, relation))
    raw_roundabouts = raw.get("roundabout_priorities", ())
    if not isinstance(raw_roundabouts, (list, tuple)):
        errors.append("vehicle_yield_roundabout_priorities_invalid")
    else:
        for index, item in enumerate(raw_roundabouts):
            if not isinstance(item, Mapping):
                errors.append(f"vehicle_yield_roundabout_invalid:{index}")
                continue
            values = tuple(
                item.get(field)
                for field in ("component_id", "entry_lane_id", "circulating_lane_id")
            )
            if any(not isinstance(value, str) or not value for value in values):
                errors.append(f"vehicle_yield_roundabout_invalid:{index}")
                continue
            roundabouts.append(RoundaboutPriorityRecord(*values))
    return tuple(priorities), tuple(roundabouts), tuple(errors)


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


def _group_signal_controls(
    controls: tuple[TrafficControlRecord, ...],
) -> tuple[TrafficControlRecord, ...]:
    """Group physical signal heads that govern one movement/control line."""

    grouped: list[TrafficControlRecord] = []
    pending = sorted(controls, key=lambda control: control.control_group_id)
    while pending:
        control = pending.pop(0)
        if control.control_type is not ApproachControl.SIGNAL:
            grouped.append(control)
            continue
        compatible = [control]
        retained: list[TrafficControlRecord] = []
        for candidate in pending:
            if (
                candidate.control_type is ApproachControl.SIGNAL
                and candidate.movement_key == control.movement_key
                and abs(candidate.route_s_m - control.route_s_m) <= 5.0e-2
                and abs(candidate.elevation_m - control.elevation_m) <= 1.0e-3
                and candidate.control_line.equals(control.control_line)
            ):
                compatible.append(candidate)
            else:
                retained.append(candidate)
        pending = retained
        physical_ids = tuple(
            sorted(
                {physical_id for item in compatible for physical_id in item.physical_control_ids}
            )
        )
        if not physical_ids:
            raise ValueError("Signal control group requires physical IDs")
        grouped.append(
            TrafficControlRecord(
                control_group_id="signal:" + ":".join(physical_ids),
                control_type=ApproachControl.SIGNAL,
                controlled_lane_ids=tuple(
                    sorted({lane_id for item in compatible for lane_id in item.controlled_lane_ids})
                ),
                movement_key=control.movement_key,
                control_line=control.control_line,
                route_s_m=control.route_s_m,
                elevation_m=control.elevation_m,
                physical_control_ids=physical_ids,
            )
        )
    return tuple(sorted(grouped, key=lambda control: control.control_group_id))


def normalize_static_records(
    *,
    scenario_uid: str,
    task_route: TaskRouteRecord,
    route_lanes: tuple[RouteLaneRecord, ...],
    map_features: tuple[MapFeatureRecord, ...],
    traffic_controls: tuple[TrafficControlRecord, ...],
    movement_priority_records: tuple[MovementPriorityRecord, ...] = (),
    roundabout_priority_records: tuple[RoundaboutPriorityRecord, ...] = (),
    unmapped_feature_types: tuple[str, ...] = (),
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
        if (
            feature.feature_class
            in {
                MapFeatureClass.CROSSWALK,
                MapFeatureClass.LANE_MARKING_SOLID,
                MapFeatureClass.LANE_MARKING_DASHED,
            }
            and feature.elevation_m is None
        ):
            errors.append(f"missing_map_feature_elevation:{feature.feature_id}")
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
    normalized_controls = _group_signal_controls(traffic_controls)
    priority_pairs: set[tuple[object, object]] = set()
    for record in movement_priority_records:
        pair = (record.ego_movement_key, record.other_movement_key)
        if pair in priority_pairs:
            errors.append("duplicate_movement_priority_record")
        priority_pairs.add(pair)
        for lane_id in (
            record.ego_movement_key.approach_lane_id,
            record.ego_movement_key.exit_lane_id,
            record.other_movement_key.approach_lane_id,
            record.other_movement_key.exit_lane_id,
        ):
            if lane_id not in route_lane_ids:
                errors.append(f"movement_priority_lane_missing:{lane_id}")
    for record in roundabout_priority_records:
        if not record.component_id or not record.entry_lane_id or not record.circulating_lane_id:
            errors.append("incomplete_roundabout_priority_record")
        for lane_id in (record.entry_lane_id, record.circulating_lane_id):
            if lane_id not in route_lane_ids:
                errors.append(f"roundabout_priority_lane_missing:{lane_id}")
    control_ids: set[str] = set()
    for control in normalized_controls:
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
        traffic_controls=normalized_controls,
        movement_priority_records=movement_priority_records,
        roundabout_priority_records=roundabout_priority_records,
        validation_errors=tuple(errors),
        unmapped_feature_types=unmapped_feature_types,
    )
