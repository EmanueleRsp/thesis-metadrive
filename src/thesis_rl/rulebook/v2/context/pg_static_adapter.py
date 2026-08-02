"""Offline converter for the procedural ScenarioDescription schema."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from shapely.geometry import LineString, Polygon

from thesis_rl.rulebook.v2.context.map_matching import (
    OfflineTrackSample,
    map_match_sdc_track_to_task_route,
    reachable_lane_ids,
)
from thesis_rl.rulebook.v2.context.task_route import build_task_route_record
from thesis_rl.rulebook.v2.context.static_adapter import (
    StaticAdapterResult,
    normalize_static_records,
    vehicle_yield_records_from_metadata,
)
from thesis_rl.rulebook.v2.geometry.controls import (
    ControlLineOffRouteError,
    derive_control_line,
)
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord, derive_lane_movement_key
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline, build_assigned_route_polyline
from thesis_rl.rulebook.v2.types import (
    ApproachControl,
    MapFeatureClass,
    MapFeatureRecord,
    TrafficControlRecord,
)

_FEATURE_CLASSES = {
    "CROSSWALK": MapFeatureClass.CROSSWALK,
    "ROAD_EDGE_BOUNDARY": MapFeatureClass.ROAD_BOUNDARY,
    "ROAD_LINE_SOLID_SINGLE_WHITE": MapFeatureClass.LANE_MARKING_SOLID,
    "ROAD_LINE_SOLID_DOUBLE_YELLOW": MapFeatureClass.LANE_MARKING_SOLID,
    # ROAD_LINE_SOLID_SINGLE_YELLOW is the continuous carriageway centreline
    # PGMap.get_line_type emits for a continuous yellow line (REQ-RBCOST-004).
    "ROAD_LINE_SOLID_SINGLE_YELLOW": MapFeatureClass.LANE_MARKING_SOLID,
    "ROAD_LINE_BROKEN_SINGLE_WHITE": MapFeatureClass.LANE_MARKING_DASHED,
    "ROAD_LINE_BROKEN_SINGLE_YELLOW": MapFeatureClass.LANE_MARKING_DASHED,
}
# Feature type prefixes that are recognised but intentionally not lane
# markings/boundaries (e.g. lane surfaces, stop signs, dynamic states),
# excluded from the REQ-RBCOST-011 unmapped-type diagnostic so it reports
# only genuine coverage gaps in the marking/boundary/crosswalk mapping.
_NON_MARKING_FEATURE_PREFIXES = ("LANE_",)
_NON_MARKING_FEATURE_TYPES = frozenset({"STOP_SIGN"})


def _array_points(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("PG geometry must contain an XY point array")
    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)))
    return points[:, :3]


def _track_samples(
    track: Mapping[str, Any], *, z_origin_m: float = 0.0
) -> tuple[OfflineTrackSample, ...]:
    state = track.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("PG track state must be a mapping")
    positions = _array_points(state.get("position"))
    headings = np.asarray(state.get("heading"), dtype=float)
    valid = np.asarray(state.get("valid", np.ones(len(positions), dtype=bool)), dtype=bool)
    if len(headings) != len(positions) or len(valid) != len(positions):
        raise ValueError("PG SDC track arrays have inconsistent lengths")
    return tuple(
        OfflineTrackSample((float(p[0]), float(p[1])), float(p[2] - z_origin_m), float(h))
        for p, h, ok in zip(positions, headings, valid)
        if ok
    )


def _lane_record(
    lane_id: str, lane: Mapping[str, Any], *, z_origin_m: float = 0.0
) -> RouteLaneRecord:
    points = _array_points(lane.get("polyline"))
    centerline = RoutePolyline(
        tuple((float(point[0]), float(point[1]), float(point[2] - z_origin_m)) for point in points)
    )
    polygon_value = lane.get("polygon")
    if polygon_value is not None:
        polygon_points = _array_points(polygon_value)
        polygon = Polygon(tuple((float(x), float(y)) for x, y, _ in polygon_points))
    else:
        polygon = LineString(tuple((float(x), float(y)) for x, y, _ in points)).buffer(
            1.75, cap_style="flat", join_style="mitre"
        )
    successors = tuple(str(successor) for successor in lane.get("exit_lanes", ()))
    return RouteLaneRecord(lane_id, polygon, centerline, successors)


def _lane_successors(features: Mapping[Any, Any]) -> dict[str, tuple[str, ...]]:
    return {
        str(feature_id): tuple(str(successor) for successor in feature.get("exit_lanes", ()))
        for feature_id, feature in features.items()
        if isinstance(feature, Mapping) and str(feature.get("type", "")).startswith("LANE_")
    }


def _sdc_z_origin(scenario: Mapping[str, Any], metadata: Mapping[str, Any]) -> float:
    """Match MetaDrive's reset convention while retaining relative elevation."""

    tracks = scenario.get("tracks")
    sdc_id = str(metadata.get("sdc_id", ""))
    if not isinstance(tracks, Mapping) or sdc_id not in tracks:
        return 0.0
    state = tracks[sdc_id].get("state") if isinstance(tracks[sdc_id], Mapping) else None
    positions = state.get("position") if isinstance(state, Mapping) else None
    if positions is None:
        return 0.0
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3 or len(values) == 0 or not np.isfinite(values[0, 2]):
        return 0.0
    return float(values[0, 2])


def build_pg_static_adapter_result(
    scenario: Mapping[str, Any], *, scenario_uid: str, adapter_version: str = "pg-v2"
) -> StaticAdapterResult:
    """Convert one procedural ScenarioDescription to canonical static records."""
    features = scenario.get("map_features")
    metadata = scenario.get("metadata")
    if not isinstance(features, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("PG scenario requires map_features and metadata mappings")
    z_origin_m = _sdc_z_origin(scenario, metadata)
    lanes: dict[str, RouteLaneRecord] = {}
    for feature_id, feature in features.items():
        if isinstance(feature, Mapping) and str(feature.get("type", "")).startswith("LANE_"):
            lanes[str(feature_id)] = _lane_record(str(feature_id), feature, z_origin_m=z_origin_m)
    if not lanes:
        raise ValueError("PG scenario has no lane geometry")
    source_hash = hashlib.sha256(json.dumps(sorted(lanes), separators=(",", ":")).encode()).digest()
    assigned_route = metadata.get("assigned_route_lane_ids")
    if assigned_route is not None:
        if not isinstance(assigned_route, (list, tuple)):
            raise ValueError("PG assigned_route_lane_ids must be a sequence")
        task_route = build_task_route_record(
            scenario_uid=scenario_uid,
            lane_ids=tuple(str(lane_id) for lane_id in assigned_route),
            provenance="persisted_assigned_route_metadata",
            adapter_version=adapter_version,
            source_geometry_bytes=source_hash,
            route_assignment_source=str(
                metadata.get("assigned_route_source") or "pg_sdc_offline_task_annotation"
            ),
        )
    else:
        sdc_id = str(metadata.get("sdc_id", ""))
        tracks = scenario.get("tracks")
        if not sdc_id or not isinstance(tracks, Mapping) or sdc_id not in tracks:
            raise ValueError("PG scenario has no valid SDC track for offline annotation")
        task_route = map_match_sdc_track_to_task_route(
            scenario_uid=scenario_uid,
            track=_track_samples(tracks[sdc_id], z_origin_m=z_origin_m),
            route_lanes=lanes,
            source_geometry_bytes=source_hash,
            adapter_version=adapter_version,
            route_assignment_source="pg_sdc_offline_task_annotation",
        )
    # REQ-EF-05: traffic-control route coordinates are canonical route
    # coordinates, so the assigned route must exist before controls are built.
    try:
        assigned_route = build_assigned_route_polyline(task_route.lane_ids, lanes)
    except ValueError:
        # ``route_s`` is by definition a coordinate on the assigned route; with
        # no buildable route it does not exist, and emitting a control carrying
        # a lane-local value instead is exactly the defect REQ-EF-05 removes.
        # The scenario is already ineligible in this state (the same failure is
        # reported by ``normalize_static_records``); record why controls are
        # absent rather than leaving it implicit.
        assigned_route = None
    map_records: list[MapFeatureRecord] = []
    feature_errors: list[str] = []
    unmapped_feature_types: set[str] = set()
    for feature_id, feature in features.items():
        if not isinstance(feature, Mapping):
            continue
        raw_type = str(feature.get("type", ""))
        feature_class = _FEATURE_CLASSES.get(raw_type)
        raw_geometry = feature.get("polygon", feature.get("polyline"))
        if feature_class is None or raw_geometry is None:
            if (
                feature_class is None
                and raw_geometry is not None
                and raw_type not in _NON_MARKING_FEATURE_TYPES
                and not raw_type.startswith(_NON_MARKING_FEATURE_PREFIXES)
            ):
                unmapped_feature_types.add(raw_type)
            continue
        points = _array_points(raw_geometry)
        polygonal = feature.get("polygon") is not None
        if len(points) < (3 if polygonal else 2):
            feature_errors.append(f"invalid_map_feature_geometry:{feature_id}")
            continue
        geometry = (
            Polygon(tuple((float(x), float(y)) for x, y, _ in points))
            if polygonal
            else LineString(tuple((float(x), float(y)) for x, y, _ in points))
        )
        map_records.append(
            MapFeatureRecord(
                str(feature_id),
                feature_class,
                geometry,
                float(np.median(points[:, 2] - z_origin_m)),
                elevation_profile_xyz=tuple(
                    (float(x), float(y), float(z - z_origin_m)) for x, y, z in points
                ),
            )
        )
    controls: list[TrafficControlRecord] = []
    dropped_control_line_off_route_count = 0
    signal_errors: list[str] = []
    for feature_id, feature in features.items():
        if not isinstance(feature, Mapping) or feature.get("type") != "STOP_SIGN":
            continue
        position_value = feature.get("position", feature.get("stop_point"))
        lane_ids = feature.get("lane", ())
        if position_value is None or not isinstance(lane_ids, (list, tuple)):
            continue
        point = _array_points(np.asarray(position_value, dtype=float).reshape(1, -1))[0]
        for lane_id_value in lane_ids:
            lane_id = str(lane_id_value)
            lane = lanes.get(lane_id)
            if lane is None:
                continue
            movement = derive_lane_movement_key(lane, assigned_route_lane_ids=task_route.lane_ids)
            if movement is None:
                feature_errors.append(f"movement_key_ambiguous:{feature_id}:{lane_id}")
                continue
            if assigned_route is None:
                continue
            try:
                line = derive_control_line(
                    control_point_xy=(float(point[0]), float(point[1])),
                    control_point_z=float(point[2]),
                    controlled_lane=lane,
                    route=assigned_route,
                )
            except ControlLineOffRouteError:
                # Governs another approach of the same junction, not the ego's
                # task route: correctly absent, not a data defect.
                dropped_control_line_off_route_count += 1
                continue
            except ValueError:
                # OPEN-EF-04: a §2.9.6 geometry ambiguity ("multiple unresolved
                # components") is skipped, not escalated to a validation error.
                # Escalating it disqualified scenarios that trained fine before,
                # including the smoke preset, because the ambiguity is
                # pre-existing map data rather than anything this plan changed.
                # REQ-EF-05/06 (canonical coordinate, movement scoping) do not
                # depend on the escalation.
                continue
            controls.append(
                TrafficControlRecord(
                    f"{feature_id}:{lane_id}",
                    ApproachControl.STOP,
                    (lane_id,),
                    movement,
                    line.geometry,
                    line.route_s_m,
                    float(point[2] - z_origin_m),
                    (),
                )
            )
    if assigned_route is None:
        feature_errors.append("traffic_controls_skipped_unbuildable_assigned_route")
    relevant_lane_ids = reachable_lane_ids(
        route_lane_ids=task_route.lane_ids,
        lane_successors=_lane_successors(features),
    )
    dynamic_states = scenario.get("dynamic_map_states", {})
    if isinstance(dynamic_states, Mapping):
        for physical_id, dynamic in dynamic_states.items():
            if not isinstance(dynamic, Mapping) or dynamic.get("type") != "TRAFFIC_LIGHT":
                continue
            lane_id = str(dynamic.get("lane", ""))
            point_value = dynamic.get("stop_point")
            if point_value is None or lane_id not in lanes:
                continue
            states = (
                dynamic.get("state", {}).get("object_state")
                if isinstance(dynamic.get("state"), Mapping)
                else None
            )
            if lane_id in relevant_lane_ids:
                if not isinstance(states, (list, tuple, np.ndarray)) or len(states) != int(
                    scenario.get("length", 0)
                ):
                    signal_errors.append(f"signal_sequence_invalid:{physical_id}")
                elif any(
                    str(state) in {"LANE_STATE_UNKNOWN", "TRAFFIC_LIGHT_UNKNOWN"}
                    for state in states
                ):
                    signal_errors.append(f"signal_state_unknown:{physical_id}")
            point = _array_points(np.asarray(point_value).reshape(1, -1))[0]
            movement = derive_lane_movement_key(
                lanes[lane_id], assigned_route_lane_ids=task_route.lane_ids
            )
            if movement is None:
                signal_errors.append(f"movement_key_ambiguous:{physical_id}:{lane_id}")
                continue
            if assigned_route is None:
                continue
            try:
                line = derive_control_line(
                    control_point_xy=(float(point[0]), float(point[1])),
                    control_point_z=float(point[2]),
                    controlled_lane=lanes[lane_id],
                    route=assigned_route,
                )
            except ControlLineOffRouteError:
                dropped_control_line_off_route_count += 1
                continue
            except ValueError:
                # OPEN-EF-04, as above.
                continue
            controls.append(
                TrafficControlRecord(
                    str(physical_id),
                    ApproachControl.SIGNAL,
                    (lane_id,),
                    movement,
                    line.geometry,
                    line.route_s_m,
                    float(point[2] - z_origin_m),
                    (str(physical_id),),
                )
            )
    priority_records, roundabout_records, priority_errors = vehicle_yield_records_from_metadata(
        metadata
    )
    result = normalize_static_records(
        scenario_uid=scenario_uid,
        task_route=task_route,
        route_lanes=tuple(lanes.values()),
        map_features=tuple(map_records),
        traffic_controls=tuple(controls),
        movement_priority_records=priority_records,
        roundabout_priority_records=roundabout_records,
        unmapped_feature_types=tuple(sorted(unmapped_feature_types)),
        dropped_control_line_off_route_count=dropped_control_line_off_route_count,
    )
    return replace(
        result,
        validation_errors=tuple(
            (*result.validation_errors, *feature_errors, *signal_errors, *priority_errors)
        ),
    )
