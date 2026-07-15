"""Offline converter for the checked-in ScenarioNet/Waymo mapping schema."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from shapely.geometry import LineString, Polygon

from thesis_rl.rulebook.v2.context.map_matching import OfflineTrackSample, map_match_sdc_track_to_task_route
from thesis_rl.rulebook.v2.context.static_adapter import StaticAdapterResult, normalize_static_records
from thesis_rl.rulebook.v2.geometry.controls import derive_control_line
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import ApproachControl, MapFeatureClass, MapFeatureRecord, MovementKey, TrafficControlRecord

_FEATURE_CLASSES = {
    "CROSSWALK": MapFeatureClass.CROSSWALK,
    "ROAD_EDGE_BOUNDARY": MapFeatureClass.ROAD_BOUNDARY,
    "ROAD_LINE_SOLID_SINGLE_WHITE": MapFeatureClass.LANE_MARKING_SOLID,
    "ROAD_LINE_SOLID_DOUBLE_YELLOW": MapFeatureClass.LANE_MARKING_SOLID,
    "ROAD_LINE_SOLID_SINGLE_YELLOW": MapFeatureClass.LANE_MARKING_SOLID,
    "ROAD_LINE_BROKEN_SINGLE_WHITE": MapFeatureClass.LANE_MARKING_DASHED,
    "DRIVEWAY": MapFeatureClass.OTHER_NON_DRIVABLE,
}


def _points(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("Waymo geometry must be a 2D array with XY coordinates")
    if array.shape[1] == 2:
        array = np.pad(array, ((0, 0), (0, 1)))
    return array[:, :3]


def _lane_record(lane_id: str, lane: Mapping[str, Any]) -> RouteLaneRecord:
    points = _points(lane.get("polyline"))
    centerline = RoutePolyline(
        tuple((float(point[0]), float(point[1]), float(point[2])) for point in points)
    )
    width = lane.get("width")
    width_m = float(np.nanmedian(np.asarray(width, dtype=float))) if width is not None else 3.5
    if not np.isfinite(width_m) or width_m <= 0.0:
        raise ValueError(f"Waymo lane {lane_id!r} has invalid width")
    polygon = LineString(tuple((float(x), float(y)) for x, y, _ in points)).buffer(width_m / 2.0, cap_style="flat", join_style="mitre")
    return RouteLaneRecord(lane_id, polygon, centerline)


def _track_samples(track: Mapping[str, Any]) -> tuple[OfflineTrackSample, ...]:
    state = track.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("Waymo track state must be a mapping")
    positions = np.asarray(state.get("position"), dtype=float)
    headings = np.asarray(state.get("heading"), dtype=float)
    valid = np.asarray(state.get("valid", np.ones(len(positions), dtype=bool)), dtype=bool)
    if positions.ndim != 2 or len(positions) != len(headings) or len(valid) != len(positions):
        raise ValueError("Waymo SDC track arrays have inconsistent lengths")
    return tuple(
        OfflineTrackSample((float(position[0]), float(position[1])), float(position[2]), float(heading))
        for position, heading, is_valid in zip(positions, headings, valid) if is_valid
    )


def build_waymo_static_adapter_result(scenario: Mapping[str, Any], *, scenario_uid: str, adapter_version: str = "waymo-v2") -> StaticAdapterResult:
    """Convert one converted Waymo scenario to canonical static v2 records."""
    features = scenario.get("map_features")
    metadata = scenario.get("metadata")
    if not isinstance(features, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("Waymo scenario requires map_features and metadata mappings")
    lanes: dict[str, RouteLaneRecord] = {}
    for feature_id, feature in features.items():
        if not isinstance(feature, Mapping) or not str(feature.get("type", "")).startswith("LANE_"):
            continue
        try:
            lanes[str(feature_id)] = _lane_record(str(feature_id), feature)
        except ValueError:
            # Invalid lane geometry is excluded offline; it can never become
            # an online fallback or silently widen the drivable surface.
            continue
    if not lanes:
        raise ValueError("Waymo scenario has no lane geometry")
    sdc_id = str(metadata.get("sdc_id", ""))
    tracks = scenario.get("tracks")
    if not sdc_id or not isinstance(tracks, Mapping) or sdc_id not in tracks:
        raise ValueError("Waymo scenario has no valid SDC track")
    source_hash = hashlib.sha256(json.dumps(sorted(lanes), separators=(",", ":")).encode()).digest()
    route = map_match_sdc_track_to_task_route(scenario_uid=scenario_uid, track=_track_samples(tracks[sdc_id]), route_lanes=lanes, source_geometry_bytes=source_hash, adapter_version=adapter_version)
    map_records: list[MapFeatureRecord] = []
    for feature_id, feature in features.items():
        if not isinstance(feature, Mapping):
            continue
        feature_class = _FEATURE_CLASSES.get(str(feature.get("type", "")))
        geometry_values = feature.get("polygon", feature.get("polyline"))
        if feature_class is None or geometry_values is None:
            continue
        points = _points(geometry_values)
        geometry = Polygon(tuple((float(x), float(y)) for x, y, _ in points)) if feature.get("polygon") is not None else LineString(tuple((float(x), float(y)) for x, y, _ in points))
        map_records.append(MapFeatureRecord(str(feature_id), feature_class, geometry, float(np.median(points[:, 2]))))
    controls: list[TrafficControlRecord] = []
    signal_errors: list[str] = []
    for feature_id, feature in features.items():
        if not isinstance(feature, Mapping) or feature.get("type") != "STOP_SIGN":
            continue
        position = feature.get("position")
        lane_ids = feature.get("lane", ())
        if position is None or not isinstance(lane_ids, (list, tuple)):
            continue
        point = _points(np.asarray(position, dtype=float).reshape(1, -1))[0]
        for lane_id_value in lane_ids:
            lane_id = str(lane_id_value)
            lane = lanes.get(lane_id)
            if lane is None:
                continue
            movement = MovementKey(lane_id, f"control:{feature_id}", lane_id)
            try:
                line = derive_control_line(control_point_xy=(float(point[0]), float(point[1])), control_point_z=float(point[2]), controlled_lane=lane)
            except ValueError:
                continue
            controls.append(TrafficControlRecord(f"{feature_id}:{lane_id}", ApproachControl.STOP, (lane_id,), movement, line.geometry, line.route_s_m, float(point[2]), ()))
    dynamic_states = scenario.get("dynamic_map_states", {})
    if isinstance(dynamic_states, Mapping):
        for physical_id, dynamic in dynamic_states.items():
            if not isinstance(dynamic, Mapping) or dynamic.get("type") != "TRAFFIC_LIGHT":
                continue
            lane_id = str(dynamic.get("lane", ""))
            point = _points(np.asarray(dynamic.get("stop_point"), dtype=float).reshape(1, -1))[0]
            lane = lanes.get(lane_id)
            if lane is None:
                continue
            states = dynamic.get("state", {}).get("object_state") if isinstance(dynamic.get("state"), Mapping) else None
            if not isinstance(states, (list, tuple, np.ndarray)) or len(states) != int(scenario.get("length", 0)):
                signal_errors.append(f"signal_sequence_invalid:{physical_id}")
            elif any(str(state) == "LANE_STATE_UNKNOWN" for state in states):
                signal_errors.append(f"signal_state_unknown:{physical_id}")
            movement = MovementKey(lane_id, f"control:{physical_id}", lane_id)
            try:
                line = derive_control_line(control_point_xy=(float(point[0]), float(point[1])), control_point_z=float(point[2]), controlled_lane=lane)
            except ValueError:
                continue
            controls.append(TrafficControlRecord(str(physical_id), ApproachControl.SIGNAL, (lane_id,), movement, line.geometry, line.route_s_m, float(point[2]), (str(physical_id),)))
    result = normalize_static_records(scenario_uid=scenario_uid, task_route=route, route_lanes=tuple(lanes.values()), map_features=tuple(map_records), traffic_controls=tuple(controls), movement_priority_records=())
    return replace(result, validation_errors=tuple((*result.validation_errors, *signal_errors)))
