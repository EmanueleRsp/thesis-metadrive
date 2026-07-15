"""Offline converter for the procedural ScenarioDescription schema."""

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
    "ROAD_LINE_BROKEN_SINGLE_WHITE": MapFeatureClass.LANE_MARKING_DASHED,
}


def _array_points(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("PG geometry must contain an XY point array")
    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)))
    return points[:, :3]


def _track_samples(track: Mapping[str, Any]) -> tuple[OfflineTrackSample, ...]:
    state = track.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("PG track state must be a mapping")
    positions = _array_points(state.get("position"))
    headings = np.asarray(state.get("heading"), dtype=float)
    valid = np.asarray(state.get("valid", np.ones(len(positions), dtype=bool)), dtype=bool)
    if len(headings) != len(positions) or len(valid) != len(positions):
        raise ValueError("PG SDC track arrays have inconsistent lengths")
    return tuple(OfflineTrackSample((float(p[0]), float(p[1])), float(p[2]), float(h)) for p, h, ok in zip(positions, headings, valid) if ok)


def _lane_record(lane_id: str, lane: Mapping[str, Any]) -> RouteLaneRecord:
    points = _array_points(lane.get("polyline"))
    centerline = RoutePolyline(tuple(tuple(float(x) for x in point) for point in points))
    polygon_value = lane.get("polygon")
    if polygon_value is not None:
        polygon_points = _array_points(polygon_value)
        polygon = Polygon(tuple((float(x), float(y)) for x, y, _ in polygon_points))
    else:
        polygon = LineString(tuple((float(x), float(y)) for x, y, _ in points)).buffer(1.75, cap_style="flat", join_style="mitre")
    return RouteLaneRecord(lane_id, polygon, centerline)


def build_pg_static_adapter_result(scenario: Mapping[str, Any], *, scenario_uid: str, adapter_version: str = "pg-v2") -> StaticAdapterResult:
    """Convert one procedural ScenarioDescription to canonical static records."""
    features = scenario.get("map_features")
    metadata = scenario.get("metadata")
    if not isinstance(features, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("PG scenario requires map_features and metadata mappings")
    lanes: dict[str, RouteLaneRecord] = {}
    for feature_id, feature in features.items():
        if isinstance(feature, Mapping) and str(feature.get("type", "")).startswith("LANE_"):
            lanes[str(feature_id)] = _lane_record(str(feature_id), feature)
    if not lanes:
        raise ValueError("PG scenario has no lane geometry")
    sdc_id = str(metadata.get("sdc_id", ""))
    tracks = scenario.get("tracks")
    if not sdc_id or not isinstance(tracks, Mapping) or sdc_id not in tracks:
        raise ValueError("PG scenario has no valid SDC track")
    source_hash = hashlib.sha256(json.dumps(sorted(lanes), separators=(",", ":")).encode()).digest()
    task_route = map_match_sdc_track_to_task_route(scenario_uid=scenario_uid, track=_track_samples(tracks[sdc_id]), route_lanes=lanes, source_geometry_bytes=source_hash, adapter_version=adapter_version)
    map_records: list[MapFeatureRecord] = []
    for feature_id, feature in features.items():
        if not isinstance(feature, Mapping):
            continue
        feature_class = _FEATURE_CLASSES.get(str(feature.get("type", "")))
        raw_geometry = feature.get("polygon", feature.get("polyline"))
        if feature_class is None or raw_geometry is None:
            continue
        points = _array_points(raw_geometry)
        geometry = Polygon(tuple((float(x), float(y)) for x, y, _ in points)) if feature.get("polygon") is not None else LineString(tuple((float(x), float(y)) for x, y, _ in points))
        map_records.append(MapFeatureRecord(str(feature_id), feature_class, geometry, float(np.median(points[:, 2]))))
    controls: list[TrafficControlRecord] = []
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
            point_value = dynamic.get("stop_point")
            if point_value is None or lane_id not in lanes:
                continue
            states = dynamic.get("state", {}).get("object_state") if isinstance(dynamic.get("state"), Mapping) else None
            if not isinstance(states, (list, tuple, np.ndarray)) or len(states) != int(scenario.get("length", 0)):
                signal_errors.append(f"signal_sequence_invalid:{physical_id}")
            elif any(str(state) == "LANE_STATE_UNKNOWN" for state in states):
                signal_errors.append(f"signal_state_unknown:{physical_id}")
            point = _array_points(np.asarray(point_value).reshape(1, -1))[0]
            movement = MovementKey(lane_id, f"control:{physical_id}", lane_id)
            try:
                line = derive_control_line(control_point_xy=(float(point[0]), float(point[1])), control_point_z=float(point[2]), controlled_lane=lanes[lane_id])
            except ValueError:
                continue
            controls.append(TrafficControlRecord(str(physical_id), ApproachControl.SIGNAL, (lane_id,), movement, line.geometry, line.route_s_m, float(point[2]), (str(physical_id),)))
    result = normalize_static_records(scenario_uid=scenario_uid, task_route=task_route, route_lanes=tuple(lanes.values()), map_features=tuple(map_records), traffic_controls=tuple(controls), movement_priority_records=())
    return replace(result, validation_errors=tuple((*result.validation_errors, *signal_errors)))
