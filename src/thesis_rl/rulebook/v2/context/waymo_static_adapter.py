"""Offline converter for the checked-in ScenarioNet/Waymo mapping schema."""

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
from thesis_rl.rulebook.v2.geometry.controls import derive_control_line
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord, derive_lane_movement_key
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
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


def _lane_polygon_from_widths(points: np.ndarray, width: Any, *, lane_id: str) -> Polygon:
    """Build a lane polygon from ScenarioNet's left/right per-point widths."""

    widths = np.asarray(width, dtype=float)
    if widths.ndim == 1:
        width_m = float(np.nanmedian(widths))
        if not np.isfinite(width_m) or width_m <= 0.0:
            raise ValueError(f"Waymo lane {lane_id!r} has invalid width")
        return LineString(tuple((float(x), float(y)) for x, y, _ in points)).buffer(
            width_m / 2.0, cap_style="flat", join_style="mitre"
        )
    if widths.ndim != 2 or widths.shape != (len(points), 2):
        raise ValueError(f"Waymo lane {lane_id!r} width must have one left/right pair per point")

    # ScenarioNet's Waymo converter derives a width independently at every
    # centerline sample. Missing boundary spans remain zero; linearly repair
    # only those missing samples from the lane's own valid widths.
    sample_index = np.arange(len(points), dtype=float)
    resolved_widths = np.empty_like(widths)
    for side in range(2):
        valid = np.isfinite(widths[:, side]) & (widths[:, side] > 0.0)
        if not np.any(valid):
            raise ValueError(f"Waymo lane {lane_id!r} has invalid left/right widths")
        resolved_widths[:, side] = np.interp(
            sample_index,
            sample_index[valid],
            widths[valid, side],
        )

    xy = points[:, :2]
    tangents = np.empty_like(xy)
    tangents[0] = xy[1] - xy[0]
    tangents[-1] = xy[-1] - xy[-2]
    if len(xy) > 2:
        tangents[1:-1] = xy[2:] - xy[:-2]
    norms = np.linalg.norm(tangents, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError(f"Waymo lane {lane_id!r} has repeated centerline points")
    left_normals = np.column_stack((-tangents[:, 1] / norms, tangents[:, 0] / norms))
    left_boundary = xy + left_normals * resolved_widths[:, :1]
    right_boundary = xy - left_normals * resolved_widths[:, 1:]
    polygon = Polygon(np.concatenate((left_boundary, right_boundary[::-1]), axis=0))
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError(f"Waymo lane {lane_id!r} width polygon is invalid")
    return polygon


def _lane_record(
    lane_id: str, lane: Mapping[str, Any], *, z_origin_m: float = 0.0
) -> RouteLaneRecord:
    points = _points(lane.get("polyline"))
    centerline = RoutePolyline(
        tuple((float(point[0]), float(point[1]), float(point[2] - z_origin_m)) for point in points)
    )
    width = lane.get("width")
    polygon = _lane_polygon_from_widths(
        points,
        3.5 if width is None else width,
        lane_id=lane_id,
    )
    successors = tuple(str(successor) for successor in lane.get("exit_lanes", ()))
    return RouteLaneRecord(lane_id, polygon, centerline, successors)


def _track_samples(
    track: Mapping[str, Any], *, z_origin_m: float = 0.0
) -> tuple[OfflineTrackSample, ...]:
    state = track.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("Waymo track state must be a mapping")
    positions = np.asarray(state.get("position"), dtype=float)
    headings = np.asarray(state.get("heading"), dtype=float)
    valid = np.asarray(state.get("valid", np.ones(len(positions), dtype=bool)), dtype=bool)
    if positions.ndim != 2 or len(positions) != len(headings) or len(valid) != len(positions):
        raise ValueError("Waymo SDC track arrays have inconsistent lengths")
    return tuple(
        OfflineTrackSample(
            (float(position[0]), float(position[1])),
            float(position[2] - z_origin_m),
            float(heading),
        )
        for position, heading, is_valid in zip(positions, headings, valid)
        if is_valid
    )


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


def build_waymo_static_adapter_result(
    scenario: Mapping[str, Any], *, scenario_uid: str, adapter_version: str = "waymo-v2"
) -> StaticAdapterResult:
    """Convert one converted Waymo scenario to canonical static v2 records."""
    features = scenario.get("map_features")
    metadata = scenario.get("metadata")
    if not isinstance(features, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("Waymo scenario requires map_features and metadata mappings")
    z_origin_m = _sdc_z_origin(scenario, metadata)
    lanes: dict[str, RouteLaneRecord] = {}
    for feature_id, feature in features.items():
        if not isinstance(feature, Mapping) or not str(feature.get("type", "")).startswith("LANE_"):
            continue
        try:
            lanes[str(feature_id)] = _lane_record(str(feature_id), feature, z_origin_m=z_origin_m)
        except ValueError:
            # Invalid lane geometry is excluded offline; it can never become
            # an online fallback or silently widen the drivable surface.
            continue
    if not lanes:
        raise ValueError("Waymo scenario has no lane geometry")
    source_hash = hashlib.sha256(json.dumps(sorted(lanes), separators=(",", ":")).encode()).digest()
    assigned_route = metadata.get("assigned_route_lane_ids")
    if assigned_route is not None:
        if not isinstance(assigned_route, (list, tuple)):
            raise ValueError("Waymo assigned_route_lane_ids must be a sequence")
        route = build_task_route_record(
            scenario_uid=scenario_uid,
            lane_ids=tuple(str(lane_id) for lane_id in assigned_route),
            provenance="persisted_assigned_route_metadata",
            adapter_version=adapter_version,
            source_geometry_bytes=source_hash,
            route_assignment_source=str(
                metadata.get("assigned_route_source") or "waymo_sdc_offline_task_annotation"
            ),
        )
    else:
        sdc_id = str(metadata.get("sdc_id", ""))
        tracks = scenario.get("tracks")
        if not sdc_id or not isinstance(tracks, Mapping) or sdc_id not in tracks:
            raise ValueError("Waymo scenario has no valid SDC track for offline annotation")
        route = map_match_sdc_track_to_task_route(
            scenario_uid=scenario_uid,
            track=_track_samples(tracks[sdc_id], z_origin_m=z_origin_m),
            route_lanes=lanes,
            source_geometry_bytes=source_hash,
            adapter_version=adapter_version,
            route_assignment_source="waymo_sdc_offline_task_annotation",
        )
    map_records: list[MapFeatureRecord] = []
    feature_errors: list[str] = []
    for feature_id, feature in features.items():
        if not isinstance(feature, Mapping):
            continue
        feature_class = _FEATURE_CLASSES.get(str(feature.get("type", "")))
        geometry_values = feature.get("polygon", feature.get("polyline"))
        if feature_class is None or geometry_values is None:
            continue
        points = _points(geometry_values)
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
            movement = derive_lane_movement_key(lane, assigned_route_lane_ids=route.lane_ids)
            if movement is None:
                feature_errors.append(f"movement_key_ambiguous:{feature_id}:{lane_id}")
                continue
            try:
                line = derive_control_line(
                    control_point_xy=(float(point[0]), float(point[1])),
                    control_point_z=float(point[2] - z_origin_m),
                    controlled_lane=lane,
                )
            except ValueError:
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
    relevant_lane_ids = reachable_lane_ids(
        route_lane_ids=route.lane_ids,
        lane_successors=_lane_successors(features),
    )
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
            movement = derive_lane_movement_key(lane, assigned_route_lane_ids=route.lane_ids)
            if movement is None:
                signal_errors.append(f"movement_key_ambiguous:{physical_id}:{lane_id}")
                continue
            try:
                line = derive_control_line(
                    control_point_xy=(float(point[0]), float(point[1])),
                    control_point_z=float(point[2] - z_origin_m),
                    controlled_lane=lane,
                )
            except ValueError:
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
        task_route=route,
        route_lanes=tuple(lanes.values()),
        map_features=tuple(map_records),
        traffic_controls=tuple(controls),
        movement_priority_records=priority_records,
        roundabout_priority_records=roundabout_records,
    )
    return replace(
        result,
        validation_errors=tuple(
            (*result.validation_errors, *feature_errors, *signal_errors, *priority_errors)
        ),
    )
