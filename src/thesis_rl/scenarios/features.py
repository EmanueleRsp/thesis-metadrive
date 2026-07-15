from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from thesis_rl.scenarios.conflicts import TrackConflict, closest_approach_conflict
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioSource
from thesis_rl.scenarios.waymo_topology import infer_waymo_topology


_DYNAMIC_TYPES = {"VEHICLE", "PEDESTRIAN", "CYCLIST"}


def _state_array(track: Mapping[str, Any], name: str, length: int) -> np.ndarray:
    state = track.get("state")
    if not isinstance(state, Mapping) or name not in state:
        raise ValueError(f"track is missing state.{name}")
    array = np.asarray(state[name])
    if len(array) != length:
        raise ValueError(f"state.{name} length {len(array)} does not match scenario length {length}")
    return array


def _valid_mask(track: Mapping[str, Any], length: int) -> np.ndarray:
    state = track.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("track is missing state mapping")
    if "valid" not in state:
        return np.ones(length, dtype=bool)
    valid = np.asarray(state["valid"], dtype=bool)
    if valid.shape != (length,):
        raise ValueError("track state.valid must have shape (scenario_length,)")
    return valid


def _positions(track: Mapping[str, Any], length: int) -> np.ndarray:
    position = np.asarray(_state_array(track, "position", length), dtype=np.float64)
    if position.ndim != 2 or position.shape[1] < 2:
        raise ValueError("track state.position must have shape (length, >=2)")
    if position.shape[1] == 2:
        position = np.pad(position, ((0, 0), (0, 1)))
    return position[:, :3]


def _route_length(route: np.ndarray) -> float:
    if len(route) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1).sum())


def _route_z_range(route: np.ndarray) -> float:
    if len(route) == 0:
        return 0.0
    z = route[:, 2]
    return float(z.max() - z.min())


def _timestamps(metadata: Mapping[str, Any], length: int) -> np.ndarray:
    values = np.asarray(metadata.get("ts", np.arange(length) * 0.1), dtype=np.float64)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError("scenario metadata.ts must be a finite length-sized array")
    if np.any(np.diff(values) <= 0):
        raise ValueError("scenario timestamps must be strictly increasing")
    return values


def _minimum_conflict(
    conflicts: list[TrackConflict],
) -> tuple[int, float | None, float | None]:
    active = [conflict for conflict in conflicts if conflict.is_conflict]
    if not active:
        return 0, None, None
    selected = min(
        active,
        key=lambda conflict: float("inf")
        if conflict.min_dcpa_m is None
        else conflict.min_dcpa_m,
    )
    return len(active), selected.min_dcpa_m, selected.min_tcpa_s


def _minimum_distance(points: np.ndarray, route: np.ndarray) -> float | None:
    if len(points) == 0 or len(route) == 0:
        return None
    minimum = np.inf
    # Chunking avoids constructing an unbounded agents-by-route matrix.
    for start in range(0, len(points), 256):
        chunk = points[start : start + 256, :2]
        distances = np.linalg.norm(chunk[:, None, :] - route[None, :, :2], axis=-1)
        minimum = min(minimum, float(distances.min()))
    return None if not np.isfinite(minimum) else minimum


def _optional_bool(mapping: Mapping[str, Any], key: str) -> bool | None:
    if key not in mapping or mapping[key] is None:
        return None
    if not isinstance(mapping[key], (bool, np.bool_)):
        raise ValueError(f"realized metadata {key!r} must be boolean or null")
    return bool(mapping[key])


def _topology(
    realized_generation_metadata: Mapping[str, Any] | None,
) -> tuple[bool | None, bool | None, str]:
    topology: Mapping[str, Any] = {}
    if realized_generation_metadata is not None:
        candidate = realized_generation_metadata.get("realized_topology", {})
        if not isinstance(candidate, Mapping):
            raise ValueError("realized_topology must be a mapping")
        topology = candidate
    intersection = _optional_bool(topology, "has_intersection")
    merge = _optional_bool(topology, "has_merge_or_roundabout")
    if intersection is True and merge is True:
        tag = "mixed"
    elif intersection is True:
        tag = "intersection"
    elif merge is True:
        tag = "merge_or_roundabout"
    elif intersection is False and merge is False:
        tag = "simple"
    else:
        tag = "unknown"
    return intersection, merge, tag


def _route_controls(
    scenario: Mapping[str, Any],
    realized_generation_metadata: Mapping[str, Any] | None,
) -> tuple[bool | None, bool | None, bool | None, str]:
    map_features = scenario.get("map_features", {})
    dynamic_states = scenario.get("dynamic_map_states", {})
    if not isinstance(map_features, Mapping) or not isinstance(dynamic_states, Mapping):
        raise ValueError("map_features and dynamic_map_states must be mappings")
    map_types = {
        str(feature.get("type"))
        for feature in map_features.values()
        if isinstance(feature, Mapping)
    }

    controls: Mapping[str, Any] = {}
    if realized_generation_metadata is not None:
        candidate = realized_generation_metadata.get("route_traffic_controls", {})
        if not isinstance(candidate, Mapping):
            raise ValueError("route_traffic_controls must be a mapping")
        controls = candidate

    route_light = _optional_bool(controls, "has_traffic_light")
    route_stop = _optional_bool(controls, "has_stop_sign")
    route_crosswalk = _optional_bool(controls, "has_crosswalk")

    if route_stop is None and "STOP_SIGN" not in map_types:
        route_stop = False
    if route_crosswalk is None and "CROSSWALK" not in map_types:
        route_crosswalk = False

    if route_light is True:
        states_complete = _optional_bool(controls, "traffic_light_states_complete")
        if states_complete is True:
            reliability = "complete"
        elif states_complete is False:
            reliability = "missing"
        else:
            reliability = "partial"
    elif route_light is False:
        reliability = "not_applicable"
    elif not dynamic_states:
        route_light = False
        reliability = "not_applicable"
    else:
        # Lights exist, but their route relevance cannot be established safely.
        reliability = "partial"
    return route_light, route_stop, route_crosswalk, reliability


def extract_scenario_features(
    scenario: dict[str, Any],
    source: ScenarioSource,
    realized_generation_metadata: dict[str, Any] | None = None,
    *,
    relevant_radius_m: float = 50.0,
    vertical_tolerance_m: float = 3.0,
    temporal_quantile: float = 0.90,
    vru_max_distance_to_route_m: float = 8.0,
    waymo_lane_route_distance_m: float = 6.0,
    waymo_control_route_distance_m: float = 15.0,
    waymo_merge_min_entry_lanes: int = 2,
    conflict_horizon_s: float = 5.0,
    vehicle_conflict_distance_m: float = 4.0,
    vru_conflict_distance_m: float = 3.0,
) -> ScenarioFeatures:
    if source not in {"waymo", "pg"}:
        raise ValueError(f"unsupported scenario source: {source!r}")
    if relevant_radius_m <= 0 or vertical_tolerance_m <= 0:
        raise ValueError("relevance distance thresholds must be positive")
    if not 0 <= temporal_quantile <= 1:
        raise ValueError("temporal_quantile must be in [0, 1]")

    length = int(scenario.get("length", 0))
    if length <= 0:
        raise ValueError("scenario length must be positive")
    tracks = scenario.get("tracks")
    metadata = scenario.get("metadata")
    if not isinstance(tracks, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("scenario tracks and metadata must be mappings")
    sdc_id = str(metadata.get("sdc_id", ""))
    if not sdc_id or sdc_id not in tracks:
        raise ValueError("scenario metadata references a missing SDC track")

    ego_track = tracks[sdc_id]
    if not isinstance(ego_track, Mapping):
        raise ValueError("SDC track must be a mapping")
    ego_positions = _positions(ego_track, length)
    ego_valid = _valid_mask(ego_track, length)
    timestamps = _timestamps(metadata, length)
    ego_route = ego_positions[ego_valid]
    if len(ego_route) < 2 or not np.isfinite(ego_route).all():
        raise ValueError("SDC route is missing, non-finite, or degenerate")
    map_features = scenario.get("map_features", {})
    if not isinstance(map_features, Mapping):
        raise ValueError("scenario map_features must be a mapping")
    dynamic_object_count = sum(
        1
        for track_value in tracks.values()
        if isinstance(track_value, Mapping)
        and str(track_value.get("type", "")) in _DYNAMIC_TYPES
    )

    relevant_agents = np.zeros(length, dtype=np.int64)
    relevant_vehicles = np.zeros(length, dtype=np.int64)
    relevant_vrus = np.zeros(length, dtype=np.int64)
    vehicle_distances: list[np.ndarray] = []
    vru_points: list[np.ndarray] = []
    vehicle_conflicts: list[TrackConflict] = []
    vru_conflicts: list[TrackConflict] = []
    present_types: set[str] = set()

    for track_id, track_value in tracks.items():
        if str(track_id) == sdc_id or not isinstance(track_value, Mapping):
            continue
        object_type = str(track_value.get("type", ""))
        if object_type not in _DYNAMIC_TYPES:
            continue
        present_types.add(object_type)
        positions = _positions(track_value, length)
        track_valid = _valid_mask(track_value, length)
        valid = track_valid & ego_valid
        finite = np.isfinite(positions).all(axis=1) & np.isfinite(ego_positions).all(axis=1)
        valid &= finite
        planar_distance = np.linalg.norm(positions[:, :2] - ego_positions[:, :2], axis=1)
        vertical_distance = np.abs(positions[:, 2] - ego_positions[:, 2])
        relevant = (
            valid
            & (planar_distance <= relevant_radius_m)
            & (vertical_distance < vertical_tolerance_m)
        )
        relevant_agents += relevant.astype(np.int64)
        if object_type == "VEHICLE":
            relevant_vehicles += relevant.astype(np.int64)
            vehicle_distances.append(planar_distance[valid])
            vehicle_conflicts.append(
                closest_approach_conflict(
                    ego_positions,
                    positions,
                    valid,
                    timestamps,
                    horizon_s=conflict_horizon_s,
                    distance_threshold_m=vehicle_conflict_distance_m,
                    relevance_radius_m=relevant_radius_m,
                    vertical_tolerance_m=vertical_tolerance_m,
                )
            )
        elif object_type in {"PEDESTRIAN", "CYCLIST"}:
            relevant_vrus += relevant.astype(np.int64)
            vru_points.append(positions[track_valid & finite])
            vru_conflicts.append(
                closest_approach_conflict(
                    ego_positions,
                    positions,
                    valid,
                    timestamps,
                    horizon_s=conflict_horizon_s,
                    distance_threshold_m=vru_conflict_distance_m,
                    relevance_radius_m=relevant_radius_m,
                    vertical_tolerance_m=vertical_tolerance_m,
                )
            )

    min_vehicle_distance = None
    if vehicle_distances and any(len(values) for values in vehicle_distances):
        min_vehicle_distance = float(
            min(values.min() for values in vehicle_distances if len(values))
        )
    all_vru_points = np.concatenate(vru_points, axis=0) if vru_points else np.empty((0, 3))
    min_vru_distance = _minimum_distance(all_vru_points, ego_route)
    vru_interaction = bool(
        min_vru_distance is not None and min_vru_distance <= vru_max_distance_to_route_m
    )
    vehicle_conflict_count, vehicle_dcpa, vehicle_tcpa = _minimum_conflict(
        vehicle_conflicts
    )
    vru_conflict_count, vru_dcpa, vru_tcpa = _minimum_conflict(vru_conflicts)

    topology_confidence = "unknown"
    topology_evidence: tuple[str, ...] = ()
    if source == "waymo" and realized_generation_metadata is None:
        inferred = infer_waymo_topology(
            scenario,
            ego_route,
            lane_route_distance_m=waymo_lane_route_distance_m,
            control_route_distance_m=waymo_control_route_distance_m,
            vertical_tolerance_m=vertical_tolerance_m,
            merge_min_entry_lanes=waymo_merge_min_entry_lanes,
        )
        has_intersection = inferred.has_intersection
        has_merge = inferred.has_merge_or_roundabout
        route_light = inferred.has_route_traffic_light
        route_stop = inferred.has_route_stop_sign
        route_crosswalk = inferred.has_route_crosswalk
        signal_reliability = inferred.signal_reliability
        topology_confidence = inferred.confidence
        topology_evidence = inferred.evidence
        if has_intersection is True and has_merge is True:
            topology_tag = "mixed"
        elif has_intersection is True:
            topology_tag = "intersection"
        elif has_merge is True:
            topology_tag = "merge_or_roundabout"
        elif has_intersection is False and has_merge is False:
            topology_tag = "simple"
        else:
            topology_tag = "unknown"
    else:
        has_intersection, has_merge, topology_tag = _topology(
            realized_generation_metadata
        )
        route_light, route_stop, route_crosswalk, signal_reliability = _route_controls(
            scenario, realized_generation_metadata
        )
    scenario_id = str(scenario.get("id") or metadata.get("scenario_id") or "")
    if not scenario_id:
        raise ValueError("scenario has no id")

    return ScenarioFeatures(
        scenario_id=scenario_id,
        source=source,
        length=length,
        route_length_m=_route_length(ego_route),
        topology_tag=topology_tag,  # type: ignore[arg-type]
        has_intersection=has_intersection,
        has_merge_or_roundabout=has_merge,
        has_route_traffic_light=route_light,
        has_route_stop_sign=route_stop,
        has_route_crosswalk=route_crosswalk,
        signal_reliability=signal_reliability,  # type: ignore[arg-type]
        has_vehicle="VEHICLE" in present_types,
        has_pedestrian="PEDESTRIAN" in present_types,
        has_cyclist="CYCLIST" in present_types,
        relevant_agents_q90=float(np.quantile(relevant_agents, temporal_quantile)),
        relevant_vehicles_q90=float(np.quantile(relevant_vehicles, temporal_quantile)),
        min_vehicle_distance_m=min_vehicle_distance,
        min_vru_distance_to_route_m=min_vru_distance,
        low_traffic=False,
        dense_traffic=False,
        vru_interaction=vru_interaction,
        topology_confidence=topology_confidence,  # type: ignore[arg-type]
        topology_evidence=topology_evidence,
        relevant_vrus_q90=float(np.quantile(relevant_vrus, temporal_quantile)),
        vehicle_conflict_count=vehicle_conflict_count,
        vru_conflict_count=vru_conflict_count,
        min_vehicle_conflict_dcpa_m=vehicle_dcpa,
        min_vehicle_conflict_tcpa_s=vehicle_tcpa,
        min_vru_conflict_dcpa_m=vru_dcpa,
        min_vru_conflict_tcpa_s=vru_tcpa,
        sdc_valid_ratio=float(np.mean(ego_valid)),
        sdc_initial_valid=bool(ego_valid[0]),
        sdc_route_z_range_m=_route_z_range(ego_route),
        map_feature_count=len(map_features),
        dynamic_object_count=dynamic_object_count,
    )
