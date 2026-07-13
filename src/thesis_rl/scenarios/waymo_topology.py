"""Offline, route-aware topology evidence for converted Waymo scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class WaymoTopologyEvidence:
    has_intersection: bool | None
    has_merge_or_roundabout: bool | None
    has_route_traffic_light: bool | None
    has_route_stop_sign: bool | None
    has_route_crosswalk: bool | None
    signal_reliability: str
    confidence: str
    evidence: tuple[str, ...]


class _RouteSpatialIndex:
    """Small dependency-free grid index for repeated route proximity queries."""

    def __init__(self, route: np.ndarray, *, cell_size_m: float) -> None:
        finite = route[np.isfinite(route).all(axis=1)]
        self._cell_size_m = cell_size_m
        cells: dict[tuple[int, int], list[np.ndarray]] = {}
        for point in finite:
            cells.setdefault(self._cell(point), []).append(point)
        self._cells = {
            key: np.asarray(values, dtype=np.float64) for key, values in cells.items()
        }

    def _cell(self, point: np.ndarray) -> tuple[int, int]:
        indices = np.floor(point[:2] / self._cell_size_m).astype(np.int64)
        return int(indices[0]), int(indices[1])

    def near(
        self,
        points: np.ndarray,
        *,
        radius_m: float,
        vertical_tolerance_m: float,
    ) -> bool:
        finite_points = points[np.isfinite(points).all(axis=1)]
        span = int(ceil(radius_m / self._cell_size_m))
        for point in finite_points:
            cell_x, cell_y = self._cell(point)
            candidates = [
                self._cells[(cell_x + dx, cell_y + dy)]
                for dx in range(-span, span + 1)
                for dy in range(-span, span + 1)
                if (cell_x + dx, cell_y + dy) in self._cells
            ]
            if not candidates:
                continue
            route_points = np.concatenate(candidates, axis=0)
            planar = np.linalg.norm(route_points[:, :2] - point[:2], axis=1)
            vertical = np.abs(route_points[:, 2] - point[2])
            if np.any((planar <= radius_m) & (vertical <= vertical_tolerance_m)):
                return True
        return False


def _points(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1 and array.size >= 2:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 2:
        return np.empty((0, 3), dtype=np.float64)
    if array.shape[1] == 2:
        array = np.pad(array, ((0, 0), (0, 1)))
    return array[:, :3]


def _feature_points(feature: Mapping[str, Any]) -> np.ndarray:
    for key in ("polyline", "polygon", "position", "stop_point"):
        if key in feature:
            points = _points(feature[key])
            if len(points):
                return points
    return np.empty((0, 3), dtype=np.float64)


def _lane_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (str, int, np.integer)):
        return {str(value)}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {str(item) for item in value}
    return set()


def infer_waymo_topology(
    scenario: Mapping[str, Any],
    ego_route: np.ndarray,
    *,
    lane_route_distance_m: float = 6.0,
    control_route_distance_m: float = 15.0,
    vertical_tolerance_m: float = 3.0,
    merge_min_entry_lanes: int = 2,
) -> WaymoTopologyEvidence:
    """Infer conservative topology from converted map data near the ego route.

    The result is an offline catalog label. It is never exposed as future state
    to the online policy.
    """

    if lane_route_distance_m <= 0 or control_route_distance_m <= 0:
        raise ValueError("route-aware topology distances must be positive")
    if vertical_tolerance_m <= 0 or merge_min_entry_lanes < 2:
        raise ValueError("invalid topology evidence thresholds")
    map_features = scenario.get("map_features", {})
    dynamic_states = scenario.get("dynamic_map_states", {})
    if not isinstance(map_features, Mapping) or not isinstance(dynamic_states, Mapping):
        raise ValueError("map_features and dynamic_map_states must be mappings")
    route_index = _RouteSpatialIndex(
        ego_route, cell_size_m=min(lane_route_distance_m, control_route_distance_m)
    )

    lanes: dict[str, Mapping[str, Any]] = {
        str(feature_id): feature
        for feature_id, feature in map_features.items()
        if isinstance(feature, Mapping)
        and str(feature.get("type", "")).startswith("LANE_")
    }
    route_lanes = {
        lane_id: lane
        for lane_id, lane in lanes.items()
        if route_index.near(
            _feature_points(lane),
            radius_m=lane_route_distance_m,
            vertical_tolerance_m=vertical_tolerance_m,
        )
    }
    route_lane_ids = set(route_lanes)
    evidence: list[str] = []
    if route_lanes:
        evidence.append("route_lane_match")

    merge_lanes = [
        lane_id
        for lane_id, lane in route_lanes.items()
        if len(_lane_ids(lane.get("entry_lanes"))) >= merge_min_entry_lanes
    ]
    has_merge: bool | None
    if merge_lanes:
        has_merge = True
        evidence.append("converging_route_lanes")
    elif route_lanes:
        has_merge = False
    else:
        has_merge = None

    route_crosswalk = False
    route_stop = False
    for feature in map_features.values():
        if not isinstance(feature, Mapping):
            continue
        feature_type = str(feature.get("type", ""))
        linked_lanes = _lane_ids(feature.get("lane"))
        linked_to_route = bool(linked_lanes & route_lane_ids)
        geometrically_near = route_index.near(
            _feature_points(feature),
            radius_m=control_route_distance_m,
            vertical_tolerance_m=vertical_tolerance_m,
        )
        if feature_type == "CROSSWALK" and geometrically_near:
            route_crosswalk = True
        elif feature_type == "STOP_SIGN" and (linked_to_route or geometrically_near):
            route_stop = True

    route_light = False
    has_any_light = False
    light_has_known_state = False
    light_has_unknown_state = False
    light_sequence_invalid = False
    for dynamic in dynamic_states.values():
        if not isinstance(dynamic, Mapping) or dynamic.get("type") != "TRAFFIC_LIGHT":
            continue
        has_any_light = True
        linked_to_route = bool(_lane_ids(dynamic.get("lane")) & route_lane_ids)
        geometrically_near = route_index.near(
            _feature_points(dynamic),
            radius_m=control_route_distance_m,
            vertical_tolerance_m=vertical_tolerance_m,
        )
        if not (linked_to_route or geometrically_near):
            continue
        route_light = True
        state = dynamic.get("state", {})
        object_states = state.get("object_state") if isinstance(state, Mapping) else None
        if not isinstance(object_states, (list, tuple, np.ndarray)):
            light_sequence_invalid = True
        elif len(object_states) != int(scenario.get("length", 0)):
            light_sequence_invalid = True
        else:
            normalized_states = [str(value) for value in object_states]
            light_has_unknown_state |= any(
                value == "LANE_STATE_UNKNOWN" for value in normalized_states
            )
            light_has_known_state |= any(
                value != "LANE_STATE_UNKNOWN" for value in normalized_states
            )

    if route_light:
        evidence.append("route_traffic_light")
    if route_stop:
        evidence.append("route_stop_sign")
    if route_crosswalk:
        evidence.append("route_crosswalk")
    has_intersection: bool | None
    if route_light or route_stop or route_crosswalk:
        has_intersection = True
    elif route_lanes:
        has_intersection = False
    else:
        has_intersection = None

    # Multiple incoming lanes are also common inside an intersection. Treat
    # route-local controls as the stronger, mutually exclusive explanation;
    # otherwise almost every controlled junction becomes spuriously "mixed".
    if has_intersection is True and has_merge is True:
        has_merge = False
        evidence.remove("converging_route_lanes")
        evidence.append("convergence_at_controlled_junction")

    if route_light:
        if not light_has_known_state:
            reliability = "missing"
        elif light_sequence_invalid or light_has_unknown_state:
            reliability = "partial"
        else:
            reliability = "complete"
        route_light_value: bool | None = True
    elif has_any_light and not route_lanes:
        reliability = "partial"
        route_light_value = None
    else:
        reliability = "not_applicable"
        route_light_value = False
    positive_evidence = has_intersection is True or has_merge is True
    if not route_lanes:
        confidence = "unknown"
    elif positive_evidence:
        confidence = "high" if len(evidence) >= 2 else "medium"
    else:
        confidence = "medium"
    return WaymoTopologyEvidence(
        has_intersection=has_intersection,
        has_merge_or_roundabout=has_merge,
        has_route_traffic_light=route_light_value,
        has_route_stop_sign=route_stop if route_lanes or route_stop else None,
        has_route_crosswalk=(
            route_crosswalk if route_lanes or route_crosswalk else None
        ),
        signal_reliability=reliability,
        confidence=confidence,
        evidence=tuple(evidence),
    )


__all__ = ["WaymoTopologyEvidence", "infer_waymo_topology"]
