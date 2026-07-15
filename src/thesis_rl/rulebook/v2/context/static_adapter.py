"""Source-neutral normalization of static map records for F3 adapters."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from thesis_rl.rulebook.v2.geometry.canonical import canonicalize_geometry, stable_geometry_id
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.types import MapFeatureRecord, TaskRouteRecord, TrafficControlRecord


@dataclass(frozen=True, slots=True)
class StaticAdapterResult:
    scenario_uid: str
    task_route: TaskRouteRecord
    route_lanes: tuple[RouteLaneRecord, ...]
    map_features: dict[str, MapFeatureRecord]
    traffic_controls: tuple[TrafficControlRecord, ...]
    validation_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "map_features", MappingProxyType(dict(self.map_features)))


def normalize_static_records(
    *,
    scenario_uid: str,
    task_route: TaskRouteRecord,
    route_lanes: tuple[RouteLaneRecord, ...],
    map_features: tuple[MapFeatureRecord, ...],
    traffic_controls: tuple[TrafficControlRecord, ...],
) -> StaticAdapterResult:
    """Canonicalize adapter output and return typed validation errors, never fallbacks."""

    if not scenario_uid or task_route.scenario_uid != scenario_uid:
        raise ValueError("Static adapter scenario UID does not match task route")
    errors: list[str] = []
    route_lane_ids = {lane.lane_id for lane in route_lanes}
    missing = [lane_id for lane_id in task_route.lane_ids if lane_id not in route_lane_ids]
    if missing:
        errors.append("task_route_lane_missing:" + ",".join(missing))
    normalized_features: dict[str, MapFeatureRecord] = {}
    for feature in map_features:
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
    return StaticAdapterResult(
        scenario_uid=scenario_uid,
        task_route=task_route,
        route_lanes=route_lanes,
        map_features=normalized_features,
        traffic_controls=traffic_controls,
        validation_errors=tuple(errors),
    )
