"""Strict source-neutral hooks for offline PG/Waymo static adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from thesis_rl.rulebook.v2.context.static_adapter import StaticAdapterResult, normalize_static_records
from thesis_rl.rulebook.v2.types import MapFeatureRecord, TaskRouteRecord, TrafficControlRecord
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord


@dataclass(frozen=True, slots=True)
class StaticRecordSources:
    """Offline providers shared by PG and Waymo adapters."""

    task_route: Callable[[Any], TaskRouteRecord]
    route_lanes: Callable[[Any], tuple[RouteLaneRecord, ...]]
    map_features: Callable[[Any], tuple[MapFeatureRecord, ...]]
    traffic_controls: Callable[[Any], tuple[TrafficControlRecord, ...]]

    def __post_init__(self) -> None:
        if any(not callable(provider) for provider in (
            self.task_route, self.route_lanes, self.map_features, self.traffic_controls,
        )):
            raise TypeError("Every StaticRecordSources field must be callable")

    @classmethod
    def from_mapping(cls, providers: Mapping[str, Callable[[Any], object]]) -> "StaticRecordSources":
        required = tuple(cls.__dataclass_fields__)
        missing = tuple(name for name in required if name not in providers)
        unknown = tuple(sorted(set(providers).difference(required)))
        if missing:
            raise ValueError(f"Missing static record providers: {missing}")
        if unknown:
            raise ValueError(f"Unknown static record providers: {unknown}")
        return cls(**cast(Any, {name: providers[name] for name in required}))


class StaticRecordAdapter:
    """Normalize one offline source through the common static contract."""

    def __init__(self, sources: StaticRecordSources) -> None:
        self.sources = sources

    def normalize(self, source_record: Any, *, scenario_uid: str) -> StaticAdapterResult:
        sources = self.sources
        task_route = sources.task_route(source_record)
        if task_route.scenario_uid != scenario_uid:
            raise ValueError("Static source task route scenario UID mismatch")
        return normalize_static_records(
            scenario_uid=scenario_uid,
            task_route=task_route,
            route_lanes=sources.route_lanes(source_record),
            map_features=sources.map_features(source_record),
            traffic_controls=sources.traffic_controls(source_record),
        )
