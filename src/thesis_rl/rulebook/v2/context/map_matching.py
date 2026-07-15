"""Offline-only SDC-to-lane map matching for static task-route creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from thesis_rl.rulebook.v2.context.task_route import build_task_route_record
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord, associate_route_lane
from thesis_rl.rulebook.v2.types import TaskRouteRecord


@dataclass(frozen=True, slots=True)
class OfflineTrackSample:
    """Input accepted only by the offline matcher; never stored in the record."""

    position_xy: tuple[float, float]
    position_z: float
    heading_rad: float


def map_match_sdc_track_to_task_route(
    *,
    scenario_uid: str,
    track: tuple[OfflineTrackSample, ...],
    route_lanes: Mapping[str, RouteLaneRecord],
    source_geometry_bytes: bytes,
    adapter_version: str,
) -> TaskRouteRecord:
    """Map-match an offline SDC track and retain only its lane topology."""

    if not track:
        raise ValueError("Offline SDC track must contain at least one sample")
    if not route_lanes:
        raise ValueError("Offline map matcher requires route lane records")
    lane_ids: list[str] = []
    for sample in track:
        association = associate_route_lane(
            position_xy=sample.position_xy,
            position_z=sample.position_z,
            heading_rad=sample.heading_rad,
            route_lanes=tuple(route_lanes.values()),
        )
        if association is None:
            raise ValueError("SDC track lane association is ambiguous or unavailable")
        if not lane_ids or lane_ids[-1] != association.lane_id:
            lane_ids.append(association.lane_id)
    return build_task_route_record(
        scenario_uid=scenario_uid,
        lane_ids=tuple(lane_ids),
        provenance="offline_sdc_map_match",
        adapter_version=adapter_version,
        source_geometry_bytes=source_geometry_bytes,
    )
