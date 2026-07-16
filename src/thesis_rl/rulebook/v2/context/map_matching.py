"""Offline-only SDC-to-lane map matching for static task-route creation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Mapping

from shapely.geometry import Point

from thesis_rl.rulebook.v2.context.task_route import build_task_route_record
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord, associate_route_lane
from thesis_rl.rulebook.v2.geometry.route import GEOMETRY_EPSILON_M
from thesis_rl.rulebook.v2.geometry.vertical import VERTICAL_COMPATIBILITY_TOLERANCE_M
from thesis_rl.rulebook.v2.types import TaskRouteRecord


@dataclass(frozen=True, slots=True)
class OfflineTrackSample:
    """Input accepted only by the offline matcher; never stored in the record."""

    position_xy: tuple[float, float]
    position_z: float
    heading_rad: float


class TaskRouteMapMatchError(ValueError):
    """Typed offline exclusion for a task route that cannot be recovered uniquely."""

    def __init__(self, validation_error: str, *, sample_index: int | None = None) -> None:
        self.validation_error = validation_error
        self.sample_index = sample_index
        detail = validation_error
        if sample_index is not None:
            detail += f" at sample {sample_index}"
        super().__init__(detail)


def reachable_lane_ids(
    *,
    route_lane_ids: tuple[str, ...],
    lane_successors: Mapping[str, tuple[str, ...]],
) -> frozenset[str]:
    """Return route lanes and every lane reachable through frozen topology."""

    reachable = set(route_lane_ids)
    pending = list(route_lane_ids)
    while pending:
        lane_id = pending.pop()
        for successor in lane_successors.get(lane_id, ()):
            if successor in reachable:
                continue
            reachable.add(successor)
            pending.append(successor)
    return frozenset(reachable)


def _is_canonical_lane_transition(
    *,
    sample: OfflineTrackSample,
    previous_lane: RouteLaneRecord,
    next_lane: RouteLaneRecord,
) -> bool:
    """Recognize one unresolved sample exactly on a contiguous lane boundary."""

    if previous_lane.lane_id == next_lane.lane_id:
        return False
    previous_end = previous_lane.centerline.points_xyz[-1]
    next_start = next_lane.centerline.points_xyz[0]
    if hypot(previous_end[0] - next_start[0], previous_end[1] - next_start[1]) > GEOMETRY_EPSILON_M:
        return False
    if abs(previous_end[2] - next_start[2]) > VERTICAL_COMPATIBILITY_TOLERANCE_M:
        return False
    for lane in (previous_lane, next_lane):
        if not lane.polygon_xy.buffer(GEOMETRY_EPSILON_M).covers(Point(sample.position_xy)):
            return False
        try:
            lane.centerline.project(sample.position_xy, position_z=sample.position_z)
        except ValueError:
            return False
    return True


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
        raise TaskRouteMapMatchError("task_route_track_empty")
    if not route_lanes:
        raise TaskRouteMapMatchError("task_route_lane_records_missing")
    associations = tuple(
        associate_route_lane(
            position_xy=sample.position_xy,
            position_z=sample.position_z,
            heading_rad=sample.heading_rad,
            route_lanes=tuple(route_lanes.values()),
        )
        for sample in track
    )
    lane_ids: list[str] = []
    for index, (sample, association) in enumerate(zip(track, associations)):
        if association is None:
            previous = associations[index - 1] if index > 0 else None
            following = associations[index + 1] if index + 1 < len(associations) else None
            if (
                previous is not None
                and following is not None
                and _is_canonical_lane_transition(
                    sample=sample,
                    previous_lane=route_lanes[previous.lane_id],
                    next_lane=route_lanes[following.lane_id],
                )
            ):
                continue
            raise TaskRouteMapMatchError(
                "task_route_lane_association_ambiguous_or_unavailable",
                sample_index=index,
            )
        if not lane_ids or lane_ids[-1] != association.lane_id:
            lane_ids.append(association.lane_id)
    return build_task_route_record(
        scenario_uid=scenario_uid,
        lane_ids=tuple(lane_ids),
        provenance="offline_sdc_map_match",
        adapter_version=adapter_version,
        source_geometry_bytes=source_geometry_bytes,
    )
