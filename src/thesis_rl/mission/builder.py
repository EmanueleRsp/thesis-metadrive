"""Deterministic offline construction of mission sections and directed gates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from shapely.geometry import LineString, Point

from thesis_rl.mission.gates import GateGeometry
from thesis_rl.mission.types import (
    DirectedGate,
    DrivingMissionRecord,
    FinalGateSegment,
    LaneSpan,
    MissionSection,
)
from thesis_rl.rulebook.v2.geometry.route import GEOMETRY_EPSILON_M, RoutePolyline


FINAL_GATE_BUILDER_EPSILON_M = 0.01


@dataclass(frozen=True, slots=True)
class NormalizedLane:
    lane_id: str
    length_m: float
    successor_lane_ids: tuple[str, ...] = ()
    lateral_lane_ids: tuple[str, ...] = ()
    centerline: RoutePolyline | None = None
    polygon_xy: Any | None = None

    def __post_init__(self) -> None:
        if not self.lane_id or self.length_m <= 0.0:
            raise ValueError("normalized lane ID and positive length are required")
        if self.centerline is not None and abs(self.centerline.length_m - self.length_m) > 1.0e-6:
            raise ValueError("normalized lane centerline length must match length_m")

    def gate_geometry_at(self, s_m: float) -> GateGeometry | None:
        if self.centerline is None:
            return None
        point = self.centerline.point_at(s_m)
        projection = self.centerline.project((point[0], point[1]), position_z=point[2])
        tangent = projection.tangent_xy
        normal = (-tangent[1], tangent[0])
        return GateGeometry(
            (
                (point[0] - normal[0] * 100.0, point[1] - normal[1] * 100.0),
                (point[0] + normal[0] * 100.0, point[1] + normal[1] * 100.0),
            ),
            tangent,
            point[2],
        )


def _reachable_to(target_lane_id: str, lanes: Mapping[str, NormalizedLane]) -> frozenset[str]:
    reverse: dict[str, set[str]] = {}
    for lane in lanes.values():
        for successor in lane.successor_lane_ids:
            reverse.setdefault(successor, set()).add(lane.lane_id)
    reachable = {target_lane_id}
    pending = [target_lane_id]
    while pending:
        current = pending.pop()
        for predecessor in sorted(reverse.get(current, ())):
            if predecessor not in reachable:
                reachable.add(predecessor)
                pending.append(predecessor)
    return frozenset(reachable)


def build_driving_mission(
    scenario_uid: str,
    preferred_lane_ids: tuple[str, ...],
    lanes: Mapping[str, NormalizedLane],
    *,
    final_goal_lane_id: str,
    final_goal_s_m: float,
    builder_version: str = "mission-builder-v1",
) -> DrivingMissionRecord:
    """Build sections at mandatory branch boundaries; never infer a new route."""
    if not preferred_lane_ids or any(lane_id not in lanes for lane_id in preferred_lane_ids):
        raise ValueError("preferred route must contain known lanes")
    if final_goal_lane_id not in lanes:
        raise ValueError("final goal lane must be known")
    if not 0.0 <= final_goal_s_m <= lanes[final_goal_lane_id].length_m:
        raise ValueError("final goal must lie on its lane")
    for current, following in zip(preferred_lane_ids, preferred_lane_ids[1:]):
        if following not in lanes[current].successor_lane_ids:
            raise ValueError(f"preferred route is not contiguous: {current}->{following}")
    sections: list[MissionSection] = []
    section_start = 0
    for index, lane_id in enumerate(preferred_lane_ids[:-1]):
        if len(lanes[lane_id].successor_lane_ids) == 1:
            continue
        target = lanes[preferred_lane_ids[index + 1]]
        reachable = _reachable_to(target.lane_id, lanes)
        allowed_ids = set(preferred_lane_ids[section_start : index + 1])
        for preferred in preferred_lane_ids[section_start : index + 1]:
            allowed_ids.update(
                neighbor for neighbor in lanes[preferred].lateral_lane_ids if neighbor in reachable
            )
        allowed = tuple(LaneSpan(lane, 0.0, lanes[lane].length_m) for lane in sorted(allowed_ids))
        preferred_span = LaneSpan(
            preferred_lane_ids[section_start],
            0.0,
            lanes[preferred_lane_ids[section_start]].length_m,
        )
        gate_span = LaneSpan(lane_id, 0.0, lanes[lane_id].length_m)
        gate = DirectedGate(
            f"gate:{index}:{lane_id}",
            (gate_span,),
            lane_id,
            lanes[lane_id].length_m,
            lanes[lane_id].gate_geometry_at(lanes[lane_id].length_m),
        )
        sections.append(MissionSection(f"section:{len(sections)}", preferred_span, allowed, gate))
        section_start = index + 1
    final_span = LaneSpan(final_goal_lane_id, 0.0, lanes[final_goal_lane_id].length_m)
    final_goal = DirectedGate(
        "goal:final",
        (final_span,),
        final_goal_lane_id,
        final_goal_s_m,
        lanes[final_goal_lane_id].gate_geometry_at(final_goal_s_m),
    )
    if not sections:
        preferred = preferred_lane_ids[0]
        span = LaneSpan(preferred, 0.0, lanes[preferred].length_m)
        sections.append(
            MissionSection(
                "section:0",
                span,
                tuple(LaneSpan(lane, 0.0, lanes[lane].length_m) for lane in preferred_lane_ids),
                final_goal,
            )
        )
    return DrivingMissionRecord(scenario_uid, builder_version, tuple(sections), final_goal)


def build_driving_mission_from_source(
    scenario: Mapping[str, Any],
    *,
    scenario_uid: str,
    source: str,
    assigned_route_lane_ids: tuple[str, ...],
) -> DrivingMissionRecord:
    """Materialize one source-neutral immutable mission from static source data only."""
    if source == "pg":
        from thesis_rl.rulebook.v2.context.pg_static_adapter import _lane_record as make_lane
    elif source == "waymo":
        from thesis_rl.rulebook.v2.context.waymo_static_adapter import _lane_record as make_lane
    else:
        raise ValueError(f"unsupported mission source: {source!r}")
    metadata = scenario.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("scenario metadata is required")
    metadata_copy = dict(metadata)
    features = scenario.get("map_features")
    if not isinstance(features, Mapping):
        raise ValueError("scenario map_features are required")
    tracks = scenario.get("tracks")
    if not isinstance(tracks, Mapping):
        raise ValueError("scenario tracks are required")
    sdc_track = tracks.get(metadata_copy.get("sdc_id"))
    state = sdc_track.get("state") if isinstance(sdc_track, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("SDC state is required")
    positions = state.get("position")
    try:
        has_positions = len(positions) > 0
    except TypeError:
        has_positions = False
    if not has_positions:
        raise ValueError("SDC positions are required")
    z_origin = float(positions[0][2]) if len(positions[0]) > 2 else 0.0
    lane_map = {}
    for lane_id, raw in features.items():
        if not isinstance(raw, Mapping) or not str(raw.get("type", "")).startswith("LANE_"):
            continue
        try:
            lane_map[str(lane_id)] = make_lane(str(lane_id), raw, z_origin_m=z_origin)
        except ValueError:
            continue
    lanes: dict[str, NormalizedLane] = {}
    for lane_id, lane in lane_map.items():
        raw = features.get(lane_id, {})
        neighbors: list[str] = []
        if isinstance(raw, Mapping):
            for key in ("left_neighbor", "right_neighbor"):
                values = raw.get(key, ())
                if not isinstance(values, (list, tuple)):
                    values = (values,)
                for value in values:
                    neighbor_id = value.get("feature_id") if isinstance(value, Mapping) else value
                    if str(neighbor_id) in lane_map:
                        neighbors.append(str(neighbor_id))
        lanes[lane_id] = NormalizedLane(
            lane_id,
            lane.centerline.length_m,
            lane.successor_lane_ids,
            tuple(sorted(set(neighbors))),
            lane.centerline,
            lane.polygon_xy,
        )
    valid = state.get("valid", ())
    reset_index = next((index for index, value in enumerate(valid) if bool(value)), -1)
    if reset_index < 0:
        raise ValueError("SDC trajectory has no valid reset pose")
    reset_position = state["position"][reset_index]
    reset_xy = (float(reset_position[0]), float(reset_position[1]))
    reset_z = (float(reset_position[2]) if len(reset_position) > 2 else 0.0) - z_origin
    assigned_route_lane_ids = _repair_reset_route(
        tuple(str(value) for value in assigned_route_lane_ids),
        lanes,
        reset_xy,
        reset_z,
    )
    terminal_index = max((index for index, value in enumerate(valid) if bool(value)), default=-1)
    if terminal_index < 0:
        raise ValueError("SDC trajectory has no valid terminal pose")
    position = state["position"][terminal_index]
    heading = float(state["heading"][terminal_index])
    del heading
    return _build_source_route_mission(
        scenario_uid,
        source,
        assigned_route_lane_ids,
        lanes,
        reset_xy=reset_xy,
        reset_z=reset_z,
        terminal_position=(float(position[0]), float(position[1])),
        terminal_z=(float(position[2]) if len(position) > 2 else 0.0) - z_origin,
    )


def _line_parts(geometry: Any) -> tuple[Any, ...]:
    if geometry.is_empty:
        return ()
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return (geometry,)
    if hasattr(geometry, "geoms"):
        return tuple(part for item in geometry.geoms for part in _line_parts(item))
    return ()


def _build_source_route_mission(
    scenario_uid: str,
    source: str,
    route_lane_ids: tuple[str, ...],
    lanes: Mapping[str, NormalizedLane],
    *,
    reset_xy: tuple[float, float],
    reset_z: float,
    terminal_position: tuple[float, float],
    terminal_z: float,
) -> DrivingMissionRecord:
    """Build the v1.1 normalized record and its frozen anchor-based gate."""
    route_lanes = tuple(lanes[lane_id] for lane_id in route_lane_ids)
    if any(lane.centerline is None or lane.polygon_xy is None for lane in route_lanes):
        raise ValueError("source route lanes require centerline and polygon geometry")
    canonical = RoutePolyline.from_lane_centerlines(
        tuple(lane.centerline.points_xyz for lane in route_lanes if lane.centerline is not None)
    )
    reset_projection = canonical.project(reset_xy, position_z=reset_z)
    final_projection = route_lanes[-1].centerline.project(terminal_position, position_z=terminal_z)
    final_prefix_m = sum(lane.length_m for lane in route_lanes[:-1])
    raw_goal_m = final_prefix_m + final_projection.s_m
    s_goal_m = raw_goal_m - reset_projection.s_m
    if s_goal_m <= 0.0:
        raise ValueError("normalized route has non-positive goal station")
    goal_xyz = canonical.point_at(raw_goal_m)
    goal = canonical.project(goal_xyz[:2], position_z=goal_xyz[2])
    tangent = goal.tangent_xy
    normal = (-tangent[1], tangent[0])
    cross = LineString(
        (
            (goal_xyz[0] - 100.0 * normal[0], goal_xyz[1] - 100.0 * normal[1]),
            (goal_xyz[0] + 100.0 * normal[0], goal_xyz[1] + 100.0 * normal[1]),
        )
    )
    intervals: list[tuple[float, float, str]] = []
    final_intervals: list[tuple[float, float, str]] = []
    for lane in lanes.values():
        if lane.centerline is None or lane.polygon_xy is None:
            continue
        try:
            projection = lane.centerline.project(goal_xyz[:2], position_z=goal_xyz[2])
        except ValueError as error:
            if "vertically compatible" in str(error):
                continue
            raise
        if projection.tangent_xy[0] * tangent[0] + projection.tangent_xy[1] * tangent[1] <= 0.0:
            continue
        for part in _line_parts(cross.intersection(lane.polygon_xy)):
            first, last = part.coords[0], part.coords[-1]
            lo = cross.project(Point(first)) - 100.0
            hi = cross.project(Point(last)) - 100.0
            lo, hi = min(lo, hi), max(lo, hi)
            if hi - lo <= FINAL_GATE_BUILDER_EPSILON_M:
                continue
            item = (lo, hi, lane.lane_id)
            intervals.append(item)
            if lane.lane_id == route_lane_ids[-1]:
                final_intervals.append(item)
    merged: list[list[Any]] = []
    for lo, hi, lane_id in sorted(intervals):
        if not merged or lo > merged[-1][1] + FINAL_GATE_BUILDER_EPSILON_M:
            merged.append([lo, hi, [lane_id]])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
            merged[-1][2].append(lane_id)
    containing = [
        item
        for item in merged
        if item[0] - FINAL_GATE_BUILDER_EPSILON_M <= 0.0 <= item[1] + FINAL_GATE_BUILDER_EPSILON_M
    ]
    if len(containing) != 1:
        raise ValueError(f"final gate anchor component count is {len(containing)}")
    selected = containing[0]
    if not any(
        selected[0] - FINAL_GATE_BUILDER_EPSILON_M <= lo
        and hi <= selected[1] + FINAL_GATE_BUILDER_EPSILON_M
        for lo, hi, _ in final_intervals
    ):
        raise ValueError("final occurrence is inconsistent with the anchor gate")
    source_geometry_hash = hashlib.sha256(
        json.dumps(
            {"route": route_lane_ids, "points": canonical.points_xyz},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    gate = FinalGateSegment(
        (
            (goal_xyz[0] + selected[0] * normal[0], goal_xyz[1] + selected[0] * normal[1]),
            (goal_xyz[0] + selected[1] * normal[0], goal_xyz[1] + selected[1] * normal[1]),
        ),
        tangent,
        goal_xyz[2],
        route_lane_ids[-1],
        f"offline:{source}:anchor_cross_section",
        source_geometry_hash,
        "driving-mission-v1.1-anchor-builder",
    )
    legacy_span = LaneSpan(route_lane_ids[-1], 0.0, route_lanes[-1].length_m)
    legacy_goal = DirectedGate(
        "goal:final", (legacy_span,), route_lane_ids[-1], final_projection.s_m, None
    )
    return DrivingMissionRecord(
        scenario_uid,
        "mission-builder-v1.1",
        (MissionSection("section:route", legacy_span, (legacy_span,), legacy_goal),),
        legacy_goal,
        route_lane_ids=route_lane_ids,
        canonical_route_points_xyz=canonical.points_xyz,
        start_occurrence_id=route_lane_ids[0],
        final_occurrence_id=route_lane_ids[-1],
        s_start_m=0.0,
        s_goal_m=s_goal_m,
        final_gate_segment=gate,
    )


def _repair_reset_route(
    route_lane_ids: tuple[str, ...],
    lanes: Mapping[str, NormalizedLane],
    reset_xy: tuple[float, float],
    reset_z: float,
) -> tuple[str, ...]:
    """Apply the approved offline correction-first reset normalization."""
    candidates: list[int] = []
    for index, lane_id in enumerate(route_lane_ids):
        lane = lanes[lane_id]
        if lane.polygon_xy is None or not lane.polygon_xy.buffer(GEOMETRY_EPSILON_M).covers(Point(reset_xy)):
            continue
        try:
            projection = lane.centerline.project(reset_xy, position_z=reset_z)
        except ValueError:
            continue
        if index == 0 or projection.s_m <= GEOMETRY_EPSILON_M:
            candidates.append(index)
    if not candidates:
        raise ValueError("reset pose is not contained by the first occurrence or shared boundary")
    if len(set(candidates)) != 1:
        raise ValueError("reset pose has multiple offline route-occurrence associations")
    return route_lane_ids[candidates[0] :]
