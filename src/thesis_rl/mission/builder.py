"""Deterministic offline construction of mission sections and directed gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from thesis_rl.mission.gates import GateGeometry
from thesis_rl.mission.types import DirectedGate, DrivingMissionRecord, LaneSpan, MissionSection
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.geometry.lanes import associate_route_lane


@dataclass(frozen=True, slots=True)
class NormalizedLane:
    lane_id: str
    length_m: float
    successor_lane_ids: tuple[str, ...] = ()
    lateral_lane_ids: tuple[str, ...] = ()
    centerline: RoutePolyline | None = None

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
        )
    valid = state.get("valid", ())
    terminal_index = max((index for index, value in enumerate(valid) if bool(value)), default=-1)
    if terminal_index < 0:
        raise ValueError("SDC trajectory has no valid terminal pose")
    position = state["position"][terminal_index]
    heading = float(state["heading"][terminal_index])
    association = associate_route_lane(
        position_xy=(float(position[0]), float(position[1])),
        position_z=(float(position[2]) if len(position) > 2 else 0.0) - z_origin,
        heading_rad=heading,
        route_lanes=tuple(lane_map[lane_id] for lane_id in assigned_route_lane_ids),
    )
    if association is None:
        raise ValueError("terminal SDC pose cannot be projected to the assigned route")
    return build_driving_mission(
        scenario_uid,
        assigned_route_lane_ids,
        lanes,
        final_goal_lane_id=association.lane_id,
        final_goal_s_m=association.route_projection_s_m,
    )
