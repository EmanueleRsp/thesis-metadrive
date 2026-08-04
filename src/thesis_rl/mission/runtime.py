"""Causal reset/transition integration for one frozen driving mission."""

from __future__ import annotations

from dataclasses import replace
from math import hypot
from typing import Iterable

from shapely.geometry import LineString
from shapely.ops import unary_union

from thesis_rl.mission.distance import LaneGraph
from thesis_rl.mission.gates import GateGeometry
from thesis_rl.mission.tracker import MissionTracker, RouteCoordinateMissionTracker
from thesis_rl.mission.types import (
    DirectedGate,
    DrivingMissionRecord,
    MissionSnapshot,
    ordered_mission_gates,
)
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord, associate_route_lane
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import EnvSnapshot


GATE_ROAD_ENVELOPE_MARGIN_M = 0.5


def _materialize_gate(
    gate: DirectedGate, lanes: dict[str, RouteLaneRecord]
) -> DirectedGate:
    lane = lanes.get(gate.lane_id)
    if lane is None:
        raise ValueError(f"mission gate references unknown live lane: {gate.lane_id}")
    point = lane.centerline.point_at(gate.s_m)
    projection = lane.centerline.project((point[0], point[1]), position_z=point[2])
    tangent = projection.tangent_xy
    normal = (-tangent[1], tangent[0])
    envelope = unary_union(
        [candidate.polygon_xy for candidate in _lateral_road_envelope(gate.lane_id, lanes)]
    ).buffer(GATE_ROAD_ENVELOPE_MARGIN_M)
    min_x, min_y, max_x, max_y = envelope.bounds
    probe_half_width = max(
        hypot(min_x - point[0], min_y - point[1]),
        hypot(max_x - point[0], max_y - point[1]),
    ) + GATE_ROAD_ENVELOPE_MARGIN_M
    probe = LineString(
        (
            (point[0] - normal[0] * probe_half_width, point[1] - normal[1] * probe_half_width),
            (point[0] + normal[0] * probe_half_width, point[1] + normal[1] * probe_half_width),
        )
    )
    cross_section = probe.intersection(envelope)
    coordinates = _line_coordinates(cross_section)
    if len(coordinates) < 2:
        raise ValueError(f"mission gate cannot derive road envelope: {gate.gate_id}")
    offsets = tuple(
        (x - point[0]) * normal[0] + (y - point[1]) * normal[1] for x, y in coordinates
    )
    geometry = GateGeometry(
        (
            (point[0] + normal[0] * min(offsets), point[1] + normal[1] * min(offsets)),
            (point[0] + normal[0] * max(offsets), point[1] + normal[1] * max(offsets)),
        ),
        tangent,
        point[2],
    )
    return replace(gate, geometry=geometry)


def _lateral_road_envelope(
    gate_lane_id: str, lanes: dict[str, RouteLaneRecord]
) -> tuple[RouteLaneRecord, ...]:
    """Return the declared lateral carriageway component of a gate lane.

    This intentionally relies on source-map adjacency rather than proximity:
    a crossing road or a nearby parallel service road is not part of a gate
    merely because it lies close to the same transverse line.
    """

    adjacent: dict[str, set[str]] = {lane_id: set() for lane_id in lanes}
    for lane in lanes.values():
        for neighbor in lane.lateral_lane_ids:
            if neighbor in lanes:
                adjacent[lane.lane_id].add(neighbor)
                adjacent[neighbor].add(lane.lane_id)
    selected = {gate_lane_id}
    pending = [gate_lane_id]
    while pending:
        lane_id = pending.pop()
        for neighbor in sorted(adjacent[lane_id]):
            if neighbor not in selected:
                selected.add(neighbor)
                pending.append(neighbor)
    return tuple(lanes[lane_id] for lane_id in sorted(selected))


def _line_coordinates(geometry) -> tuple[tuple[float, float], ...]:
    if geometry.is_empty:
        return ()
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return tuple((float(x), float(y)) for x, y, *_ in geometry.coords)
    if hasattr(geometry, "geoms"):
        return tuple(
            coordinate
            for part in geometry.geoms
            for coordinate in _line_coordinates(part)
        )
    return ()


def _lanes_reaching_any_gate(
    lanes: dict[str, RouteLaneRecord], gate_lane_ids: set[str]
) -> frozenset[str]:
    """Return live lanes from which a pending mission gate can be reached.

    Section spans name the task's admissible surfaces, while the static map
    topology also contains necessary connector lanes between two ordered gate
    frontiers.  Those connectors must remain available to the recovery graph;
    disconnected map lanes must not become association candidates merely
    because they exist in the source map.
    """

    predecessors: dict[str, set[str]] = {}
    for lane in lanes.values():
        for successor in lane.successor_lane_ids:
            if successor in lanes:
                predecessors.setdefault(successor, set()).add(lane.lane_id)
    reachable = set(gate_lane_ids)
    pending = list(sorted(gate_lane_ids))
    while pending:
        current = pending.pop()
        for predecessor in sorted(predecessors.get(current, ())):
            if predecessor not in reachable:
                reachable.add(predecessor)
                pending.append(predecessor)
    return frozenset(reachable)


class MissionRuntime:
    """Own a tracker and its static live-map realization for one episode."""

    def __init__(
        self,
        mission: DrivingMissionRecord,
        route_lanes: Iterable[RouteLaneRecord],
        initial_snapshot: EnvSnapshot,
    ) -> None:
        all_lanes = tuple(route_lanes)
        all_by_id = {lane.lane_id: lane for lane in all_lanes}
        if len(all_by_id) != len(all_lanes):
            raise ValueError("mission runtime requires unique live lane IDs")
        if mission.route_lane_ids:
            missing = sorted(set(mission.route_lane_ids).difference(all_by_id))
            if missing:
                raise ValueError(f"mission route references unavailable lanes: {missing[:5]}")
            route = RoutePolyline(tuple(mission.canonical_route_points_xyz))
            initial_route_station = _initial_route_station(mission, all_by_id, initial_snapshot)
            self._route_lanes = tuple(all_lanes)
            self._tracker = RouteCoordinateMissionTracker(
                mission, route, 0.0, route_offset_m=initial_route_station
            )
            return
        referenced_lane_ids = {
            span.lane_id for section in mission.sections for span in section.allowed_spans
        }
        referenced_lane_ids.update(
            span.lane_id
            for gate in (*[section.exit_gate for section in mission.sections], mission.final_goal)
            for span in gate.compatible_spans
        )
        missing = sorted(referenced_lane_ids.difference(all_by_id))
        if missing:
            raise ValueError(f"mission references unavailable live lanes: {missing[:5]}")
        gates_to_materialize = ordered_mission_gates(mission)
        gates = tuple(_materialize_gate(gate, all_by_id) for gate in gates_to_materialize)
        graph = LaneGraph(
            lengths_m={lane.lane_id: lane.centerline.length_m for lane in all_lanes},
            successors={
                lane.lane_id: tuple(
                    successor for successor in lane.successor_lane_ids if successor in all_by_id
                )
                for lane in all_lanes
            },
        )
        traversable_lane_ids = _lanes_reaching_any_gate(
            all_by_id, {gate.lane_id for gate in gates_to_materialize}
        )
        association_lanes = tuple(
            lane for lane in all_lanes if lane.lane_id in traversable_lane_ids
        )
        association = _associate(initial_snapshot, association_lanes)
        self._route_lanes = association_lanes
        self._tracker = MissionTracker(
            mission,
            graph,
            "" if association is None else association.lane_id,
            0.0 if association is None else association.route_projection_s_m,
            materialized_gates=gates,
            initial_footprint=initial_snapshot.ego.footprint,
            initial_heading_rad=initial_snapshot.ego.heading_rad,
        )

    @property
    def snapshot(self) -> MissionSnapshot:
        return self._tracker.snapshot()

    @property
    def gates(self) -> tuple[DirectedGate, ...]:
        """Return materialized ordered gates for read-only diagnostics."""

        return self._tracker.gates

    def update(self, pre: EnvSnapshot, post: EnvSnapshot) -> MissionSnapshot:
        if isinstance(self._tracker, RouteCoordinateMissionTracker):
            pre_projection = self._tracker.project(pre.ego.position_xy, pre.ego.position_z)
            post_projection = self._tracker.project(post.ego.position_xy, post.ego.position_z)
            return self._tracker.update(
                pre_projection.s_m,
                post_projection.s_m,
                pre_footprint=pre.ego.footprint,
                post_footprint=post.ego.footprint,
                pre_heading_rad=pre.ego.heading_rad,
                post_heading_rad=post.ego.heading_rad,
                post_ego_z_m=post.ego.position_z,
            )
        pre_association = _associate(pre, self._route_lanes)
        post_association = _associate(post, self._route_lanes)
        return self._tracker.update(
            "" if pre_association is None else pre_association.lane_id,
            0.0 if pre_association is None else pre_association.route_projection_s_m,
            "" if post_association is None else post_association.lane_id,
            0.0 if post_association is None else post_association.route_projection_s_m,
            pre_footprint=pre.ego.footprint,
            post_footprint=post.ego.footprint,
            pre_heading_rad=pre.ego.heading_rad,
            post_heading_rad=post.ego.heading_rad,
            post_ego_z_m=post.ego.position_z,
        )


def _associate(snapshot: EnvSnapshot, route_lanes: tuple[RouteLaneRecord, ...]):
    return associate_route_lane(
        position_xy=snapshot.ego.position_xy,
        position_z=snapshot.ego.position_z,
        heading_rad=snapshot.ego.heading_rad,
        route_lanes=route_lanes,
    )


def _initial_route_station(
    mission: DrivingMissionRecord,
    lanes: dict[str, RouteLaneRecord],
    snapshot: EnvSnapshot,
) -> float:
    """Apply the v1.1 reset invariant: first lane, or second only at boundary."""
    first = lanes[mission.route_lane_ids[0]]
    try:
        projection = first.centerline.project(snapshot.ego.position_xy, position_z=snapshot.ego.position_z)
        return projection.s_m
    except ValueError:
        pass
    if len(mission.route_lane_ids) < 2:
        raise ValueError("reset pose is not vertically compatible with first occurrence")
    second = lanes[mission.route_lane_ids[1]]
    projection = second.centerline.project(snapshot.ego.position_xy, position_z=snapshot.ego.position_z)
    if projection.s_m > 0.01:
        raise ValueError("reset pose is outside first occurrence and shared boundary")
    return first.centerline.length_m + projection.s_m
