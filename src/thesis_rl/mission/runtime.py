"""Causal reset/transition integration for one frozen driving mission."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from thesis_rl.mission.builder import NormalizedLane
from thesis_rl.mission.distance import LaneGraph
from thesis_rl.mission.tracker import MissionTracker
from thesis_rl.mission.types import DirectedGate, DrivingMissionRecord, MissionSnapshot
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord, associate_route_lane
from thesis_rl.rulebook.v2.types import EnvSnapshot


def _materialize_gate(gate: DirectedGate, lanes: dict[str, NormalizedLane]) -> DirectedGate:
    lane = lanes.get(gate.lane_id)
    if lane is None:
        raise ValueError(f"mission gate references unknown live lane: {gate.lane_id}")
    geometry = lane.gate_geometry_at(gate.s_m)
    if geometry is None:
        raise ValueError(f"mission gate cannot be materialized: {gate.gate_id}")
    return replace(gate, geometry=geometry)


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
        gates_to_materialize = (
            *[section.exit_gate for section in mission.sections],
            mission.final_goal,
        )
        normalized = {
            lane.lane_id: NormalizedLane(
                lane.lane_id,
                lane.centerline.length_m,
                lane.successor_lane_ids,
                centerline=lane.centerline,
            )
            for lane in all_lanes
        }
        gates = tuple(_materialize_gate(gate, normalized) for gate in gates_to_materialize)
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
        )

    @property
    def snapshot(self) -> MissionSnapshot:
        return self._tracker.snapshot()

    def update(self, pre: EnvSnapshot, post: EnvSnapshot) -> MissionSnapshot:
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
