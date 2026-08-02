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
        legal_lane_ids = {
            span.lane_id for section in mission.sections for span in section.allowed_spans
        }
        legal_lane_ids.update(
            span.lane_id
            for gate in (*[section.exit_gate for section in mission.sections], mission.final_goal)
            for span in gate.compatible_spans
        )
        missing = sorted(legal_lane_ids.difference(all_by_id))
        if missing:
            raise ValueError(f"mission references unavailable live lanes: {missing[:5]}")
        lanes = tuple(lane for lane in all_lanes if lane.lane_id in legal_lane_ids)
        by_id = {lane.lane_id: lane for lane in lanes}
        normalized = {
            lane.lane_id: NormalizedLane(
                lane.lane_id,
                lane.centerline.length_m,
                lane.successor_lane_ids,
                centerline=lane.centerline,
            )
            for lane in lanes
        }
        gates = tuple(
            _materialize_gate(gate, normalized)
            for gate in (*[section.exit_gate for section in mission.sections], mission.final_goal)
        )
        graph = LaneGraph(
            lengths_m={lane.lane_id: lane.centerline.length_m for lane in lanes},
            successors={
                lane.lane_id: tuple(
                    successor for successor in lane.successor_lane_ids if successor in by_id
                )
                for lane in lanes
            },
        )
        association = _associate(initial_snapshot, lanes)
        if association is None:
            raise ValueError("reset ego state cannot be associated with the frozen mission graph")
        self._route_lanes = lanes
        self._tracker = MissionTracker(
            mission,
            graph,
            association.lane_id,
            association.route_projection_s_m,
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
