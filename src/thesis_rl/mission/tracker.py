"""Episode-local, idempotent mission progress tracker."""

from __future__ import annotations

from shapely.geometry import Polygon

from thesis_rl.mission.distance import LaneGraph
from thesis_rl.mission.gates import directed_gate_crossed
from thesis_rl.mission.types import DirectedGate, DrivingMissionRecord, MissionSnapshot


class MissionTracker:
    def __init__(
        self,
        mission: DrivingMissionRecord,
        graph: LaneGraph,
        lane_id: str,
        s_m: float,
        *,
        materialized_gates: tuple[DirectedGate, ...] | None = None,
    ) -> None:
        self._mission, self._graph = mission, graph
        frozen_gates: tuple[DirectedGate, ...] = tuple(
            section.exit_gate for section in mission.sections
        ) + (mission.final_goal,)
        self._gates = frozen_gates if materialized_gates is None else materialized_gates
        if len(self._gates) != len(frozen_gates) or any(
            current.gate_id != frozen.gate_id
            for current, frozen in zip(self._gates, frozen_gates, strict=True)
        ):
            raise ValueError("materialized mission gates must match frozen gate identity")
        self._pending, self._step, self._completion = 0, 0, 0.0
        initial = self._remaining(lane_id, s_m)
        if initial is None:
            self._initial_distance = 0.0
            self._snapshot = self._make_snapshot(0.0, False, False, True)
            self._snapshots = {0: self._snapshot}
            return
        self._initial_distance = initial
        self._snapshot = self._make_snapshot(initial, True, False, False)
        self._snapshots = {0: self._snapshot}

    def snapshot(self) -> MissionSnapshot:
        return self._snapshot

    def snapshot_at(self, step_index: int) -> MissionSnapshot:
        return self._snapshots[step_index]

    def _remaining(self, lane_id: str, s_m: float) -> float | None:
        gate = self._gates[self._pending]
        distance = self._graph.distance(lane_id, s_m, gate.lane_id, gate.s_m)
        if distance is None:
            return None
        for current, following in zip(
            self._gates[self._pending :], self._gates[self._pending + 1 :]
        ):
            leg = self._graph.distance(
                current.lane_id, current.s_m, following.lane_id, following.s_m
            )
            if leg is None:
                return None
            distance += leg
        return distance

    def _make_snapshot(
        self, remaining: float, reachable: bool, success: bool, unreachable: bool
    ) -> MissionSnapshot:
        if success:
            self._completion = 1.0
        elif reachable and self._initial_distance > 0.0:
            self._completion = max(
                self._completion,
                max(0.0, min(1.0, 1.0 - remaining / self._initial_distance)),
            )
        return MissionSnapshot(
            self._mission.mission_hash,
            self._step,
            self._pending,
            remaining,
            self._completion,
            reachable,
            success,
            unreachable,
            "mission_unreachable" if unreachable else None,
        )

    def update(
        self,
        pre_lane_id: str,
        pre_s_m: float,
        post_lane_id: str,
        post_s_m: float,
        *,
        pre_footprint: Polygon,
        post_footprint: Polygon,
        pre_heading_rad: float,
        post_heading_rad: float,
        post_ego_z_m: float,
    ) -> MissionSnapshot:
        if self._snapshot.mission_success or self._snapshot.mission_unreachable:
            return self._snapshot
        gate = self._gates[self._pending]
        if gate.geometry is None:
            raise ValueError("pending mission gate has no materialized geometry")
        directed_gate_crossing = directed_gate_crossed(
            gate.geometry,
            pre_footprint=pre_footprint,
            post_footprint=post_footprint,
            pre_heading_rad=pre_heading_rad,
            post_heading_rad=post_heading_rad,
            post_ego_z_m=post_ego_z_m,
        )
        if directed_gate_crossing and pre_lane_id == gate.lane_id and post_lane_id == gate.lane_id:
            self._pending += 1
        self._step += 1
        if self._pending == len(self._gates):
            self._snapshot = self._make_snapshot(0.0, True, True, False)
        else:
            remaining = self._remaining(post_lane_id, post_s_m)
            self._snapshot = self._make_snapshot(
                remaining or 0.0, remaining is not None, False, remaining is None
            )
        self._snapshots[self._step] = self._snapshot
        return self._snapshot
