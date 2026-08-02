"""Episode-local, idempotent mission progress tracker."""

from __future__ import annotations
from thesis_rl.mission.distance import LaneGraph
from thesis_rl.mission.types import DirectedGate, DrivingMissionRecord, MissionSnapshot


class MissionTracker:
    def __init__(
        self, mission: DrivingMissionRecord, graph: LaneGraph, lane_id: str, s_m: float
    ) -> None:
        self._mission, self._graph = mission, graph
        self._gates: tuple[DirectedGate, ...] = tuple(
            section.exit_gate for section in mission.sections
        ) + (mission.final_goal,)
        self._pending, self._step, self._completion = 0, 0, 0.0
        initial = self._remaining(lane_id, s_m)
        if initial is None:
            raise ValueError("reset ego state cannot reach the pending mission gate")
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
        self._completion = (
            1.0
            if success
            else max(self._completion, max(0.0, min(1.0, 1.0 - remaining / self._initial_distance)))
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
        self, pre_lane_id: str, pre_s_m: float, post_lane_id: str, post_s_m: float
    ) -> MissionSnapshot:
        if self._snapshot.mission_success or self._snapshot.mission_unreachable:
            return self._snapshot
        gate = self._gates[self._pending]
        if (
            pre_lane_id == gate.lane_id
            and post_lane_id == gate.lane_id
            and pre_s_m < gate.s_m <= post_s_m
        ):
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
