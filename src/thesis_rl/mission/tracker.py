"""Episode-local, idempotent mission progress tracker."""

from __future__ import annotations

from math import hypot

from shapely.geometry import Polygon

from thesis_rl.mission.distance import LaneGraph
from thesis_rl.mission.gates import directed_gate_crossed
from thesis_rl.rulebook.v2.geometry.footprint import front_bumper_segment
from thesis_rl.mission.types import (
    DirectedGate,
    DrivingMissionRecord,
    MissionSnapshot,
    ordered_mission_gates,
)


class MissionTracker:
    def __init__(
        self,
        mission: DrivingMissionRecord,
        graph: LaneGraph,
        lane_id: str,
        s_m: float,
        *,
        materialized_gates: tuple[DirectedGate, ...] | None = None,
        initial_footprint: Polygon | None = None,
        initial_heading_rad: float | None = None,
    ) -> None:
        self._mission, self._graph = mission, graph
        frozen_gates = ordered_mission_gates(mission)
        self._gates = frozen_gates if materialized_gates is None else materialized_gates
        if len(self._gates) != len(frozen_gates) or any(
            current.gate_id != frozen.gate_id
            for current, frozen in zip(self._gates, frozen_gates, strict=True)
        ):
            raise ValueError("materialized mission gates must match frozen gate identity")
        self._pending, self._step, self._completion = 0, 0, 0.0
        initial = (
            self._remaining_from_footprint(initial_footprint, initial_heading_rad)
            if initial_footprint is not None and initial_heading_rad is not None
            else self._remaining(lane_id, s_m)
        )
        if initial is None:
            raise ValueError("mission tracker requires an initial geometric or graph distance")
        self._initial_distance = initial
        self._snapshot = self._make_snapshot(initial, True, False, False)
        self._snapshots = {0: self._snapshot}

    def snapshot(self) -> MissionSnapshot:
        return self._snapshot

    @property
    def gates(self) -> tuple[DirectedGate, ...]:
        """Return read-only materialized task boundaries."""

        return self._gates

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

    def _remaining_from_footprint(self, footprint: Polygon, heading_rad: float) -> float:
        """Return remaining ordered-gate distance without lane admissibility gates."""

        gate = self._gates[self._pending]
        if gate.geometry is None:
            raise ValueError("pending mission gate has no materialized geometry")
        front = front_bumper_segment(footprint, heading_rad=heading_rad).centroid
        anchor_x, anchor_y = _gate_anchor(gate)
        tangent_x, tangent_y = gate.geometry.forward_tangent_xy
        tangent_norm = hypot(tangent_x, tangent_y)
        upstream_distance = max(
            0.0,
            -((front.x - anchor_x) * tangent_x + (front.y - anchor_y) * tangent_y)
            / tangent_norm,
        )
        downstream_distance = 0.0
        for current, following in zip(
            self._gates[self._pending :], self._gates[self._pending + 1 :]
        ):
            if current.geometry is None or following.geometry is None:
                raise ValueError("mission gate has no materialized geometry")
            current_anchor = _gate_anchor(current)
            following_anchor = _gate_anchor(following)
            downstream_distance += hypot(
                following_anchor[0] - current_anchor[0],
                following_anchor[1] - current_anchor[1],
            )
        return upstream_distance + downstream_distance
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
        if self._snapshot.mission_success:
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
        if directed_gate_crossing:
            self._pending += 1
        self._step += 1
        if self._pending == len(self._gates):
            self._snapshot = self._make_snapshot(0.0, True, True, False)
        else:
            remaining = self._remaining_from_footprint(post_footprint, post_heading_rad)
            self._snapshot = self._make_snapshot(remaining, True, False, False)
        self._snapshots[self._step] = self._snapshot
        return self._snapshot


def _gate_anchor(gate: DirectedGate) -> tuple[float, float]:
    """Return the midpoint of a materialized gate's road-crossing segment."""

    if gate.geometry is None:
        raise ValueError("mission gate has no materialized geometry")
    start, end = gate.geometry.line_xy
    return ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
