"""Deterministic offline construction of mission sections and directed gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from thesis_rl.mission.types import DirectedGate, DrivingMissionRecord, LaneSpan, MissionSection


@dataclass(frozen=True, slots=True)
class NormalizedLane:
    lane_id: str
    length_m: float
    successor_lane_ids: tuple[str, ...] = ()
    lateral_lane_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.lane_id or self.length_m <= 0.0:
            raise ValueError("normalized lane ID and positive length are required")


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
            allowed_ids.update(neighbor for neighbor in lanes[preferred].lateral_lane_ids if neighbor in reachable)
        allowed = tuple(LaneSpan(lane, 0.0, lanes[lane].length_m) for lane in sorted(allowed_ids))
        preferred_span = LaneSpan(preferred_lane_ids[section_start], 0.0, lanes[preferred_lane_ids[section_start]].length_m)
        gate_span = LaneSpan(lane_id, 0.0, lanes[lane_id].length_m)
        gate = DirectedGate(f"gate:{index}:{lane_id}", (gate_span,), lane_id, lanes[lane_id].length_m)
        sections.append(MissionSection(f"section:{len(sections)}", preferred_span, allowed, gate))
        section_start = index + 1
    final_span = LaneSpan(final_goal_lane_id, 0.0, lanes[final_goal_lane_id].length_m)
    final_goal = DirectedGate("goal:final", (final_span,), final_goal_lane_id, final_goal_s_m)
    if not sections:
        preferred = preferred_lane_ids[0]
        span = LaneSpan(preferred, 0.0, lanes[preferred].length_m)
        sections.append(MissionSection("section:0", span, tuple(LaneSpan(lane, 0.0, lanes[lane].length_m) for lane in preferred_lane_ids), final_goal))
    return DrivingMissionRecord(scenario_uid, builder_version, tuple(sections), final_goal)
