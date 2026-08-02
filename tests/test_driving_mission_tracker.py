from __future__ import annotations

from thesis_rl.mission.distance import LaneGraph
from thesis_rl.mission.tracker import MissionTracker
from thesis_rl.mission.types import DirectedGate, DrivingMissionRecord, LaneSpan, MissionSection


def _mission() -> DrivingMissionRecord:
    a, b = LaneSpan("a", 0.0, 10.0), LaneSpan("b", 0.0, 10.0)
    intermediate = DirectedGate("gate:a", (a,), "a", 10.0)
    final = DirectedGate("goal:final", (b,), "b", 8.0)
    return DrivingMissionRecord("uid", "mission-builder-v1", (MissionSection("section:0", a, (a,), intermediate),), final)


def test_tracker_advances_directed_gate_and_is_idempotent() -> None:
    tracker = MissionTracker(_mission(), LaneGraph({"a": 10.0, "b": 10.0}, {"a": ("b",)}), "a", 2.0)
    before = tracker.snapshot()
    after = tracker.update("a", 9.0, "a", 10.0)
    assert before == tracker.snapshot_at(0)
    assert after.pending_gate_index == 1
    assert after.remaining_distance_m < before.remaining_distance_m
    assert after == tracker.snapshot()


def test_tracker_reports_unreachable_and_completion_never_decreases() -> None:
    tracker = MissionTracker(_mission(), LaneGraph({"a": 10.0, "b": 10.0}, {"a": ("b",)}), "a", 2.0)
    progressed = tracker.update("a", 2.0, "a", 8.0)
    unreachable = tracker.update("a", 8.0, "lost", 0.0)
    assert unreachable.mission_unreachable
    assert unreachable.route_completion >= progressed.route_completion
