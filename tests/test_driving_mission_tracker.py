from __future__ import annotations

from thesis_rl.mission.distance import LaneGraph
from thesis_rl.mission.gates import GateGeometry
from thesis_rl.mission.tracker import MissionTracker
from thesis_rl.mission.types import DirectedGate, DrivingMissionRecord, LaneSpan, MissionSection
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box


def _mission() -> DrivingMissionRecord:
    a, b = LaneSpan("a", 0.0, 10.0), LaneSpan("b", 0.0, 10.0)
    geometry = GateGeometry(((10.0, -3.0), (10.0, 3.0)), (1.0, 0.0), 0.0)
    intermediate = DirectedGate("gate:a", (a,), "a", 10.0, geometry)
    final = DirectedGate("goal:final", (b,), "b", 8.0, geometry)
    return DrivingMissionRecord(
        "uid", "mission-builder-v1", (MissionSection("section:0", a, (a,), intermediate),), final
    )


def test_tracker_advances_directed_gate_and_is_idempotent() -> None:
    tracker = MissionTracker(_mission(), LaneGraph({"a": 10.0, "b": 10.0}, {"a": ("b",)}), "a", 2.0)
    before = tracker.snapshot()
    pre = oriented_bounding_box(center_xy=(8.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    post = oriented_bounding_box(center_xy=(11.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    after = tracker.update(
        "a",
        9.0,
        "a",
        10.0,
        pre_footprint=pre,
        post_footprint=post,
        pre_heading_rad=0.0,
        post_heading_rad=0.0,
        post_ego_z_m=0.0,
    )
    assert before == tracker.snapshot_at(0)
    assert after.pending_gate_index == 1
    assert after.remaining_distance_m < before.remaining_distance_m
    assert after == tracker.snapshot()


def test_tracker_reports_unreachable_and_completion_never_decreases() -> None:
    tracker = MissionTracker(_mission(), LaneGraph({"a": 10.0, "b": 10.0}, {"a": ("b",)}), "a", 2.0)
    footprint = oriented_bounding_box(
        center_xy=(2.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0
    )
    progressed = tracker.update(
        "a",
        2.0,
        "a",
        8.0,
        pre_footprint=footprint,
        post_footprint=footprint,
        pre_heading_rad=0.0,
        post_heading_rad=0.0,
        post_ego_z_m=0.0,
    )
    unreachable = tracker.update(
        "a",
        8.0,
        "lost",
        0.0,
        pre_footprint=footprint,
        post_footprint=footprint,
        pre_heading_rad=0.0,
        post_heading_rad=0.0,
        post_ego_z_m=0.0,
    )
    assert unreachable.mission_unreachable
    assert unreachable.route_completion >= progressed.route_completion


def test_graph_recovery_tie_break_and_cycle_are_deterministic() -> None:
    graph = LaneGraph(
        {"a": 10.0, "b": 10.0, "c": 10.0, "d": 10.0},
        {"a": ("c", "b"), "b": ("a", "d"), "c": ("d",)},
    )

    assert graph.shortest_path("a", 5.0, "d", 4.0) == (19.0, ("a", "b", "d"))


def test_graph_accepts_legal_lateral_recovery() -> None:
    graph = LaneGraph(
        {"a": 10.0, "parallel": 10.0, "goal": 10.0},
        {"parallel": ("goal",)},
        {"a": ("parallel",)},
    )

    assert graph.shortest_path("a", 5.0, "goal", 2.0) == (17.0, ("a", "parallel", "goal"))


def test_tracker_reset_instances_are_isolated_and_gate_distance_is_continuous() -> None:
    graph = LaneGraph({"a": 10.0, "b": 10.0}, {"a": ("b",)})
    first = MissionTracker(_mission(), graph, "a", 2.0)
    second = MissionTracker(_mission(), graph, "a", 2.0)
    pre = oriented_bounding_box(center_xy=(8.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    post = oriented_bounding_box(center_xy=(11.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)

    advanced = first.update(
        "a",
        9.0,
        "a",
        10.0,
        pre_footprint=pre,
        post_footprint=post,
        pre_heading_rad=0.0,
        post_heading_rad=0.0,
        post_ego_z_m=0.0,
    )

    assert advanced.remaining_distance_m == 8.0
    assert second.snapshot().pending_gate_index == 0
    assert second.snapshot().remaining_distance_m == 16.0
