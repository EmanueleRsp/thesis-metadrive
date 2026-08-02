from __future__ import annotations

from shapely.geometry import box

from thesis_rl.mission.runtime import MissionRuntime
from thesis_rl.mission.types import DirectedGate, DrivingMissionRecord, LaneSpan, MissionSection
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, EnvSnapshot


def _snapshot(step: int, x: float) -> EnvSnapshot:
    ego = ActorSnapshot(
        "ego",
        ActorClass.VEHICLE,
        (x, 0.0),
        0.0,
        0.0,
        (5.0, 0.0),
        oriented_bounding_box(center_xy=(x, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        None,
        None,
    )
    return EnvSnapshot("scenario", step, step * 0.1, ego, (), (), frozenset(), {})


def test_runtime_materializes_live_gate_geometry_without_changing_frozen_hash() -> None:
    lane_a = RouteLaneRecord(
        "a", box(0.0, -2.0, 10.0, 2.0), RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))), ("b",)
    )
    lane_b = RouteLaneRecord(
        "b", box(10.0, -2.0, 20.0, 2.0), RoutePolyline(((10.0, 0.0, 0.0), (20.0, 0.0, 0.0)))
    )
    span_a, span_b = LaneSpan("a", 0.0, 10.0), LaneSpan("b", 0.0, 10.0)
    intermediate = DirectedGate("gate:a", (span_a,), "a", 10.0)
    final = DirectedGate("goal:final", (span_b,), "b", 8.0)
    mission = DrivingMissionRecord(
        "scenario",
        "mission-builder-v1",
        (MissionSection("section:0", span_a, (span_a,), intermediate),),
        final,
    )

    runtime = MissionRuntime(mission, (lane_a, lane_b), _snapshot(0, 2.0))
    after_gate = runtime.update(_snapshot(0, 8.0), _snapshot(1, 9.5))
    success = runtime.update(_snapshot(1, 16.0), _snapshot(2, 17.0))

    assert runtime.snapshot.mission_hash == mission.mission_hash
    assert after_gate.pending_gate_index == 1
    assert success.mission_success is True
    assert success.route_completion == 1.0


def test_runtime_ignores_unavailable_static_successors_but_preserves_required_path() -> None:
    lane = RouteLaneRecord(
        "a",
        box(0.0, -2.0, 10.0, 2.0),
        RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
        ("missing",),
    )
    span = LaneSpan("a", 0.0, 10.0)
    mission = DrivingMissionRecord(
        "scenario",
        "mission-builder-v1",
        (MissionSection("section:0", span, (span,), DirectedGate("gate", (span,), "a", 8.0)),),
        DirectedGate("goal:final", (span,), "a", 9.0),
    )

    runtime = MissionRuntime(mission, (lane,), _snapshot(0, 2.0))

    assert runtime.snapshot.reachable is True


def test_runtime_excludes_non_mission_lanes_from_association_graph() -> None:
    legal_lane = RouteLaneRecord(
        "a", box(0.0, -2.0, 10.0, 2.0), RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    )
    foreign_lane = RouteLaneRecord(
        "foreign",
        box(100.0, -2.0, 110.0, 2.0),
        RoutePolyline(((100.0, 0.0, 0.0), (110.0, 0.0, 0.0))),
    )
    span = LaneSpan("a", 0.0, 10.0)
    mission = DrivingMissionRecord(
        "scenario",
        "mission-builder-v1",
        (MissionSection("section:0", span, (span,), DirectedGate("gate", (span,), "a", 8.0)),),
        DirectedGate("goal:final", (span,), "a", 9.0),
    )

    runtime = MissionRuntime(mission, (legal_lane, foreign_lane), _snapshot(0, 2.0))

    assert runtime.snapshot.reachable is True
