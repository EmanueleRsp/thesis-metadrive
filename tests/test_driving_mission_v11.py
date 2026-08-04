from __future__ import annotations

from shapely.geometry import box

from thesis_rl.mission.builder import build_driving_mission_from_source
from thesis_rl.mission.runtime import MissionRuntime
from thesis_rl.mission.types import DrivingMissionRecord
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, EnvSnapshot


def _scenario() -> dict[str, object]:
    return {
        "metadata": {"sdc_id": "ego"},
        "tracks": {"ego": {"state": {"position": [[1.0, 0.0, 0.0], [18.0, 0.0, 0.0]], "heading": [0.0, 0.0], "valid": [True, True]}}},
        "map_features": {
            "a": {"type": "LANE_SURFACE_STREET", "polyline": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], "polygon": [[0.0, -2.0, 0.0], [10.0, -2.0, 0.0], [10.0, 2.0, 0.0], [0.0, 2.0, 0.0]], "exit_lanes": ["b"], "left_neighbor": [], "right_neighbor": []},
            "b": {"type": "LANE_SURFACE_STREET", "polyline": [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], "polygon": [[10.0, -2.0, 0.0], [20.0, -2.0, 0.0], [20.0, 2.0, 0.0], [10.0, 2.0, 0.0]], "exit_lanes": [], "left_neighbor": [], "right_neighbor": []},
        },
    }


def _snapshot(x: float) -> EnvSnapshot:
    ego = ActorSnapshot("ego", ActorClass.VEHICLE, (x, 0.0), 0.0, 0.0, (1.0, 0.0), oriented_bounding_box(center_xy=(x, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0), None, None)
    return EnvSnapshot("uid", 0, 0.0, ego, (), (), frozenset(), {})


def test_v11_builder_freezes_anchor_gate_and_round_trips() -> None:
    mission = build_driving_mission_from_source(_scenario(), scenario_uid="uid", source="pg", assigned_route_lane_ids=("a", "b"))
    assert mission.s_start_m == 0.0
    assert mission.s_goal_m == 17.0
    assert mission.final_gate_segment is not None
    assert mission.final_gate_segment.builder_identity.endswith("anchor-builder")
    restored = DrivingMissionRecord.from_dict(mission.to_dict())
    assert restored.mission_hash == mission.mission_hash


def test_v11_runtime_uses_exact_delta_s_and_one_passive_final_gate() -> None:
    mission = build_driving_mission_from_source(_scenario(), scenario_uid="uid", source="pg", assigned_route_lane_ids=("a", "b"))
    lanes = (
        RouteLaneRecord("a", box(0.0, -2.0, 10.0, 2.0), RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))), ("b",)),
        RouteLaneRecord("b", box(10.0, -2.0, 20.0, 2.0), RoutePolyline(((10.0, 0.0, 0.0), (20.0, 0.0, 0.0)))),
    )
    runtime = MissionRuntime(mission, lanes, _snapshot(1.0))
    snapshot = runtime.update(_snapshot(1.0), _snapshot(5.0))
    assert snapshot.s_m == 4.0
    assert snapshot.delta_s_m == 4.0
    assert snapshot.completion_instant == snapshot.completion_max
    assert snapshot.mission_success is False
    assert len(runtime.gates) == 1
