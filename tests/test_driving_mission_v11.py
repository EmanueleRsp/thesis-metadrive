from __future__ import annotations

import pytest
from shapely.geometry import box

from thesis_rl.mission.builder import (
    NormalizedLane,
    _associate_temporal_route_occurrences,
    build_driving_mission_from_source,
)
from thesis_rl.mission.runtime import MissionRuntime
from thesis_rl.mission.types import DrivingMissionRecord
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, EnvSnapshot


def _scenario() -> dict[str, object]:
    return {
        "metadata": {"sdc_id": "ego"},
        "tracks": {
            "ego": {
                "state": {
                    "position": [[1.0, 0.0, 0.0], [18.0, 0.0, 0.0]],
                    "heading": [0.0, 0.0],
                    "valid": [True, True],
                }
            }
        },
        "map_features": {
            "a": {
                "type": "LANE_SURFACE_STREET",
                "polyline": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                "polygon": [[0.0, -2.0, 0.0], [10.0, -2.0, 0.0], [10.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
                "exit_lanes": ["b"],
                "left_neighbor": [],
                "right_neighbor": [],
            },
            "b": {
                "type": "LANE_SURFACE_STREET",
                "polyline": [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                "polygon": [
                    [10.0, -2.0, 0.0],
                    [20.0, -2.0, 0.0],
                    [20.0, 2.0, 0.0],
                    [10.0, 2.0, 0.0],
                ],
                "exit_lanes": [],
                "left_neighbor": [],
                "right_neighbor": [],
            },
        },
    }


def _snapshot(x: float) -> EnvSnapshot:
    ego = ActorSnapshot(
        "ego",
        ActorClass.VEHICLE,
        (x, 0.0),
        0.0,
        0.0,
        (1.0, 0.0),
        oriented_bounding_box(center_xy=(x, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        None,
        None,
    )
    return EnvSnapshot("uid", 0, 0.0, ego, (), (), frozenset(), {})


def test_v11_builder_freezes_anchor_gate_and_round_trips() -> None:
    mission = build_driving_mission_from_source(
        _scenario(), scenario_uid="uid", source="pg", assigned_route_lane_ids=("a", "b")
    )
    assert mission.s_start_m == 0.0
    assert mission.s_goal_m == 17.0
    assert mission.final_gate_segment is not None
    assert mission.final_gate_segment.builder_identity.endswith("anchor-builder")
    restored = DrivingMissionRecord.from_dict(mission.to_dict())
    assert restored.mission_hash == mission.mission_hash


def test_v11_runtime_uses_exact_delta_s_and_one_passive_final_gate() -> None:
    mission = build_driving_mission_from_source(
        _scenario(), scenario_uid="uid", source="pg", assigned_route_lane_ids=("a", "b")
    )
    lanes = (
        RouteLaneRecord(
            "a",
            box(0.0, -2.0, 10.0, 2.0),
            RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
            ("b",),
        ),
        RouteLaneRecord(
            "b", box(10.0, -2.0, 20.0, 2.0), RoutePolyline(((10.0, 0.0, 0.0), (20.0, 0.0, 0.0)))
        ),
    )
    runtime = MissionRuntime(mission, lanes, _snapshot(1.0))
    initial_snapshot = runtime.snapshot
    assert initial_snapshot.s_m == 0.0
    assert runtime.route.length_m == mission.s_goal_m
    snapshot = runtime.update(_snapshot(1.0), _snapshot(5.0))
    assert snapshot.s_m == 4.0
    assert snapshot.delta_s_m == 4.0
    assert snapshot.completion_instant == snapshot.completion_max
    assert snapshot.mission_success is False
    assert len(runtime.gates) == 1
    assert runtime.snapshot.s_m == snapshot.s_m


def test_v11_global_orientation_reverses_source_centerline_without_local_fallback() -> None:
    scenario = _scenario()
    scenario["tracks"]["ego"]["state"]["position"] = [[9.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    mission = build_driving_mission_from_source(
        scenario,
        scenario_uid="reverse-uid",
        source="pg",
        assigned_route_lane_ids=("a",),
    )
    assert mission.route_occurrences[0].orientation == "REVERSED"
    assert mission.s_start_m == 0.0
    assert mission.s_goal_m == 8.0
    assert mission.canonical_route_points_xyz[0][:2] == (9.0, 0.0)
    assert mission.canonical_route_points_xyz[-1][:2] == (1.0, 0.0)


def _snapshot_with_z(x: float, z: float) -> EnvSnapshot:
    ego = ActorSnapshot(
        "ego",
        ActorClass.VEHICLE,
        (x, 0.0),
        z,
        0.0,
        (1.0, 0.0),
        oriented_bounding_box(center_xy=(x, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        None,
        None,
    )
    return EnvSnapshot("uid", 0, 0.0, ego, (), (), frozenset(), {})


def test_v11_route_coordinate_runtime_aligns_frozen_elevation_to_live_datum() -> None:
    """Regression: ``MissionRuntime`` reused a route-coordinate mission's
    frozen (offline-datum) elevation verbatim instead of aligning it to the
    live MetaDrive world the way ``align_episode_cache_to_live_elevation``
    aligns every other episode-cache geometry. On a source with a nonzero
    live/offline datum offset (documented as common for Waymo), the frozen
    final gate's elevation then never matched the live ego's z within
    ``directed_gate_crossed``'s hard tolerance, so the mission could never
    succeed even after physically reaching the final gate.
    """
    mission = build_driving_mission_from_source(
        _scenario(), scenario_uid="uid", source="pg", assigned_route_lane_ids=("a", "b")
    )
    assert mission.final_gate_segment is not None
    assert mission.final_gate_segment.elevation_m == 0.0

    lanes = (
        RouteLaneRecord(
            "a",
            box(0.0, -2.0, 10.0, 2.0),
            RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
            ("b",),
        ),
        RouteLaneRecord(
            "b", box(10.0, -2.0, 20.0, 2.0), RoutePolyline(((10.0, 0.0, 0.0), (20.0, 0.0, 0.0)))
        ),
    )
    # MetaDrive's live spawn sits at z=5.0 here, while the frozen mission's
    # own offline datum is z=0.0 -- the same kind of datum mismatch
    # `align_episode_cache_to_live_elevation` corrects for the rest of the
    # episode cache, documented there as common for Waymo sources.
    runtime = MissionRuntime(mission, lanes, _snapshot_with_z(1.0, 5.0))

    assert runtime.route.points_xyz[0][2] == 5.0
    assert runtime._tracker.gate.elevation_m == 5.0

    success = runtime.update(_snapshot_with_z(15.0, 5.0), _snapshot_with_z(19.0, 5.0))

    assert success.mission_success is True


def _translated_scenario(dx: float, dy: float) -> dict[str, object]:
    scenario = _scenario()
    for feature in scenario["map_features"].values():
        feature["polyline"] = [[x + dx, y + dy, z] for x, y, z in feature["polyline"]]
        feature["polygon"] = [[x + dx, y + dy, z] for x, y, z in feature["polygon"]]
    state = scenario["tracks"]["ego"]["state"]
    state["position"] = [[x + dx, y + dy, z] for x, y, z in state["position"]]
    return scenario


def test_v11_route_coordinate_runtime_aligns_frozen_mission_to_centralized_live_frame() -> None:
    """Regression: the frozen mission is built offline from the *raw* source
    file, but ``ScenarioDataManager`` loads every scenario with
    ``centralize=True``, translating the live map and ego by the SDC's first
    raw position. ``MissionRuntime`` used the frozen geometry verbatim, so the
    canonical route and the final gate sat in a different coordinate frame than
    the vehicle they measure -- 5.9 m away for PG and 47.5 km away for Waymo's
    global coordinates. The route station, the completion ratio and the gate
    crossing were consequently pinned at their initial values, and the drawn
    route was offset from the road (PG) or entirely off-canvas (Waymo).
    """
    raw_dx, raw_dy = 1000.0, 2000.0
    mission = build_driving_mission_from_source(
        _translated_scenario(raw_dx, raw_dy),
        scenario_uid="uid",
        source="pg",
        assigned_route_lane_ids=("a", "b"),
    )
    # MetaDrive centralizes on the SDC's first raw position, (1001, 2000) here,
    # and records the inverse translation as `old_origin_in_current_coordinate`.
    origin_offset_xy = (-1001.0, -2000.0)
    # Live lanes are what the static adapter derives from the centralized
    # scenario, i.e. the raw geometry moved by exactly that offset.
    lanes = (
        RouteLaneRecord(
            "a",
            box(-1.0, -2.0, 9.0, 2.0),
            RoutePolyline(((-1.0, 0.0, 0.0), (9.0, 0.0, 0.0))),
            ("b",),
        ),
        RouteLaneRecord(
            "b", box(9.0, -2.0, 19.0, 2.0), RoutePolyline(((9.0, 0.0, 0.0), (19.0, 0.0, 0.0)))
        ),
    )

    # Without the offset the mission stays in the raw frame: fail closed rather
    # than silently measuring a vehicle against a route 2 km away.
    with pytest.raises(ValueError, match="not in the live scenario frame"):
        MissionRuntime(mission, lanes, _snapshot(0.0))

    runtime = MissionRuntime(mission, lanes, _snapshot(0.0), origin_offset_xy=origin_offset_xy)

    assert runtime.route.points_xyz[0][0] == pytest.approx(0.0, abs=1e-6)
    assert runtime.route.points_xyz[0][1] == pytest.approx(0.0, abs=1e-6)
    assert runtime.route.length_m == pytest.approx(mission.s_goal_m)
    # The frozen identity is provenance and must survive a pure datum change.
    assert runtime.snapshot.mission_hash == mission.mission_hash
    # The ego now projects onto its own route instead of a distant branch.
    assert runtime.route.project((0.0, 0.0)).lateral_distance_m == pytest.approx(0.0, abs=1e-6)

    success = runtime.update(_snapshot(15.0), _snapshot(19.0))

    assert success.mission_success is True


def test_v11_temporal_association_does_not_revisit_overlapping_prior_occurrence() -> None:
    lanes = (
        NormalizedLane(
            "a",
            10.0,
            centerline=RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
            polygon_xy=box(0.0, -1.0, 10.0, 1.0),
        ),
        NormalizedLane(
            "b",
            14.0,
            centerline=RoutePolyline(((10.0, 0.0, 0.0), (12.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
            polygon_xy=box(-1.0, -1.0, 13.0, 1.0),
        ),
    )
    samples = _associate_temporal_route_occurrences(
        lanes,
        (((9.0, 0.0), 0.0), ((12.0, 0.0), 0.0), ((0.0, 0.0), 0.0)),
    )
    assert samples[0] == (9.0,)
    assert samples[1][0] == 2.0
    assert samples[1][-1] == 14.0
