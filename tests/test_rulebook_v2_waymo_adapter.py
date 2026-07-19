from __future__ import annotations

import glob
import pickle

import pytest

from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result


def _minimal_scenario(*, signal_lane_reachable: bool) -> dict:
    return {
        "id": "minimal",
        "length": 2,
        "metadata": {"sdc_id": "ego"},
        "tracks": {
            "ego": {
                "state": {
                    "position": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    "heading": [0.0, 0.0],
                    "valid": [True, True],
                }
            }
        },
        "map_features": {
            "lane-a": {
                "type": "LANE_SURFACE_STREET",
                "polyline": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                "width": [3.5, 3.5],
                "exit_lanes": ["lane-b"] if signal_lane_reachable else [],
            },
            "lane-b": {
                "type": "LANE_SURFACE_STREET",
                "polyline": [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                "width": [3.5, 3.5],
                "exit_lanes": [],
            },
            "edge": {
                "type": "ROAD_EDGE_BOUNDARY",
                "polyline": [[3.0, -2.0, 0.0]],
            },
        },
        "dynamic_map_states": {
            "signal": {
                "type": "TRAFFIC_LIGHT",
                "lane": "lane-b",
                "stop_point": [11.0, 0.0, 0.0],
                "state": {"object_state": ["LANE_STATE_UNKNOWN", "LANE_STATE_UNKNOWN"]},
            }
        },
    }


def test_bundled_waymo_fixture_converts_to_canonical_static_records():
    paths = sorted(glob.glob("third_party/metadrive/metadrive/assets/waymo/sd_*.pkl"))
    if not paths:
        pytest.skip("bundled Waymo fixture unavailable")
    with open(paths[0], "rb") as handle:
        scenario = pickle.load(handle)
    result = build_waymo_static_adapter_result(scenario, scenario_uid="waymo-fixture")
    assert result.task_route.lane_ids
    assert result.route_lanes
    assert result.scenario_uid == "waymo-fixture"
    assert not any(
        error.startswith("task_route_lane_missing") for error in result.validation_errors
    )
    assert any(control.control_type.value == "stop" for control in result.traffic_controls)
    assert result.movement_priority_records == ()


def test_waymo_adapter_fails_fast_without_lane_geometry():
    with pytest.raises(ValueError, match="no lane geometry"):
        build_waymo_static_adapter_result({"map_features": {}, "metadata": {}}, scenario_uid="s")


def test_waymo_adapter_types_single_point_map_feature_instead_of_raising_geos():
    result = build_waymo_static_adapter_result(
        _minimal_scenario(signal_lane_reachable=False),
        scenario_uid="minimal",
    )
    assert "invalid_map_feature_geometry:edge" in result.validation_errors


def test_waymo_adapter_prefers_persisted_route_over_future_sdc_track() -> None:
    scenario = _minimal_scenario(signal_lane_reachable=False)
    scenario["metadata"]["assigned_route_lane_ids"] = ["lane-a", "lane-b"]
    scenario["metadata"]["assigned_route_source"] = "waymo_sdc_offline_task_annotation"
    scenario["tracks"]["ego"]["state"]["position"] = [[1000.0, 1000.0, 0.0]]
    scenario["tracks"]["ego"]["state"]["heading"] = [3.14]
    scenario["tracks"]["ego"]["state"]["valid"] = [True]

    result = build_waymo_static_adapter_result(scenario, scenario_uid="metadata-route")

    assert result.task_route.lane_ids == ("lane-a", "lane-b")
    assert result.task_route.route_assignment_source == "waymo_sdc_offline_task_annotation"


def test_waymo_adapter_preserves_lane_successors() -> None:
    result = build_waymo_static_adapter_result(
        _minimal_scenario(signal_lane_reachable=True), scenario_uid="topology"
    )
    lanes = {lane.lane_id: lane for lane in result.route_lanes}
    assert lanes["lane-a"].successor_lane_ids == ("lane-b",)
    assert lanes["lane-b"].successor_lane_ids == ()


@pytest.mark.parametrize(
    ("reachable", "expected"),
    ((False, False), (True, True)),
)
def test_waymo_adapter_validates_unknown_signal_only_when_topologically_relevant(
    reachable: bool,
    expected: bool,
) -> None:
    result = build_waymo_static_adapter_result(
        _minimal_scenario(signal_lane_reachable=reachable),
        scenario_uid="minimal",
    )
    assert ("signal_state_unknown:signal" in result.validation_errors) is expected
