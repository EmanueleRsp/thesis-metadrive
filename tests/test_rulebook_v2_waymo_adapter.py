from __future__ import annotations

import glob
import pickle

import pytest

from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result


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
    assert not any(error.startswith("task_route_lane_missing") for error in result.validation_errors)
    assert any(control.control_type.value == "stop" for control in result.traffic_controls)
    assert result.movement_priority_records == ()


def test_waymo_adapter_fails_fast_without_lane_geometry():
    with pytest.raises(ValueError, match="no lane geometry"):
        build_waymo_static_adapter_result({"map_features": {}, "metadata": {}}, scenario_uid="s")
