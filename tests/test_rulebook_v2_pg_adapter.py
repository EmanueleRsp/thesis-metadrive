from __future__ import annotations

import glob
import pickle

import pytest

from thesis_rl.rulebook.v2.context.pg_static_adapter import build_pg_static_adapter_result


def _minimal_pg_scenario() -> dict:
    return {
        "metadata": {
            "sdc_id": "ego",
            "assigned_route_lane_ids": ["lane-a", "lane-b"],
            "assigned_route_source": "pg_sdc_offline_task_annotation",
        },
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
                "exit_lanes": ["lane-b"],
            },
            "lane-b": {
                "type": "LANE_SURFACE_STREET",
                "polyline": [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                "exit_lanes": [],
            },
        },
    }


def test_bundled_pg_fixture_converts_to_canonical_static_records():
    paths = sorted(glob.glob("data/scenarionet/pg/database/**/*.pkl", recursive=True))
    paths = [path for path in paths if "dataset_" not in path]
    if not paths:
        pytest.skip("bundled PG fixture unavailable")
    with open(paths[0], "rb") as handle:
        scenario = pickle.load(handle)
    result = build_pg_static_adapter_result(scenario, scenario_uid="pg-fixture")
    assert result.task_route.lane_ids
    assert result.route_lanes
    assert result.scenario_uid == "pg-fixture"


def test_pg_adapter_fails_fast_without_lane_geometry():
    with pytest.raises(ValueError, match="no lane geometry"):
        build_pg_static_adapter_result({"map_features": {}, "metadata": {}}, scenario_uid="s")


def test_pg_adapter_prefers_persisted_route_over_future_sdc_track() -> None:
    scenario = _minimal_pg_scenario()
    scenario["tracks"]["ego"]["state"]["position"] = [[1000.0, 1000.0, 0.0]]
    scenario["tracks"]["ego"]["state"]["heading"] = [3.14]
    scenario["tracks"]["ego"]["state"]["valid"] = [True]

    result = build_pg_static_adapter_result(scenario, scenario_uid="metadata-route")

    assert result.task_route.lane_ids == ("lane-a", "lane-b")
    assert result.task_route.route_assignment_source == "pg_sdc_offline_task_annotation"
