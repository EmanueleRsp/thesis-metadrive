from __future__ import annotations

import glob
import pickle

import pytest

from thesis_rl.rulebook.v2.context.pg_static_adapter import build_pg_static_adapter_result


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
