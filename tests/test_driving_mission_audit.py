from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from thesis_rl.scenarios.driving_mission_audit import audit_frozen_index, write_audit_report


def _record(*, relative_path: str = "pg/one.pkl") -> dict[str, object]:
    return {
        "scenario_uid": "pg:test:one",
        "source": "pg",
        "split": "train",
        "primary_arm": "simple",
        "relative_path": relative_path,
        "length": 2,
        "assigned_route_lane_ids": ["lane_a", "lane_b"],
    }


def _scenario(*, terminal_x: float = 11.0) -> dict[str, object]:
    return {
        "length": 2,
        "metadata": {"sdc_id": "ego"},
        "tracks": {
            "ego": {
                "state": {
                    "position": [[0.0, 0.0, 0.0], [terminal_x, 0.0, 0.0]],
                    "heading": [0.0, 0.0],
                    "valid": [True, True],
                }
            }
        },
        "map_features": {
            "lane_a": {
                "type": "LANE_SURFACE_STREET",
                "polyline": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                "polygon": [[0.0, -2.0, 0.0], [10.0, -2.0, 0.0], [10.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
                "exit_lanes": ["lane_b"],
                "left_neighbor": [],
                "right_neighbor": [],
            },
            "lane_b": {
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


def test_audit_builds_goal_gate_and_allowed_route_spans(tmp_path: Path) -> None:
    source = tmp_path / "pg" / "one.pkl"
    source.parent.mkdir()
    with source.open("wb") as handle:
        pickle.dump(_scenario(), handle)

    result = audit_frozen_index(
        {"schema": "scenarionet_frozen_selection_v1", "records": [_record()]}, tmp_path
    )

    assert result.passed
    item = result.records[0]
    assert item.outcome == "pass"
    assert item.goal_lane_id == "lane_b"
    assert item.gate_count == 1
    assert item.allowed_span_count == 2


def test_audit_reason_codes_noncontiguous_route_and_never_mutates_source(tmp_path: Path) -> None:
    source = tmp_path / "pg" / "one.pkl"
    source.parent.mkdir()
    scenario = _scenario()
    scenario["map_features"]["lane_a"]["exit_lanes"] = []  # type: ignore[index]
    with source.open("wb") as handle:
        pickle.dump(scenario, handle)
    before = source.read_bytes()

    result = audit_frozen_index(
        {"schema": "scenarionet_frozen_selection_v1", "records": [_record()]}, tmp_path
    )

    assert not result.passed
    assert result.records[0].outcome == "non_contiguous_gate_order"
    assert source.read_bytes() == before


def test_audit_accepts_numpy_backed_sdc_arrays(tmp_path: Path) -> None:
    source = tmp_path / "pg" / "one.pkl"
    source.parent.mkdir()
    scenario = _scenario()
    state = scenario["tracks"]["ego"]["state"]  # type: ignore[index]
    state["position"] = np.asarray(state["position"], dtype=float)
    state["heading"] = np.asarray(state["heading"], dtype=float)
    state["valid"] = np.asarray(state["valid"], dtype=bool)
    with source.open("wb") as handle:
        pickle.dump(scenario, handle)

    result = audit_frozen_index(
        {"schema": "scenarionet_frozen_selection_v1", "records": [_record()]}, tmp_path
    )

    assert result.records[0].outcome == "pass"


def test_write_audit_report_refuses_to_overwrite(tmp_path: Path) -> None:
    result = audit_frozen_index(
        {
            "schema": "scenarionet_frozen_selection_v1",
            "records": [_record(relative_path="missing.pkl")],
        },
        tmp_path,
    )
    output_dir = tmp_path / "output"

    write_audit_report(result, output_dir)

    assert (output_dir / "driving_mission_audit.json").is_file()
    try:
        write_audit_report(result, output_dir)
    except FileExistsError:
        pass
    else:
        raise AssertionError("audit report writer overwrote an existing directory")
