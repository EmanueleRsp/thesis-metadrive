from __future__ import annotations

import json
import pickle
from pathlib import Path

from thesis_rl.mission.materialize import (
    build_candidate_index,
    combine_candidate_indexes,
    promote_candidate_index,
    validate_candidate_index,
    write_candidate_index,
)


def _scenario() -> dict[str, object]:
    return {
        "length": 2,
        "metadata": {"sdc_id": "ego"},
        "tracks": {
            "ego": {
                "state": {
                    "position": [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
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
                "exit_lanes": [],
                "left_neighbor": [],
                "right_neighbor": [],
            }
        },
    }


def test_candidate_materialization_preserves_identity_and_never_writes_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pg" / "one.pkl"
    source.parent.mkdir()
    with source.open("wb") as handle:
        pickle.dump(_scenario(), handle)
    before = source.read_bytes()
    index = {
        "schema": "scenarionet_frozen_selection_v1",
        "selection_hash": "parent",
        "records": [
            {
                "scenario_uid": "pg:test:one",
                "source": "pg",
                "split": "train",
                "relative_path": "pg/one.pkl",
                "assigned_route_lane_ids": ["a"],
            }
        ],
    }

    candidate = build_candidate_index(index, tmp_path)

    assert candidate["schema"] == "scenarionet_frozen_mission_candidate_v1_1"
    assert candidate["parent_selection_hash"] == "parent"
    assert candidate["records"][0]["scenario_uid"] == "pg:test:one"
    assert candidate["records"][0]["driving_mission"]["scenario_uid"] == "pg:test:one"
    assert source.read_bytes() == before


def test_candidate_writer_refuses_overwrite(tmp_path: Path) -> None:
    payload = {"schema": "scenarionet_frozen_mission_candidate_v1_1", "records": []}
    output = tmp_path / "candidate.json"
    write_candidate_index(payload, output)
    assert json.loads(output.read_text()) == payload
    try:
        write_candidate_index(payload, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("candidate writer overwrote evidence")


def test_candidate_batches_combine_only_when_contiguous(tmp_path: Path) -> None:
    for name in ("one", "two"):
        source = tmp_path / "pg" / f"{name}.pkl"
        source.parent.mkdir(exist_ok=True)
        with source.open("wb") as handle:
            pickle.dump(_scenario(), handle)
    index = {
        "schema": "scenarionet_frozen_selection_v1",
        "selection_hash": "parent",
        "records": [
            {
                "scenario_uid": f"pg:test:{name}",
                "source": "pg",
                "split": "train",
                "relative_path": f"pg/{name}.pkl",
                "assigned_route_lane_ids": ["a"],
            }
            for name in ("one", "two")
        ],
    }
    first = build_candidate_index(index, tmp_path, record_start=0, record_end=1)
    second = build_candidate_index(index, tmp_path, record_start=1, record_end=2)

    combined = combine_candidate_indexes([first, second])

    assert [record["scenario_uid"] for record in combined["records"]] == [
        "pg:test:one",
        "pg:test:two",
    ]
    try:
        combine_candidate_indexes([second, first])
    except ValueError:
        pass
    else:
        raise AssertionError("candidate combiner accepted non-contiguous batches")


def test_candidate_validator_checks_identity_hash_and_mission_payload(tmp_path: Path) -> None:
    source = tmp_path / "pg" / "one.pkl"
    source.parent.mkdir()
    with source.open("wb") as handle:
        pickle.dump(_scenario(), handle)
    index = {
        "schema": "scenarionet_frozen_selection_v1",
        "selection_hash": "parent",
        "records": [
            {
                "scenario_uid": "pg:test:one",
                "scenario_id": "one",
                "source": "pg",
                "split": "train",
                "relative_path": "pg/one.pkl",
                "assigned_route_lane_ids": ["a"],
                "assigned_route_source": "test",
                "dataset_version": "test",
                "primary_arm": "test",
                "length": 2,
                "pg_seed": 1,
                "official_split": None,
                "source_log_id": None,
                "source_scenario_id": None,
                "converter_version": None,
                "runtime_index": None,
                "pg_profile": None,
                "map_id": None,
                "tags": [],
                "signal_reliability": "not_applicable",
                "validation_status": "valid",
                "validation_warnings": [],
                "dense_traffic": False,
            }
        ],
    }
    candidate = build_candidate_index(index, tmp_path)

    assert validate_candidate_index(candidate, index) == {"records": 1, "pg": 1}
    candidate["mission_selection_hash"] = "tampered"
    try:
        validate_candidate_index(candidate, index)
    except ValueError as error:
        assert "hash" in str(error)
    else:
        raise AssertionError("candidate validator accepted a tampered hash")


def test_promoted_candidate_preserves_parent_metadata_and_uses_new_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pg" / "one.pkl"
    source.parent.mkdir()
    with source.open("wb") as handle:
        pickle.dump(_scenario(), handle)
    index = {
        "schema": "scenarionet_frozen_selection_v1",
        "selection_hash": "parent",
        "split_manifest": {"counts": {}},
        "source_file_paths": ["pg/one.pkl"],
        "records": [
            {
                "scenario_uid": "pg:test:one",
                "scenario_id": "one",
                "source": "pg",
                "split": "train",
                "relative_path": "pg/one.pkl",
                "assigned_route_lane_ids": ["a"],
                "assigned_route_source": "test",
                "dataset_version": "test",
                "primary_arm": "test",
                "length": 2,
                "pg_seed": 1,
                "official_split": None,
                "source_log_id": None,
                "source_scenario_id": None,
                "converter_version": None,
                "runtime_index": None,
                "pg_profile": None,
                "map_id": None,
                "tags": [],
                "signal_reliability": "not_applicable",
                "validation_status": "valid",
                "validation_warnings": [],
                "dense_traffic": False,
            }
        ],
    }
    candidate = build_candidate_index(index, tmp_path)

    promoted = promote_candidate_index(candidate, index)

    assert promoted["schema"] == "scenarionet_frozen_selection_mission_v1_1"
    assert promoted["parent_selection_hash"] == "parent"
    assert promoted["selection_hash"] == candidate["mission_selection_hash"]
    assert promoted["split_manifest"] == {"counts": {}}
