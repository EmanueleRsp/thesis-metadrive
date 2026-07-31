from __future__ import annotations

from pathlib import Path

import pytest

from thesis_rl.scenarios.waymo import (
    build_converter_command,
    load_converted_waymo_entries,
    validate_training_20s_source,
    waymo_dependency_status,
    waymo_group_id,
    waymo_map_fingerprint,
)


def test_training_20s_source_check_is_strict(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "training_20s.tfrecord-00000-of-01000").touch()

    assert len(validate_training_20s_source(raw)) == 1
    command = build_converter_command(raw_data_path=raw, database_path=tmp_path / "db", num_files=1)
    assert command[command.index("--version") + 1] == "training_20s"
    assert command[command.index("--num_files") + 1] == "1"


def test_training_source_rejects_other_waymo_variant(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "validation.tfrecord-00000-of-00001").touch()

    with pytest.raises(ValueError, match="training_20s"):
        validate_training_20s_source(raw)


def test_waymo_group_id_prefers_source_log_then_source_file() -> None:
    assert waymo_group_id({"id": "s", "metadata": {"source_log_id": "log"}}) == "log"
    assert waymo_group_id({"id": "s", "metadata": {"source_file": "shard"}}) == "shard"
    assert (
        waymo_group_id({"id": "s", "metadata": {"source_file": "/tmp/training_20s.tfrecord-00000"}})
        == "training_20s.tfrecord-00000"
    )
    assert waymo_group_id({"id": "s", "metadata": {}}) == "scenario:s"


def test_conversion_dependency_status_is_explicit() -> None:
    assert "tensorflow" in waymo_dependency_status()


def _polyline_feature(points: list[tuple[float, float]]) -> dict:
    return {"type": "road_edge", "polyline": [[x, y, 0.0] for x, y in points]}


def test_map_fingerprint_matches_for_co_located_geometry() -> None:
    # TEST-008 (SCENARIONET-INTEGRATION v1.2 REQ-006): two scenarios sharing
    # map geometry receive the same fingerprint even under a different
    # feature key ordering and a sub-meter positional jitter.
    shared_points = [(float(index) * 2.0, float(index) * 3.0) for index in range(60)]
    scenario_a = {
        "map_features": {
            "1": _polyline_feature(shared_points),
            "2": _polyline_feature([(100.0, 200.0), (101.0, 201.0)]),
        }
    }
    scenario_b = {
        "map_features": {
            "2": _polyline_feature([(100.05, 200.02), (101.0, 201.0)]),
            "1": _polyline_feature([(x + 0.01, y - 0.01) for x, y in shared_points]),
        }
    }
    assert waymo_map_fingerprint(scenario_a) == waymo_map_fingerprint(scenario_b)


def test_map_fingerprint_differs_for_different_geometry() -> None:
    scenario_a = {"map_features": {"1": _polyline_feature([(0.0, 0.0), (10.0, 10.0)])}}
    scenario_b = {"map_features": {"1": _polyline_feature([(500.0, 500.0), (510.0, 510.0)])}}
    assert waymo_map_fingerprint(scenario_a) != waymo_map_fingerprint(scenario_b)


def test_map_fingerprint_is_none_without_usable_geometry() -> None:
    assert waymo_map_fingerprint({}) is None
    assert waymo_map_fingerprint({"map_features": {}}) is None
    assert waymo_map_fingerprint({"map_features": {"1": {"type": "road_edge"}}}) is None


def test_bundled_converted_waymo_fixture_loads_with_training_origin() -> None:
    database = Path("third_party/metadrive/metadrive/assets/waymo")
    entries, groups = load_converted_waymo_entries(
        database,
        data_root=Path("."),
    )

    assert len(entries) == 3
    assert len(groups) == 3
    assert all(entry.record.official_split == "training_20s" for entry in entries)
    assert all(entry.record.relative_path.startswith("third_party/metadrive/") for entry in entries)
