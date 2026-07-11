from __future__ import annotations

import pytest

from thesis_rl.scenarios.catalog import ScenarioCatalogEntry
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord
from thesis_rl.scenarios.reports import compute_arm_distribution, compute_feature_statistics
from thesis_rl.scenarios.arms import classify_catalog_entry
from thesis_rl.scenarios.thresholds import (
    apply_traffic_thresholds,
    compute_arm_thresholds,
    read_arm_thresholds,
    write_arm_thresholds,
)


def _entry(index: int, source: str, value: float, *, split: str = "train") -> ScenarioCatalogEntry:
    record = ScenarioRecord(
        scenario_uid=f"{source}:v1:{index}",
        scenario_id=str(index),
        source=source,  # type: ignore[arg-type]
        relative_path=f"{source}/database/{index}.pkl",
        official_split="training_20s" if source == "waymo" else None,
        source_log_id=None,
        source_scenario_id=None,
        dataset_version="v1",
        converter_version=None,
        split=split,  # type: ignore[arg-type]
        runtime_index=index,
        length=10,
        pg_profile=None if source == "waymo" else "P0_simple",
        pg_seed=None if source == "waymo" else index,
        map_id=None,
        primary_arm="A0_simple_lane_follow",
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )
    features = ScenarioFeatures(
        scenario_id=str(index),
        source=source,  # type: ignore[arg-type]
        length=10,
        route_length_m=9.0,
        topology_tag="unknown",
        has_intersection=None,
        has_merge_or_roundabout=None,
        has_route_traffic_light=False,
        has_route_stop_sign=False,
        has_route_crosswalk=False,
        signal_reliability="not_applicable",
        has_vehicle=value > 0,
        has_pedestrian=False,
        has_cyclist=False,
        relevant_agents_q90=value,
        relevant_vehicles_q90=value,
        min_vehicle_distance_m=None,
        min_vru_distance_to_route_m=None,
        low_traffic=False,
        dense_traffic=False,
        vru_interaction=False,
    )
    return ScenarioCatalogEntry(record, features)


def test_thresholds_are_computed_on_balanced_train_sources() -> None:
    entries = [
        _entry(0, "waymo", 0),
        _entry(1, "waymo", 2),
        _entry(2, "pg", 4),
        _entry(3, "pg", 8),
    ]
    thresholds = compute_arm_thresholds(entries)

    assert thresholds.tau_low == 2
    assert thresholds.tau_dense == 5
    applied = apply_traffic_thresholds(entries[0], thresholds)
    assert applied.features.low_traffic is True
    assert applied.features.dense_traffic is False


def test_thresholds_reject_evaluation_and_unbalanced_sources() -> None:
    with pytest.raises(ValueError, match="train-only"):
        compute_arm_thresholds([_entry(0, "waymo", 1), _entry(1, "pg", 1, split="test")])
    with pytest.raises(ValueError, match="balanced"):
        compute_arm_thresholds([_entry(0, "waymo", 1)])


def test_threshold_file_round_trip_and_catalog_classification(tmp_path) -> None:
    entries = [
        _entry(0, "waymo", 0),
        _entry(1, "waymo", 2),
        _entry(2, "pg", 4),
        _entry(3, "pg", 8),
    ]
    thresholds = compute_arm_thresholds(entries)
    path = write_arm_thresholds(thresholds, tmp_path / "arm_thresholds.json")

    restored = read_arm_thresholds(path)
    classified = classify_catalog_entry(entries[3], restored)

    assert restored == thresholds
    assert classified.record.primary_arm == "A1_vehicle_interaction"
    assert "has_dense_traffic" in classified.record.tags

    distribution = compute_arm_distribution(
        [classify_catalog_entry(entry, restored) for entry in entries]
    )
    assert distribution["total"] == 4
    assert distribution["by_source"]["waymo"]
    assert distribution["by_source"]["pg"]
    statistics = compute_feature_statistics(entries)
    assert statistics["total"] == 4
    assert statistics["relevant_agents_q90"]["max"] == 8
