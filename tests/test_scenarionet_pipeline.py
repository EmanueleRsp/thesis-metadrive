from __future__ import annotations

from thesis_rl.scenarios.catalog import ScenarioCatalogEntry
from thesis_rl.scenarios.pipeline import (
    assign_catalog_runtime_indices,
    assign_source_splits,
    assign_source_splits_to_targets,
)
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord


def _entry(source: str, index: int) -> ScenarioCatalogEntry:
    scenario_id = f"{source}-{index}"
    record = ScenarioRecord(
        scenario_uid=f"{source}:v1:{index}",
        scenario_id=scenario_id,
        source=source,  # type: ignore[arg-type]
        relative_path=f"{source}/database/{scenario_id}.pkl",
        official_split="training_20s" if source == "waymo" else None,
        source_log_id=f"log-{index}" if source == "waymo" else None,
        source_scenario_id=scenario_id if source == "waymo" else None,
        dataset_version="v1",
        converter_version=None,
        split="train",
        runtime_index=None,
        length=10,
        pg_profile=None if source == "waymo" else "P0_simple",
        pg_seed=None if source == "waymo" else index,
        map_id="S",
        primary_arm="A0_simple_lane_follow",
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )
    features = ScenarioFeatures(
        scenario_id=scenario_id,
        source=source,  # type: ignore[arg-type]
        length=10,
        route_length_m=10.0,
        topology_tag="simple",
        has_intersection=False,
        has_merge_or_roundabout=False,
        has_route_traffic_light=False,
        has_route_stop_sign=False,
        has_route_crosswalk=False,
        signal_reliability="not_applicable",
        has_vehicle=False,
        has_pedestrian=False,
        has_cyclist=False,
        relevant_agents_q90=0.0,
        relevant_vehicles_q90=0.0,
        min_vehicle_distance_m=None,
        min_vru_distance_to_route_m=None,
        low_traffic=True,
        dense_traffic=False,
        vru_interaction=False,
    )
    return ScenarioCatalogEntry(record, features)


def test_pipeline_assigns_source_splits_and_runtime_indices() -> None:
    entries = tuple(_entry(source, index) for source in ("waymo", "pg") for index in range(2))
    split = assign_source_splits(
        entries,
        counts={
            "waymo": {"train": 1, "validation": 0, "test": 1},
            "pg": {"train": 1, "validation": 0, "test": 1},
        },
        seed=7,
    )
    assert {entry.record.split for entry in split} == {"train", "test"}

    indexed = assign_catalog_runtime_indices(split)
    for split_name in ("train", "test"):
        indices = sorted(
            entry.record.runtime_index
            for entry in indexed
            if entry.record.split == split_name
        )
        assert indices == [0, 1]


def test_pipeline_auto_split_preserves_whole_groups() -> None:
    entries = tuple(
        _entry(source, index)
        for source in ("waymo", "pg")
        for index in range(10)
    )
    split = assign_source_splits_to_targets(
        entries,
        targets={
            "waymo": {"train": 2, "validation": 2, "test": 2},
            "pg": {"train": 2, "validation": 2, "test": 2},
        },
        seed=3,
    )
    for source in ("waymo", "pg"):
        source_entries = [entry for entry in split if entry.record.source == source]
        assert len(source_entries) == 6
        assert {
            entry.record.split
            for entry in source_entries
        } == {"train", "validation", "test"}
