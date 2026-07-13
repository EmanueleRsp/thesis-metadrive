from __future__ import annotations

from pathlib import Path

import pytest

from thesis_rl.scenarios.catalog import (
    ScenarioCatalog,
    ScenarioCatalogEntry,
    read_scenario_catalog,
    write_scenario_catalog,
)
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord


def _entry(uid: str, *, runtime_index: int, source: str = "pg") -> ScenarioCatalogEntry:
    scenario_id = uid.rsplit(":", 1)[-1]
    record = ScenarioRecord(
        scenario_uid=uid,
        scenario_id=scenario_id,
        source=source,  # type: ignore[arg-type]
        relative_path=f"{source}/database/{scenario_id}.pkl",
        official_split="training_20s" if source == "waymo" else None,
        source_log_id="log-1" if source == "waymo" else None,
        source_scenario_id=scenario_id if source == "waymo" else None,
        dataset_version="v1",
        converter_version="converter" if source == "waymo" else None,
        split="train",
        runtime_index=runtime_index,
        length=100,
        pg_profile=None if source == "waymo" else "P0_simple",
        pg_seed=None if source == "waymo" else runtime_index,
        map_id="S",
        primary_arm="A0_simple_low_traffic",
        tags=("has_crosswalk",),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )
    features = ScenarioFeatures(
        scenario_id=scenario_id,
        source=source,  # type: ignore[arg-type]
        length=100,
        route_length_m=50.0,
        topology_tag="simple",
        has_intersection=False,
        has_merge_or_roundabout=False,
        has_route_traffic_light=False,
        has_route_stop_sign=False,
        has_route_crosswalk=True,
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


def test_catalog_parquet_round_trip_and_runtime_resolution(tmp_path: Path) -> None:
    entries = [_entry("pg:v1:one", runtime_index=0), _entry("waymo:v1:two", runtime_index=1, source="waymo")]
    path = write_scenario_catalog(entries, tmp_path / "scenario_catalog.parquet")

    catalog = read_scenario_catalog(path)

    assert catalog.entries == tuple(entries)
    assert catalog.get_by_uid("pg:v1:one").record.runtime_index == 0
    assert catalog.get_by_runtime_index(split="train", runtime_index=1).record.source == "waymo"


def test_catalog_rejects_duplicate_uid_and_runtime_index() -> None:
    first = _entry("pg:v1:one", runtime_index=0)
    with pytest.raises(ValueError, match="duplicate scenario_uid"):
        ScenarioCatalog([first, first])
    with pytest.raises(ValueError, match="duplicate runtime_index"):
        ScenarioCatalog([first, _entry("pg:v1:two", runtime_index=0)])


def test_catalog_writer_does_not_overwrite_by_default(tmp_path: Path) -> None:
    path = tmp_path / "scenario_catalog.parquet"
    write_scenario_catalog([_entry("pg:v1:one", runtime_index=0)], path)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_scenario_catalog([_entry("pg:v1:one", runtime_index=0)], path)
