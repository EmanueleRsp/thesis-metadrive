"""Integration test for scripts/build_named_panel_manifest.py against a
synthetic v1.2-style catalog (records carrying holdout_pool)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from thesis_rl.scenarios.arms import ARMS
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry, write_scenario_catalog
from thesis_rl.scenarios.panel_manifest import load_panel_manifest
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_named_panel_manifest.py"
_SPEC = importlib.util.spec_from_file_location("build_named_panel_manifest", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
build_named_panel_manifest = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_named_panel_manifest)


def _entry(
    source: str, arm: str, index: int, *, split: str, holdout_pool: str | None
) -> ScenarioCatalogEntry:
    scenario_id = f"{source}-{arm}-{index}"
    record = ScenarioRecord(
        scenario_uid=f"{source}:v1:{scenario_id}",
        scenario_id=scenario_id,
        source=source,  # type: ignore[arg-type]
        relative_path=f"{source}/database/{scenario_id}.pkl",
        official_split="training_20s" if source == "waymo" else None,
        source_log_id=f"log-{scenario_id}" if source == "waymo" else None,
        source_scenario_id=scenario_id if source == "waymo" else None,
        dataset_version="v1",
        converter_version=None,
        split=split,  # type: ignore[arg-type]
        runtime_index=index,
        length=10,
        pg_profile=None if source == "waymo" else "P0_simple",
        pg_seed=None if source == "waymo" else index,
        map_id=None,
        primary_arm=arm,
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
        rulebook_eligible=True,
        holdout_pool=holdout_pool,  # type: ignore[arg-type]
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


def test_build_named_panel_manifest_for_empirical_waymo_test(tmp_path: Path, monkeypatch) -> None:
    entries = []
    counter = 0
    for arm in ARMS:
        for _ in range(20):
            entries.append(_entry("waymo", arm, counter, split="test", holdout_pool="empirical"))
            counter += 1
    catalog_path = tmp_path / "catalog.parquet"
    write_scenario_catalog(entries, catalog_path)

    output_path = tmp_path / "test_waymo_empirical_panel.json"
    argv = [
        "build_named_panel_manifest",
        "--panel-name",
        "test_waymo_empirical",
        "--size",
        "50",
        "--seed",
        "7",
        "--catalog-path",
        str(catalog_path),
        "--output",
        str(output_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    build_named_panel_manifest.main()

    manifest = load_panel_manifest(output_path)
    assert manifest.draw_policy == "empirical"
    assert manifest.source == "waymo"
    assert manifest.split == "test"
    assert len(manifest.scenario_uids) == 50


def test_build_named_panel_manifest_for_arm_stratified_test(tmp_path: Path, monkeypatch) -> None:
    entries = []
    counter = 0
    for source in ("waymo", "pg"):
        for arm in ARMS:
            for _ in range(10):
                entries.append(
                    _entry(source, arm, counter, split="test", holdout_pool="stratified")
                )
                counter += 1
    catalog_path = tmp_path / "catalog.parquet"
    write_scenario_catalog(entries, catalog_path)

    output_path = tmp_path / "test_arm_stratified_panel.json"
    argv = [
        "build_named_panel_manifest",
        "--panel-name",
        "test_arm_stratified",
        "--size",
        "36",
        "--seed",
        "3",
        "--catalog-path",
        str(catalog_path),
        "--output",
        str(output_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    build_named_panel_manifest.main()

    manifest = load_panel_manifest(output_path)
    assert manifest.draw_policy == "arm_balanced"
    assert manifest.source == "combined"
    assert manifest.arms == ARMS
    assert sum(manifest.per_arm_counts) == 36


def test_build_named_panel_manifest_fails_closed_without_matching_holdout_pool(
    tmp_path: Path, monkeypatch
) -> None:
    entries = [
        _entry("waymo", ARMS[0], 0, split="test", holdout_pool="stratified"),
    ]
    catalog_path = tmp_path / "catalog.parquet"
    write_scenario_catalog(entries, catalog_path)
    argv = [
        "build_named_panel_manifest",
        "--panel-name",
        "test_waymo_empirical",
        "--size",
        "1",
        "--seed",
        "0",
        "--catalog-path",
        str(catalog_path),
        "--output",
        str(tmp_path / "out.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        build_named_panel_manifest.main()
        raised = False
    except SystemExit:
        raised = True
    assert raised
