"""Integration test for the SCENARIONET-INTEGRATION v1.2 holdout-first
`build_splits_v1_2` CLI: runs `main()` directly against a synthetic parquet
catalog and checks the produced catalog and split manifest.
"""

from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from thesis_rl.cli.scenarios import build_splits_v1_2
from thesis_rl.scenarios.arms import ARMS
from thesis_rl.scenarios.catalog import (
    ScenarioCatalogEntry,
    read_scenario_catalog,
    write_scenario_catalog,
)
from thesis_rl.scenarios.manifests import validate_split_manifest
from thesis_rl.scenarios.pg.profiles import PG_HOLDOUT_EQUIPROBABLE_MIXTURE
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord

_PG_PROFILE_CYCLE = sorted(PG_HOLDOUT_EQUIPROBABLE_MIXTURE)


def _entry(source: str, arm: str, index: int) -> ScenarioCatalogEntry:
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
        split="train",
        runtime_index=None,
        length=10,
        pg_profile=None if source == "waymo" else _PG_PROFILE_CYCLE[index % len(_PG_PROFILE_CYCLE)],
        pg_seed=None if source == "waymo" else index,
        map_id=None,
        primary_arm=arm,
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
        rulebook_eligible=True,
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


def _build_catalog(per_arm_per_source: int) -> list[ScenarioCatalogEntry]:
    entries = []
    counter = 0
    for source in ("waymo", "pg"):
        for arm in ARMS:
            for _ in range(per_arm_per_source):
                entries.append(_entry(source, arm, counter))
                counter += 1
    return entries


def test_build_splits_v1_2_cli_end_to_end(tmp_path: Path, monkeypatch) -> None:
    entries = _build_catalog(per_arm_per_source=80)
    catalog_path = tmp_path / "catalog.parquet"
    write_scenario_catalog(entries, catalog_path)

    output_path = tmp_path / "split_catalog.parquet"
    manifest_path = tmp_path / "split_manifest.yaml"
    pg_report_path = tmp_path / "pg_replenishment_report.json"

    argv = [
        "build_splits_v1_2",
        "--catalog",
        str(catalog_path),
        "--output",
        str(output_path),
        "--split-manifest",
        str(manifest_path),
        "--pg-replenishment-report",
        str(pg_report_path),
        "--split-seed",
        "11",
        "--waymo-test-empirical",
        "40",
        "--pg-test-empirical",
        "40",
        "--waymo-validation",
        "20",
        "--pg-validation",
        "20",
        "--waymo-train",
        "300",
        "--pg-train",
        "300",
        "--stratified-total",
        "24",
        # 60/180 pg draws carry sampling noise around the exact 20% mixture
        # (multinomial std ~5pp at this sample size); loosen the tolerance
        # for this synthetic fixture. assert_pg_holdout_profile_mixture's
        # exact-fraction behavior is already unit-tested directly.
        "--pg-holdout-mixture-tolerance",
        "0.2",
        "--overwrite",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = build_splits_v1_2.main()
    assert exit_code == 0

    result_catalog = read_scenario_catalog(output_path)
    assert len(result_catalog.entries) == 744

    manifest_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    validated = validate_split_manifest(manifest_payload)
    assert validated["split_policy"] == build_splits_v1_2.SPLIT_POLICY
    assert validated["counts"]["test"]["waymo"] > 0
    assert validated["counts"]["validation"]["waymo"] == 20
    assert "holdout_policy" in manifest_payload
    assert manifest_payload["holdout_policy"]["pg_holdout_mixture_error"] is None

    split_counts: dict[str, int] = {}
    for entry in result_catalog.entries:
        split_counts[entry.record.split] = split_counts.get(entry.record.split, 0) + 1
    assert split_counts["train"] == 600
    assert split_counts["validation"] == 40
    test_empirical = sum(
        1
        for entry in result_catalog.entries
        if entry.record.split == "test" and entry.record.holdout_pool == "empirical"
    )
    test_stratified = sum(
        1
        for entry in result_catalog.entries
        if entry.record.split == "test" and entry.record.holdout_pool == "stratified"
    )
    assert test_empirical == 80
    assert test_stratified == 24
    assert {
        source: sum(
            entry.record.split == "train" and entry.record.source == source
            for entry in result_catalog.entries
        )
        for source in ("waymo", "pg")
    } == {"waymo": 300, "pg": 300}
    train_by_arm = {
        arm: sum(
            entry.record.split == "train" and entry.record.primary_arm == arm
            for entry in result_catalog.entries
        )
        for arm in ARMS
    }
    assert max(train_by_arm.values()) - min(train_by_arm.values()) <= 1


def test_build_splits_v1_2_cli_reports_pg_replenishment_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    entries = _build_catalog(per_arm_per_source=2)  # far too few for the requested counts
    catalog_path = tmp_path / "catalog.parquet"
    write_scenario_catalog(entries, catalog_path)

    output_path = tmp_path / "split_catalog.parquet"
    manifest_path = tmp_path / "split_manifest.yaml"
    pg_report_path = tmp_path / "pg_replenishment_report.json"
    argv = [
        "build_splits_v1_2",
        "--catalog",
        str(catalog_path),
        "--output",
        str(output_path),
        "--split-manifest",
        str(manifest_path),
        "--pg-replenishment-report",
        str(pg_report_path),
        "--split-seed",
        "0",
        "--waymo-test-empirical",
        "10000",
        "--pg-test-empirical",
        "10000",
        "--waymo-validation",
        "1",
        "--pg-validation",
        "1",
        "--waymo-train",
        "1",
        "--pg-train",
        "1",
        "--stratified-total",
        "6",
        "--overwrite",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        build_splits_v1_2.main()
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert pg_report_path.is_file()


def test_build_splits_v1_2_cli_restricts_pg_empirical_holdouts_to_declared_seeds(
    tmp_path: Path, monkeypatch
) -> None:
    entries = _build_catalog(per_arm_per_source=60)
    seed_start = 2_000_000
    per_profile = 4
    declared_holdouts = []
    for profile_index, profile in enumerate(_PG_PROFILE_CYCLE):
        for offset in range(per_profile):
            entry = _entry(
                "pg",
                "A0_simple_low_traffic",
                10_000 + profile_index * per_profile + offset,
            )
            declared_holdouts.append(
                ScenarioCatalogEntry(
                    record=replace(
                        entry.record,
                        pg_profile=profile,
                        pg_seed=seed_start + profile_index * 1_000_000 + offset,
                    ),
                    features=entry.features,
                )
            )
    catalog_path = tmp_path / "catalog.parquet"
    write_scenario_catalog(entries + declared_holdouts, catalog_path)
    output_path = tmp_path / "split_catalog.parquet"
    manifest_path = tmp_path / "split_manifest.yaml"
    report_path = tmp_path / "pg_replenishment_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_splits_v1_2",
            "--catalog",
            str(catalog_path),
            "--output",
            str(output_path),
            "--split-manifest",
            str(manifest_path),
            "--pg-replenishment-report",
            str(report_path),
            "--waymo-test-empirical",
            "20",
            "--pg-test-empirical",
            "10",
            "--waymo-validation",
            "10",
            "--pg-validation",
            "10",
            "--waymo-train",
            "180",
            "--pg-train",
            "180",
            "--stratified-total",
            "0",
            "--pg-holdout-seed-start",
            str(seed_start),
            "--pg-holdout-count-per-profile",
            "auto",
            "--overwrite",
        ],
    )
    assert build_splits_v1_2.main() == 0

    result = read_scenario_catalog(output_path).entries
    empirical_pg = [
        entry
        for entry in result
        if entry.record.source == "pg" and entry.record.holdout_pool == "empirical"
    ]
    assert len(empirical_pg) == 20
    assert {entry.record.pg_seed for entry in empirical_pg} == {
        seed_start + profile_index * 1_000_000 + offset
        for profile_index in range(len(_PG_PROFILE_CYCLE))
        for offset in range(per_profile)
    }
    assert any(
        entry.record.source == "pg"
        and entry.record.pg_seed is not None
        and entry.record.pg_seed < seed_start
        and entry.record.split == "train"
        for entry in result
    )
