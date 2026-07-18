from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import thesis_rl.scenarios.pg.report as pg_report
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry, write_scenario_catalog
from thesis_rl.scenarios.frozen import (
    build_frozen_index,
    frozen_catalog,
    load_frozen_index,
    verify_frozen_sources,
)
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord


def _entry(
    source: str, relative_path: str, *, runtime_index: int, seed: int | None = None
) -> ScenarioCatalogEntry:
    scenario_id = f"{source}-{runtime_index}"
    record = ScenarioRecord(
        scenario_uid=f"{source}:training_20s:{scenario_id}",
        scenario_id=scenario_id,
        source=source,  # type: ignore[arg-type]
        relative_path=relative_path,
        official_split="training_20s" if source == "waymo" else None,
        source_log_id="training_20s.tfrecord-00000-of-01000" if source == "waymo" else None,
        source_scenario_id=scenario_id,
        dataset_version="training_20s" if source == "waymo" else "pg",
        converter_version=None,
        split="train",
        runtime_index=runtime_index,
        length=10,
        pg_profile="P0_simple" if source == "pg" else None,
        pg_seed=seed,
        map_id=None,
        primary_arm="A0_simple_low_traffic",
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
        has_route_traffic_light=None,
        has_route_stop_sign=None,
        has_route_crosswalk=None,
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
    return ScenarioCatalogEntry(record=record, features=features)


def _split_manifest() -> dict[str, object]:
    return {
        "split_seed": 0,
        "split_policy": "balanced_arm_source",
        "source_policy": {},
        "grouping": {},
        "targets": {
            "waymo": {"train": 1, "validation": 0, "test": 0},
            "pg": {"train": 1, "validation": 0, "test": 0},
        },
        "balancing": {
            "arm_targets": "near_uniform",
            "max_arm_count_difference": 1,
            "source_target_within_arm": "best_effort_50_50",
            "preserve_exact_source_totals": True,
            "structural_empty_cells": {},
            "allow_cross_source_fill_within_same_arm": True,
            "allow_relabeling": False,
            "allow_duplicate_records": False,
            "allow_quality_filter_relaxation": False,
        },
        "waymo_acquisition": {"ordering_seed": 0, "batch_size_shards": 128, "max_new_shards": 256},
        "counts": {
            "train": {"waymo": 1, "pg": 1},
            "validation": {"waymo": 0, "pg": 0},
            "test": {"waymo": 0, "pg": 0},
        },
        "catalog_hash": None,
        "created_at": None,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, tuple[ScenarioCatalogEntry, ...]]:
    root = tmp_path / "scenarionet"
    waymo = root / "waymo/database/batches/batch_00000_00127/sd_waymo.pkl"
    pg = root / "pg/database/P0_simple/920000/sd_pg.pkl"
    waymo.parent.mkdir(parents=True)
    pg.parent.mkdir(parents=True)
    waymo.write_bytes(b"waymo")
    pg.write_bytes(b"pg")
    ledger = root / "waymo/acquisition/converted_shards.txt"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("training_20s.tfrecord-00000-of-01000\n", encoding="utf-8")
    manifest = root / "splits/split_manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(yaml.safe_dump(_split_manifest()), encoding="utf-8")
    entries = (
        _entry("pg", "pg/database/P0_simple/920000/sd_pg.pkl", runtime_index=0, seed=920000),
        _entry("waymo", "waymo/database/batches/batch_00000_00127/sd_waymo.pkl", runtime_index=1),
    )
    catalog = root / "catalog/scenario_catalog.parquet"
    write_scenario_catalog(entries, catalog)
    return root, catalog, ledger, entries


def test_freeze_index_captures_waymo_shards_and_pg_generation(tmp_path: Path) -> None:
    root, catalog, ledger, entries = _fixture(tmp_path)
    output = root / "frozen/scenario_selection_index.json"

    build_frozen_index(
        catalog_path=catalog,
        split_manifest_path=root / "splits/split_manifest.yaml",
        data_root=root,
        shard_ledger_path=ledger,
        output_path=output,
    )

    payload = load_frozen_index(output)
    assert payload["source_inventory"]["waymo"]["selected_shards"] == [
        "training_20s.tfrecord-00000-of-01000"
    ]
    assert payload["source_inventory"]["pg"]["generations"][0]["seed"] == 920000
    assert len(frozen_catalog(payload).entries) == len(entries)


def test_frozen_replay_verifies_source_files_without_search(tmp_path: Path) -> None:
    root, catalog, ledger, _entries = _fixture(tmp_path)
    output = root / "frozen/scenario_selection_index.json"
    build_frozen_index(
        catalog_path=catalog,
        split_manifest_path=root / "splits/split_manifest.yaml",
        data_root=root,
        shard_ledger_path=ledger,
        output_path=output,
    )
    payload = load_frozen_index(output)

    assert len(verify_frozen_sources(payload, root)) == 2
    (root / "pg/database/P0_simple/920000/sd_pg.pkl").unlink()
    with pytest.raises(FileNotFoundError, match="source files are missing"):
        verify_frozen_sources(payload, root)


def test_repository_index_contains_the_completed_source_selection() -> None:
    index_path = Path("data/scenarionet/frozen/scenario_selection_index.json")
    payload = load_frozen_index(index_path)

    assert len(payload["records"]) == 3500
    assert len(payload["source_inventory"]["waymo"]["selected_shards"]) == 768
    assert len(payload["source_inventory"]["pg"]["generations"]) == 1750
    assert payload["split_manifest"]["targets"] == {
        "waymo": {"train": 1000, "validation": 250, "test": 500},
        "pg": {"train": 1000, "validation": 250, "test": 500},
    }


def test_frozen_materializer_uses_exact_sources_without_remote_discovery() -> None:
    materializer = Path("scripts/materialize_frozen_scenarionet.sh").read_text(encoding="utf-8")
    expansion = Path("scripts/expand_waymo_pool.sh").read_text(encoding="utf-8")
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "data/scenarionet/frozen/scenario_selection_index.json" in materializer
    assert "WAYMO_FROZEN_SHARDS_FILE" in materializer
    assert "gcloud storage ls" not in materializer
    assert "generate_pg_from_frozen" in materializer
    assert "replay_frozen_dataset" in materializer
    assert "loaded frozen Waymo shard inventory" in expansion
    assert "scenarionet-materialize-frozen:" in makefile


def test_explicit_pg_task_runner_preserves_profile_seed_pairs(monkeypatch) -> None:
    seen: list[tuple[str, int]] = []

    def fake_run(task):
        seen.append((task.profile, task.seed))
        return (
            task,
            None,
            {
                "profile": task.profile,
                "seed": task.seed,
                "error_type": "SyntheticFailure",
                "error": "test",
            },
        )

    monkeypatch.setattr(pg_report, "_run_pg_task", fake_run)
    report, results = pg_report.run_pg_tasks(
        [("P5_complex_mixed", 920123), ("P0_simple", 920456)],
        data_root="/tmp/scenarionet",
        workers=1,
    )

    assert seen == [("P5_complex_mixed", 920123), ("P0_simple", 920456)]
    assert report.requested == 2
    assert report.generated == 0
    assert report.failed == 2
    assert results == ()
