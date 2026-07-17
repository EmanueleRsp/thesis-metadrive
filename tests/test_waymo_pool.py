from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from thesis_rl.cli.scenarios.waymo_pool_status import main as waymo_pool_status_main
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry, write_scenario_catalog
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord
from thesis_rl.scenarios.waymo_pool import (
    WAYMO_POOL_POLICY_VERSION,
    fingerprint_waymo_database,
    summarize_waymo_pool,
)


def _entry(index: int, *, reliability: str, arm: str) -> ScenarioCatalogEntry:
    scenario_id = f"scenario-{index}"
    record = ScenarioRecord(
        scenario_uid=f"waymo:training_20s:{scenario_id}",
        scenario_id=scenario_id,
        source="waymo",
        relative_path=f"waymo/database/{scenario_id}.pkl",
        official_split="training_20s",
        source_log_id=f"training_20s.tfrecord-{index:05d}-of-01000",
        source_scenario_id=scenario_id,
        dataset_version="training_20s",
        converter_version=None,
        split="train",
        runtime_index=None,
        length=10,
        pg_profile=None,
        pg_seed=None,
        map_id=None,
        primary_arm=arm,
        tags=(),
        signal_reliability=reliability,  # type: ignore[arg-type]
        validation_status="valid",
        validation_warnings=(),
    )
    features = ScenarioFeatures(
        scenario_id=scenario_id,
        source="waymo",
        length=10,
        route_length_m=10.0,
        topology_tag="simple",
        has_intersection=False,
        has_merge_or_roundabout=False,
        has_route_traffic_light=False,
        has_route_stop_sign=False,
        has_route_crosswalk=False,
        signal_reliability=reliability,  # type: ignore[arg-type]
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


def test_waymo_pool_counts_only_allowed_signal_reliabilities() -> None:
    entries = (
        _entry(0, reliability="complete", arm="A1_traffic"),
        _entry(1, reliability="not_applicable", arm="A2_junction"),
        _entry(2, reliability="partial", arm="A3_complex_junction"),
    )
    status = summarize_waymo_pool(
        entries,
        allowed_signal_reliabilities=("complete", "not_applicable"),
        required=3,
    )

    assert status.total == 3
    assert status.eligible == 2
    assert status.deficit == 1
    assert status.complete is False
    assert status.eligible_by_arm["A3_complex_junction"] == 0
    assert len(status.source_shards) == 3


def test_waymo_pool_excludes_invalid_quality_records() -> None:
    valid = _entry(0, reliability="complete", arm="A1_traffic")
    invalid = ScenarioCatalogEntry(
        replace(
            valid.record,
            scenario_uid="waymo:training_20s:invalid",
            validation_status="invalid",
        ),
        valid.features,
    )

    status = summarize_waymo_pool(
        (valid, invalid),
        allowed_signal_reliabilities=("complete",),
        required=2,
    )

    assert status.eligible == 1
    assert status.deficit == 1


def test_waymo_pool_requires_configured_arm_counts() -> None:
    entries = (
        _entry(0, reliability="complete", arm="A1_traffic"),
        _entry(1, reliability="complete", arm="A4_vru"),
    )

    status = summarize_waymo_pool(
        entries,
        allowed_signal_reliabilities=("complete",),
        required=2,
        required_by_arm={"A4_vru": 2},
    )

    assert status.deficit == 0
    assert status.arm_deficits == {"A4_vru": 1}
    assert status.complete is False


def test_waymo_pool_can_require_rulebook_eligibility() -> None:
    eligible = _entry(0, reliability="complete", arm="A1_traffic")
    unverified = _entry(1, reliability="complete", arm="A1_traffic")
    status = summarize_waymo_pool(
        (
            ScenarioCatalogEntry(
                replace(eligible.record, rulebook_eligible=True),
                eligible.features,
            ),
            unverified,
        ),
        allowed_signal_reliabilities=("complete",),
        required=2,
        require_rulebook_eligible=True,
    )

    assert status.eligible == 1
    assert status.deficit == 1


def test_waymo_pool_status_cli_reads_rulebook_annotated_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    eligible = _entry(0, reliability="complete", arm="A1_traffic")
    catalog_path = write_scenario_catalog(
        (
            ScenarioCatalogEntry(
                replace(eligible.record, rulebook_eligible=True),
                eligible.features,
            ),
        ),
        tmp_path / "catalog.parquet",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "waymo_pool_status",
            "--catalog",
            str(catalog_path),
            "--data-root",
            str(tmp_path),
            "--required",
            "1",
            "--allowed-signal-reliability",
            "complete",
            "--require-rulebook-eligible",
        ],
    )

    assert waymo_pool_status_main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["eligible"] == 1
    assert payload["require_rulebook_eligible"] is True


def test_waymo_pool_rejects_duplicate_scenarios() -> None:
    entry = _entry(0, reliability="complete", arm="A1_traffic")
    duplicate = ScenarioCatalogEntry(
        replace(entry.record, relative_path="waymo/database/duplicate.pkl"),
        entry.features,
    )
    with pytest.raises(ValueError, match="duplicate scenarios"):
        summarize_waymo_pool(
            (entry, duplicate),
            allowed_signal_reliabilities=("complete",),
            required=1,
        )


def test_waymo_database_fingerprint_changes_with_candidate_files(tmp_path) -> None:
    database = tmp_path / "database"
    database.mkdir()
    empty = fingerprint_waymo_database(database)
    candidate = database / "sd_example.pkl"
    candidate.write_bytes(b"first")
    first = fingerprint_waymo_database(database)
    candidate.write_bytes(b"second")

    assert first != empty
    assert fingerprint_waymo_database(database) != first


def test_waymo_pool_policy_version_invalidates_old_status_cache() -> None:
    assert WAYMO_POOL_POLICY_VERSION == "waymo_pool_v6"
