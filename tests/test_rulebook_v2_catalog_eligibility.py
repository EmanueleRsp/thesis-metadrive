from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from thesis_rl.cli.scenarios.filter_rulebook_v2_catalog import main as filter_catalog_main
from thesis_rl.rulebook.v2.calibration import write_calibration_artifact
from thesis_rl.rulebook.v2.config import geometry_config_hash
from thesis_rl.rulebook.v2.context.catalog_eligibility import (
    evaluate_catalog_entries,
    evaluate_catalog_entry,
)
from thesis_rl.rulebook.v2.components.rss import RSSCalibrationArtifact
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry
from thesis_rl.scenarios.catalog import read_scenario_catalog, write_scenario_catalog
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord


def _entry(
    *,
    path: str,
    scenario_id: str = "eligibility-1",
    pg_seed: int = 1,
) -> ScenarioCatalogEntry:
    record = ScenarioRecord(
        scenario_uid=f"pg:eligibility:{scenario_id}",
        scenario_id=scenario_id,
        source="pg",
        relative_path=path,
        official_split=None,
        source_log_id=None,
        source_scenario_id=scenario_id,
        dataset_version="scenarionet_v1",
        converter_version=None,
        split="train",
        runtime_index=None,
        length=2,
        pg_profile="P0_simple",
        pg_seed=pg_seed,
        map_id="S",
        primary_arm="A0_simple_low_traffic",
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )
    features = ScenarioFeatures(
        scenario_id=scenario_id,
        source="pg",
        length=2,
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


def _scenario(*, off_lane: bool = False) -> dict:
    y = 100.0 if off_lane else 0.0
    return {
        "id": "eligibility-1",
        "length": 2,
        "metadata": {"sdc_id": "ego"},
        "tracks": {
            "ego": {
                "state": {
                    "position": [[1.0, y, 0.0], [2.0, y, 0.0]],
                    "heading": [0.0, 0.0],
                    "valid": [True, True],
                }
            }
        },
        "map_features": {
            "lane": {
                "type": "LANE_SURFACE_STREET",
                "polyline": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                "polygon": [[0.0, -2.0, 0.0], [10.0, -2.0, 0.0], [10.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
                "exit_lanes": [],
            }
        },
        "dynamic_map_states": {},
    }


def _write_scenario(
    root: Path,
    *,
    off_lane: bool = False,
    scenario_id: str = "1",
) -> str:
    relative = f"pg/database/P0_simple/{scenario_id}/scenario.pkl"
    path = root / relative
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
        pickle.dump(_scenario(off_lane=off_lane), handle)
    return relative


def test_geometry_config_hash_is_canonical_for_the_frozen_defaults() -> None:
    assert (
        geometry_config_hash() == "71d5cdb8e9a67c7f8b3b790d21687656b8cd95fe35686b6e5773968e679208d6"
    )


def test_catalog_eligibility_accepts_static_rulebook_compatible_entry(tmp_path: Path) -> None:
    relative = _write_scenario(tmp_path)
    result = evaluate_catalog_entry(
        _entry(path=relative),
        data_root=tmp_path,
        geometry_config_hash=geometry_config_hash(),
        calibration_hash="ego-hash",
    )
    assert result.rulebook_eligible
    assert result.validation_errors == ()
    assert result.assigned_route_lane_ids == ("lane",)
    assert result.assigned_route_source == "pg_sdc_offline_task_annotation"


def test_catalog_eligibility_excludes_unmappable_route_without_runtime_fallback(
    tmp_path: Path,
) -> None:
    relative = _write_scenario(tmp_path, off_lane=True)
    result = evaluate_catalog_entry(
        _entry(path=relative),
        data_root=tmp_path,
        geometry_config_hash=geometry_config_hash(),
        calibration_hash="ego-hash",
    )
    assert not result.rulebook_eligible
    assert result.validation_errors == ("task_route_lane_association_ambiguous_or_unavailable",)
    assert result.assigned_route_lane_ids == ()


def test_catalog_eligibility_parallel_path_matches_sequential_and_reports_progress(
    tmp_path: Path,
) -> None:
    eligible_path = _write_scenario(tmp_path, scenario_id="1")
    excluded_path = _write_scenario(tmp_path, off_lane=True, scenario_id="2")
    entries = (
        _entry(path=excluded_path, scenario_id="2", pg_seed=2),
        _entry(path=eligible_path, scenario_id="1", pg_seed=1),
    )
    kwargs = {
        "data_root": tmp_path,
        "geometry_config_hash": geometry_config_hash(),
        "calibration_hash": "ego-hash",
    }
    sequential = evaluate_catalog_entries(entries, workers=1, **kwargs)
    progress: list[tuple[int, int]] = []
    parallel = evaluate_catalog_entries(
        entries,
        workers=2,
        progress_callback=lambda completed, total: progress.append((completed, total)),
        **kwargs,
    )

    assert parallel == sequential
    assert [result.scenario_uid for result in parallel] == [
        "pg:eligibility:1",
        "pg:eligibility:2",
    ]
    assert progress == [(1, 2), (2, 2)]


def test_catalog_eligibility_rejects_non_positive_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers must be positive"):
        evaluate_catalog_entries(
            (),
            data_root=tmp_path,
            geometry_config_hash=geometry_config_hash(),
            calibration_hash="ego-hash",
            workers=0,
        )


def test_catalog_filter_cli_writes_audit_artifact_and_filters_split_input(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    relative = _write_scenario(tmp_path)
    catalog_path = write_scenario_catalog((_entry(path=relative),), tmp_path / "raw.parquet")
    ego_config = tmp_path / "ego_config.json"
    ego_config.write_text(
        json.dumps({"vehicle_config": {"vehicle_model": "default"}}), encoding="utf-8"
    )
    calibration_path = write_calibration_artifact(
        RSSCalibrationArtifact(
            config_hash="af7ab58234038ae1123049ca38dd24a9da9d5dd0bf8354d5a0bdbcd3fafac44c",
            ego_min_brake_mps2=4.0,
        ),
        tmp_path / "calibration.json",
    )
    output_catalog = tmp_path / "rulebook.parquet"
    artifact = tmp_path / "eligibility.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "filter_rulebook_v2_catalog",
            "--catalog",
            str(catalog_path),
            "--data-root",
            str(tmp_path),
            "--output-catalog",
            str(output_catalog),
            "--eligibility-output",
            str(artifact),
            "--ego-config",
            str(ego_config),
            "--calibration",
            str(calibration_path),
            "--workers",
            "2",
            "--overwrite",
        ],
    )
    assert filter_catalog_main() == 0
    selected_catalog = read_scenario_catalog(output_catalog)
    assert len(selected_catalog.entries) == 1
    selected_record = selected_catalog.entries[0].record
    assert selected_record.rulebook_eligible is True
    assert selected_record.rulebook_validation_errors == ()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["eligible_records"] == 1
    assert payload["excluded_records"] == 0
    captured = capsys.readouterr()
    assert "Evaluating 1 of 1 catalog entries" in captured.err
    assert "worker process(es)" in captured.err
    assert "Rulebook v2 filtering complete:" in captured.err
    assert "eligible=1, excluded=0" in captured.err

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("compatible eligibility should be reused")

    monkeypatch.setattr(
        "thesis_rl.cli.scenarios.filter_rulebook_v2_catalog.evaluate_catalog_entries",
        fail_if_recomputed,
    )
    assert filter_catalog_main() == 0
    cached_output = capsys.readouterr().err
    assert "Evaluating 0 of 1" in cached_output
    assert "reusing" in cached_output
    assert "compatible" in cached_output
