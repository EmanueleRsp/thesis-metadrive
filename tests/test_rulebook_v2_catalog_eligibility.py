from __future__ import annotations

import json
import pickle
from pathlib import Path

from thesis_rl.cli.scenarios.filter_rulebook_v2_catalog import main as filter_catalog_main
from thesis_rl.rulebook.v2.calibration import write_calibration_artifact
from thesis_rl.rulebook.v2.config import geometry_config_hash
from thesis_rl.rulebook.v2.context.catalog_eligibility import evaluate_catalog_entry
from thesis_rl.rulebook.v2.components.rss import RSSCalibrationArtifact
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry
from thesis_rl.scenarios.catalog import read_scenario_catalog, write_scenario_catalog
from thesis_rl.scenarios.records import ScenarioFeatures, ScenarioRecord


def _entry(*, path: str) -> ScenarioCatalogEntry:
    record = ScenarioRecord(
        scenario_uid="pg:eligibility:1",
        scenario_id="eligibility-1",
        source="pg",
        relative_path=path,
        official_split=None,
        source_log_id=None,
        source_scenario_id="eligibility-1",
        dataset_version="scenarionet_v1",
        converter_version=None,
        split="train",
        runtime_index=None,
        length=2,
        pg_profile="P0_simple",
        pg_seed=1,
        map_id="S",
        primary_arm="A0_simple_low_traffic",
        tags=(),
        signal_reliability="not_applicable",
        validation_status="valid",
        validation_warnings=(),
    )
    features = ScenarioFeatures(
        scenario_id="eligibility-1",
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


def _write_scenario(root: Path, *, off_lane: bool = False) -> str:
    relative = "pg/database/P0_simple/1/scenario.pkl"
    path = root / relative
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
        pickle.dump(_scenario(off_lane=off_lane), handle)
    return relative


def test_geometry_config_hash_is_canonical_for_the_frozen_defaults() -> None:
    assert geometry_config_hash() == "f08ef3fb4790d532275974aa14bf08f91b62e9f0b55cc1a05d8d60ec4070eb97"


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


def test_catalog_eligibility_excludes_unmappable_route_without_runtime_fallback(tmp_path: Path) -> None:
    relative = _write_scenario(tmp_path, off_lane=True)
    result = evaluate_catalog_entry(
        _entry(path=relative),
        data_root=tmp_path,
        geometry_config_hash=geometry_config_hash(),
        calibration_hash="ego-hash",
    )
    assert not result.rulebook_eligible
    assert result.validation_errors == ("task_route_lane_association_ambiguous_or_unavailable",)


def test_catalog_filter_cli_writes_audit_artifact_and_filters_split_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    relative = _write_scenario(tmp_path)
    catalog_path = write_scenario_catalog((_entry(path=relative),), tmp_path / "raw.parquet")
    ego_config = tmp_path / "ego_config.json"
    ego_config.write_text(json.dumps({"vehicle_config": {"vehicle_model": "default"}}), encoding="utf-8")
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
        ],
    )
    assert filter_catalog_main() == 0
    assert len(read_scenario_catalog(output_catalog).entries) == 1
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["eligible_records"] == 1
    assert payload["excluded_records"] == 0
