from __future__ import annotations

from pathlib import Path

import pytest

from thesis_rl.scenarios.manifests import (
    ManifestValidationError,
    load_yaml_manifest,
    validate_dataset_manifest,
    validate_split_manifest,
)


def _dataset_manifest() -> dict[str, object]:
    return {
        "dataset_id": "scenarionet_v1",
        "software": {
            "project_commit": None,
            "metadrive_version": "0.4.3",
            "metadrive_commit": "md",
            "scenarionet_commit": "sn",
            "python_version": "3.10.20",
        },
        "waymo": {
            "release": None,
            "source_variant": "training_20s",
            "converter_commit": "sn",
            "source_directory": None,
            "converted_directory": None,
        },
        "procedural": {
            "generator_commit": "md",
            "exporter_commit": "md",
            "profiles_version": "pg_profiles_v1",
        },
        "scenario_description": {"version": None},
        "creation": {"created_at": None, "created_by_command": None},
    }


def _split_manifest() -> dict[str, object]:
    return {
        "split_seed": 0,
        "source_policy": {},
        "grouping": {},
        "counts": {
            "train": {"waymo": 1000, "pg": 1000},
            "validation": {"waymo": 250, "pg": 250},
            "test": {"waymo": 500, "pg": 500},
        },
        "catalog_hash": None,
        "created_at": None,
    }


def test_validate_dataset_manifest_accepts_incomplete_bootstrap_values() -> None:
    assert validate_dataset_manifest(_dataset_manifest())["dataset_id"] == "scenarionet_v1"


def test_validate_dataset_manifest_rejects_wrong_waymo_variant() -> None:
    payload = _dataset_manifest()
    payload["waymo"] = {**payload["waymo"], "source_variant": "validation"}  # type: ignore[misc]

    with pytest.raises(ManifestValidationError, match="training_20s"):
        validate_dataset_manifest(payload)


def test_validate_split_manifest_checks_all_source_counts() -> None:
    payload = _split_manifest()
    assert validate_split_manifest(payload)["split_seed"] == 0
    payload["counts"] = {**payload["counts"], "test": {"waymo": 1, "pg": -1}}  # type: ignore[misc]

    with pytest.raises(ManifestValidationError, match="test.pg"):
        validate_split_manifest(payload)


def test_load_yaml_manifest_dispatches_validation(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    import yaml  # type: ignore[import-untyped]

    path.write_text(yaml.safe_dump(_dataset_manifest()), encoding="utf-8")
    assert load_yaml_manifest(path, kind="dataset")["waymo"]["source_variant"] == "training_20s"
