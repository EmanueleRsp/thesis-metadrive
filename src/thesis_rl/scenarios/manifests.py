from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]


class ManifestValidationError(ValueError):
    pass


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{path} must be a mapping")
    return value


def _require_keys(mapping: Mapping[str, Any], path: str, keys: set[str]) -> None:
    missing = sorted(keys.difference(mapping))
    if missing:
        raise ManifestValidationError(f"{path} is missing required keys: {missing}")


def validate_dataset_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(payload)
    _require_keys(
        manifest,
        "manifest",
        {"dataset_id", "software", "waymo", "procedural", "scenario_description", "creation"},
    )
    if not isinstance(manifest["dataset_id"], str) or not manifest["dataset_id"].strip():
        raise ManifestValidationError("manifest.dataset_id must be a non-empty string")

    software = _mapping(manifest["software"], "manifest.software")
    _require_keys(
        software,
        "manifest.software",
        {
            "project_commit",
            "metadrive_version",
            "metadrive_commit",
            "scenarionet_commit",
            "python_version",
        },
    )
    waymo = _mapping(manifest["waymo"], "manifest.waymo")
    _require_keys(
        waymo,
        "manifest.waymo",
        {
            "release",
            "source_variant",
            "converter_commit",
            "source_directory",
            "converted_directory",
        },
    )
    if waymo["source_variant"] != "training_20s":
        raise ManifestValidationError("manifest.waymo.source_variant must be 'training_20s'")
    procedural = _mapping(manifest["procedural"], "manifest.procedural")
    _require_keys(
        procedural,
        "manifest.procedural",
        {"generator_commit", "exporter_commit", "profiles_version"},
    )
    scenario_description = _mapping(
        manifest["scenario_description"], "manifest.scenario_description"
    )
    _require_keys(scenario_description, "manifest.scenario_description", {"version"})
    creation = _mapping(manifest["creation"], "manifest.creation")
    _require_keys(creation, "manifest.creation", {"created_at", "created_by_command"})
    return manifest


def validate_split_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(payload)
    _require_keys(
        manifest,
        "split_manifest",
        {
            "split_seed",
            "split_policy",
            "source_policy",
            "grouping",
            "targets",
            "balancing",
            "waymo_acquisition",
            "counts",
            "catalog_hash",
            "created_at",
        },
    )
    if not isinstance(manifest["split_seed"], int) or manifest["split_seed"] < 0:
        raise ManifestValidationError("split_manifest.split_seed must be a non-negative integer")
    counts = _mapping(manifest["counts"], "split_manifest.counts")
    _require_keys(counts, "split_manifest.counts", {"train", "validation", "test"})
    for split in ("train", "validation", "test"):
        split_counts = _mapping(counts[split], f"split_manifest.counts.{split}")
        _require_keys(split_counts, f"split_manifest.counts.{split}", {"waymo", "pg"})
        for source in ("waymo", "pg"):
            value = split_counts[source]
            if not isinstance(value, int) or value < 0:
                raise ManifestValidationError(
                    f"split_manifest.counts.{split}.{source} must be non-negative"
                )
    targets = _mapping(manifest["targets"], "split_manifest.targets")
    _require_keys(targets, "split_manifest.targets", {"waymo", "pg"})
    for source in ("waymo", "pg"):
        source_targets = _mapping(targets[source], f"split_manifest.targets.{source}")
        _require_keys(
            source_targets,
            f"split_manifest.targets.{source}",
            {"train", "validation", "test"},
        )
        for split in ("train", "validation", "test"):
            value = source_targets[split]
            if not isinstance(value, int) or value < 0:
                raise ManifestValidationError(
                    f"split_manifest.targets.{source}.{split} must be non-negative"
                )
    split_policy = manifest["split_policy"]
    if split_policy not in {
        "balanced_arm_source",
        "grouped_target",
        "exact",
        # SCENARIONET-INTEGRATION v1.2 SS3.5: empirical holdouts reserved
        # before an arm-stratified test pool, before a train pool built from
        # the residual under per-arm minimums. `record.split` remains one of
        # the three canonical values checked below; the empirical/stratified
        # distinction lives in `record.holdout_pool`, invisible to this
        # count-only validator by design.
        "holdout_first_empirical_then_stratified_then_balanced_train",
    }:
        raise ManifestValidationError("split_manifest.split_policy is unsupported")
    balancing = _mapping(manifest["balancing"], "split_manifest.balancing")
    _require_keys(
        balancing,
        "split_manifest.balancing",
        {
            "arm_targets",
            "max_arm_count_difference",
            "source_target_within_arm",
            "preserve_exact_source_totals",
            "structural_empty_cells",
            "allow_cross_source_fill_within_same_arm",
            "allow_relabeling",
            "allow_duplicate_records",
            "allow_quality_filter_relaxation",
        },
    )
    waymo_acquisition = _mapping(manifest["waymo_acquisition"], "split_manifest.waymo_acquisition")
    _require_keys(
        waymo_acquisition,
        "split_manifest.waymo_acquisition",
        {"ordering_seed", "batch_size_shards", "max_new_shards"},
    )
    for key in ("ordering_seed", "batch_size_shards", "max_new_shards"):
        value = waymo_acquisition[key]
        if not isinstance(value, int) or value < 0:
            raise ManifestValidationError(
                f"split_manifest.waymo_acquisition.{key} must be a non-negative integer"
            )
    if waymo_acquisition["batch_size_shards"] < 1 or waymo_acquisition["max_new_shards"] < 1:
        raise ManifestValidationError(
            "split_manifest.waymo_acquisition batch_size_shards and max_new_shards must be positive"
        )
    return manifest


def load_yaml_manifest(path: str | Path, *, kind: str) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    mapping = _mapping(payload, str(manifest_path))
    if kind == "dataset":
        return validate_dataset_manifest(mapping)
    if kind == "split":
        return validate_split_manifest(mapping)
    raise ValueError(f"unsupported manifest kind: {kind!r}")
