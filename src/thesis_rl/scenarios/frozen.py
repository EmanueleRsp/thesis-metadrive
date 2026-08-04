"""Freeze and replay an already selected ScenarioNet population."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from thesis_rl.scenarios.catalog import (
    ScenarioCatalog,
    ScenarioCatalogEntry,
    read_scenario_catalog,
)
from thesis_rl.scenarios.manifests import validate_split_manifest
from thesis_rl.scenarios.panel_manifest import (
    NAMED_PANELS,
    PanelManifest,
    PROFILE_SUBSET_SEEDS,
    PROFILE_SUBSET_SIZES,
    build_profile_subset_panel,
    named_panel_manifest_path,
    profile_panel_manifest_path,
    save_panel_manifest,
)
from thesis_rl.scenarios.paths import ScenarioDataPaths
from thesis_rl.scenarios.pipeline import SPLITS, assign_runtime_indices
from thesis_rl.scenarios.runtime_database import sha256_file


FROZEN_INDEX_SCHEMA = "scenarionet_frozen_selection_v1"
MISSION_FROZEN_INDEX_SCHEMA = "scenarionet_frozen_selection_mission_v1_1_1"
_BATCH_PATTERN = re.compile(r"(?:^|/)batch_(\d{5})_(\d{5})(?:/|$)")
_SHARD_PATTERN = re.compile(r"training_20s\.tfrecord-(\d+)-of-(\d+)$")


def _canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_shard_ledger(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Waymo shard ledger is missing: {path}")
    return tuple(
        sorted(
            {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        )
    )


def _shard_index(name: str) -> int | None:
    match = _SHARD_PATTERN.fullmatch(Path(name).name)
    return None if match is None else int(match.group(1))


def _batch_ranges(relative_paths: Sequence[str]) -> tuple[tuple[int, int, str], ...]:
    ranges: set[tuple[int, int, str]] = set()
    for relative_path in relative_paths:
        match = _BATCH_PATTERN.search(relative_path)
        if match is not None:
            first, last = int(match.group(1)), int(match.group(2))
            ranges.add((first, last, f"batch_{first:05d}_{last:05d}"))
    return tuple(sorted(ranges))


def _artifact_payload(data_root: Path, path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {
        "relative_path": ScenarioDataPaths(data_root).make_relative(resolved),
        "sha256": sha256_file(resolved),
    }


def _panel_from_payload(payload: Mapping[str, Any]) -> PanelManifest:
    """Load a panel manifest embedded in a frozen selection index."""
    return PanelManifest(
        schema_version=str(payload["schema_version"]),
        split=str(payload["split"]),
        seed=int(payload["seed"]),
        size=int(payload["size"]),
        arms=tuple(str(value) for value in payload["arms"]),
        per_arm_counts=tuple(int(value) for value in payload["per_arm_counts"]),
        scenario_uids=tuple(str(value) for value in payload["scenario_uids"]),
        sha256=str(payload["sha256"]),
        tracked_subset_uids=tuple(str(value) for value in payload.get("tracked_subset_uids", ())),
        draw_policy=str(payload.get("draw_policy", "arm_balanced")),
        source=str(payload.get("source", "combined")),
        parent_panel=(
            None if payload.get("parent_panel") in (None, "") else str(payload["parent_panel"])
        ),
        parent_sha256=(
            None if payload.get("parent_sha256") in (None, "") else str(payload["parent_sha256"])
        ),
        scope=str(payload.get("scope", "full")),
    )


def _manifest_payload(manifest: PanelManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "split": manifest.split,
        "seed": manifest.seed,
        "size": manifest.size,
        "arms": list(manifest.arms),
        "per_arm_counts": list(manifest.per_arm_counts),
        "scenario_uids": list(manifest.scenario_uids),
        "sha256": manifest.sha256,
        "tracked_subset_uids": list(manifest.tracked_subset_uids),
        "draw_policy": manifest.draw_policy,
        "source": manifest.source,
        "parent_panel": manifest.parent_panel,
        "parent_sha256": manifest.parent_sha256,
        "scope": manifest.scope,
    }


def _freeze_named_panels(
    catalog: ScenarioCatalog, root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate complete panels and materialize/hash their diagnostic children."""
    by_uid = {entry.record.scenario_uid: entry.record for entry in catalog.entries}
    frozen_panels: dict[str, dict[str, Any]] = {}
    for name, specification in NAMED_PANELS.items():
        path = named_panel_manifest_path(name, data_root=root.parent)
        if not path.is_file():
            raise FileNotFoundError(
                f"v1.2 frozen selection requires named panel {name!r}: {path}. "
                "Run `make scenarionet-v1-2-build-panels` before freezing."
            )
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = _panel_from_payload(raw_payload)
        manifest.verify_self_consistent()
        if (
            manifest.split != specification["split"]
            or manifest.source != specification["source"]
            or manifest.draw_policy != specification["draw_policy"]
        ):
            raise ValueError(f"named panel {name!r} does not match its declared v1.2 policy")
        expected_size = int(specification["full_size"])
        if manifest.size != expected_size:
            raise ValueError(
                f"named panel {name!r} has size {manifest.size}, expected complete pool "
                f"size {expected_size} under ScenarioNet v1.3"
            )
        invalid_uids = [
            uid
            for uid in manifest.scenario_uids
            if uid not in by_uid
            or by_uid[uid].split != specification["split"]
            or by_uid[uid].holdout_pool != specification["holdout_pool"]
            or (
                specification["source"] != "combined"
                and by_uid[uid].source != specification["source"]
            )
        ]
        if invalid_uids:
            raise ValueError(
                f"named panel {name!r} contains UIDs outside its frozen endpoint: "
                f"{invalid_uids[:5]}"
            )
        frozen_panels[name] = {
            "artifact": _artifact_payload(root, path),
            "manifest": raw_payload,
        }
    profile_panels: dict[str, dict[str, Any]] = {}
    for scope, panel_sizes in PROFILE_SUBSET_SIZES.items():
        scope_payload: dict[str, dict[str, Any]] = {}
        for name, size in panel_sizes.items():
            parent = _panel_from_payload(frozen_panels[name]["manifest"])
            child = build_profile_subset_panel(
                parent,
                tuple(entry.record for entry in catalog.entries),
                parent_name=name,
                scope=scope,
                size=size,
                seed=PROFILE_SUBSET_SEEDS[scope],
            )
            child_path = profile_panel_manifest_path(name, scope, data_root=root.parent)
            save_panel_manifest(child, child_path)
            scope_payload[name] = {
                "artifact": _artifact_payload(root, child_path),
                "manifest": _manifest_payload(child),
            }
        profile_panels[scope] = scope_payload
    return frozen_panels, profile_panels


def restore_frozen_panels(
    payload: Mapping[str, Any], data_root: str | Path, *, overwrite: bool
) -> tuple[Path, ...]:
    """Restore complete and profile-subset panel files and verify their hashes."""
    frozen_panels = payload.get("panel_manifests")
    if not frozen_panels:
        return ()
    if not isinstance(frozen_panels, Mapping):
        raise ValueError("frozen panel_manifests must be a mapping")
    root = Path(data_root).expanduser().resolve()
    restored: list[Path] = []
    for name in sorted(NAMED_PANELS):
        item = frozen_panels.get(name)
        if not isinstance(item, Mapping) or not isinstance(item.get("manifest"), Mapping):
            raise ValueError(f"frozen index is missing embedded named panel {name!r}")
        manifest = _panel_from_payload(item["manifest"])
        manifest.verify_self_consistent()
        target = named_panel_manifest_path(name, data_root=root.parent)
        content = json.dumps(item["manifest"], indent=2, sort_keys=True) + "\n"
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite frozen panel manifest: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        artifact = item.get("artifact")
        if not isinstance(artifact, Mapping) or sha256_file(target) != artifact.get("sha256"):
            raise ValueError(f"restored panel {name!r} does not match its frozen file hash")
        restored.append(target)
    profile_panels = payload.get("profile_panel_manifests", {})
    if not isinstance(profile_panels, Mapping):
        raise ValueError("frozen profile_panel_manifests must be a mapping")
    # v1.2 indexes remain replayable for historical reproducibility. They are
    # not accepted by the v1.3 runtime resolver, which separately requires
    # the profile children for an official run.
    if not profile_panels:
        return tuple(restored)
    for scope, expected_sizes in PROFILE_SUBSET_SIZES.items():
        scope_items = profile_panels.get(scope)
        if not isinstance(scope_items, Mapping):
            raise ValueError(f"frozen index is missing profile panel scope {scope!r}")
        for name, expected_size in expected_sizes.items():
            item = scope_items.get(name)
            if not isinstance(item, Mapping) or not isinstance(item.get("manifest"), Mapping):
                raise ValueError(f"frozen index is missing {scope} panel {name!r}")
            manifest = _panel_from_payload(item["manifest"])
            manifest.verify_self_consistent()
            if (
                manifest.scope != scope
                or manifest.parent_panel != name
                or manifest.size != expected_size
            ):
                raise ValueError(f"frozen {scope} panel {name!r} violates its profile contract")
            target = profile_panel_manifest_path(name, scope, data_root=root.parent)
            content = json.dumps(item["manifest"], indent=2, sort_keys=True) + "\n"
            if target.exists() and not overwrite:
                raise FileExistsError(f"refusing to overwrite frozen panel manifest: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            artifact = item.get("artifact")
            if not isinstance(artifact, Mapping) or sha256_file(target) != artifact.get("sha256"):
                raise ValueError(f"restored {scope} panel {name!r} does not match frozen file hash")
            restored.append(target)
    return tuple(restored)


def _validate_selected_population(
    catalog: ScenarioCatalog,
    split_manifest: Mapping[str, Any],
    data_root: Path | None,
) -> None:
    expected_counts = split_manifest["counts"]
    for split in SPLITS:
        for source in ("waymo", "pg"):
            actual = sum(
                entry.record.split == split and entry.record.source == source
                for entry in catalog.entries
            )
            expected = int(expected_counts[split][source])
            if actual != expected:
                raise ValueError(
                    f"frozen population count mismatch for {split}/{source}: "
                    f"manifest={expected}, catalog={actual}"
                )

    missing_runtime_index = [
        entry.record.scenario_uid for entry in catalog.entries if entry.record.runtime_index is None
    ]
    if missing_runtime_index:
        raise ValueError(
            f"cannot freeze a catalog without runtime indices: {missing_runtime_index[:5]}"
        )
    rejected = [
        entry.record.scenario_uid
        for entry in catalog.entries
        if entry.record.validation_status not in {"valid", "warning"}
        or entry.record.rulebook_eligible is not True
    ]
    if rejected:
        raise ValueError(
            f"cannot freeze records outside the validated runtime population: {rejected[:5]}"
        )

    expected_indices: dict[str, int | None] = {}
    for split in SPLITS:
        split_records = tuple(record for record in catalog.records if record.split == split)
        if not split_records:
            continue
        assigned = assign_runtime_indices(split_records)
        expected_indices.update({record.scenario_uid: record.runtime_index for record in assigned})
    actual_indices = {record.scenario_uid: record.runtime_index for record in catalog.records}
    if actual_indices != expected_indices:
        raise ValueError("catalog runtime indices are not reproducible from the selected records")

    if data_root is not None:
        paths = ScenarioDataPaths(data_root)
        missing_files = [
            entry.record.relative_path
            for entry in catalog.entries
            if not paths.resolve_relative(entry.record.relative_path).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"selected scenario files are missing from the data root: {missing_files[:5]}"
            )


def build_frozen_index(
    *,
    catalog_path: str | Path,
    split_manifest_path: str | Path,
    data_root: str | Path,
    shard_ledger_path: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Write an immutable selection index from the final ScenarioNet catalog."""

    root = Path(data_root).expanduser().resolve()
    catalog_file = Path(catalog_path).expanduser().resolve()
    split_file = Path(split_manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    selected_catalog = read_scenario_catalog(catalog_file)
    split_payload = validate_split_manifest(yaml.safe_load(split_file.read_text(encoding="utf-8")))
    _validate_selected_population(selected_catalog, split_payload, root)

    waymo_entries = [entry for entry in selected_catalog.entries if entry.record.source == "waymo"]
    waymo_paths = tuple(entry.record.relative_path for entry in waymo_entries)
    batch_ranges = _batch_ranges(waymo_paths)
    acquired_shards = _read_shard_ledger(Path(shard_ledger_path).expanduser().resolve())
    selected_shards = tuple(
        shard
        for shard in acquired_shards
        if (index := _shard_index(shard)) is not None
        and any(first <= index <= last for first, last, _batch_id in batch_ranges)
    )
    if waymo_entries and not selected_shards:
        raise ValueError("selected Waymo records do not map to any shard in the acquisition ledger")

    relative_source_files = tuple(
        sorted({entry.record.relative_path for entry in selected_catalog.entries})
    )
    pg_generations = [
        {
            "profile": entry.record.pg_profile,
            "seed": entry.record.pg_seed,
            "scenario_uid": entry.record.scenario_uid,
            "relative_path": entry.record.relative_path,
        }
        for entry in selected_catalog.entries
        if entry.record.source == "pg"
    ]
    waymo_scenarios = [
        {
            "scenario_uid": entry.record.scenario_uid,
            "relative_path": entry.record.relative_path,
            "batch_id": next(
                (
                    batch_id
                    for _first, _last, batch_id in batch_ranges
                    if f"/{batch_id}/" in f"/{entry.record.relative_path}/"
                ),
                None,
            ),
        }
        for entry in waymo_entries
    ]
    artifacts: dict[str, dict[str, str]] = {
        "catalog": _artifact_payload(root, catalog_file),
        "split_manifest": _artifact_payload(root, split_file),
    }
    for name, candidate in {
        "groups": root / "splits" / "scenario_groups.json",
        "thresholds": root / "splits" / "arm_thresholds.json",
        "rulebook_eligibility": root / "rulebook_v2" / "catalog_eligibility.json",
        "shard_ledger": Path(shard_ledger_path).expanduser().resolve(),
    }.items():
        if candidate.is_file():
            artifacts[name] = _artifact_payload(root, candidate)

    panel_manifests: dict[str, dict[str, Any]] = {}
    profile_panel_manifests: dict[str, dict[str, Any]] = {}
    if any(entry.record.holdout_pool is not None for entry in selected_catalog.entries):
        panel_manifests, profile_panel_manifests = _freeze_named_panels(selected_catalog, root)

    payload = {
        "schema": FROZEN_INDEX_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_policy": {
            "waymo_dataset_version": "training_20s",
            "source_files_are_not_copied": True,
            "selection_is_replayed_without_search": True,
        },
        "artifacts": artifacts,
        "panel_manifests": panel_manifests,
        "profile_panel_manifests": profile_panel_manifests,
        "split_manifest": split_payload,
        "records": [entry.to_flat_dict() for entry in selected_catalog.entries],
        "source_inventory": {
            "waymo": {
                "selected_batches": [batch_id for _first, _last, batch_id in batch_ranges],
                "selected_shards": list(selected_shards),
                "scenarios": waymo_scenarios,
                "selected_scenario_uids": [entry.record.scenario_uid for entry in waymo_entries],
            },
            "pg": {
                "generations": pg_generations,
            },
        },
        "source_file_paths": list(relative_source_files),
        "catalog_hash": sha256_file(catalog_file),
        "selection_hash": _canonical_json_hash(
            {
                "records": [entry.to_flat_dict() for entry in selected_catalog.entries],
                "split_manifest": split_payload,
            }
        ),
    }
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen selection index: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def load_frozen_index(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if payload.get("schema") not in {FROZEN_INDEX_SCHEMA, MISSION_FROZEN_INDEX_SCHEMA}:
        raise ValueError(f"unsupported frozen selection index schema: {payload.get('schema')!r}")
    if not isinstance(payload.get("records"), list) or not payload["records"]:
        raise ValueError("frozen selection index must contain a non-empty records list")
    return payload


def frozen_catalog(payload: Mapping[str, Any]) -> ScenarioCatalog:
    entries = tuple(ScenarioCatalogEntry.from_flat_dict(row) for row in payload["records"])
    catalog = ScenarioCatalog(entries)
    _validate_selected_population(catalog, payload["split_manifest"], None)
    return catalog


def verify_frozen_sources(payload: Mapping[str, Any], data_root: str | Path) -> tuple[str, ...]:
    paths = ScenarioDataPaths(Path(data_root).expanduser().resolve())
    missing = [
        relative_path
        for relative_path in payload["source_file_paths"]
        if not paths.resolve_relative(relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"frozen selection source files are missing: {missing[:5]}")
    catalog = ScenarioCatalog(
        tuple(ScenarioCatalogEntry.from_flat_dict(row) for row in payload["records"])
    )
    _validate_selected_population(catalog, payload["split_manifest"], paths.root)
    return tuple(payload["source_file_paths"])


__all__ = [
    "FROZEN_INDEX_SCHEMA",
    "MISSION_FROZEN_INDEX_SCHEMA",
    "build_frozen_index",
    "frozen_catalog",
    "load_frozen_index",
    "restore_frozen_panels",
    "verify_frozen_sources",
]
