"""Load exported procedural ScenarioNet records into the common catalog."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from thesis_rl.scenarios.arms import assign_primary_arm, derive_scenario_tags
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry
from thesis_rl.scenarios.features import extract_scenario_features
from thesis_rl.scenarios.parallel import ProgressCallback, ordered_process_map
from thesis_rl.scenarios.quality import apply_catalog_quality_policy
from thesis_rl.scenarios.pg.validation import validate_exported_scenario
from thesis_rl.scenarios.pg.profiles import PG_PROFILES
from thesis_rl.scenarios.records import ScenarioRecord


def _manifest_for(root: Path, scenario_id: str) -> dict[str, Any]:
    path = root / "pg" / "generation_manifests" / f"{scenario_id}.yaml"
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_exported_pg_entries(
    database_path: str | Path,
    *,
    data_root: str | Path,
    split: str = "train",
    dataset_version: str = "scenarionet_v1",
    seed_start: int | None = None,
    count_per_profile: int | None = None,
    workers: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> tuple[ScenarioCatalogEntry, ...]:
    """Load PG files while preserving realized topology from generation manifests."""

    database = Path(database_path).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    if not database.is_dir():
        raise FileNotFoundError(f"PG database does not exist: {database}")
    files = tuple(
        sorted(
            path
            for path in database.rglob("*.pkl")
            if path.name not in {"dataset_summary.pkl", "dataset_mapping.pkl"}
        )
    )
    if not files:
        raise ValueError(f"PG database contains no scenario files: {database}")

    if seed_start is not None or count_per_profile is not None:
        if seed_start is None or count_per_profile is None or count_per_profile < 1:
            raise ValueError(
                "seed_start and count_per_profile must be provided together and count_per_profile > 0"
            )
    tasks = tuple(
        (
            path,
            str(root),
            split,
            dataset_version,
            seed_start,
            count_per_profile,
        )
        for path in files
    )
    loaded = ordered_process_map(
        tasks,
        _load_pg_entry,
        workers=workers,
        progress_callback=progress_callback,
    )
    return tuple(entry for entry in loaded if entry is not None)


def _load_pg_entry(
    task: tuple[Path, str, str, str, int | None, int | None],
) -> ScenarioCatalogEntry | None:
    """Load and catalog one exported PG scenario in a worker process."""

    path, root_value, split, dataset_version, seed_start, count_per_profile = task
    root = Path(root_value)
    with path.open("rb") as handle:
        scenario = pickle.load(handle)
    if not isinstance(scenario, dict):
        raise ValueError(f"PG scenario is not a mapping: {path}")
    scenario_id = str(scenario.get("id", ""))
    if not scenario_id:
        raise ValueError(f"PG scenario has no id: {path}")
    manifest = _manifest_for(root, scenario_id)
    realized = manifest.get("realized")
    realized_metadata = realized if isinstance(realized, dict) else None
    features = extract_scenario_features(
        scenario,
        "pg",
        realized_generation_metadata=realized_metadata,
    )
    validation = validate_exported_scenario(scenario)
    generation = manifest.get("generation", {})
    if not isinstance(generation, dict):
        generation = {}
    profile = str(generation.get("profile") or path.parent.parent.name)
    seed_value = generation.get("seed")
    seed = int(seed_value) if seed_value is not None else int(path.parent.name)
    if seed_start is not None and count_per_profile is not None:
        profile_stride = 1_000_000
        in_window = any(
            seed_start + profile_index * profile_stride
            <= seed
            < seed_start + profile_index * profile_stride + count_per_profile
            for profile_index in range(len(PG_PROFILES))
        )
        if not in_window:
            return None
    block_sequence = realized_metadata.get("block_sequence", []) if realized_metadata else []
    map_id = "".join(str(token) for token in block_sequence) or profile
    static_metadata = realized_metadata.get("static_obstacle", {}) if realized_metadata else {}
    static_obstacle = isinstance(static_metadata, dict) and static_metadata.get("realized") is True
    record = ScenarioRecord(
        scenario_uid=f"pg:{dataset_version}:{scenario_id}",
        scenario_id=scenario_id,
        source="pg",
        relative_path=path.relative_to(root).as_posix(),
        official_split=None,
        source_log_id=None,
        source_scenario_id=scenario_id,
        dataset_version=dataset_version,
        converter_version=None,
        split=split,  # type: ignore[arg-type]
        runtime_index=None,
        length=int(validation.scenario_length),
        pg_profile=profile,
        pg_seed=seed,
        map_id=map_id,
        primary_arm=assign_primary_arm(features, has_static_obstacle=static_obstacle),
        tags=derive_scenario_tags(features, has_static_obstacle=static_obstacle),
        signal_reliability=features.signal_reliability,
        validation_status=validation.status,  # type: ignore[arg-type]
        validation_warnings=validation.warnings,
    )
    record = apply_catalog_quality_policy(record, features)
    return ScenarioCatalogEntry(record=record, features=features)


__all__ = ["load_exported_pg_entries"]
