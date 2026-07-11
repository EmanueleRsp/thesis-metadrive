from __future__ import annotations

import hashlib
import os
import pickle
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from thesis_rl.scenarios.records import ScenarioRecord


def assign_runtime_indices(records: Sequence[ScenarioRecord]) -> tuple[ScenarioRecord, ...]:
    if not records:
        raise ValueError("cannot assign runtime indices to an empty record sequence")
    splits = {record.split for record in records}
    if len(splits) != 1:
        raise ValueError(f"runtime database records must belong to one split, got {sorted(splits)}")
    ordered = sorted(records, key=lambda record: (record.source, record.scenario_uid))
    return tuple(
        replace(record, runtime_index=index) for index, record in enumerate(ordered)
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_dataset_metadata(database: Path) -> tuple[dict[str, Any], dict[str, str]]:
    summary_path = database / "dataset_summary.pkl"
    mapping_path = database / "dataset_mapping.pkl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"ScenarioNet summary is missing: {summary_path}")
    with summary_path.open("rb") as handle:
        summary = pickle.load(handle)
    if mapping_path.is_file():
        with mapping_path.open("rb") as handle:
            mapping = pickle.load(handle)
    else:
        mapping = {key: "" for key in summary}
    return dict(summary), dict(mapping)


def _atomic_pickle(path: Path, payload: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    temporary.replace(path)


def build_runtime_database(
    records: Sequence[ScenarioRecord],
    *,
    data_root: str | Path,
    runtime_directory: str | Path,
    overwrite: bool = False,
) -> Path:
    valid_records = tuple(
        record for record in records if record.validation_status in {"valid", "warning"}
    )
    if not valid_records:
        raise ValueError("runtime database requires at least one record")
    assigned_records = assign_runtime_indices(valid_records)
    root = Path(data_root).expanduser().resolve()
    runtime = Path(runtime_directory).expanduser().resolve()
    if runtime.exists() and not overwrite:
        summary_exists = (runtime / "dataset_summary.pkl").exists()
        if summary_exists:
            raise FileExistsError(f"refusing to overwrite runtime database: {runtime}")
    runtime.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    mapping: dict[str, str] = {}
    used_filenames: set[str] = set()
    ordered = sorted(
        assigned_records,
        key=lambda record: (
            record.runtime_index is None,
            record.runtime_index if record.runtime_index is not None else 0,
            record.scenario_uid,
        ),
    )
    for record in ordered:
        source_file = (root / record.relative_path).resolve()
        if not source_file.is_file():
            raise FileNotFoundError(f"catalog scenario file is missing: {source_file}")
        filename = source_file.name
        if filename in used_filenames:
            raise ValueError(f"duplicate runtime scenario filename: {filename}")
        used_filenames.add(filename)
        source_database = source_file.parent
        source_summary, source_mapping = _read_dataset_metadata(source_database)
        source_key = filename
        if source_key not in source_summary:
            # Some manually assembled fixtures use a summary key with a different
            # basename; resolve it through ScenarioNet's mapping when unambiguous.
            candidates = [
                key
                for key, subdir in source_mapping.items()
                if source_database / subdir / key == source_file
            ]
            if len(candidates) != 1:
                raise ValueError(f"scenario file is not represented in its summary: {source_file}")
            source_key = candidates[0]
        relative_source_directory = os.path.relpath(source_database, runtime)
        mapping[filename] = relative_source_directory
        summary[filename] = source_summary[source_key]

    _atomic_pickle(runtime / "dataset_summary.pkl", summary)
    _atomic_pickle(runtime / "dataset_mapping.pkl", mapping)
    return runtime


def verify_runtime_mapping(runtime_directory: str | Path) -> tuple[str, ...]:
    runtime = Path(runtime_directory).expanduser().resolve()
    summary, mapping = _read_dataset_metadata(runtime)
    missing: list[str] = []
    for filename in summary:
        path = runtime / mapping.get(filename, "") / filename
        if not path.is_file():
            missing.append(filename)
    if missing:
        raise FileNotFoundError(f"runtime mapping has missing files: {missing}")
    return tuple(summary)
