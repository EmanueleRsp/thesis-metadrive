from __future__ import annotations

import os
import pickle
import subprocess
import sys
import importlib.util
from collections import deque
from pathlib import Path
from typing import Any, Sequence

from thesis_rl.scenarios.catalog import ScenarioCatalogEntry
from thesis_rl.scenarios.arms import assign_primary_arm, derive_scenario_tags
from thesis_rl.scenarios.features import extract_scenario_features
from thesis_rl.scenarios.quality import apply_catalog_quality_policy
from thesis_rl.scenarios.records import ScenarioRecord
from thesis_rl.scenarios.splits import (
    assert_no_group_overlap,
    assert_waymo_training_20s,
    assign_grouped_splits,
)
from thesis_rl.scenarios.waymo_monitor import NullProgressSink, ProgressSink


class WaymoConversionError(RuntimeError):
    pass


def waymo_dependency_status() -> dict[str, bool]:
    return {"tensorflow": importlib.util.find_spec("tensorflow") is not None}


def validate_training_20s_source(raw_data_path: str | Path) -> tuple[Path, ...]:
    root = Path(raw_data_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Waymo raw data directory does not exist: {root}")
    files = tuple(sorted(root.glob("training_20s.tfrecord*")))
    if not files:
        raise ValueError(
            "Waymo source must contain files named training_20s.tfrecord*; "
            f"none found in {root}"
        )
    return files


def build_converter_command(
    *,
    raw_data_path: str | Path,
    database_path: str | Path,
    num_workers: int = 8,
    num_files: int | None = None,
    overwrite: bool = False,
) -> list[str]:
    validate_training_20s_source(raw_data_path)
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    command = [
        sys.executable,
        "-m",
        "scenarionet.convert_waymo",
        "--raw_data_path",
        str(Path(raw_data_path).expanduser().resolve()),
        "--database_path",
        str(Path(database_path).expanduser().resolve()),
        "--dataset_name",
        "waymo",
        "--version",
        "training_20s",
        "--num_workers",
        str(num_workers),
    ]
    if num_files is not None:
        if num_files < 1:
            raise ValueError("num_files must be positive")
        command.extend(["--num_files", str(num_files)])
    if overwrite:
        command.append("--overwrite")
    return command


def convert_waymo_training_20s(
    *,
    raw_data_path: str | Path,
    database_path: str | Path,
    num_workers: int = 8,
    num_files: int | None = None,
    overwrite: bool = False,
    progress_sink: ProgressSink | None = None,
) -> list[str]:
    dependency_status = waymo_dependency_status()
    if not dependency_status["tensorflow"]:
        raise WaymoConversionError(
            "TensorFlow is required by the checked-out ScenarioNet Waymo converter "
            "but is not installed. Use the dedicated Waymo conversion environment "
            "or install the project conversion extra before retrying."
        )
    raw_files = validate_training_20s_source(raw_data_path)
    command = build_converter_command(
        raw_data_path=raw_data_path,
        database_path=database_path,
        num_workers=num_workers,
        num_files=num_files,
        overwrite=overwrite,
    )
    monitor = progress_sink or NullProgressSink()
    total_files = len(raw_files) if num_files is None else min(num_files, len(raw_files))
    monitor.emit({"kind": "started", "total_files": total_files})
    child_env = os.environ.copy()
    # Waymo conversion is intentionally CPU-only. Keep expected TensorFlow
    # CUDA/TensorRT diagnostics out of the user-facing terminal; failures are
    # still reported through the monitor and the exception below.
    child_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tail: deque[str] = deque(maxlen=40)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
        )
        assert process.stdout is not None
        from thesis_rl.scenarios.waymo_monitor import parse_converter_line

        for line in process.stdout:
            clean_line = line.strip()
            if clean_line:
                tail.append(clean_line)
            for event in parse_converter_line(line):
                monitor.emit(event)
        return_code = process.wait()
        monitor.emit({"kind": "finished", "exit_code": return_code})
        if return_code:
            details = "\n".join(tail)
            raise WaymoConversionError(
                "ScenarioNet Waymo conversion failed. "
                f"Check converter output (exit code {return_code}).\n{details}"
            )
    except FileNotFoundError as exc:
        raise WaymoConversionError(
            "Python executable for ScenarioNet converter is unavailable"
        ) from exc
    return command


def waymo_group_id(scenario: dict[str, Any]) -> str:
    metadata = scenario.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("source_log_id", "segment_id", "source_file_id", "source_file"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            normalized = str(value)
            if key in {"source_file_id", "source_file"}:
                normalized = Path(normalized).name
            return normalized
    scenario_id = str(scenario.get("id") or metadata.get("scenario_id") or "")
    if not scenario_id:
        raise ValueError("Waymo scenario has no grouping metadata or scenario id")
    return f"scenario:{scenario_id}"


def load_converted_waymo_entries(
    database_path: str | Path,
    *,
    data_root: str | Path | None = None,
    dataset_version: str = "training_20s",
) -> tuple[tuple[ScenarioCatalogEntry, ...], dict[str, str]]:
    root = Path(database_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"converted Waymo database does not exist: {root}")
    files = tuple(
        sorted(
            path
            for path in root.rglob("*.pkl")
            if path.name not in {"dataset_summary.pkl", "dataset_mapping.pkl"}
        )
    )
    if not files:
        raise ValueError(f"converted Waymo database contains no scenario files: {root}")
    base = Path(data_root).expanduser().resolve() if data_root is not None else root.parent.parent
    entries: list[ScenarioCatalogEntry] = []
    groups: dict[str, str] = {}
    for path in files:
        with path.open("rb") as handle:
            scenario = pickle.load(handle)
        scenario_id = str(scenario.get("id", ""))
        if not scenario_id:
            raise ValueError(f"scenario file has no id: {path}")
        features = extract_scenario_features(scenario, "waymo")
        group_id = waymo_group_id(scenario)
        relative_path = path.relative_to(base).as_posix()
        record = ScenarioRecord(
            scenario_uid=f"waymo:{dataset_version}:{scenario_id}",
            scenario_id=scenario_id,
            source="waymo",
            relative_path=relative_path,
            official_split="training_20s",
            source_log_id=group_id,
            source_scenario_id=scenario_id,
            dataset_version=dataset_version,
            converter_version=None,
            split="train",
            runtime_index=None,
            length=int(scenario["length"]),
            pg_profile=None,
            pg_seed=None,
            map_id=None,
            primary_arm=assign_primary_arm(features),
            tags=derive_scenario_tags(features),
            signal_reliability=features.signal_reliability,
            validation_status="valid",
            validation_warnings=(),
        )
        record = apply_catalog_quality_policy(record, features)
        entries.append(ScenarioCatalogEntry(record=record, features=features))
        groups[record.scenario_uid] = group_id
    assert_waymo_training_20s([entry.record for entry in entries])
    return tuple(entries), groups


def assign_waymo_internal_splits(
    entries: Sequence[ScenarioCatalogEntry],
    groups: dict[str, str],
    *,
    counts: dict[str, int],
    seed: int = 0,
) -> tuple[ScenarioCatalogEntry, ...]:
    records = assign_grouped_splits(
        [entry.record for entry in entries],
        group_id_by_uid=groups,
        counts=counts,
        seed=seed,
    )
    record_by_uid = {record.scenario_uid: record for record in records}
    result = tuple(
        ScenarioCatalogEntry(
            record=record_by_uid[entry.record.scenario_uid], features=entry.features
        )
        for entry in entries
    )
    assert_no_group_overlap([entry.record for entry in result], groups)
    return result
