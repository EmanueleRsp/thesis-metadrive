"""Read-only construction of a non-canonical mission-aware frozen-index candidate."""

from __future__ import annotations

import hashlib
import json
import pickle
import gc
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from thesis_rl.mission.builder import build_driving_mission_from_source
from thesis_rl.mission.types import DrivingMissionRecord


CANDIDATE_SCHEMA = "scenarionet_frozen_mission_candidate_v1_1_1"
MISSION_FROZEN_INDEX_SCHEMA = "scenarionet_frozen_selection_mission_v1_1_1"


def build_candidate_index(
    index: Mapping[str, Any],
    data_root: str | Path,
    *,
    record_start: int = 0,
    record_end: int | None = None,
) -> dict[str, Any]:
    """Build a contiguous, read-only candidate batch from frozen source paths."""
    if index.get("schema") != "scenarionet_frozen_selection_v1":
        raise ValueError("M7a requires scenarionet_frozen_selection_v1 input")
    records = index.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("frozen index must contain records")
    if record_start < 0:
        raise ValueError("record_start must be non-negative")
    effective_end = len(records) if record_end is None else record_end
    if effective_end < record_start or effective_end > len(records):
        raise ValueError("record range is outside the frozen index")
    selected_records = records[record_start:effective_end]
    if not selected_records:
        raise ValueError("record range must contain at least one record")
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"scenario data root does not exist: {root}")
    candidate_records: list[dict[str, Any]] = []
    for raw_record in selected_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("frozen index contains a non-mapping record")
        record = deepcopy(dict(raw_record))
        relative_path = record.get("relative_path")
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise ValueError(f"unsafe frozen relative path: {relative_path!r}")
        source_path = root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"frozen source file is missing: {source_path}")
        with source_path.open("rb") as handle:
            scenario = pickle.load(handle)
        mission = build_driving_mission_from_source(
            scenario,
            scenario_uid=str(record["scenario_uid"]),
            source=str(record["source"]),
            assigned_route_lane_ids=tuple(
                str(value) for value in record["assigned_route_lane_ids"]
            ),
        )
        record["driving_mission"] = mission.to_dict()
        candidate_records.append(record)
        del scenario, mission
        gc.collect()
    identity = [
        (record["scenario_uid"], record["split"], record["relative_path"])
        for record in candidate_records
    ]
    if identity != [
        (record["scenario_uid"], record["split"], record["relative_path"])
        for record in selected_records
    ]:
        raise AssertionError("candidate materialization changed frozen identity")
    payload = {
        "schema": CANDIDATE_SCHEMA,
        "parent_selection_hash": str(index.get("selection_hash", "")),
        "parent_record_count": len(records),
        "record_start": record_start,
        "record_end": effective_end,
        "records": candidate_records,
    }
    payload["mission_selection_hash"] = hashlib.sha256(
        json.dumps(candidate_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def combine_candidate_indexes(candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine contiguous M7a batches without changing their frozen order."""
    if not candidates:
        raise ValueError("at least one candidate batch is required")
    parent_hash: str | None = None
    parent_count: int | None = None
    expected_start = 0
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("schema") != CANDIDATE_SCHEMA:
            raise ValueError("candidate batch has an unsupported schema")
        candidate_parent_hash = candidate.get("parent_selection_hash")
        candidate_parent_count = candidate.get("parent_record_count")
        start = candidate.get("record_start")
        end = candidate.get("record_end")
        batch_records = candidate.get("records")
        if (
            not isinstance(candidate_parent_hash, str)
            or not isinstance(candidate_parent_count, int)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not isinstance(batch_records, list)
            or start != expected_start
            or end <= start
            or end - start != len(batch_records)
        ):
            raise ValueError("candidate batches must be contiguous and well-formed")
        if parent_hash is None:
            parent_hash = candidate_parent_hash
            parent_count = candidate_parent_count
        elif candidate_parent_hash != parent_hash or candidate_parent_count != parent_count:
            raise ValueError("candidate batches have different frozen parents")
        records.extend(deepcopy(batch_records))
        expected_start = end
    if parent_count is None or expected_start != parent_count:
        raise ValueError("candidate batches do not cover the complete frozen index")
    payload = {
        "schema": CANDIDATE_SCHEMA,
        "parent_selection_hash": parent_hash,
        "records": records,
    }
    payload["mission_selection_hash"] = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def validate_candidate_index(
    candidate: Mapping[str, Any], parent_index: Mapping[str, Any]
) -> dict[str, int]:
    """Validate a complete candidate against its immutable frozen parent index."""
    if parent_index.get("schema") != "scenarionet_frozen_selection_v1":
        raise ValueError("M7a requires scenarionet_frozen_selection_v1 parent input")
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("candidate has an unsupported schema")
    parent_records = parent_index.get("records")
    candidate_records = candidate.get("records")
    if not isinstance(parent_records, list) or not isinstance(candidate_records, list):
        raise ValueError("parent and candidate must contain record lists")
    if candidate.get("parent_selection_hash") != parent_index.get("selection_hash"):
        raise ValueError("candidate parent selection hash does not match frozen index")
    if len(candidate_records) != len(parent_records):
        raise ValueError("candidate record count does not match frozen index")
    expected_hash = hashlib.sha256(
        json.dumps(candidate_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if candidate.get("mission_selection_hash") != expected_hash:
        raise ValueError("candidate mission selection hash is invalid")

    sources: dict[str, int] = {}
    for parent_record, candidate_record in zip(parent_records, candidate_records, strict=True):
        if not isinstance(parent_record, Mapping) or not isinstance(candidate_record, dict):
            raise ValueError("parent and candidate records must be mappings")
        parent_identity = tuple(
            parent_record.get(key) for key in ("scenario_uid", "split", "relative_path")
        )
        candidate_identity = tuple(
            candidate_record.get(key) for key in ("scenario_uid", "split", "relative_path")
        )
        if candidate_identity != parent_identity:
            raise ValueError("candidate changed frozen UID, split, path, or order")
        if not isinstance(candidate_record.get("driving_mission"), dict):
            raise ValueError("candidate record is missing a driving mission")
        mission = DrivingMissionRecord.from_dict(candidate_record["driving_mission"])
        if mission.scenario_uid != candidate_record["scenario_uid"]:
            raise ValueError("candidate mission UID does not match its scenario record")
        source = candidate_record.get("source")
        if not isinstance(source, str):
            raise ValueError("candidate record source must be a string")
        sources[source] = sources.get(source, 0) + 1
    return {"records": len(candidate_records), **sources}


def promote_candidate_index(
    candidate: Mapping[str, Any], parent_index: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a versioned canonical index from an approved M7a candidate.

    The frozen parent contributes selection provenance, source inventory, split
    manifests, and panel identities. The candidate contributes only validated
    mission-bearing records. Neither input mapping is mutated.
    """
    validate_candidate_index(candidate, parent_index)
    promoted = deepcopy(dict(parent_index))
    promoted["schema"] = MISSION_FROZEN_INDEX_SCHEMA
    promoted["parent_selection_schema"] = str(parent_index["schema"])
    promoted["parent_selection_hash"] = str(parent_index["selection_hash"])
    promoted["mission_selection_hash"] = str(candidate["mission_selection_hash"])
    promoted["selection_hash"] = str(candidate["mission_selection_hash"])
    promoted["records"] = deepcopy(candidate["records"])
    return promoted


def write_candidate_index(payload: Mapping[str, Any], output_path: str | Path) -> None:
    """Write one evidence artifact, refusing replacement or source-root output."""
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite candidate index: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
