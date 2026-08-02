"""Read-only full-catalog preflight for the unified driving-mission contract."""

from __future__ import annotations

import csv
import json
import math
import pickle
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord, associate_route_lane


FROZEN_INDEX_SCHEMA = "scenarionet_frozen_selection_v1"
AUDIT_SCHEMA = "driving_mission_full_catalog_audit_v1"


@dataclass(frozen=True, slots=True)
class AuditRecordResult:
    scenario_uid: str
    source: str
    split: str
    primary_arm: str
    relative_path: str
    outcome: str
    detail: str | None
    goal_lane_id: str | None = None
    goal_s_m: float | None = None
    gate_count: int = 0
    allowed_span_count: int = 0
    lateral_relation_count: int = 0


@dataclass(frozen=True, slots=True)
class AuditResult:
    records: tuple[AuditRecordResult, ...]

    @property
    def passed(self) -> bool:
        return all(item.outcome == "pass" for item in self.records)

    def summary(self) -> dict[str, Any]:
        return {
            "schema": AUDIT_SCHEMA,
            "records_checked": len(self.records),
            "passed": self.passed,
            "outcomes": dict(sorted(Counter(item.outcome for item in self.records).items())),
            "by_source": _group_counts(self.records, "source"),
            "by_split": _group_counts(self.records, "split"),
            "by_arm": _group_counts(self.records, "primary_arm"),
        }


def _group_counts(records: Sequence[AuditRecordResult], field: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = {}
    for record in records:
        grouped.setdefault(str(getattr(record, field)), Counter())[record.outcome] += 1
    return {name: dict(sorted(counts.items())) for name, counts in sorted(grouped.items())}


def _failure(record: Mapping[str, Any], outcome: str, detail: str) -> AuditRecordResult:
    return AuditRecordResult(
        scenario_uid=str(record.get("scenario_uid", "")),
        source=str(record.get("source", "")),
        split=str(record.get("split", "")),
        primary_arm=str(record.get("primary_arm", "")),
        relative_path=str(record.get("relative_path", "")),
        outcome=outcome,
        detail=detail,
    )


def _lane_record(source: str, lane_id: str, lane: Mapping[str, Any], z_origin_m: float) -> RouteLaneRecord:
    if source == "pg":
        from thesis_rl.rulebook.v2.context.pg_static_adapter import _lane_record as build_lane
    elif source == "waymo":
        from thesis_rl.rulebook.v2.context.waymo_static_adapter import _lane_record as build_lane
    else:
        raise ValueError(f"unsupported source: {source!r}")
    return build_lane(lane_id, lane, z_origin_m=z_origin_m)


def _terminal_pose(scenario: Mapping[str, Any], expected_length: int) -> tuple[float, float, float, float]:
    metadata = scenario.get("metadata")
    tracks = scenario.get("tracks")
    if not isinstance(metadata, Mapping) or not isinstance(tracks, Mapping):
        raise ValueError("missing_metadata_or_tracks")
    track = tracks.get(metadata.get("sdc_id"))
    state = track.get("state") if isinstance(track, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("missing_sdc_state")
    positions, headings, valid = state.get("position"), state.get("heading"), state.get("valid")
    if not all(_has_length(value, expected_length) for value in (positions, headings, valid)):
        raise ValueError("invalid_sdc_trajectory_length")
    valid_indices = [index for index, item in enumerate(valid) if bool(item)]
    if not valid_indices:
        raise ValueError("no_valid_sdc_pose")
    index = valid_indices[-1]
    position = positions[index]
    if not _has_length_at_least(position, 2):
        raise ValueError("invalid_terminal_position")
    values = (float(position[0]), float(position[1]), float(position[2]) if len(position) > 2 else 0.0, float(headings[index]))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite_terminal_pose")
    return values


def _has_length(value: Any, expected_length: int) -> bool:
    try:
        return len(value) == expected_length
    except TypeError:
        return False


def _has_length_at_least(value: Any, minimum_length: int) -> bool:
    try:
        return len(value) >= minimum_length
    except TypeError:
        return False


def _neighbor_ids(lane: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("left_neighbor", "right_neighbor"):
        raw = lane.get(key, ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raw = (raw,)
        for item in raw:
            if isinstance(item, Mapping):
                feature_id = item.get("feature_id")
            else:
                feature_id = item
            if feature_id not in (None, ""):
                values.append(str(feature_id))
    return tuple(sorted(set(values)))


def _audit_record(record: Mapping[str, Any], data_root: Path) -> AuditRecordResult:
    relative_path = record.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        return _failure(record, "invalid_relative_path", "record has no relative_path")
    source_path = data_root / relative_path
    if not source_path.is_file():
        return _failure(record, "missing_file", f"source file does not exist: {source_path}")
    try:
        with source_path.open("rb") as handle:
            scenario = pickle.load(handle)
    except Exception as exc:
        return _failure(record, "unloadable", f"{type(exc).__name__}: {exc}")
    if not isinstance(scenario, Mapping):
        return _failure(record, "invalid_description", "ScenarioDescription is not a mapping")
    expected_length = record.get("length")
    if not isinstance(expected_length, int) or expected_length <= 0 or scenario.get("length") != expected_length:
        return _failure(record, "horizon_mismatch", "catalog and description horizon differ")
    route_ids = record.get("assigned_route_lane_ids")
    if not isinstance(route_ids, Sequence) or isinstance(route_ids, (str, bytes)) or not route_ids:
        return _failure(record, "missing_assigned_route", "assigned route is empty")
    route = tuple(str(value) for value in route_ids)
    features = scenario.get("map_features")
    if not isinstance(features, Mapping):
        return _failure(record, "missing_map_features", "map_features is not a mapping")
    lane_features: dict[str, Mapping[str, Any]] = {}
    for lane_id in route:
        feature = features.get(lane_id)
        if not isinstance(feature, Mapping) or not str(feature.get("type", "")).startswith("LANE_"):
            return _failure(record, "missing_preferred_lane", f"route lane unavailable: {lane_id}")
        lane_features[lane_id] = feature
    for previous, following in zip(route, route[1:]):
        successors = lane_features[previous].get("exit_lanes", ())
        if not isinstance(successors, Sequence) or isinstance(successors, (str, bytes)) or following not in {str(value) for value in successors}:
            return _failure(record, "non_contiguous_gate_order", f"{previous} does not lead to {following}")
    try:
        terminal_x, terminal_y, terminal_z, heading = _terminal_pose(scenario, expected_length)
        metadata = scenario["metadata"]
        tracks = scenario["tracks"]
        initial_position = tracks[metadata["sdc_id"]]["state"]["position"][0]
        z_origin_m = float(initial_position[2]) if len(initial_position) > 2 else 0.0
        lanes = tuple(
            _lane_record(str(record.get("source", "")), lane_id, lane, z_origin_m)
            for lane_id, lane in lane_features.items()
        )
        association = associate_route_lane(
            position_xy=(terminal_x, terminal_y),
            position_z=terminal_z - z_origin_m,
            heading_rad=heading,
            route_lanes=lanes,
        )
    except ValueError as exc:
        return _failure(record, "invalid_terminal_projection", str(exc))
    if association is None:
        return _failure(record, "invalid_terminal_projection", "terminal SDC pose has no unique compatible route lane")
    lateral_ids: set[str] = set()
    for lane in lane_features.values():
        for neighbor_id in _neighbor_ids(lane):
            neighbor = features.get(neighbor_id)
            if not isinstance(neighbor, Mapping) or not str(neighbor.get("type", "")).startswith("LANE_"):
                return _failure(record, "invalid_lateral_relation", f"unavailable neighbor lane: {neighbor_id}")
            lateral_ids.add(neighbor_id)
    return AuditRecordResult(
        scenario_uid=str(record["scenario_uid"]),
        source=str(record["source"]),
        split=str(record["split"]),
        primary_arm=str(record["primary_arm"]),
        relative_path=relative_path,
        outcome="pass",
        detail=None,
        goal_lane_id=association.lane_id,
        goal_s_m=association.route_projection_s_m,
        gate_count=max(0, len(route) - 1),
        allowed_span_count=len(route),
        lateral_relation_count=len(lateral_ids),
    )


def audit_frozen_index(payload: Mapping[str, Any], data_root: str | Path) -> AuditResult:
    """Audit exactly the immutable frozen-index records without writing source data."""
    if payload.get("schema") != FROZEN_INDEX_SCHEMA:
        raise ValueError(f"unsupported frozen index schema: {payload.get('schema')!r}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("frozen index must contain a non-empty records list")
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"scenario data root does not exist: {root}")
    return AuditResult(tuple(_audit_record(record, root) for record in records if isinstance(record, Mapping)))


def write_audit_report(result: AuditResult, output_dir: str | Path) -> None:
    """Write immutable audit artifacts and refuse to overwrite prior evidence."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {destination}")
    destination.mkdir(parents=True)
    summary = result.summary()
    (destination / "driving_mission_audit.json").write_text(
        json.dumps({**summary, "records": [asdict(item) for item in result.records]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (destination / "driving_mission_audit_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(result.records[0])))
        writer.writeheader()
        writer.writerows(asdict(item) for item in result.records)
    lines = ["# Unified Driving Mission Full-Catalog Audit", "", f"Status: `{'PASS' if result.passed else 'FAIL'}`", "", "## Summary", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in summary.items() if key not in {"by_source", "by_split", "by_arm"})
    for group in ("by_source", "by_split", "by_arm"):
        lines.extend(["", f"## Outcomes by {group[3:]}", "", "```json", json.dumps(summary[group], indent=2, sort_keys=True), "```"])
    (destination / "driving_mission_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
