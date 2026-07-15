"""Task-route records and offline eligibility artifacts.

The functions here intentionally accept only static lane topology.  A future
SDC trajectory, timestamps, poses and velocities cannot be represented by the
record or consumed by runtime validation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

from thesis_rl.rulebook.v2.types import TaskRouteRecord


RULEBOOK_V2_ADAPTER_CONTRACT_VERSION = "task-route-v1"


@dataclass(frozen=True, slots=True)
class TaskRouteEligibility:
    scenario_uid: str
    rulebook_version: str
    adapter_version: str
    geometry_config_hash: str
    calibration_hash: str
    rulebook_eligible: bool
    validation_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskRouteEligibilityIndex:
    """Deterministic offline index keyed by scenario UID."""

    records: tuple[TaskRouteEligibility, ...]

    def __post_init__(self) -> None:
        ids = tuple(record.scenario_uid for record in self.records)
        if any(not scenario_uid for scenario_uid in ids):
            raise ValueError("Eligibility index requires non-empty scenario UIDs")
        if len(ids) != len(set(ids)):
            raise ValueError("Eligibility index contains duplicate scenario UIDs")

    @property
    def by_scenario_uid(self) -> Mapping[str, TaskRouteEligibility]:
        return {record.scenario_uid: record for record in self.records}

    def eligible(self, scenario_uid: str) -> TaskRouteEligibility:
        try:
            record = self.by_scenario_uid[scenario_uid]
        except KeyError as error:
            raise KeyError(f"Scenario UID is absent from eligibility index: {scenario_uid!r}") from error
        if not record.rulebook_eligible:
            raise ValueError(f"Scenario UID is not Rulebook v2 eligible: {scenario_uid!r}")
        return record


@dataclass(frozen=True, slots=True)
class TaskRouteExclusionReport:
    total_records: int
    eligible_records: int
    excluded_records: int
    excluded_by_adapter: Mapping[str, int]
    excluded_by_cause: Mapping[str, int]


def build_task_route_exclusion_report(index: TaskRouteEligibilityIndex) -> TaskRouteExclusionReport:
    """Summarize offline exclusions deterministically for audit artifacts."""
    by_adapter: dict[str, int] = {}
    by_cause: dict[str, int] = {}
    excluded = 0
    for record in index.records:
        if record.rulebook_eligible:
            continue
        excluded += 1
        by_adapter[record.adapter_version] = by_adapter.get(record.adapter_version, 0) + 1
        for cause in record.validation_errors:
            by_cause[cause] = by_cause.get(cause, 0) + 1
    return TaskRouteExclusionReport(
        total_records=len(index.records),
        eligible_records=len(index.records) - excluded,
        excluded_records=excluded,
        excluded_by_adapter=dict(sorted(by_adapter.items())),
        excluded_by_cause=dict(sorted(by_cause.items())),
    )


def build_task_route_eligibility_index(
    records: Iterable[TaskRouteEligibility],
) -> TaskRouteEligibilityIndex:
    """Build an immutable index while preserving deterministic UID order."""
    ordered = tuple(sorted(records, key=lambda record: record.scenario_uid))
    return TaskRouteEligibilityIndex(ordered)


def build_task_route_record(
    *,
    scenario_uid: str,
    lane_ids: Iterable[str],
    provenance: str,
    adapter_version: str = RULEBOOK_V2_ADAPTER_CONTRACT_VERSION,
    source_geometry_bytes: bytes,
) -> TaskRouteRecord:
    """Build a route record from topology and canonical source geometry bytes."""

    lanes = tuple(str(lane_id) for lane_id in lane_ids)
    if not scenario_uid or not provenance or not adapter_version or not lanes:
        raise ValueError("TaskRouteRecord requires non-empty identity and lane topology")
    if any(not lane_id for lane_id in lanes):
        raise ValueError("TaskRouteRecord lane IDs must be non-empty")
    if not source_geometry_bytes:
        raise ValueError("TaskRouteRecord requires source geometry bytes")
    return TaskRouteRecord(
        scenario_uid=scenario_uid,
        lane_ids=lanes,
        provenance=provenance,
        adapter_version=adapter_version,
        source_geometry_hash=hashlib.sha256(source_geometry_bytes).hexdigest(),
    )


def validate_task_route(
    record: TaskRouteRecord,
    *,
    available_lane_ids: Mapping[str, object],
    rulebook_version: str,
    geometry_config_hash: str,
    calibration_hash: str,
) -> TaskRouteEligibility:
    """Produce an offline eligibility artifact without reading future tracks."""

    errors: list[str] = []
    if not record.scenario_uid or not record.lane_ids:
        errors.append("route_identity_or_lane_sequence_missing")
    if not record.provenance or not record.adapter_version:
        errors.append("route_adapter_metadata_missing")
    if not record.source_geometry_hash:
        errors.append("source_geometry_hash_missing")
    if not rulebook_version:
        errors.append("rulebook_version_missing")
    missing = [lane_id for lane_id in record.lane_ids if lane_id not in available_lane_ids]
    if missing:
        errors.append("missing_lane_ids:" + ",".join(missing))
    if not geometry_config_hash:
        errors.append("geometry_config_hash_missing")
    if not calibration_hash:
        errors.append("calibration_hash_missing")
    return TaskRouteEligibility(
        scenario_uid=record.scenario_uid,
        rulebook_version=rulebook_version,
        adapter_version=record.adapter_version,
        geometry_config_hash=geometry_config_hash,
        calibration_hash=calibration_hash,
        rulebook_eligible=not errors,
        validation_errors=tuple(errors),
    )
