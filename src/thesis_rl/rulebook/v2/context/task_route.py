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
