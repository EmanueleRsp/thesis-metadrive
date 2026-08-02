"""Versioned immutable contracts for the unified driving mission."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


MISSION_SCHEMA_VERSION = "driving_mission_v1"


def _finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class LaneSpan:
    lane_id: str
    start_s_m: float
    end_s_m: float

    def __post_init__(self) -> None:
        if not self.lane_id:
            raise ValueError("lane_id must be non-empty")
        _finite(self.start_s_m, "start_s_m")
        _finite(self.end_s_m, "end_s_m")
        if self.start_s_m < 0.0 or self.end_s_m <= self.start_s_m:
            raise ValueError("end_s_m must be finite and greater than start_s_m")


@dataclass(frozen=True, slots=True)
class DirectedGate:
    gate_id: str
    compatible_spans: tuple[LaneSpan, ...]
    lane_id: str
    s_m: float

    def __post_init__(self) -> None:
        if not self.gate_id or not self.lane_id or not self.compatible_spans:
            raise ValueError("gate ID, lane ID, and compatible spans are required")
        _finite(self.s_m, "s_m")
        if self.s_m < 0.0:
            raise ValueError("s_m must be non-negative")
        if not any(span.lane_id == self.lane_id and span.start_s_m <= self.s_m <= span.end_s_m for span in self.compatible_spans):
            raise ValueError("gate must be located on a compatible lane span")


@dataclass(frozen=True, slots=True)
class MissionSection:
    section_id: str
    preferred_span: LaneSpan
    allowed_spans: tuple[LaneSpan, ...]
    exit_gate: DirectedGate

    def __post_init__(self) -> None:
        if not self.section_id or not self.allowed_spans:
            raise ValueError("section ID and allowed spans are required")
        if self.preferred_span not in self.allowed_spans:
            raise ValueError("preferred span must be allowed")


@dataclass(frozen=True, slots=True)
class DrivingMissionRecord:
    scenario_uid: str
    builder_version: str
    sections: tuple[MissionSection, ...]
    final_goal: DirectedGate
    schema_version: str = MISSION_SCHEMA_VERSION
    mission_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.scenario_uid or not self.builder_version or not self.sections:
            raise ValueError("scenario UID, builder version, and sections are required")
        if self.schema_version != MISSION_SCHEMA_VERSION:
            raise ValueError(f"unsupported mission schema: {self.schema_version!r}")
        payload = self.to_dict(include_hash=False)
        object.__setattr__(self, "mission_hash", hashlib.sha256(_canonical_json(payload)).hexdigest())

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "scenario_uid": self.scenario_uid,
            "builder_version": self.builder_version,
            "sections": [asdict(section) for section in self.sections],
            "final_goal": asdict(self.final_goal),
            "schema_version": self.schema_version,
        }
        if include_hash:
            payload["mission_hash"] = self.mission_hash
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DrivingMissionRecord":
        def span(data: dict[str, Any]) -> LaneSpan:
            return LaneSpan(**data)

        def gate(data: dict[str, Any]) -> DirectedGate:
            return DirectedGate(data["gate_id"], tuple(span(item) for item in data["compatible_spans"]), data["lane_id"], data["s_m"])

        sections = tuple(MissionSection(item["section_id"], span(item["preferred_span"]), tuple(span(value) for value in item["allowed_spans"]), gate(item["exit_gate"])) for item in payload["sections"])
        record = cls(payload["scenario_uid"], payload["builder_version"], sections, gate(payload["final_goal"]), payload.get("schema_version", MISSION_SCHEMA_VERSION))
        expected = payload.get("mission_hash")
        if expected is not None and expected != record.mission_hash:
            raise ValueError("mission_hash does not match immutable payload")
        return record


@dataclass(frozen=True, slots=True)
class MissionSnapshot:
    mission_hash: str
    step_index: int
    pending_gate_index: int
    remaining_distance_m: float
    route_completion: float
    reachable: bool
    mission_success: bool
    mission_unreachable: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.step_index < 0 or self.pending_gate_index < 0:
            raise ValueError("mission snapshot indices must be non-negative")
        _finite(self.remaining_distance_m, "remaining_distance_m")
        _finite(self.route_completion, "route_completion")
        if self.remaining_distance_m < 0.0 or not 0.0 <= self.route_completion <= 1.0:
            raise ValueError("mission snapshot values are outside their valid ranges")
        if self.mission_success and self.mission_unreachable:
            raise ValueError("success and unreachable cannot both be true")


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
