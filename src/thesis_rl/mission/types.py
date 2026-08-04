"""Versioned immutable contracts for the unified driving mission."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from thesis_rl.mission.gates import GateGeometry


MISSION_SCHEMA_VERSION = "driving_mission_v1_1_1"


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
    geometry: GateGeometry | None = None

    def __post_init__(self) -> None:
        if not self.gate_id or not self.lane_id or not self.compatible_spans:
            raise ValueError("gate ID, lane ID, and compatible spans are required")
        _finite(self.s_m, "s_m")
        if self.s_m < 0.0:
            raise ValueError("s_m must be non-negative")
        if not any(
            span.lane_id == self.lane_id and span.start_s_m <= self.s_m <= span.end_s_m
            for span in self.compatible_spans
        ):
            raise ValueError("gate must be located on a compatible lane span")


@dataclass(frozen=True, slots=True)
class FinalGateSegment:
    """Frozen offline terminal segment consumed passively by runtime."""

    line_xy: tuple[tuple[float, float], tuple[float, float]]
    static_tangent_xy: tuple[float, float]
    elevation_m: float
    final_occurrence_id: str
    provenance: str
    source_geometry_hash: str
    builder_identity: str

    def __post_init__(self) -> None:
        values = (*self.line_xy[0], *self.line_xy[1], *self.static_tangent_xy, self.elevation_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("final gate segment values must be finite")
        if self.line_xy[0] == self.line_xy[1] or math.hypot(*self.static_tangent_xy) <= 0.0:
            raise ValueError("final gate segment must have non-zero geometry and tangent")
        if not all((self.final_occurrence_id, self.provenance, self.source_geometry_hash, self.builder_identity)):
            raise ValueError("final gate segment identity fields are required")


@dataclass(frozen=True, slots=True)
class RouteOccurrence:
    """One frozen assigned-route occurrence with offline mission orientation."""

    occurrence_index: int
    lane_id: str
    orientation: str
    oriented_centerline_points_xyz: tuple[tuple[float, float, float], ...]
    source_start_s_m: float
    source_end_s_m: float
    orientation_provenance: str
    source_geometry_hash: str

    def __post_init__(self) -> None:
        if self.occurrence_index < 0 or not self.lane_id:
            raise ValueError("route occurrence index and lane ID are required")
        if self.orientation not in {"FORWARD", "REVERSED"}:
            raise ValueError("route occurrence orientation must be FORWARD or REVERSED")
        if len(self.oriented_centerline_points_xyz) < 2:
            raise ValueError("route occurrence requires at least two centerline points")
        if not all(math.isfinite(value) for point in self.oriented_centerline_points_xyz for value in point):
            raise ValueError("route occurrence geometry must be finite")
        _finite(self.source_start_s_m, "source_start_s_m")
        _finite(self.source_end_s_m, "source_end_s_m")
        if self.source_start_s_m < 0.0 or self.source_end_s_m <= self.source_start_s_m:
            raise ValueError("route occurrence source station bounds are invalid")
        if not self.orientation_provenance or not self.source_geometry_hash:
            raise ValueError("route occurrence provenance and geometry hash are required")


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
    route_lane_ids: tuple[str, ...] = ()
    canonical_route_points_xyz: tuple[tuple[float, float, float], ...] = ()
    start_occurrence_id: str = ""
    final_occurrence_id: str = ""
    s_start_m: float = 0.0
    s_goal_m: float = 0.0
    final_gate_segment: FinalGateSegment | None = None
    route_occurrences: tuple[RouteOccurrence, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario_uid or not self.builder_version or not self.sections:
            raise ValueError("scenario UID, builder version, and sections are required")
        if self.schema_version != MISSION_SCHEMA_VERSION:
            raise ValueError(f"unsupported mission schema: {self.schema_version!r}")
        if self.route_lane_ids:
            if not self.canonical_route_points_xyz or not self.final_occurrence_id or not self.route_occurrences:
                raise ValueError("route mission requires canonical route, occurrences, and final occurrence")
            if self.s_start_m != 0.0:
                raise ValueError("route mission requires s_start_m=0")
            if not math.isfinite(self.s_goal_m) or self.s_goal_m <= 0.0:
                raise ValueError("route mission requires positive finite s_goal_m")
            if self.final_gate_segment is None:
                raise ValueError("route mission requires a frozen final gate segment")
        payload = self.to_dict(include_hash=False)
        object.__setattr__(
            self, "mission_hash", hashlib.sha256(_canonical_json(payload)).hexdigest()
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "scenario_uid": self.scenario_uid,
            "builder_version": self.builder_version,
            "sections": [asdict(section) for section in self.sections],
            "final_goal": asdict(self.final_goal),
            "schema_version": self.schema_version,
            "route_lane_ids": list(self.route_lane_ids),
            "canonical_route_points_xyz": [list(point) for point in self.canonical_route_points_xyz],
            "start_occurrence_id": self.start_occurrence_id,
            "final_occurrence_id": self.final_occurrence_id,
            "s_start_m": self.s_start_m,
            "s_goal_m": self.s_goal_m,
            "final_gate_segment": None if self.final_gate_segment is None else asdict(self.final_gate_segment),
            "route_occurrences": [asdict(occurrence) for occurrence in self.route_occurrences],
        }
        if include_hash:
            payload["mission_hash"] = self.mission_hash
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DrivingMissionRecord":
        def span(data: dict[str, Any]) -> LaneSpan:
            return LaneSpan(**data)

        def gate(data: dict[str, Any]) -> DirectedGate:
            geometry = data.get("geometry")
            return DirectedGate(
                data["gate_id"],
                tuple(span(item) for item in data["compatible_spans"]),
                data["lane_id"],
                data["s_m"],
                None if geometry is None else GateGeometry(**geometry),
            )

        sections = tuple(
            MissionSection(
                item["section_id"],
                span(item["preferred_span"]),
                tuple(span(value) for value in item["allowed_spans"]),
                gate(item["exit_gate"]),
            )
            for item in payload["sections"]
        )
        final_segment = payload.get("final_gate_segment")
        occurrences = tuple(
            RouteOccurrence(
                int(item["occurrence_index"]),
                str(item["lane_id"]),
                str(item["orientation"]),
                tuple(tuple(float(value) for value in point) for point in item["oriented_centerline_points_xyz"]),
                float(item["source_start_s_m"]),
                float(item["source_end_s_m"]),
                str(item["orientation_provenance"]),
                str(item["source_geometry_hash"]),
            )
            for item in payload.get("route_occurrences", ())
        )
        record = cls(
            scenario_uid=payload["scenario_uid"],
            builder_version=payload["builder_version"],
            sections=sections,
            final_goal=gate(payload["final_goal"]),
            schema_version=payload.get("schema_version", MISSION_SCHEMA_VERSION),
            route_lane_ids=tuple(str(value) for value in payload.get("route_lane_ids", ())),
            canonical_route_points_xyz=tuple(tuple(float(value) for value in point) for point in payload.get("canonical_route_points_xyz", ())),
            start_occurrence_id=str(payload.get("start_occurrence_id", "")),
            final_occurrence_id=str(payload.get("final_occurrence_id", "")),
            s_start_m=float(payload.get("s_start_m", 0.0)),
            s_goal_m=float(payload.get("s_goal_m", 0.0)),
            final_gate_segment=None if final_segment is None else FinalGateSegment(
                tuple(tuple(float(value) for value in point) for point in final_segment["line_xy"]),
                tuple(float(value) for value in final_segment["static_tangent_xy"]),
                float(final_segment["elevation_m"]),
                str(final_segment["final_occurrence_id"]),
                str(final_segment["provenance"]),
                str(final_segment["source_geometry_hash"]),
                str(final_segment["builder_identity"]),
            ),
            route_occurrences=occurrences,
        )
        expected = payload.get("mission_hash")
        if expected is not None and expected != record.mission_hash:
            raise ValueError("mission_hash does not match immutable payload")
        return record


def ordered_mission_gates(mission: DrivingMissionRecord) -> tuple[DirectedGate, ...]:
    """Return the ordered task boundaries, retaining a final goal only once.

    A no-branch mission has one section whose exit is the final goal. Frozen
    v1 records preserve that section structure, so the goal appears in both
    fields. It denotes one physical boundary, never two consecutive crossings.
    """

    exits = tuple(section.exit_gate for section in mission.sections)
    if exits and exits[-1].gate_id == mission.final_goal.gate_id:
        return exits
    return (*exits, mission.final_goal)


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
    s_m: float | None = None
    delta_s_m: float | None = None
    completion_instant: float | None = None
    completion_max: float | None = None

    def __post_init__(self) -> None:
        if self.step_index < 0 or self.pending_gate_index < 0:
            raise ValueError("mission snapshot indices must be non-negative")
        _finite(self.remaining_distance_m, "remaining_distance_m")
        _finite(self.route_completion, "route_completion")
        if self.remaining_distance_m < 0.0 or not 0.0 <= self.route_completion <= 1.0:
            raise ValueError("mission snapshot values are outside their valid ranges")
        if self.mission_success and self.mission_unreachable:
            raise ValueError("success and unreachable cannot both be true")
        if self.s_m is not None:
            _finite(self.s_m, "s_m")
        if self.delta_s_m is not None:
            _finite(self.delta_s_m, "delta_s_m")
        instant = self.route_completion if self.completion_instant is None else self.completion_instant
        maximum = self.route_completion if self.completion_max is None else self.completion_max
        _finite(instant, "completion_instant")
        _finite(maximum, "completion_max")
        if not 0.0 <= instant <= 1.0 or not 0.0 <= maximum <= 1.0:
            raise ValueError("completion values must be within [0, 1]")
        if maximum < instant and self.completion_max is not None:
            raise ValueError("completion_max cannot be below completion_instant")
        object.__setattr__(self, "completion_instant", instant)
        object.__setattr__(self, "completion_max", maximum)

    @property
    def instantaneous_completion(self) -> float:
        return float(self.completion_instant)

    @property
    def maximum_completion(self) -> float:
        return float(self.completion_max)


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
