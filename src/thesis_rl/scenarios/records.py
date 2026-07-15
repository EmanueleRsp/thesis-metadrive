from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Literal


ScenarioSource = Literal["waymo", "pg"]
ScenarioSplit = Literal["train", "validation", "test"]
TopologyTag = Literal["simple", "merge_or_roundabout", "intersection", "mixed", "unknown"]
SignalReliability = Literal["not_applicable", "complete", "partial", "missing"]
TopologyConfidence = Literal["unknown", "medium", "high"]
ValidationStatus = Literal["valid", "warning", "invalid"]

SOURCES = frozenset({"waymo", "pg"})
SPLITS = frozenset({"train", "validation", "test"})
TOPOLOGY_TAGS = frozenset(
    {"simple", "merge_or_roundabout", "intersection", "mixed", "unknown"}
)
SIGNAL_RELIABILITIES = frozenset({"not_applicable", "complete", "partial", "missing"})
TOPOLOGY_CONFIDENCES = frozenset({"unknown", "medium", "high"})
VALIDATION_STATUSES = frozenset({"valid", "warning", "invalid"})


def _require_non_empty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _validate_relative_posix_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value != path.as_posix()
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError(f"relative_path must be a normalized relative POSIX path: {value!r}")
    if "\\" in value:
        raise ValueError("relative_path must use POSIX separators")


@dataclass(frozen=True, slots=True)
class ScenarioRecord:
    scenario_uid: str
    scenario_id: str
    source: ScenarioSource
    relative_path: str
    official_split: str | None
    source_log_id: str | None
    source_scenario_id: str | None
    dataset_version: str
    converter_version: str | None
    split: ScenarioSplit
    runtime_index: int | None
    length: int
    pg_profile: str | None
    pg_seed: int | None
    map_id: str | None
    primary_arm: str
    tags: tuple[str, ...]
    signal_reliability: SignalReliability
    validation_status: ValidationStatus
    validation_warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("scenario_uid", "scenario_id", "dataset_version", "primary_arm"):
            _require_non_empty(name, getattr(self, name))
        if self.source not in SOURCES:
            raise ValueError(f"unsupported scenario source: {self.source!r}")
        if self.split not in SPLITS:
            raise ValueError(f"unsupported scenario split: {self.split!r}")
        if self.signal_reliability not in SIGNAL_RELIABILITIES:
            raise ValueError(f"unsupported signal reliability: {self.signal_reliability!r}")
        if self.validation_status not in VALIDATION_STATUSES:
            raise ValueError(f"unsupported validation status: {self.validation_status!r}")
        _validate_relative_posix_path(self.relative_path)
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.runtime_index is not None and self.runtime_index < 0:
            raise ValueError("runtime_index must be non-negative")
        if self.pg_seed is not None and self.pg_seed < 0:
            raise ValueError("pg_seed must be non-negative")
        if self.source == "waymo" and self.pg_seed is not None:
            raise ValueError("Waymo records cannot carry a PG seed")
        if self.source == "pg" and self.pg_seed is None:
            raise ValueError("PG records require a generation seed")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("tags must not contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        payload["validation_warnings"] = list(self.validation_warnings)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScenarioRecord":
        data = dict(payload)
        data["tags"] = tuple(data.get("tags", ()))
        data["validation_warnings"] = tuple(data.get("validation_warnings", ()))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ScenarioFeatures:
    scenario_id: str
    source: ScenarioSource
    length: int
    route_length_m: float
    topology_tag: TopologyTag
    has_intersection: bool | None
    has_merge_or_roundabout: bool | None
    has_route_traffic_light: bool | None
    has_route_stop_sign: bool | None
    has_route_crosswalk: bool | None
    signal_reliability: SignalReliability
    has_vehicle: bool
    has_pedestrian: bool
    has_cyclist: bool
    relevant_agents_q90: float
    relevant_vehicles_q90: float
    min_vehicle_distance_m: float | None
    min_vru_distance_to_route_m: float | None
    low_traffic: bool
    dense_traffic: bool
    vru_interaction: bool
    topology_confidence: TopologyConfidence = "unknown"
    topology_evidence: tuple[str, ...] = ()
    relevant_vrus_q90: float = 0.0
    vehicle_conflict_count: int = 0
    vru_conflict_count: int = 0
    min_vehicle_conflict_dcpa_m: float | None = None
    min_vehicle_conflict_tcpa_s: float | None = None
    min_vru_conflict_dcpa_m: float | None = None
    min_vru_conflict_tcpa_s: float | None = None
    sdc_valid_ratio: float = 1.0
    sdc_initial_valid: bool = True
    sdc_route_z_range_m: float = 0.0
    map_feature_count: int = 0
    dynamic_object_count: int = 0

    def __post_init__(self) -> None:
        _require_non_empty("scenario_id", self.scenario_id)
        if self.source not in SOURCES:
            raise ValueError(f"unsupported scenario source: {self.source!r}")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.route_length_m < 0:
            raise ValueError("route_length_m must be non-negative")
        if self.topology_tag not in TOPOLOGY_TAGS:
            raise ValueError(f"unsupported topology tag: {self.topology_tag!r}")
        if self.signal_reliability not in SIGNAL_RELIABILITIES:
            raise ValueError(f"unsupported signal reliability: {self.signal_reliability!r}")
        if self.topology_confidence not in TOPOLOGY_CONFIDENCES:
            raise ValueError(
                f"unsupported topology confidence: {self.topology_confidence!r}"
            )
        if len(set(self.topology_evidence)) != len(self.topology_evidence):
            raise ValueError("topology evidence must not contain duplicates")
        if (
            self.relevant_agents_q90 < 0
            or self.relevant_vehicles_q90 < 0
            or self.relevant_vrus_q90 < 0
        ):
            raise ValueError("relevant-agent quantiles must be non-negative")
        if self.vehicle_conflict_count < 0 or self.vru_conflict_count < 0:
            raise ValueError("conflict counts must be non-negative")
        if not 0 <= self.sdc_valid_ratio <= 1:
            raise ValueError("sdc_valid_ratio must be in [0, 1]")
        if self.sdc_route_z_range_m < 0:
            raise ValueError("sdc_route_z_range_m must be non-negative")
        if self.map_feature_count < 0 or self.dynamic_object_count < 0:
            raise ValueError("scenario feature counts must be non-negative")
        for name in (
            "min_vehicle_distance_m",
            "min_vru_distance_to_route_m",
            "min_vehicle_conflict_dcpa_m",
            "min_vehicle_conflict_tcpa_s",
            "min_vru_conflict_dcpa_m",
            "min_vru_conflict_tcpa_s",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def has_unknown_signal(self) -> bool:
        return self.signal_reliability in {"partial", "missing"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScenarioFeatures":
        data = dict(payload)
        data["topology_evidence"] = tuple(data.get("topology_evidence") or ())
        return cls(**data)
