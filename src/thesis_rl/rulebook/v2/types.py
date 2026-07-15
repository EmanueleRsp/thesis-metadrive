"""Canonical, immutable contracts at the Rulebook v2 boundary.

These records deliberately contain no source-specific ScenarioNet or
MetaDrive objects.  Adapters introduced in later phases construct them from
their respective sources before the monitor is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping, TypeAlias

from shapely.geometry.base import BaseGeometry


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | tuple["JSONValue", ...] | Mapping[str, "JSONValue"]


def freeze_mapping(values: Mapping[str, object] | None = None) -> Mapping[str, object]:
    """Return an immutable copy suitable for frozen normative records."""

    return MappingProxyType(dict(values or {}))


def _require_finite(record_name: str, **values: float) -> None:
    invalid = [name for name, value in values.items() if not isfinite(value)]
    if invalid:
        raise ValueError(f"{record_name} requires finite values: {', '.join(invalid)}")


class ActorClass(str, Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    STATIC_COLLIDABLE = "static_collidable"
    INFRASTRUCTURE_NON_COLLIDABLE = "infrastructure_non_collidable"


class MapFeatureClass(str, Enum):
    DRIVABLE_LANE = "drivable_lane"
    SIDEWALK = "sidewalk"
    CROSSWALK = "crosswalk"
    ROAD_BOUNDARY = "road_boundary"
    LANE_MARKING_SOLID = "lane_marking_solid"
    LANE_MARKING_DASHED = "lane_marking_dashed"
    OTHER_NON_DRIVABLE = "other_non_drivable"


class ApproachControl(str, Enum):
    NONE = "none"
    STOP = "stop"
    SIGNAL = "signal"
    UNKNOWN = "unknown"


class MovementPriority(str, Enum):
    OTHER_HAS_PRIORITY = "other_has_priority"
    EGO_HAS_PRIORITY = "ego_has_priority"
    UNDEFINED = "undefined"


class ComponentStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    NOT_EVALUABLE = "not_evaluable"
    SATISFIED = "satisfied"
    VIOLATED = "violated"


class MacroRule(str, Enum):
    COLLISION_IMPACT = "collision_impact"
    DYNAMIC_INTERACTION_SAFETY = "dynamic_interaction_safety"
    ROAD_TRAFFIC_COMPLIANCE = "road_traffic_compliance"
    ROUTE_PROGRESS = "route_progress"


MACRO_RULE_ORDER: tuple[MacroRule, ...] = (
    MacroRule.COLLISION_IMPACT,
    MacroRule.DYNAMIC_INTERACTION_SAFETY,
    MacroRule.ROAD_TRAFFIC_COMPLIANCE,
    MacroRule.ROUTE_PROGRESS,
)


@dataclass(frozen=True, slots=True, order=True)
class MovementKey:
    approach_lane_id: str
    conflict_node_id: str
    exit_lane_id: str


@dataclass(frozen=True, slots=True)
class TaskRouteRecord:
    """Static task input; it never carries a future SDC trajectory."""

    scenario_uid: str
    lane_ids: tuple[str, ...]
    provenance: str
    adapter_version: str
    source_geometry_hash: str

    def __post_init__(self) -> None:
        if not self.scenario_uid or not self.lane_ids:
            raise ValueError("TaskRouteRecord requires a scenario UID and non-empty lane sequence")
        if any(not lane_id for lane_id in self.lane_ids):
            raise ValueError("TaskRouteRecord lane IDs must be non-empty")
        if not self.provenance or not self.adapter_version or not self.source_geometry_hash:
            raise ValueError("TaskRouteRecord identity metadata must be non-empty")


@dataclass(frozen=True, slots=True)
class ActorSnapshot:
    actor_id: str
    actor_class: ActorClass
    position_xy: tuple[float, float]
    position_z: float
    heading_rad: float
    velocity_xy: tuple[float, float]
    footprint: BaseGeometry
    live_lane_id: str | None
    configured_speed_cap_mps: float | None

    def __post_init__(self) -> None:
        _require_finite(
            "ActorSnapshot",
            position_x=self.position_xy[0],
            position_y=self.position_xy[1],
            position_z=self.position_z,
            heading_rad=self.heading_rad,
            velocity_x=self.velocity_xy[0],
            velocity_y=self.velocity_xy[1],
        )
        if (
            self.configured_speed_cap_mps is not None
            and not isfinite(self.configured_speed_cap_mps)
        ):
            raise ValueError("ActorSnapshot configured_speed_cap_mps must be finite when supplied")


@dataclass(frozen=True, slots=True)
class ContactOnsetRecord:
    actor_id: str
    actor_class: ActorClass
    contact_point_xy: tuple[float, float]
    normal_ego_to_other_xy: tuple[float, float]


@dataclass(frozen=True, slots=True)
class EnvSnapshot:
    scenario_id: str
    step_index: int
    sim_time_s: float
    ego: ActorSnapshot
    actors: tuple[ActorSnapshot, ...]
    contact_onset_records: tuple[ContactOnsetRecord, ...]
    active_contact_ids: frozenset[str]
    signal_states_by_physical_id: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "signal_states_by_physical_id",
            freeze_mapping(self.signal_states_by_physical_id),
        )


@dataclass(frozen=True, slots=True)
class MapFeatureRecord:
    feature_id: str
    feature_class: MapFeatureClass
    geometry: BaseGeometry
    elevation_m: float | None
    logical_boundary_id: str | None = None


@dataclass(frozen=True, slots=True)
class TrafficControlRecord:
    control_group_id: str
    control_type: ApproachControl
    controlled_lane_ids: tuple[str, ...]
    movement_key: MovementKey
    control_line: BaseGeometry
    route_s_m: float
    elevation_m: float
    physical_control_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MovementPriorityRecord:
    ego_movement_key: MovementKey
    other_movement_key: MovementKey
    relation: MovementPriority


@dataclass(frozen=True, slots=True)
class ConflictZoneRecord:
    zone_id: str
    polygon: BaseGeometry
    ego_movement_key: MovementKey
    other_movement_key: MovementKey | None
    route_entry_s_m: float
    route_exit_s_m: float
    elevation_m: float


@dataclass(frozen=True, slots=True)
class RulebookMemory:
    previous_contact_ids: frozenset[str] = frozenset()
    active_dashed_boundary_id: str | None = None
    dashed_line_timer_s: float = 0.0
    active_signal_group_id: str | None = None
    previous_signal_state: str | None = None
    yellow_must_stop: bool = False
    previous_signal_delta_m: float | None = None
    resolved_signal_group_ids: frozenset[str] = frozenset()
    active_stop_group_id: str | None = None
    stop_continuous_timer_s: float = 0.0
    stop_best_timer_s: float = 0.0
    previous_stop_delta_m: float | None = None
    resolved_stop_group_ids: frozenset[str] = frozenset()
    crosswalk_illegal_entries: frozenset[tuple[str, str]] = frozenset()
    vehicle_yield_illegal_entries: frozenset[tuple[str, str]] = frozenset()
    preexisting_ego_occupancy_zone_ids: frozenset[str] = frozenset()
    frozen_actor_movement_keys: tuple[tuple[str, MovementKey], ...] = ()
    previous_route_s_m: float = 0.0


@dataclass(frozen=True, slots=True)
class MemoryDelta:
    writer: str | None = None
    writes: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class CacheDelta:
    new_conflict_zones: tuple[ConflictZoneRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class EpisodeCache:
    scenario_id: str
    task_route: TaskRouteRecord
    conflict_zones: Mapping[str, ConflictZoneRecord] = field(default_factory=freeze_mapping)
    map_feature_catalog: Mapping[str, MapFeatureRecord] = field(default_factory=freeze_mapping)
    traffic_control_catalog: tuple[TrafficControlRecord, ...] = ()
    movement_priority_records: tuple[MovementPriorityRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "conflict_zones", freeze_mapping(self.conflict_zones))
        object.__setattr__(self, "map_feature_catalog", freeze_mapping(self.map_feature_catalog))


@dataclass(frozen=True, slots=True)
class RuleComponentResult:
    name: str
    cost: float
    raw: Mapping[str, JSONValue]
    applicable: bool
    evaluable: bool
    status: ComponentStatus
    diagnostics: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        _require_finite("RuleComponentResult", cost=self.cost)
        object.__setattr__(self, "raw", freeze_mapping(self.raw))
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "cost": self.cost,
            "raw": dict(self.raw),
            "applicable": self.applicable,
            "evaluable": self.evaluable,
            "status": self.status.value,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class RulebookResult:
    margins: tuple[float, float, float, float]
    costs: tuple[float, float, float]
    raw_progress_m: float
    components: Mapping[str, RuleComponentResult]
    complete_evaluation: bool

    def __post_init__(self) -> None:
        _require_finite(
            "RulebookResult",
            margin_1=self.margins[0],
            margin_2=self.margins[1],
            margin_3=self.margins[2],
            margin_4=self.margins[3],
            cost_1=self.costs[0],
            cost_2=self.costs[1],
            cost_3=self.costs[2],
            raw_progress_m=self.raw_progress_m,
        )
        object.__setattr__(self, "components", freeze_mapping(self.components))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "margins": self.margins,
            "costs": self.costs,
            "raw_progress_m": self.raw_progress_m,
            "components": {name: result.to_dict() for name, result in self.components.items()},
            "complete_evaluation": self.complete_evaluation,
        }
