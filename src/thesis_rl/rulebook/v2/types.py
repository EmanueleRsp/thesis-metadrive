"""Canonical, immutable contracts at the Rulebook v2 boundary.

These records deliberately contain no source-specific ScenarioNet or
MetaDrive objects.  Adapters introduced in later phases construct them from
their respective sources before the monitor is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot, isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, TypeAlias, cast

from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from thesis_rl.mission.types import MissionSnapshot

if TYPE_CHECKING:
    from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
    from thesis_rl.rulebook.v2.geometry.route import RoutePolyline


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


class StaticSubclass(str, Enum):
    """Sub-taxonomy of `ActorClass.STATIC_COLLIDABLE` road furniture.

    OBS-V1.3 assumes ideal semantic classification after physical admission
    (OBS-V1.3 SS10.1), which a fused camera/LiDAR stack supports: cones,
    barriers and warning triangles differ in appearance and in how a driver
    must respond to them.  The distinction exists in the source taxonomy and
    was previously discarded before reaching the observation.

    The Rulebook itself does not branch on this field; it is carried for the
    observation only.
    """

    TRAFFIC_CONE = "traffic_cone"
    TRAFFIC_BARRIER = "traffic_barrier"
    TRAFFIC_WARNING = "traffic_warning"
    OTHER = "other"


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
    route_assignment_source: str = "offline_task_annotation"

    def __post_init__(self) -> None:
        if not self.scenario_uid or not self.lane_ids:
            raise ValueError("TaskRouteRecord requires a scenario UID and non-empty lane sequence")
        if any(not lane_id for lane_id in self.lane_ids):
            raise ValueError("TaskRouteRecord lane IDs must be non-empty")
        if (
            not self.provenance
            or not self.adapter_version
            or not self.source_geometry_hash
            or not self.route_assignment_source
        ):
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
    # Observation-only refinement of STATIC_COLLIDABLE; None for every other
    # actor class and for statics whose source type is not recognised.
    static_subclass: "StaticSubclass | None" = None

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
        if self.configured_speed_cap_mps is not None and not isfinite(
            self.configured_speed_cap_mps
        ):
            raise ValueError("ActorSnapshot configured_speed_cap_mps must be finite when supplied")


@dataclass(frozen=True, slots=True)
class ContactOnsetRecord:
    actor_id: str
    actor_class: ActorClass


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
    mission_snapshot: MissionSnapshot | None = None

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
    elevation_profile_xyz: tuple[tuple[float, float, float], ...] = ()

    def elevation_at_xy(self, point_xy: tuple[float, float]) -> float | None:
        """Interpolate the source 2.5D profile at the nearest XY location."""
        if not self.elevation_profile_xyz:
            return self.elevation_m
        px, py = point_xy
        best: tuple[float, int, float] | None = None
        for index, (start, end) in enumerate(
            zip(self.elevation_profile_xyz, self.elevation_profile_xyz[1:])
        ):
            ax, ay, az = start
            bx, by, bz = end
            dx, dy = bx - ax, by - ay
            length_sq = dx * dx + dy * dy
            if length_sq <= 0.0:
                continue
            fraction = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
            x, y = ax + fraction * dx, ay + fraction * dy
            distance = hypot(px - x, py - y)
            candidate = (distance, index, az + fraction * (bz - az))
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is not None:
            return best[2]
        return self.elevation_profile_xyz[0][2] if len(self.elevation_profile_xyz) == 1 else self.elevation_m


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
class RoundaboutPriorityRecord:
    """Validated entry/circulating lane relation for one roundabout component.

    The record is source-bound metadata: topology alone must never infer a
    roundabout priority relation.
    """

    component_id: str
    entry_lane_id: str
    circulating_lane_id: str


@dataclass(frozen=True, slots=True)
class ConflictZoneRecord:
    zone_id: str
    polygon: BaseGeometry
    ego_movement_key: MovementKey
    other_movement_key: MovementKey | None
    route_entry_s_m: float
    route_exit_s_m: float
    elevation_m: float
    component_index: int = 0


@dataclass(frozen=True, slots=True)
class VehicleConflictPairRecord:
    """Complete static vehicle-zone catalogue for one ordered movement pair."""

    ego_movement_key: MovementKey
    other_movement_key: MovementKey
    candidates: tuple[ConflictZoneRecord, ...]


@dataclass(frozen=True, slots=True)
class ActorMotionSample:
    """One causal live kinematic sample retained for CTRV estimation."""

    timestamp_s: float
    position_xy_m: tuple[float, float]
    heading_rad: float
    velocity_xy_mps: tuple[float, float]

    def __post_init__(self) -> None:
        _require_finite(
            "ActorMotionSample",
            timestamp_s=self.timestamp_s,
            position_x=self.position_xy_m[0],
            position_y=self.position_xy_m[1],
            heading_rad=self.heading_rad,
            velocity_x=self.velocity_xy_mps[0],
            velocity_y=self.velocity_xy_mps[1],
        )


@dataclass(frozen=True, slots=True)
class ActorMotionHistory:
    """Immutable, ordered causal history for one episodic actor."""

    actor_id: str
    samples: tuple[ActorMotionSample, ...] = ()

    def __post_init__(self) -> None:
        if not self.actor_id:
            raise ValueError("ActorMotionHistory actor_id must be non-empty")
        timestamps = tuple(sample.timestamp_s for sample in self.samples)
        if any(previous >= current for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("ActorMotionHistory timestamps must be strictly increasing")


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
    actor_motion_histories: tuple[ActorMotionHistory, ...] = ()
    previous_sim_time_s: float | None = None


@dataclass(frozen=True, slots=True)
class MemoryDelta:
    writer: str | None = None
    writes: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class CacheDelta:
    new_conflict_zones: tuple[ConflictZoneRecord, ...] = ()
    new_vehicle_conflict_pairs: tuple[VehicleConflictPairRecord, ...] = ()
    diagnostic_timing_seconds: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EpisodeCache:
    scenario_id: str
    task_route: TaskRouteRecord
    conflict_zones: Mapping[str, ConflictZoneRecord] = field(
        default_factory=lambda: cast(Mapping[str, ConflictZoneRecord], freeze_mapping())
    )
    vehicle_conflict_pairs: Mapping[tuple[MovementKey, MovementKey], VehicleConflictPairRecord] = (
        field(
            default_factory=lambda: cast(
                Mapping[tuple[MovementKey, MovementKey], VehicleConflictPairRecord],
                freeze_mapping(),
            )
        )
    )
    map_feature_catalog: Mapping[str, MapFeatureRecord] = field(
        default_factory=lambda: cast(Mapping[str, MapFeatureRecord], freeze_mapping())
    )
    traffic_control_catalog: tuple[TrafficControlRecord, ...] = ()
    movement_priority_records: tuple[MovementPriorityRecord, ...] = ()
    roundabout_priority_records: tuple[RoundaboutPriorityRecord, ...] = ()
    route_lanes: tuple["RouteLaneRecord", ...] = ()
    route_polyline: "RoutePolyline | None" = None
    # Additive diagnostic only (F4b): count of stop/signal control candidates
    # dropped by ControlLineOffRouteError during static adapter construction,
    # threaded from StaticAdapterResult.dropped_control_line_off_route_count.
    control_line_off_route_drop_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "conflict_zones", freeze_mapping(self.conflict_zones))
        object.__setattr__(
            self, "vehicle_conflict_pairs", freeze_mapping(self.vehicle_conflict_pairs)
        )
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
