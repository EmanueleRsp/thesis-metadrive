"""Causal semantic v1.1 batch construction.

The builder is intentionally independent from the legacy semantic observation.
It consumes one committed :class:`CausalSceneContext`, canonical map geometry,
and the current live ego vehicle only.  It never receives a rule result or a
ScenarioDescription track array.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import atan2, cos, hypot, isfinite, sin
from typing import Any, Callable, Mapping

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, split
from thesis_rl.contracts.causal_scene_context import CausalSceneContext
from thesis_rl.contracts.observation_schema import (
    SemanticObservationBatch,
    SemanticObservationBatchV12,
)
from thesis_rl.envs.observations.perception import first_hit_lidar_sweep, mapped_signal_visibility
from thesis_rl.rulebook.v2.geometry.continuous_sat import (
    OccupancyInterval,
    predict_occupancy_interval,
)
from thesis_rl.rulebook.v2.geometry.lanes import associate_route_lane
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline, RouteProjection
from thesis_rl.rulebook.v2.transition import associated_speed_limit_mps
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    ApproachControl,
    ConflictZoneRecord,
    MapFeatureClass,
    MovementPriority,
    StaticSubclass,
    TrafficControlRecord,
)


class CausalSemanticObservationError(ValueError):
    """Raised when a required causal semantic input is unavailable or invalid."""


def _mission_s_m(context: CausalSceneContext) -> float:
    """Return the ego route station already committed by the mission tracker.

    DRIVING-MISSION-V1.1 §3/§5: semantic observation consumes the immutable
    ``MissionSnapshot`` and never independently reprojects the ego.
    """

    mission_snapshot = context.snapshot.mission_snapshot
    if mission_snapshot is None or mission_snapshot.s_m is None:
        raise CausalSemanticObservationError(
            "Semantic observation requires a committed mission snapshot with a route station"
        )
    return mission_snapshot.s_m


def _mission_station_is_measured(context: CausalSceneContext) -> bool:
    """Return whether the committed station is a tracked measurement.

    At ``step_index == 0`` the tracker has not advanced yet: `mission/runtime.py`
    constructs `RouteCoordinateMissionTracker(mission, route, 0.0)`, so the
    committed station is a **definition**, not a projection of the ego pose. It
    therefore carries no information to cross-check a geometric projection
    against; see `_ego_local_projection`.
    """

    mission_snapshot = context.snapshot.mission_snapshot
    return mission_snapshot is not None and int(mission_snapshot.step_index) > 0


_MISSION_S_CONSISTENCY_TOLERANCE_M = 1e-6


def _ego_local_projection(
    route: RoutePolyline,
    ego: ActorSnapshot,
    mission_s_m: float,
    *,
    station_is_measured: bool,
) -> RouteProjection:
    """Return route tangent/lateral offset at the ego without reprojecting its `s`.

    Anchored at the mission tracker's already-committed ``mission_s_m`` using
    the same tie-break authority the tracker itself uses, and asserts the
    result agrees with the committed snapshot rather than silently accepting
    an independently derived station (DRIVING-MISSION-V1.1 §3/§5).

    The invariant this guard protects is one of **authority**: the observation
    must consume the tracker's station instead of deriving its own. That is
    meaningful only once the tracker has actually tracked something. On the
    reset observation the committed value is a hard-coded ``0.0``
    (`mission/runtime.py`), so comparing a projection against it tests nothing
    and fails on the ordinary centimetre-scale offset between the spawn pose
    and the route origin -- measured at **0.0229 m** on a 23.17 m route, against
    a tolerance of 1e-6 m, and consistent with the projection jitter ADR-073
    measured at up to 0.053 m. The check is therefore enforced from the first
    tracked step onward and skipped at reset; the tolerance is unchanged, so no
    mid-episode divergence is newly tolerated.
    """

    projection = route.project(ego.position_xy, position_z=ego.position_z, previous_s_m=mission_s_m)
    divergence_m = abs(projection.s_m - mission_s_m)
    if station_is_measured and divergence_m > _MISSION_S_CONSISTENCY_TOLERANCE_M:
        # Report the magnitude: it is what distinguishes floating-point noise
        # from a genuine mission/route inconsistency, and the bare message
        # cannot be acted on without re-instrumenting the run.
        raise CausalSemanticObservationError(
            "Ego route geometry query diverged from the committed mission route station: "
            f"projected s={projection.s_m!r} m, committed s={mission_s_m!r} m, "
            f"divergence={divergence_m!r} m, tolerance={_MISSION_S_CONSISTENCY_TOLERANCE_M!r} m, "
            f"route length={route.length_m!r} m"
        )
    return projection


@dataclass(frozen=True, slots=True)
class SemanticOverflowDiagnostics:
    """Debug-only capacity diagnostics; never included in the policy tensor."""

    total_candidates: Mapping[str, int]
    selected_candidates: Mapping[str, int]
    capacity_dropped: Mapping[str, int]
    critical_dropped: int
    last_selected_key: Mapping[str, object]
    first_excluded_key: Mapping[str, object]
    route_incompatible_static_features: int = 0


@dataclass(frozen=True, slots=True)
class _EgoFrame:
    snapshot: ActorSnapshot
    steering: float
    throttle_brake: float
    acceleration_xy: tuple[float, float]
    yaw_rate: float
    lane_offset_m: float
    route_heading_error_rad: float


@dataclass(frozen=True, slots=True)
class _InteractionCandidate:
    zone_id: str
    polygon: Any
    zone_type_index: int | None
    route_entry_s_m: float
    route_exit_s_m: float
    actor: ActorSnapshot
    ego_interval: OccupancyInterval | None
    other_interval: OccupancyInterval | None
    ego_movement_key: Any | None = None
    other_movement_key: Any | None = None


def _clip(value: float, scale: float, *, lower: float = -1.0, upper: float = 1.0) -> float:
    if not isfinite(value) or not isfinite(scale) or scale <= 0.0:
        raise CausalSemanticObservationError(
            "Semantic normalization requires finite positive scales"
        )
    return float(np.clip(value / scale, lower, upper))


def _finite(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CausalSemanticObservationError(f"Missing finite semantic field: {name}") from error
    if not isfinite(result):
        raise CausalSemanticObservationError(f"Semantic field is not finite: {name}")
    return result


def _vector(value: object, *, name: str) -> tuple[float, float]:
    try:
        result = tuple(float(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError) as error:
        raise CausalSemanticObservationError(f"Missing finite semantic vector: {name}") from error
    if len(result) != 2 or not all(isfinite(item) for item in result):
        raise CausalSemanticObservationError(f"Semantic vector must be finite and 2D: {name}")
    return result


def _one_hot(index: int | None, size: int) -> list[float]:
    values = [0.0] * size
    if index is not None:
        if not 0 <= index < size:
            raise CausalSemanticObservationError("Categorical index is outside its contract")
        values[index] = 1.0
    return values


def _world_to_ego(vector: tuple[float, float], heading_rad: float) -> tuple[float, float]:
    c, s = cos(heading_rad), sin(heading_rad)
    return c * vector[0] + s * vector[1], -s * vector[0] + c * vector[1]


def _relative_position(
    position_xy: tuple[float, float], ego_position_xy: tuple[float, float], ego_heading: float
) -> tuple[float, float]:
    return _world_to_ego(
        (position_xy[0] - ego_position_xy[0], position_xy[1] - ego_position_xy[1]), ego_heading
    )


def _dimensions(footprint: Any) -> tuple[float, float]:
    if footprint.is_empty or not footprint.is_valid:
        raise CausalSemanticObservationError("Semantic footprint must be valid and non-empty")
    rectangle = footprint.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)
    lengths = [
        hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(coordinates, coordinates[1:])
    ]
    lengths = [length for length in lengths if length > 1.0e-6]
    if len(lengths) < 2:
        raise CausalSemanticObservationError("Semantic footprint dimensions are unavailable")
    return max(lengths), min(lengths)


def _heading_sincos(relative_heading: float) -> tuple[float, float]:
    return sin(relative_heading), cos(relative_heading)


def _actor_type_index(actor_class: ActorClass) -> int:
    return {
        ActorClass.VEHICLE: 0,
        ActorClass.PEDESTRIAN: 1,
        ActorClass.CYCLIST: 2,
    }.get(actor_class, 3)


# OBS-V1.3 feature offsets. They are exported so tests and the encoder can
# reference positions by name instead of re-deriving them from the field order.
DYNAMIC_ROUTE_LATERAL_INDEX = 18
STATIC_TYPE_SLICE = slice(6, 11)
CONTROL_GOVERNS_INDEX = 12
CONTROL_ACTIVE_INDEX = 13
CONTROL_STATE_VALID_INDEX = 14
INTERACTION_EGO_INSIDE_INDEX = 8
INTERACTION_EGO_APPROACH_CONTROL_SLICE = slice(21, 25)
INTERACTION_OTHER_APPROACH_CONTROL_SLICE = slice(25, 29)
INTERACTION_PREEXISTING_INDEX = 32

# Radius of the window used to describe a long map geometry locally, instead of
# by the bounding box of its whole extent (OBS-V1.3 REQ-005).
STATIC_LOCAL_WINDOW_M = 10.0

# OBS-V1.3 static taxonomy: the classes the source can actually discriminate.
# OBS-V1.3 REQ-028 static taxonomy.  Every slot is reachable: 0-2 come from the
# STATIC_COLLIDABLE sub-taxonomy MetaDrive already carries, 3-4 from the map
# feature catalogue.  The previous `stationary_vehicle` slot was unreachable by
# construction (a parked car is ActorClass.VEHICLE and is emitted in the dynamic
# group), and `unknown` was never assigned because the upstream class mapping
# fails closed on unrecognised actors.
STATIC_TYPE_TRAFFIC_CONE = 0
STATIC_TYPE_TRAFFIC_BARRIER = 1
STATIC_TYPE_OTHER_OBSTACLE = 2
STATIC_TYPE_ROAD_BOUNDARY = 3
STATIC_TYPE_OTHER_NON_DRIVABLE = 4

STATIC_SUBCLASS_TYPE_INDEX = {
    StaticSubclass.TRAFFIC_CONE: STATIC_TYPE_TRAFFIC_CONE,
    StaticSubclass.TRAFFIC_BARRIER: STATIC_TYPE_TRAFFIC_BARRIER,
    # A warning triangle is a hazard marker, not a rigid barrier; it shares the
    # generic obstacle slot with unrefined statics rather than claiming one of
    # the two physically distinct categories.
    StaticSubclass.TRAFFIC_WARNING: STATIC_TYPE_OTHER_OBSTACLE,
    StaticSubclass.OTHER: STATIC_TYPE_OTHER_OBSTACLE,
}


def _local_window(geometry: Any, ego_position_xy: tuple[float, float]) -> Any:
    """Clip a map geometry to the neighbourhood that the ego can act on."""

    window = Point(ego_position_xy).buffer(STATIC_LOCAL_WINDOW_M)
    local = geometry.intersection(window)
    return geometry if local.is_empty else local


def _local_tangent_rad(geometry: Any, point_xy: tuple[float, float]) -> float:
    """Return the orientation of the geometry segment nearest to ``point_xy``.

    Map features carry no heading of their own.  Emitting a constant zero would
    make the ego-relative sin/cos pair a direct encoding of the ego world
    heading (OBS-V1.3 REQ-006), so the local tangent is measured instead.
    """

    boundary = geometry.exterior if hasattr(geometry, "exterior") else geometry
    coordinates = list(getattr(boundary, "coords", ()))
    if len(coordinates) < 2:
        return 0.0
    target = Point(point_xy)
    best_angle = 0.0
    best_distance = float("inf")
    for first, second in zip(coordinates, coordinates[1:]):
        segment = LineString((first, second))
        distance = segment.distance(target)
        if distance < best_distance:
            best_distance = distance
            best_angle = atan2(second[1] - first[1], second[0] - first[0])
    return best_angle


def _local_extent(local: Any) -> tuple[float, float]:
    """Rotation-invariant (length, width) of a clipped map geometry.

    The axis-aligned bounding box is not invariant under a rigid rotation of
    the scene, so it cannot describe a map feature in the ego frame.  A
    line-like geometry has no minimum rotated rectangle, and falls back to its
    own length.
    """

    rectangle = local.minimum_rotated_rectangle
    boundary = getattr(rectangle, "exterior", rectangle)
    coordinates = list(getattr(boundary, "coords", ()))
    lengths = [
        hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(coordinates, coordinates[1:])
    ]
    lengths = [length for length in lengths if length > 1.0e-6]
    if len(lengths) >= 2:
        return max(lengths), min(lengths)
    return max(float(local.length), 0.1), 0.1


def _signed_footprint_clearance(geometry: Any, footprint: Any) -> float:
    """Signed clearance: positive before contact, negative during overlap.

    ``shapely`` reports distance ``0`` for any pair of intersecting geometries,
    so the negative branch required by the contract is unreachable without
    measuring how deep the footprint has crossed.  The footprint is convex, so
    the deepest point of the crossed piece is always one of its vertices.
    """

    if not geometry.intersects(footprint):
        return float(geometry.distance(footprint))
    try:
        pieces = list(split(footprint, geometry).geoms)
    except Exception:  # pragma: no cover - shapely raises several types here
        pieces = []
    if len(pieces) < 2:
        return 0.0
    centre = footprint.centroid
    beyond = [piece for piece in pieces if not piece.contains(centre)]
    if not beyond:
        return 0.0
    depth = max(
        geometry.distance(Point(coordinate))
        for piece in beyond
        for coordinate in piece.exterior.coords
    )
    return -float(depth)


class CausalSemanticBatchBuilder:
    """Build one structured semantic batch from committed causal state.

    ``commit_context`` is idempotent per scenario/step.  The builder retains
    only observations it has actually received, so a newly selected actor
    cannot acquire retroactive history from the scenario's future tracks.

    LEGACY OBS-V1.1 BUILDER.  Retained only to reproduce historical runs; it is
    no longer the repository default (``conf/config.yaml`` selects
    ``obs=semantic_v3``).  Unlike the OBS-V1.3 subclass, this builder DOES read
    Rulebook state into the policy observation: resolved signal/stop group ids
    in the control ranking, ``previous_signal_delta_m``, and the dashed/stop
    timers in the compliance row.  Because the Rulebook also produces the
    reward, that coupling is label leakage; it is a known defect recorded in
    ADR-033.  Do not select this observation for new experiments, and do not
    copy its memory reads into the perception-bounded path.
    """

    def __init__(
        self,
        *,
        route: RoutePolyline,
        route_lanes: tuple[Any, ...] = (),
        context_provider: Callable[[], CausalSceneContext] | None = None,
        history_length: int = 5,
        control_timestep_s: float = 0.1,
        dynamic_radius_m: float = 50.0,
        static_radius_m: float = 50.0,
        control_radius_m: float = 80.0,
        prediction_horizon_s: float = 3.0,
        vertical_tolerance_m: float = 3.0,
        ego_speed_cap_mps: float | None = None,
        signal_range_m: float = 80.0,
        signal_fov_degrees: float = 65.0,
        signal_camera_height_m: float = 1.2,
    ) -> None:
        if history_length != 5:
            raise ValueError("OBS-V1.1 requires history_length=5")
        if control_timestep_s <= 0.0 or prediction_horizon_s <= 0.0:
            raise ValueError("Semantic time constants must be positive")
        self.route = route
        self.route_lanes = tuple(route_lanes)
        self.context_provider = context_provider
        self.history_length = history_length
        self.dt = control_timestep_s
        self.dynamic_radius_m = dynamic_radius_m
        self.static_radius_m = static_radius_m
        self.control_radius_m = control_radius_m
        self.signal_range_m = signal_range_m
        self.signal_fov_degrees = signal_fov_degrees
        self.signal_camera_height_m = signal_camera_height_m
        self.prediction_horizon_s = prediction_horizon_s
        self.vertical_tolerance_m = vertical_tolerance_m
        self.ego_speed_cap_mps = ego_speed_cap_mps
        self._scenario_id: str | None = None
        self._last_context_step: int | None = None
        self._actor_cache: dict[str, deque[ActorSnapshot]] = {}
        self._ego_frames: deque[_EgoFrame] = deque(maxlen=history_length)
        self._slot_actor: dict[int, str] = {}
        self._slot_last_seen: dict[int, int] = {}
        self._last_diagnostics = SemanticOverflowDiagnostics({}, {}, {}, 0, {}, {})
        self._route_incompatible_static_features = 0

    @property
    def diagnostics(self) -> SemanticOverflowDiagnostics:
        return self._last_diagnostics

    def reset(self) -> None:
        """Clear every episode-local cache and slot reservation."""

        self._scenario_id = None
        self._last_context_step = None
        self._actor_cache.clear()
        self._ego_frames.clear()
        self._slot_actor.clear()
        self._slot_last_seen.clear()
        self._route_incompatible_static_features = 0

    def commit_context(self, context: CausalSceneContext) -> None:
        """Commit one observed snapshot to the causal track cache."""

        if self._scenario_id != context.snapshot.scenario_id:
            self.reset()
            self._scenario_id = context.snapshot.scenario_id
        if self._last_context_step == context.snapshot.step_index:
            return
        ego = context.snapshot.ego
        for actor in context.snapshot.actors:
            if actor.actor_id == ego.actor_id:
                continue
            distance = hypot(
                actor.position_xy[0] - ego.position_xy[0], actor.position_xy[1] - ego.position_xy[1]
            )
            if (
                distance > self.dynamic_radius_m
                and actor.actor_class != ActorClass.STATIC_COLLIDABLE
            ):
                continue
            if abs(actor.position_z - ego.position_z) > self.vertical_tolerance_m:
                continue
            history = self._actor_cache.setdefault(
                actor.actor_id, deque(maxlen=self.history_length)
            )
            if not history or history[-1] != actor:
                history.append(actor)
        self._last_context_step = context.snapshot.step_index

    def _context(self, context: CausalSceneContext | None) -> CausalSceneContext:
        selected = context
        if selected is None and self.context_provider is not None:
            selected = self.context_provider()
        if not isinstance(selected, CausalSceneContext):
            raise CausalSemanticObservationError(
                "Semantic observation requires an environment-owned committed CausalSceneContext"
            )
        if selected.mission_route != self.route:
            raise CausalSemanticObservationError(
                "Semantic route differs from committed mission route geometry"
            )
        self.commit_context(selected)
        return selected

    def __call__(self, vehicle: object) -> SemanticObservationBatch:
        return self.build(vehicle)

    def build(
        self, vehicle: object, context: CausalSceneContext | None = None
    ) -> SemanticObservationBatch:
        context = self._context(context)
        ego = context.snapshot.ego
        mission_s_m = _mission_s_m(context)
        self._record_ego_frame(
            vehicle, ego, mission_s_m, station_is_measured=_mission_station_is_measured(context)
        )
        ego_speed_cap = self.ego_speed_cap_mps or ego.configured_speed_cap_mps
        if ego_speed_cap is None or ego_speed_cap <= 0.0:
            raise CausalSemanticObservationError(
                "Ego speed cap is required for semantic normalization"
            )

        ego_history, ego_history_mask = self._build_ego_history(ego, ego_speed_cap)
        ego_current = np.asarray(
            [
                _clip(_dimensions(ego.footprint)[0], 10.0, lower=0.0),
                _clip(_dimensions(ego.footprint)[1], 5.0, lower=0.0),
                _clip(mission_s_m, max(self.route.length_m, 1.0), lower=0.0),
            ],
            dtype=np.float32,
        )
        route, route_mask = self._build_route(ego, mission_s_m)
        dynamic, dynamic_mask = self._build_dynamic(context, ego, ego_speed_cap)
        static, static_mask = self._build_static(context, ego)
        self._last_diagnostics = replace(
            self._last_diagnostics,
            route_incompatible_static_features=self._route_incompatible_static_features,
        )
        lane_road = self._build_lane_road(context, ego, mission_s_m, vehicle)
        controls, controls_mask = self._build_controls(context, ego, mission_s_m)
        interactions, interactions_mask = self._build_interactions(context, ego)
        temporal = self._build_temporal(context)
        return SemanticObservationBatch(
            ego_history=ego_history,
            ego_history_mask=ego_history_mask,
            ego_current=ego_current,
            route=route,
            route_mask=route_mask,
            dynamic=dynamic,
            dynamic_mask=dynamic_mask,
            static=static,
            static_mask=static_mask,
            lane_road=lane_road,
            controls=controls,
            controls_mask=controls_mask,
            interactions=interactions,
            interactions_mask=interactions_mask,
            temporal=temporal,
        )

    def _record_ego_frame(
        self,
        vehicle: object,
        ego: ActorSnapshot,
        mission_s_m: float,
        *,
        station_is_measured: bool,
    ) -> None:
        if self._ego_frames and self._last_context_step is not None:
            last = self._ego_frames[-1]
            if last.snapshot == ego:
                return
        velocity = ego.velocity_xy
        previous_velocity = getattr(vehicle, "last_velocity", None)
        if previous_velocity is not None and not isinstance(previous_velocity, (int, float)):
            previous_velocity = _vector(previous_velocity, name="vehicle.last_velocity")
        elif self._ego_frames:
            previous_velocity = self._ego_frames[-1].snapshot.velocity_xy
        else:
            raise CausalSemanticObservationError(
                "Current ego acceleration requires vehicle.last_velocity at reset"
            )
        acceleration = (
            (velocity[0] - previous_velocity[0]) / self.dt,
            (velocity[1] - previous_velocity[1]) / self.dt,
        )
        steering = getattr(vehicle, "steering", None)
        if steering is None:
            action = getattr(vehicle, "current_action", None)
            steering = action[0] if action is not None else None
        throttle = getattr(vehicle, "throttle_brake", None)
        if throttle is None:
            action = getattr(vehicle, "current_action", None)
            throttle = action[1] if action is not None else None
        steering_value = _finite(steering, name="ego.steering")
        throttle_value = _finite(throttle, name="ego.throttle_brake")
        yaw_rate_value = getattr(vehicle, "yaw_rate", None)
        if yaw_rate_value is None:
            angular = getattr(vehicle, "angular_velocity", None)
            if angular is not None:
                angular_values = tuple(float(value) for value in angular)
                if len(angular_values) >= 3:
                    yaw_rate_value = angular_values[2]
        if yaw_rate_value is None:
            previous_heading = getattr(vehicle, "last_heading_dir", None)
            if previous_heading is not None:
                previous_heading_xy = _vector(previous_heading, name="vehicle.last_heading_dir")
                previous_angle = atan2(previous_heading_xy[1], previous_heading_xy[0])
                yaw_rate_value = (
                    atan2(
                        sin(ego.heading_rad - previous_angle), cos(ego.heading_rad - previous_angle)
                    )
                    / self.dt
                )
        if yaw_rate_value is None and self._ego_frames:
            previous = self._ego_frames[-1].snapshot
            yaw_rate_value = (
                atan2(
                    sin(ego.heading_rad - previous.heading_rad),
                    cos(ego.heading_rad - previous.heading_rad),
                )
                / self.dt
            )
        if yaw_rate_value is None:
            raise CausalSemanticObservationError("Current ego yaw rate is unavailable")
        projection = _ego_local_projection(
            self.route, ego, mission_s_m, station_is_measured=station_is_measured
        )
        heading_error = atan2(
            sin(ego.heading_rad - atan2(*reversed(projection.tangent_xy))),
            cos(ego.heading_rad - atan2(*reversed(projection.tangent_xy))),
        )
        self._ego_frames.append(
            _EgoFrame(
                snapshot=ego,
                steering=_clip(steering_value, 1.0),
                throttle_brake=_clip(throttle_value, 1.0),
                acceleration_xy=acceleration,
                yaw_rate=_finite(yaw_rate_value, name="ego.yaw_rate"),
                lane_offset_m=projection.lateral_distance_m,
                route_heading_error_rad=heading_error,
            )
        )

    def _build_ego_history(
        self, ego: ActorSnapshot, ego_speed_cap: float
    ) -> tuple[np.ndarray, np.ndarray]:
        payload = np.zeros((self.history_length, 10), dtype=np.float32)
        mask = np.zeros(self.history_length, dtype=np.float32)
        frames = list(self._ego_frames)[-self.history_length :]
        start = self.history_length - len(frames)
        for index, frame in enumerate(frames, start):
            snapshot = frame.snapshot
            velocity = _world_to_ego(snapshot.velocity_xy, ego.heading_rad)
            acceleration = _world_to_ego(frame.acceleration_xy, ego.heading_rad)
            payload[index] = np.asarray(
                [
                    _clip(velocity[0], ego_speed_cap),
                    _clip(velocity[1], ego_speed_cap),
                    _clip(acceleration[0], 20.0),
                    _clip(acceleration[1], 20.0),
                    _clip(frame.yaw_rate, 1.0),
                    frame.steering,
                    frame.throttle_brake,
                    _clip(frame.lane_offset_m, 6.0),
                    sin(frame.route_heading_error_rad),
                    cos(frame.route_heading_error_rad),
                ],
                dtype=np.float32,
            )
            mask[index] = 1.0
        return payload, mask

    def _build_route(self, ego: ActorSnapshot, current_s: float) -> tuple[np.ndarray, np.ndarray]:
        payload = np.zeros((10, 7), dtype=np.float32)
        mask = np.zeros(10, dtype=np.float32)
        lane_width = self._lane_width(ego.live_lane_id)
        for index in range(10):
            target_s = current_s + (index + 1) * 5.0
            if target_s > self.route.length_m:
                continue
            point = self.route.point_at(target_s)
            projection = self.route.project(point[:2], position_z=point[2])
            relative = _relative_position(point[:2], ego.position_xy, ego.heading_rad)
            route_heading = atan2(projection.tangent_xy[1], projection.tangent_xy[0])
            error = route_heading - ego.heading_rad
            curvature = self._route_curvature(target_s)
            payload[index] = np.asarray(
                [
                    _clip(relative[0], 50.0),
                    _clip(relative[1], 50.0),
                    sin(error),
                    cos(error),
                    _clip(curvature, 0.2),
                    _clip(lane_width, 6.0, lower=0.0),
                    _clip(target_s - current_s, 50.0, lower=0.0),
                ],
                dtype=np.float32,
            )
            mask[index] = 1.0
        return payload, mask

    def _route_curvature(self, s_m: float) -> float:
        delta = min(1.0, self.route.length_m / 4.0)
        first = self.route.project(self.route.point_at(max(0.0, s_m - delta))[:2])
        second = self.route.project(self.route.point_at(min(self.route.length_m, s_m + delta))[:2])
        return atan2(
            first.tangent_xy[0] * second.tangent_xy[1] - first.tangent_xy[1] * second.tangent_xy[0],
            first.tangent_xy[0] * second.tangent_xy[0] + first.tangent_xy[1] * second.tangent_xy[1],
        ) / max(2.0 * delta, 1.0e-6)

    def _lane_width(self, lane_id: str | None, vehicle: object | None = None) -> float:
        if lane_id is not None:
            for lane in self.route_lanes:
                if lane.lane_id == lane_id:
                    width = float(lane.polygon_xy.bounds[3] - lane.polygon_xy.bounds[1])
                    if width > 0.0:
                        return width
        if vehicle is not None:
            navigation = getattr(vehicle, "navigation", None)
            getter = getattr(navigation, "get_current_lane_width", None)
            if callable(getter):
                width = _finite(getter(), name="navigation.current_lane_width")
                if width > 0.0:
                    return width
        if self.route_lanes:
            return max(
                float(
                    self.route_lanes[0].polygon_xy.bounds[3]
                    - self.route_lanes[0].polygon_xy.bounds[1]
                ),
                1.0,
            )
        raise CausalSemanticObservationError("Canonical lane width is unavailable")

    def _dynamic_candidates(
        self, context: CausalSceneContext, ego: ActorSnapshot
    ) -> list[ActorSnapshot]:
        result = []
        for actor in context.snapshot.actors:
            if actor.actor_id == ego.actor_id or actor.actor_class == ActorClass.STATIC_COLLIDABLE:
                continue
            distance = hypot(
                actor.position_xy[0] - ego.position_xy[0], actor.position_xy[1] - ego.position_xy[1]
            )
            if (
                distance <= self.dynamic_radius_m
                and abs(actor.position_z - ego.position_z) <= self.vertical_tolerance_m
            ):
                result.append(actor)
        return result

    def _zone_actor_ids(self, context: CausalSceneContext, ego: ActorSnapshot) -> set[str]:
        ids: set[str] = set()
        for zone in context.conflict_zones.values():
            if zone.other_movement_key is None:
                continue
            for actor in self._dynamic_candidates(context, ego):
                if actor.live_lane_id == zone.other_movement_key.approach_lane_id:
                    ids.add(actor.actor_id)
        return ids

    def _cpa(self, actor: ActorSnapshot, ego: ActorSnapshot) -> tuple[bool, float, float]:
        position = np.asarray(
            (actor.position_xy[0] - ego.position_xy[0], actor.position_xy[1] - ego.position_xy[1])
        )
        velocity = np.asarray(
            (actor.velocity_xy[0] - ego.velocity_xy[0], actor.velocity_xy[1] - ego.velocity_xy[1])
        )
        denominator = float(np.dot(velocity, velocity)) + 1.0e-8
        raw = -float(np.dot(position, velocity)) / denominator
        valid = float(np.dot(position, velocity)) < 0.0 and 0.0 <= raw <= self.prediction_horizon_s
        tau = float(np.clip(raw, 0.0, self.prediction_horizon_s))
        distance = float(np.linalg.norm(position + tau * velocity))
        return valid, tau, distance

    def _dynamic_key(
        self, actor: ActorSnapshot, ego: ActorSnapshot, conflict_ids: set[str]
    ) -> tuple[object, ...]:
        valid, t_cpa, d_cpa = self._cpa(actor, ego)
        relation = (
            0 if actor.live_lane_id == ego.live_lane_id and actor.live_lane_id is not None else 1
        )
        distance = hypot(
            actor.position_xy[0] - ego.position_xy[0], actor.position_xy[1] - ego.position_xy[1]
        )
        if actor.actor_id in conflict_ids:
            return (0, not valid, t_cpa, d_cpa, relation, distance, actor.actor_id)
        # Context quota (OBS-V1.1 SS8.1 amendment 2026-07-26, ADR-026): plain
        # distance, not lane precedence or CPA/TTC. CPA assumes constant-velocity
        # extrapolation, which is least reliable for exactly the agents (about to
        # turn/brake/accelerate) that matter most once this pool is not tiny; see
        # ADR-026 for the literature basis (interactivity beats distance only for
        # small N, per Sun et al. 2021).
        return (1, distance, actor.actor_id)

    def _route_s(self, actor: ActorSnapshot, ego: ActorSnapshot) -> float:
        try:
            return self.route.project(actor.position_xy, position_z=actor.position_z).s_m
        except ValueError:
            return -float("inf")

    def _route_lateral(self, actor: ActorSnapshot) -> float:
        try:
            return self.route.project(
                actor.position_xy, position_z=actor.position_z
            ).lateral_distance_m
        except ValueError:
            return float("inf")

    def _assign_slots(
        self, selected: list[tuple[ActorSnapshot, bool]], step: int
    ) -> dict[int, ActorSnapshot]:
        current = {actor.actor_id: is_conflict for actor, is_conflict in selected}
        for slot, actor_id in list(self._slot_actor.items()):
            if actor_id not in current:
                if step - self._slot_last_seen.get(slot, step) >= self.history_length:
                    self._slot_actor.pop(slot, None)
                    self._slot_last_seen.pop(slot, None)
        assignments: dict[int, ActorSnapshot] = {}
        occupied: set[int] = set()
        by_id = {actor.actor_id: (actor, is_conflict) for actor, is_conflict in selected}
        for actor_id, is_conflict in selected:
            previous = next(
                (slot for slot, value in self._slot_actor.items() if value == actor_id), None
            )
            if previous is not None and (
                (is_conflict and previous < 8) or (not is_conflict and previous >= 8)
            ):
                assignments[previous] = actor_id
                occupied.add(previous)
            elif previous is not None:
                self._slot_actor.pop(previous, None)
                self._slot_last_seen.pop(previous, None)
        for actor, is_conflict in selected:
            if actor.actor_id in assignments.values():
                continue
            preferred = range(0, 8) if is_conflict else range(8, 16)
            fallback = range(8, 16) if is_conflict else range(0, 8)
            available = [slot for slot in (*preferred, *fallback) if slot not in occupied]
            if not available:
                raise CausalSemanticObservationError("Dynamic slot assignment exceeded capacity")
            slot = available[0]
            old_actor = self._slot_actor.get(slot)
            if old_actor is not None:
                self._slot_actor = {
                    key: value for key, value in self._slot_actor.items() if value != old_actor
                }
            self._slot_actor[slot] = actor.actor_id
            self._slot_last_seen[slot] = 0
            assignments[slot] = actor.actor_id
            occupied.add(slot)
        result: dict[int, ActorSnapshot] = {}
        for slot, actor_id in assignments.items():
            if actor_id in by_id:
                result[slot] = by_id[actor_id][0]
                self._slot_actor[slot] = actor_id
                self._slot_last_seen[slot] = step
        return result

    def _build_dynamic(
        self, context: CausalSceneContext, ego: ActorSnapshot, ego_speed_cap: float
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = self._dynamic_candidates(context, ego)
        conflict_ids = self._zone_actor_ids(context, ego)
        ordered = sorted(candidates, key=lambda actor: self._dynamic_key(actor, ego, conflict_ids))
        conflicts = [actor for actor in ordered if actor.actor_id in conflict_ids]
        selected_conflicts = conflicts[:8]
        remaining = [actor for actor in ordered if actor not in selected_conflicts]
        selected = selected_conflicts + remaining[: max(0, 16 - len(selected_conflicts))]
        slots = self._assign_slots(
            [(actor, actor.actor_id in conflict_ids) for actor in selected],
            context.snapshot.step_index,
        )
        payload = np.zeros((16, 5, 22), dtype=np.float32)
        mask = np.zeros((16, 5), dtype=np.float32)
        for slot, actor in slots.items():
            history = list(self._actor_cache.get(actor.actor_id, ()))
            history = history[-self.history_length :]
            first = self.history_length - len(history)
            for history_index, snapshot in enumerate(history, first):
                payload[slot, history_index] = self._dynamic_features(
                    snapshot, ego, ego_speed_cap, context
                )
                mask[slot, history_index] = 1.0
        self._last_diagnostics = SemanticOverflowDiagnostics(
            {"dynamic": len(candidates)},
            {"dynamic": len(selected)},
            {"dynamic": max(0, len(candidates) - len(selected))},
            sum(actor.actor_id in conflict_ids for actor in conflicts[8:]),
            {"dynamic": self._dynamic_key(selected[-1], ego, conflict_ids) if selected else ()},
            {
                "dynamic": self._dynamic_key(ordered[len(selected)], ego, conflict_ids)
                if len(ordered) > len(selected)
                else ()
            },
        )
        return payload, mask

    def _lane_relation(
        self, actor: ActorSnapshot, ego: ActorSnapshot, relative_y: float
    ) -> list[float]:
        if actor.live_lane_id is None or ego.live_lane_id is None:
            return _one_hot(4, 5)
        if actor.live_lane_id == ego.live_lane_id:
            return _one_hot(0, 5)
        # A lateral relation is used only when the map does not expose an
        # adjacency record; it is still current-geometry-derived, not intent.
        if abs(relative_y) <= 1.5 * self._lane_width(ego.live_lane_id):
            return _one_hot(1 if relative_y > 0.0 else 2, 5)
        return _one_hot(3, 5)

    def _dynamic_features(
        self,
        actor: ActorSnapshot,
        ego: ActorSnapshot,
        ego_speed_cap: float,
        context: CausalSceneContext,
    ) -> np.ndarray:
        relative = _relative_position(actor.position_xy, ego.position_xy, ego.heading_rad)
        relative_velocity = _world_to_ego(
            (actor.velocity_xy[0] - ego.velocity_xy[0], actor.velocity_xy[1] - ego.velocity_xy[1]),
            ego.heading_rad,
        )
        dimensions = _dimensions(actor.footprint)
        # VRUs (pedestrians/cyclists) never carry a configured_speed_cap_mps
        # (metadrive_live.py only populates it for ActorClass.VEHICLE), so the
        # fallback must not depend on the actor's own instantaneous speed: that
        # would make the same physical relative velocity map to a different
        # normalized value depending on how fast the VRU happens to be moving,
        # and the scale would drift most exactly when closing speed is highest.
        actor_cap = actor.configured_speed_cap_mps or ego_speed_cap
        cpa_valid, t_cpa, d_cpa = self._cpa(actor, ego)
        relative_heading = actor.heading_rad - ego.heading_rad
        actor_projection = self._route_projection_or_raise(actor, ego, context)
        ego_projection = self._route_projection_or_raise(ego, ego, context)
        values = [
            _clip(relative[0], 50.0),
            _clip(relative[1], 50.0),
            _clip(relative_velocity[0], ego_speed_cap + actor_cap),
            _clip(relative_velocity[1], ego_speed_cap + actor_cap),
            *_heading_sincos(relative_heading),
            _clip(dimensions[0], 10.0, lower=0.0),
            _clip(dimensions[1], 5.0, lower=0.0),
            *_one_hot(_actor_type_index(actor.actor_class), 4),
            *self._lane_relation(actor, ego, relative[1]),
            _clip(actor_projection.s_m - ego_projection.s_m, 50.0),
            _clip(actor_projection.lateral_distance_m, 50.0, lower=0.0),
            float(cpa_valid),
            _clip(t_cpa, 3.0, lower=0.0),
            _clip(d_cpa, 50.0, lower=0.0),
        ]
        if len(values) != 22:
            raise RuntimeError(f"Dynamic feature contract produced {len(values)} values")
        return np.asarray(values, dtype=np.float32)

    def _route_projection_or_raise(
        self, actor: ActorSnapshot, ego: ActorSnapshot, context: CausalSceneContext
    ) -> RouteProjection:
        try:
            return self.route.project(actor.position_xy, position_z=actor.position_z)
        except ValueError as error:
            diagnostics = self.route.projection_diagnostics(
                actor.position_xy, position_z=actor.position_z
            )
            raise CausalSemanticObservationError(
                "Semantic route projection unavailable: "
                f"scenario_id={context.snapshot.scenario_id!r}, "
                f"step={context.snapshot.step_index}, actor_id={actor.actor_id!r}, "
                f"actor_z_m={actor.position_z:.6f}, ego_id={ego.actor_id!r}, "
                f"ego_z_m={ego.position_z:.6f}, "
                f"actor_ego_vertical_delta_m={abs(actor.position_z - ego.position_z):.6f}, "
                f"nearest_planar_segment_index={diagnostics.nearest_planar_segment_index}, "
                f"nearest_planar_s_m={diagnostics.nearest_planar_s_m:.6f}, "
                f"nearest_planar_z_m={diagnostics.nearest_planar_z_m:.6f}, "
                f"nearest_planar_distance_m={diagnostics.nearest_planar_distance_m:.6f}, "
                f"minimum_route_vertical_delta_m={diagnostics.minimum_vertical_difference_m:.6f}, "
                f"compatible_route_segment_count={diagnostics.vertically_compatible_segment_count}, "
                f"route_z_range_m=({diagnostics.route_min_z_m:.6f}, {diagnostics.route_max_z_m:.6f})"
            ) from error

    def _build_static(
        self, context: CausalSceneContext, ego: ActorSnapshot
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates: list[
            tuple[str, tuple[float, float], float, tuple[float, float], int, float, float]
        ] = []
        for actor in context.snapshot.actors:
            if actor.actor_class != ActorClass.STATIC_COLLIDABLE:
                continue
            distance = hypot(
                actor.position_xy[0] - ego.position_xy[0], actor.position_xy[1] - ego.position_xy[1]
            )
            if (
                distance <= self.static_radius_m
                and abs(actor.position_z - ego.position_z) <= self.vertical_tolerance_m
            ):
                route_s = self._route_s(actor, ego)
                candidates.append(
                    (
                        actor.actor_id,
                        actor.position_xy,
                        actor.heading_rad,
                        _dimensions(actor.footprint),
                        3,
                        route_s,
                        self._route_lateral(actor),
                    )
                )
        for feature_id, feature in context.episode_cache.map_feature_catalog.items():
            if feature.feature_class not in {
                MapFeatureClass.OTHER_NON_DRIVABLE,
                MapFeatureClass.ROAD_BOUNDARY,
            }:
                continue
            point = feature.geometry.representative_point()
            elevation = feature.elevation_m
            if (
                elevation is not None
                and abs(elevation - ego.position_z) > self.vertical_tolerance_m
            ):
                continue
            distance = hypot(point.x - ego.position_xy[0], point.y - ego.position_xy[1])
            if distance <= self.static_radius_m:
                projection_z = elevation if elevation is not None else ego.position_z
                try:
                    projection = self.route.project((point.x, point.y), position_z=projection_z)
                except ValueError as error:
                    diagnostics = self.route.projection_diagnostics(
                        (point.x, point.y), position_z=projection_z
                    )
                    if diagnostics.vertically_compatible_segment_count == 0:
                        self._route_incompatible_static_features += 1
                        continue
                    raise CausalSemanticObservationError(
                        "Semantic static-map route projection unavailable: "
                        f"scenario_id={context.snapshot.scenario_id}; step={context.snapshot.step_index}; "
                        f"feature_id={feature_id}; feature_class={feature.feature_class.value}; "
                        f"feature_position_xy={(point.x, point.y)}; feature_elevation_m={elevation}; "
                        f"ego_id={ego.actor_id}; ego_z_m={ego.position_z}; "
                        f"nearest_planar_segment_index={diagnostics.nearest_planar_segment_index}; "
                        f"nearest_planar_s_m={diagnostics.nearest_planar_s_m}; "
                        f"nearest_planar_z_m={diagnostics.nearest_planar_z_m}; "
                        f"nearest_planar_distance_m={diagnostics.nearest_planar_distance_m}; "
                        f"minimum_vertical_difference_m={diagnostics.minimum_vertical_difference_m}; "
                        f"vertically_compatible_segment_count={diagnostics.vertically_compatible_segment_count}; "
                        f"route_z_range_m=({diagnostics.route_min_z_m},{diagnostics.route_max_z_m})"
                    ) from error
                bounds = feature.geometry.bounds
                candidates.append(
                    (
                        feature_id,
                        (point.x, point.y),
                        0.0,
                        (max(bounds[2] - bounds[0], 0.1), max(bounds[3] - bounds[1], 0.1)),
                        3,
                        projection.s_m,
                        projection.lateral_distance_m,
                    )
                )
        candidates.sort(
            key=lambda item: (
                not (item[5] >= self._route_s(ego, ego)),
                abs(item[6]),
                hypot(item[1][0] - ego.position_xy[0], item[1][1] - ego.position_xy[1]),
                item[0],
            )
        )
        payload = np.zeros((8, 13), dtype=np.float32)
        mask = np.zeros(8, dtype=np.float32)
        for index, (_id, position, heading, dimensions, type_index, route_s, lateral) in enumerate(
            candidates[:8]
        ):
            relative = _relative_position(position, ego.position_xy, ego.heading_rad)
            payload[index] = np.asarray(
                [
                    _clip(relative[0], 50.0),
                    _clip(relative[1], 50.0),
                    *_heading_sincos(heading - ego.heading_rad),
                    _clip(dimensions[0], 10.0, lower=0.0),
                    _clip(dimensions[1], 5.0, lower=0.0),
                    *_one_hot(type_index, 5),
                    _clip(route_s - self._route_s(ego, ego), 50.0),
                    _clip(lateral, 50.0, lower=0.0),
                ],
                dtype=np.float32,
            )
            mask[index] = 1.0
        return payload, mask

    def _visible_signal_state(
        self, control: TrafficControlRecord, context: CausalSceneContext, vehicle: object
    ) -> tuple[str, bool]:
        if control.control_type == ApproachControl.STOP:
            return "not-signal", True
        for physical_id in control.physical_control_ids:
            visibility = mapped_signal_visibility(
                vehicle,
                str(physical_id),
                range_m=self.signal_range_m,
                horizontal_fov_deg=self.signal_fov_degrees,
                camera_height_m=self.signal_camera_height_m,
            )
            if not visibility.visible:
                continue
            state = str(context.snapshot.signal_states_by_physical_id.get(physical_id, "")).lower()
            if state:
                return state, state not in {"unknown", "lane_state_unknown"}
        return "unknown", False

    def _control_candidates(
        self, context: CausalSceneContext, ego: ActorSnapshot, current_s: float
    ) -> list[TrafficControlRecord]:
        candidates = [
            control
            for control in context.traffic_controls
            if abs(control.route_s_m - current_s) <= self.control_radius_m
            and abs(control.elevation_m - ego.position_z) <= self.vertical_tolerance_m
        ]
        return sorted(
            candidates,
            key=lambda control: (
                control.route_s_m < current_s,
                max(0.0, control.route_s_m - current_s),
                abs(control.route_s_m - current_s),
                control.control_group_id,
            ),
        )

    def _build_lane_road(
        self, context: CausalSceneContext, ego: ActorSnapshot, current_s: float, vehicle: object
    ) -> np.ndarray:
        lane_width = self._lane_width(ego.live_lane_id, vehicle)
        boundaries = {"left": (1.0, 3), "right": (1.0, 3)}
        for feature in context.episode_cache.map_feature_catalog.values():
            if feature.feature_class not in {
                MapFeatureClass.LANE_MARKING_SOLID,
                MapFeatureClass.LANE_MARKING_DASHED,
                MapFeatureClass.ROAD_BOUNDARY,
            }:
                continue
            point = feature.geometry.representative_point()
            distance = feature.geometry.distance(ego.footprint)
            if distance > 10.0:
                continue
            relative = _relative_position((point.x, point.y), ego.position_xy, ego.heading_rad)
            side = "left" if relative[1] >= 0.0 else "right"
            type_index = (
                0
                if feature.feature_class == MapFeatureClass.LANE_MARKING_SOLID
                else 1
                if feature.feature_class == MapFeatureClass.LANE_MARKING_DASHED
                else 2
            )
            signed = -distance if feature.geometry.intersects(ego.footprint) else distance
            if signed < boundaries[side][0] * 50.0:
                boundaries[side] = (signed, type_index)
        curvature = self._route_curvature(current_s)
        values = [
            _clip(lane_width, 6.0, lower=0.0),
            _clip(boundaries["left"][0], 50.0),
            _clip(boundaries["right"][0], 50.0),
            *_one_hot(boundaries["left"][1], 4),
            *_one_hot(boundaries["right"][1], 4),
            0.0,
            0.0,
            _clip(curvature, 0.2),
        ]
        return np.asarray(values, dtype=np.float32)

    def _control_state(
        self, control: TrafficControlRecord, context: CausalSceneContext
    ) -> tuple[str, bool]:
        values = [
            str(context.snapshot.signal_states_by_physical_id.get(identifier, "")).lower()
            for identifier in control.physical_control_ids
        ]
        if control.control_type == ApproachControl.STOP:
            return "not-signal", True
        state = next((value for value in values if value), "")
        return state, bool(state and state not in {"unknown", "lane_state_unknown"})

    def _build_controls(
        self, context: CausalSceneContext, ego: ActorSnapshot, current_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = [
            control
            for control in context.traffic_controls
            if abs(control.route_s_m - current_s) <= self.control_radius_m
            and abs(control.elevation_m - ego.position_z) <= self.vertical_tolerance_m
        ]
        candidates.sort(
            key=lambda control: (
                control.control_group_id
                not in {context.memory.active_signal_group_id, context.memory.active_stop_group_id},
                control.control_group_id
                not in context.memory.resolved_signal_group_ids
                | context.memory.resolved_stop_group_ids,
                max(0.0, control.route_s_m - current_s),
                abs(control.route_s_m - current_s),
                control.control_group_id,
            )
        )
        payload = np.zeros((8, 17), dtype=np.float32)
        mask = np.zeros(8, dtype=np.float32)
        for index, control in enumerate(candidates[:8]):
            midpoint = control.control_line.centroid
            line_coords = (
                list(control.control_line.coords) if hasattr(control.control_line, "coords") else []
            )
            if len(line_coords) >= 2:
                direction = atan2(
                    line_coords[-1][1] - line_coords[0][1], line_coords[-1][0] - line_coords[0][0]
                )
            else:
                direction = 0.0
            state, valid = self._control_state(control, context)
            state_index = {"green": 0, "yellow": 1, "red": 2, "flashing-yellow": 3}.get(state)
            signal_state = _one_hot(state_index if valid and state_index is not None else 4, 5)
            onset_delta = (
                context.memory.previous_signal_delta_m
                if context.memory.active_signal_group_id == control.control_group_id
                else None
            )
            payload[index] = np.asarray(
                [
                    _clip(midpoint.x - ego.position_xy[0], 80.0),
                    _clip(midpoint.y - ego.position_xy[1], 80.0),
                    _clip(control.route_s_m - current_s, 80.0),
                    sin(direction),
                    cos(direction),
                    *_one_hot(0 if control.control_type == ApproachControl.SIGNAL else 1, 2),
                    *signal_state,
                    float(ego.live_lane_id in control.controlled_lane_ids),
                    float(
                        control.control_group_id
                        in {
                            context.memory.active_signal_group_id,
                            context.memory.active_stop_group_id,
                        }
                    ),
                    float(valid),
                    _clip(onset_delta or 0.0, 80.0),
                    _clip(context.memory.previous_signal_delta_m or 0.0, 80.0),
                ],
                dtype=np.float32,
            )
            mask[index] = 1.0
        return payload, mask

    def _approach_control(
        self, actor: ActorSnapshot, context: CausalSceneContext
    ) -> ApproachControl:
        for control in context.traffic_controls:
            if actor.live_lane_id in control.controlled_lane_ids:
                return control.control_type
        return ApproachControl.UNKNOWN if actor.live_lane_id is None else ApproachControl.NONE

    def _interval(self, actor: ActorSnapshot, zone: ConflictZoneRecord) -> OccupancyInterval | None:
        return predict_occupancy_interval(
            actor_footprint=actor.footprint,
            actor_velocity_xy=actor.velocity_xy,
            zone=zone.polygon,
            horizon_s=self.prediction_horizon_s,
        )

    def _conflict_zone_type_index(
        self, context: CausalSceneContext, zone: ConflictZoneRecord
    ) -> int | None:
        """Return the historical type for V1.1; V1.2 overrides unsupported types."""

        del context, zone
        return 1

    def _build_interactions(
        self, context: CausalSceneContext, ego: ActorSnapshot
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates: list[_InteractionCandidate] = []
        for zone in context.conflict_zones.values():
            if zone.other_movement_key is None:
                continue
            for actor in self._dynamic_candidates(context, ego):
                if actor.live_lane_id != zone.other_movement_key.approach_lane_id:
                    continue
                ego_interval, other_interval = (
                    self._interval(ego, zone),
                    self._interval(actor, zone),
                )
                if (
                    ego_interval is None
                    and other_interval is None
                    and not ego.footprint.intersects(zone.polygon)
                    and not actor.footprint.intersects(zone.polygon)
                ):
                    continue
                candidates.append(
                    _InteractionCandidate(
                        zone_id=zone.zone_id,
                        polygon=zone.polygon,
                        zone_type_index=self._conflict_zone_type_index(context, zone),
                        route_entry_s_m=zone.route_entry_s_m,
                        route_exit_s_m=zone.route_exit_s_m,
                        actor=actor,
                        ego_interval=ego_interval,
                        other_interval=other_interval,
                        ego_movement_key=zone.ego_movement_key,
                        other_movement_key=zone.other_movement_key,
                    )
                )
        for feature in context.episode_cache.map_feature_catalog.values():
            if feature.feature_class != MapFeatureClass.CROSSWALK:
                continue
            elevation = feature.elevation_m if feature.elevation_m is not None else ego.position_z
            if abs(elevation - ego.position_z) > self.vertical_tolerance_m:
                continue
            centroid = feature.geometry.representative_point()
            try:
                projection = self.route.project((centroid.x, centroid.y), position_z=elevation)
            except ValueError:
                # Crosswalk elevation is within ego's vertical tolerance but
                # has no vertically compatible route segment nearby; exclude
                # it from interaction candidates rather than crashing the
                # whole observation build.
                continue
            for actor in self._dynamic_candidates(context, ego):
                if actor.actor_class not in {ActorClass.PEDESTRIAN, ActorClass.CYCLIST}:
                    continue
                ego_interval = predict_occupancy_interval(
                    actor_footprint=ego.footprint,
                    actor_velocity_xy=ego.velocity_xy,
                    zone=feature.geometry,
                    horizon_s=self.prediction_horizon_s,
                )
                other_interval = predict_occupancy_interval(
                    actor_footprint=actor.footprint,
                    actor_velocity_xy=actor.velocity_xy,
                    zone=feature.geometry,
                    horizon_s=self.prediction_horizon_s,
                )
                if (
                    ego_interval is None
                    and other_interval is None
                    and not ego.footprint.intersects(feature.geometry)
                    and not actor.footprint.intersects(feature.geometry)
                ):
                    continue
                candidates.append(
                    _InteractionCandidate(
                        zone_id=feature.feature_id,
                        polygon=feature.geometry,
                        zone_type_index=0,
                        route_entry_s_m=projection.s_m,
                        route_exit_s_m=projection.s_m,
                        actor=actor,
                        ego_interval=ego_interval,
                        other_interval=other_interval,
                    )
                )
        candidates.sort(key=lambda item: (item.zone_id, item.actor.actor_id))
        payload = np.zeros((8, 35), dtype=np.float32)
        mask = np.zeros(8, dtype=np.float32)
        for index, candidate in enumerate(candidates[:8]):
            zone_id, polygon, actor = candidate.zone_id, candidate.polygon, candidate.actor
            ego_interval, other_interval = candidate.ego_interval, candidate.other_interval
            centroid = polygon.centroid
            ego_s = self._route_s(ego, ego)
            relative = _relative_position(
                (centroid.x, centroid.y), ego.position_xy, ego.heading_rad
            )
            priority = MovementPriority.UNDEFINED
            for record in context.episode_cache.movement_priority_records:
                if (
                    record.ego_movement_key == candidate.ego_movement_key
                    and record.other_movement_key == candidate.other_movement_key
                ):
                    priority = record.relation
            priority_index = {
                MovementPriority.EGO_HAS_PRIORITY: 0,
                MovementPriority.OTHER_HAS_PRIORITY: 1,
                MovementPriority.UNDEFINED: 2,
            }[priority]
            ego_control = self._approach_control(ego, context)
            other_control = self._approach_control(actor, context)
            ego_entry = _clip(candidate.route_entry_s_m - ego_s, 50.0)
            ego_exit = _clip(candidate.route_exit_s_m - ego_s, 50.0)
            ego_in = ego.footprint.intersects(polygon)
            other_in = actor.footprint.intersects(polygon)
            pair = (zone_id, actor.actor_id)
            values = [
                _clip(relative[0], 50.0),
                _clip(relative[1], 50.0),
                ego_entry,
                ego_exit,
                *_one_hot(candidate.zone_type_index, 4),
                float(ego_in),
                float(other_in),
                *_one_hot(_actor_type_index(actor.actor_class), 3),
                _clip(ego_interval.start_s if ego_interval else 0.0, 3.0, lower=0.0),
                _clip(
                    ego_interval.end_s if ego_interval and ego_interval.end_s is not None else 3.0,
                    3.0,
                    lower=0.0,
                ),
                _clip(other_interval.start_s if other_interval else 0.0, 3.0, lower=0.0),
                _clip(
                    other_interval.end_s
                    if other_interval and other_interval.end_s is not None
                    else 3.0,
                    3.0,
                    lower=0.0,
                ),
                float(ego_interval is not None),
                float(ego_interval.is_open_end if ego_interval else False),
                float(other_interval is not None),
                float(other_interval.is_open_end if other_interval else False),
                *_one_hot(
                    {
                        ApproachControl.NONE: 0,
                        ApproachControl.STOP: 1,
                        ApproachControl.SIGNAL: 2,
                        ApproachControl.UNKNOWN: 3,
                    }[ego_control],
                    4,
                ),
                *_one_hot(
                    {
                        ApproachControl.NONE: 0,
                        ApproachControl.STOP: 1,
                        ApproachControl.SIGNAL: 2,
                        ApproachControl.UNKNOWN: 3,
                    }[other_control],
                    4,
                ),
                *_one_hot(priority_index, 3),
                float(candidate.zone_type_index == 3),
                float(zone_id in context.memory.preexisting_ego_occupancy_zone_ids),
                # DO NOT "FIX" THE TUPLE ORDER BELOW.  ADR-033.
                #
                # `pair` is (zone_id, actor_id) while both latch sets are keyed
                # (actor_id, zone_id), so this membership test is constant 0.0.
                # That looks like an ordinary bug and is not: the Rulebook
                # produces the reward, so feeding one of its violation latches
                # back into the policy input is label leakage, and correcting
                # the order would activate it for the first time.  The latch is
                # also set *after* the incompatible entry, so it can only
                # support reacting to a violation already committed, never
                # preventing one.
                #
                # This whole legacy OBS-V1.1 path is retained solely to
                # reproduce historical runs; it is no longer the default
                # (see conf/config.yaml).  OBS-V1.3 removes the field, and
                # `PerceptionBoundedSemanticBatchBuilder` reads no Rulebook
                # memory at all — pinned by TEST-009, which builds an
                # observation from a poisoned `RulebookMemory` and asserts it is
                # bit-identical to one built from an empty memory.
                float(
                    pair in context.memory.vehicle_yield_illegal_entries
                    or pair in context.memory.crosswalk_illegal_entries
                ),
            ]
            if len(values) != 35:
                raise RuntimeError(f"Interaction feature contract produced {len(values)} values")
            payload[index] = np.asarray(values, dtype=np.float32)
            mask[index] = 1.0
        return payload, mask

    def _build_temporal(self, context: CausalSceneContext) -> np.ndarray:
        memory = context.memory
        return np.asarray(
            [
                float(memory.active_dashed_boundary_id is not None),
                _clip(memory.dashed_line_timer_s, 2.0, lower=0.0),
                float(memory.active_stop_group_id is not None),
                _clip(memory.stop_continuous_timer_s, 1.0, lower=0.0),
                _clip(memory.stop_best_timer_s, 1.0, lower=0.0),
            ],
            dtype=np.float32,
        )


class PerceptionBoundedSemanticBatchBuilder(CausalSemanticBatchBuilder):
    """OBS-V1.2 builder with physical admission and ego-owned temporal state.

    The historical builder remains untouched for `semantic_v2`.  This subclass
    stores measurements only after a first-hit LiDAR admission at their actual
    simulation step, so a reacquired object cannot fabricate contiguous time
    history.
    """

    context_history_length = 21

    def __init__(self, *, brake_mps2: float | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if brake_mps2 is not None and brake_mps2 <= 0.0:
            raise ValueError("OBS-V1.2 brake_mps2 must be positive when available")
        self.brake_mps2 = brake_mps2
        self._v12_actor_cache: dict[str, deque[tuple[int, ActorSnapshot]]] = {}
        self._visible_actor_ids: frozenset[str] = frozenset()
        self._context_rows: deque[tuple[int, np.ndarray]] = deque(
            maxlen=self.context_history_length
        )
        self._yellow_control_id: str | None = None
        self._yellow_onset_distance_m = 0.0
        self._yellow_onset_speed_mps = 0.0
        self._previous_dashed_feature_id: str | None = None
        self._previous_control_id: str | None = None
        self._last_context_row_step: int | None = None
        self._preexisting_zone_occupancy: dict[str, bool] = {}
        self._dynamic_diagnostics: tuple[Any, ...] = (0, 0, 0, (), ())
        self._static_diagnostics: tuple[int, int] = (0, 0)
        self._control_diagnostics: tuple[int, int] = (0, 0)
        self._interaction_diagnostics: tuple[int, int] = (0, 0)

    def reset(self) -> None:
        super().reset()
        self._v12_actor_cache.clear()
        self._visible_actor_ids = frozenset()
        self._context_rows.clear()
        self._yellow_control_id = None
        self._yellow_onset_distance_m = 0.0
        self._yellow_onset_speed_mps = 0.0
        self._previous_dashed_feature_id = None
        self._previous_control_id = None
        self._last_context_row_step = None
        self._preexisting_zone_occupancy.clear()
        self._dynamic_diagnostics = (0, 0, 0, (), ())
        self._static_diagnostics = (0, 0)
        self._control_diagnostics = (0, 0)
        self._interaction_diagnostics = (0, 0)

    def commit_context(self, context: CausalSceneContext, vehicle: object | None = None) -> None:
        """Commit only currently LiDAR-admitted measurements at their true step."""

        if vehicle is None:
            # The environment publishes the context before it requests an
            # observation.  The first cache write is deliberately deferred
            # until the ego vehicle is available for the physical sweep.
            return
        if self._scenario_id != context.snapshot.scenario_id:
            self.reset()
            self._scenario_id = context.snapshot.scenario_id
        step = context.snapshot.step_index
        if self._last_context_step == step:
            return
        sweep = first_hit_lidar_sweep(vehicle)
        self._visible_actor_ids = sweep.actor_ids
        ego = context.snapshot.ego
        for actor in context.snapshot.actors:
            if actor.actor_id == ego.actor_id or actor.actor_id not in self._visible_actor_ids:
                continue
            if abs(actor.position_z - ego.position_z) > self.vertical_tolerance_m:
                continue
            history = self._v12_actor_cache.setdefault(
                actor.actor_id, deque(maxlen=self.history_length)
            )
            history.append((step, actor))
        self._last_context_step = step

    def _context_v12(
        self, context: CausalSceneContext | None, vehicle: object
    ) -> CausalSceneContext:
        selected = context
        if selected is None and self.context_provider is not None:
            selected = self.context_provider()
        if not isinstance(selected, CausalSceneContext):
            raise CausalSemanticObservationError(
                "Semantic observation requires an environment-owned committed CausalSceneContext"
            )
        if selected.mission_route != self.route:
            raise CausalSemanticObservationError(
                "Semantic route differs from committed mission route geometry"
            )
        self.commit_context(selected, vehicle)
        return selected

    def build(
        self, vehicle: object, context: CausalSceneContext | None = None
    ) -> SemanticObservationBatchV12:
        context = self._context_v12(context, vehicle)
        ego = context.snapshot.ego
        mission_s_m = _mission_s_m(context)
        self._record_ego_frame(
            vehicle, ego, mission_s_m, station_is_measured=_mission_station_is_measured(context)
        )
        ego_speed_cap = self.ego_speed_cap_mps or ego.configured_speed_cap_mps
        if ego_speed_cap is None or ego_speed_cap <= 0.0:
            raise CausalSemanticObservationError(
                "Ego speed cap is required for semantic normalization"
            )

        ego_history, ego_history_mask = self._build_ego_history(ego, ego_speed_cap)
        ego_current = np.asarray(
            [
                _clip(_dimensions(ego.footprint)[0], 10.0, lower=0.0),
                _clip(_dimensions(ego.footprint)[1], 5.0, lower=0.0),
                _clip(mission_s_m, max(self.route.length_m, 1.0), lower=0.0),
            ],
            dtype=np.float32,
        )
        route, route_mask = self._build_route_v12(ego, mission_s_m)
        dynamic, dynamic_mask = self._build_dynamic_v12(context, ego, ego_speed_cap)
        static, static_mask = self._build_static_v12(context, ego)
        lane_road = self._build_lane_road(context, ego, mission_s_m, vehicle)
        controls, controls_mask, control_trace = self._build_controls_v12(
            context, ego, mission_s_m, vehicle, ego_speed_cap
        )
        interactions, interactions_mask = self._build_interactions(context, ego)
        context_history, context_history_mask = self._append_context_row(
            context, ego, lane_road, control_trace, ego_speed_cap
        )
        self._publish_diagnostics()
        return SemanticObservationBatchV12(
            ego_history=ego_history,
            ego_history_mask=ego_history_mask,
            ego_current=ego_current,
            route=route,
            route_mask=route_mask,
            dynamic=dynamic,
            dynamic_mask=dynamic_mask,
            static=static,
            static_mask=static_mask,
            lane_road=lane_road,
            controls=controls,
            controls_mask=controls_mask,
            interactions=interactions,
            interactions_mask=interactions_mask,
            context_history=context_history,
            context_history_mask=context_history_mask,
            signal_onset_state=self._yellow_memory(ego_speed_cap),
        )

    def _dynamic_candidates(
        self, context: CausalSceneContext, ego: ActorSnapshot
    ) -> list[ActorSnapshot]:
        return [
            actor
            for actor in super()._dynamic_candidates(context, ego)
            if actor.actor_id in self._visible_actor_ids
        ]

    def _build_route_v12(
        self, ego: ActorSnapshot, current_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        payload = np.zeros((10, 7), dtype=np.float32)
        mask = np.zeros(10, dtype=np.float32)
        for index in range(10):
            target_s = current_s + (index + 1) * 5.0
            if target_s > self.route.length_m:
                continue
            point = self.route.point_at(target_s)
            projection = self.route.project(point[:2], position_z=point[2])
            relative = _relative_position(point[:2], ego.position_xy, ego.heading_rad)
            route_heading = atan2(projection.tangent_xy[1], projection.tangent_xy[0])
            lane_id = next(
                (
                    lane.lane_id
                    for lane in self.route_lanes
                    if lane.polygon_xy.distance(Point(point[:2])) <= 1.0e-6
                ),
                None,
            )
            if lane_id is None:
                raise CausalSemanticObservationError(
                    "No route-lane polygon contains an OBS-V1.2 route sample"
                )
            payload[index] = np.asarray(
                [
                    _clip(relative[0], 50.0),
                    _clip(relative[1], 50.0),
                    sin(route_heading - ego.heading_rad),
                    cos(route_heading - ego.heading_rad),
                    _clip(self._route_curvature(target_s), 0.2),
                    _clip(self._lane_width(lane_id), 6.0, lower=0.0),
                    _clip(target_s - current_s, 50.0, lower=0.0),
                ],
                dtype=np.float32,
            )
            mask[index] = 1.0
        return payload, mask

    def _assign_slots_v12(
        self, selected: list[tuple[ActorSnapshot, bool]], step: int
    ) -> dict[int, ActorSnapshot]:
        current = {actor.actor_id: conflict for actor, conflict in selected}
        for slot, actor_id in list(self._slot_actor.items()):
            if (
                actor_id not in current
                and step - self._slot_last_seen.get(slot, step) >= self.history_length
            ):
                self._slot_actor.pop(slot, None)
                self._slot_last_seen.pop(slot, None)
        assignments: dict[int, ActorSnapshot] = {}
        used: set[int] = set()
        for actor, is_conflict in selected:
            previous = next(
                (slot for slot, actor_id in self._slot_actor.items() if actor_id == actor.actor_id),
                None,
            )
            if previous is not None and (
                (is_conflict and previous < 8) or (not is_conflict and previous >= 8)
            ):
                assignments[previous] = actor
                used.add(previous)
        for actor, is_conflict in selected:
            if actor in assignments.values():
                continue
            preferred = tuple(range(0, 8)) if is_conflict else tuple(range(8, 16))
            fallback = tuple(range(8, 16)) if is_conflict else tuple(range(0, 8))
            slot = next(
                (candidate for candidate in (*preferred, *fallback) if candidate not in used), None
            )
            if slot is None:
                raise CausalSemanticObservationError("Dynamic slot assignment exceeded capacity")
            assignments[slot] = actor
            used.add(slot)
        for slot, actor in assignments.items():
            self._slot_actor[slot] = actor.actor_id
            self._slot_last_seen[slot] = step
        return assignments

    def _build_dynamic_v12(
        self, context: CausalSceneContext, ego: ActorSnapshot, ego_speed_cap: float
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = self._dynamic_candidates(context, ego)
        conflict_ids = self._zone_actor_ids(context, ego)
        ordered = sorted(candidates, key=lambda actor: self._dynamic_key(actor, ego, conflict_ids))
        conflicts = [actor for actor in ordered if actor.actor_id in conflict_ids]
        selected = (
            conflicts[:8]
            + [actor for actor in ordered if actor not in conflicts[:8]][: 16 - len(conflicts[:8])]
        )
        slots = self._assign_slots_v12(
            [(actor, actor.actor_id in conflict_ids) for actor in selected],
            context.snapshot.step_index,
        )
        payload = np.zeros((16, 5, 22), dtype=np.float32)
        mask = np.zeros((16, 5), dtype=np.float32)
        for slot, actor in slots.items():
            samples = {
                step: snapshot for step, snapshot in self._v12_actor_cache.get(actor.actor_id, ())
            }
            for history_index, sample_step in enumerate(
                range(
                    context.snapshot.step_index - self.history_length + 1,
                    context.snapshot.step_index + 1,
                )
            ):
                snapshot = samples.get(sample_step)
                if snapshot is None:
                    continue
                payload[slot, history_index] = self._dynamic_features(
                    snapshot, ego, ego_speed_cap, context
                )
                mask[slot, history_index] = 1.0
        self._dynamic_diagnostics = (
            len(candidates),
            len(selected),
            len(conflicts[8:]),
            self._dynamic_key(selected[-1], ego, conflict_ids) if selected else (),
            self._dynamic_key(ordered[len(selected)], ego, conflict_ids)
            if len(ordered) > len(selected)
            else (),
        )
        return payload, mask

    def _publish_diagnostics(self) -> None:
        """REQ-014: OBS-V1.1 SS8.5 requires all four groups, not just dynamic."""

        totals, selected, last_key, first_excluded = {}, {}, {}, {}
        groups = {
            "dynamic": self._dynamic_diagnostics[:2],
            "static": self._static_diagnostics,
            "controls": self._control_diagnostics,
            "interactions": self._interaction_diagnostics,
        }
        for name, (total, chosen) in groups.items():
            totals[name] = total
            selected[name] = chosen
        last_key["dynamic"] = self._dynamic_diagnostics[3]
        first_excluded["dynamic"] = self._dynamic_diagnostics[4]
        self._last_diagnostics = SemanticOverflowDiagnostics(
            totals,
            selected,
            {name: max(0, totals[name] - selected[name]) for name in totals},
            self._dynamic_diagnostics[2],
            last_key,
            first_excluded,
            self._route_incompatible_static_features,
        )

    def _conflict_zone_type_index(
        self, context: CausalSceneContext, zone: ConflictZoneRecord
    ) -> int | None:
        """Emit only a source-confirmed roundabout type; otherwise unknown.

        Conflict-zone records carry no deterministic merge/intersection
        taxonomy.  A roundabout priority record does carry lane relations, so
        it is the sole supported non-crosswalk classification in V1.2.
        """

        ego_key = zone.ego_movement_key
        other_key = zone.other_movement_key
        for record in context.episode_cache.roundabout_priority_records:
            lanes = {
                record.entry_lane_id,
                record.circulating_lane_id,
            }
            if {
                ego_key.approach_lane_id,
                other_key.approach_lane_id,
            } <= lanes:
                return 3
        return None

    # ------------------------------------------------------------------
    # OBS-V1.3 overrides.  The base implementations stay untouched so the
    # legacy ``semantic_v2`` path keeps its historical contract (DEC-008).
    # ------------------------------------------------------------------

    @staticmethod
    def _transverse_lane_width(lane: Any) -> float:
        """Measure the lane width along the centerline normal (REQ-001)."""

        centerline = lane.centerline
        length = float(centerline.length_m)
        if length > 0.0:
            midpoint = centerline.point_at(length / 2.0)
            tangent = centerline.project(midpoint[:2]).tangent_xy
            normal = (-tangent[1], tangent[0])
            reach = max(length, 50.0)
            probe = LineString(
                (
                    (midpoint[0] - normal[0] * reach, midpoint[1] - normal[1] * reach),
                    (midpoint[0] + normal[0] * reach, midpoint[1] + normal[1] * reach),
                )
            )
            chord = probe.intersection(lane.polygon_xy)
            if not chord.is_empty and chord.length > 0.0:
                return float(chord.length)
            area = float(lane.polygon_xy.area)
            if area > 0.0:
                return area / length
        return 0.0

    def _lane_width(self, lane_id: str | None, vehicle: object | None = None) -> float:
        if lane_id is not None:
            for lane in self.route_lanes:
                if lane.lane_id == lane_id:
                    width = self._transverse_lane_width(lane)
                    if width > 0.0:
                        return width
        if vehicle is not None:
            navigation = getattr(vehicle, "navigation", None)
            getter = getattr(navigation, "get_current_lane_width", None)
            if callable(getter):
                width = _finite(getter(), name="navigation.current_lane_width")
                if width > 0.0:
                    return width
        for lane in self.route_lanes:
            width = self._transverse_lane_width(lane)
            if width > 0.0:
                return width
        raise CausalSemanticObservationError("Canonical lane width is unavailable")

    def _front_bumper_s(self, ego: ActorSnapshot, mission_s_m: float) -> float:
        """Route abscissa of the ego front bumper (REQ-009).

        The Rulebook decides control-line crossing and zone entry with the
        swept front bumper, so a centre-based distance would offset every
        threshold the policy has to learn by half a vehicle length. The
        search is anchored at the mission tracker's committed ``mission_s_m``
        (same tie-break authority the tracker itself uses), not an
        independently chosen anchor, per DRIVING-MISSION-V1.1 §3.
        """

        half_length = _dimensions(ego.footprint)[0] / 2.0
        bumper = (
            ego.position_xy[0] + cos(ego.heading_rad) * half_length,
            ego.position_xy[1] + sin(ego.heading_rad) * half_length,
        )
        try:
            return self.route.project(
                bumper, position_z=ego.position_z, previous_s_m=mission_s_m
            ).s_m
        except ValueError:
            return mission_s_m

    def _build_lane_road(
        self, context: CausalSceneContext, ego: ActorSnapshot, current_s: float, vehicle: object
    ) -> np.ndarray:
        lane_width = self._lane_width(ego.live_lane_id, vehicle)
        boundaries: dict[str, tuple[float, int]] = {
            "left": (float("inf"), 3),
            "right": (float("inf"), 3),
        }
        for feature in context.episode_cache.map_feature_catalog.values():
            if feature.feature_class not in {
                MapFeatureClass.LANE_MARKING_SOLID,
                MapFeatureClass.LANE_MARKING_DASHED,
                MapFeatureClass.ROAD_BOUNDARY,
            }:
                continue
            elevation = feature.elevation_m if feature.elevation_m is not None else ego.position_z
            if abs(elevation - ego.position_z) > self.vertical_tolerance_m:
                continue
            signed = _signed_footprint_clearance(feature.geometry, ego.footprint)
            if signed > 10.0:
                continue
            # REQ-004: the side must be read at the same place as the distance.
            source, _target = nearest_points(feature.geometry, ego.footprint)
            relative = _relative_position(
                (float(source.x), float(source.y)), ego.position_xy, ego.heading_rad
            )
            side = "left" if relative[1] >= 0.0 else "right"
            type_index = (
                0
                if feature.feature_class == MapFeatureClass.LANE_MARKING_SOLID
                else 1
                if feature.feature_class == MapFeatureClass.LANE_MARKING_DASHED
                else 2
            )
            if signed < boundaries[side][0]:
                boundaries[side] = (signed, type_index)
        left = boundaries["left"][0]
        right = boundaries["right"][0]
        # OBS-V1.3.1 / OBS-LIDAR-V2.0.2: the posted limit of the ego's associated
        # route lane, required by the `speed_limit` sub-rule (ADR-068). Two
        # values, not one: the availability flag is an *explicit* "unavailable"
        # encoding, because a sentinel inside the normalized channel would be
        # indistinguishable from a real limit at that value. It is normalized by
        # the ego speed cap, the same scale the other speed features use, so the
        # agent can compare its own speed with the limit without a change of
        # units.
        posted_limit = self._posted_speed_limit_mps(context, ego)
        speed_scale = self.ego_speed_cap_mps or ego.configured_speed_cap_mps
        if speed_scale is None or speed_scale <= 0.0:
            raise CausalSemanticObservationError(
                "OBS-V1.3 speed-limit feature requires a positive ego speed cap"
            )
        values = [
            _clip(lane_width, 6.0, lower=0.0),
            _clip(50.0 if left == float("inf") else left, 50.0),
            _clip(50.0 if right == float("inf") else right, 50.0),
            *_one_hot(boundaries["left"][1], 4),
            *_one_hot(boundaries["right"][1], 4),
            _clip(self._route_curvature(current_s), 0.2),
            0.0 if posted_limit is None else _clip(posted_limit, speed_scale, lower=0.0),
            0.0 if posted_limit is None else 1.0,
        ]
        if len(values) != 14:
            raise RuntimeError(f"OBS-V1.3 lane/road contract produced {len(values)} values")
        return np.asarray(values, dtype=np.float32)

    def _posted_speed_limit_mps(
        self, context: CausalSceneContext, ego: ActorSnapshot
    ) -> float | None:
        """The same limit the reward reads, resolved the same way.

        RULEBOOK-V5.0 §7 requires the "unavailable" encoding to fire under
        *exactly* the condition that makes the sub-rule inapplicable. Calling the
        rulebook's own lookup is what makes that an identity rather than a
        promise: on the PG panel the feature is unavailable on every step,
        because no PG lane carries real-map provenance.
        """

        cache = context.episode_cache
        route_lanes = getattr(cache, "route_lanes", ())
        if not route_lanes:
            return None
        association = associate_route_lane(
            position_xy=ego.position_xy,
            position_z=ego.position_z,
            heading_rad=ego.heading_rad,
            route_lanes=route_lanes,
        )
        return associated_speed_limit_mps(route_lanes, association)

    def _dynamic_features(
        self,
        actor: ActorSnapshot,
        ego: ActorSnapshot,
        ego_speed_cap: float,
        context: CausalSceneContext,
    ) -> np.ndarray:
        values = super()._dynamic_features(actor, ego, ego_speed_cap, context)
        # REQ-012: the base clamps a signed offset to [0, 1], collapsing the
        # whole right half-plane onto zero.
        projection = self._route_projection_or_raise(actor, ego, context)
        values[DYNAMIC_ROUTE_LATERAL_INDEX] = _clip(projection.lateral_distance_m, 50.0)
        return values

    def _approach_control_in_horizon(
        self, actor: ActorSnapshot, context: CausalSceneContext, current_s: float
    ) -> ApproachControl:
        """Associate a control only inside the local horizon (REQ-025)."""

        if actor.live_lane_id is None:
            return ApproachControl.UNKNOWN
        for control in context.traffic_controls:
            if abs(control.route_s_m - current_s) > self.control_radius_m:
                continue
            if abs(control.elevation_m - actor.position_z) > self.vertical_tolerance_m:
                continue
            if actor.live_lane_id in control.controlled_lane_ids:
                return control.control_type
        return ApproachControl.NONE

    def _zone_route_limits(self, polygon: Any, fallback_s: float) -> tuple[float, float]:
        """True curvilinear entry/exit of a zone along the assigned route."""

        centerline = LineString([(x, y) for x, y, _z in self.route.points_xyz])
        crossing = centerline.intersection(polygon)
        if crossing.is_empty:
            return fallback_s, fallback_s
        geometries = getattr(crossing, "geoms", (crossing,))
        abscissas: list[float] = []
        for geometry in geometries:
            for coordinate in getattr(geometry, "coords", ()):
                try:
                    abscissas.append(self.route.project((coordinate[0], coordinate[1])).s_m)
                except ValueError:
                    continue
        if not abscissas:
            return fallback_s, fallback_s
        return min(abscissas), max(abscissas)

    def _build_interactions(
        self, context: CausalSceneContext, ego: ActorSnapshot
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates: list[_InteractionCandidate] = []
        ego_front_s = self._front_bumper_s(ego, _mission_s_m(context))
        for zone in context.conflict_zones.values():
            if zone.other_movement_key is None:
                continue
            for actor in self._dynamic_candidates(context, ego):
                if actor.live_lane_id != zone.other_movement_key.approach_lane_id:
                    continue
                ego_interval = self._interval(ego, zone)
                other_interval = self._interval(actor, zone)
                if (
                    ego_interval is None
                    and other_interval is None
                    and not ego.footprint.intersects(zone.polygon)
                    and not actor.footprint.intersects(zone.polygon)
                ):
                    continue
                candidates.append(
                    _InteractionCandidate(
                        zone_id=zone.zone_id,
                        polygon=zone.polygon,
                        zone_type_index=self._conflict_zone_type_index(context, zone),
                        route_entry_s_m=zone.route_entry_s_m,
                        route_exit_s_m=zone.route_exit_s_m,
                        actor=actor,
                        ego_interval=ego_interval,
                        other_interval=other_interval,
                        ego_movement_key=zone.ego_movement_key,
                        other_movement_key=zone.other_movement_key,
                    )
                )
        for feature in context.episode_cache.map_feature_catalog.values():
            if feature.feature_class != MapFeatureClass.CROSSWALK:
                continue
            elevation = feature.elevation_m if feature.elevation_m is not None else ego.position_z
            if abs(elevation - ego.position_z) > self.vertical_tolerance_m:
                continue
            centroid = feature.geometry.representative_point()
            try:
                fallback = self.route.project((centroid.x, centroid.y), position_z=elevation).s_m
            except ValueError:
                continue
            # REQ-015: real curvilinear limits instead of the centroid twice.
            entry_s, exit_s = self._zone_route_limits(feature.geometry, fallback)
            for actor in self._dynamic_candidates(context, ego):
                if actor.actor_class not in {ActorClass.PEDESTRIAN, ActorClass.CYCLIST}:
                    continue
                ego_interval = predict_occupancy_interval(
                    actor_footprint=ego.footprint,
                    actor_velocity_xy=ego.velocity_xy,
                    zone=feature.geometry,
                    horizon_s=self.prediction_horizon_s,
                )
                other_interval = predict_occupancy_interval(
                    actor_footprint=actor.footprint,
                    actor_velocity_xy=actor.velocity_xy,
                    zone=feature.geometry,
                    horizon_s=self.prediction_horizon_s,
                )
                if (
                    ego_interval is None
                    and other_interval is None
                    and not ego.footprint.intersects(feature.geometry)
                    and not actor.footprint.intersects(feature.geometry)
                ):
                    continue
                candidates.append(
                    _InteractionCandidate(
                        zone_id=feature.feature_id,
                        polygon=feature.geometry,
                        zone_type_index=0,
                        route_entry_s_m=entry_s,
                        route_exit_s_m=exit_s,
                        actor=actor,
                        ego_interval=ego_interval,
                        other_interval=other_interval,
                    )
                )

        # REQ-008: pre-existing occupancy is recorded the first time a zone
        # becomes a candidate, from geometry the observation already holds.
        for candidate in candidates:
            if candidate.zone_id not in self._preexisting_zone_occupancy:
                self._preexisting_zone_occupancy[candidate.zone_id] = bool(
                    ego.footprint.intersects(candidate.polygon)
                )

        def ranking_key(candidate: _InteractionCandidate) -> tuple[object, ...]:
            ego_in = ego.footprint.intersects(candidate.polygon)
            other_in = candidate.actor.footprint.intersects(candidate.polygon)
            overlap = (
                candidate.ego_interval is not None
                and candidate.other_interval is not None
                and candidate.ego_interval.start_s
                <= (candidate.other_interval.end_s or self.prediction_horizon_s)
                and candidate.other_interval.start_s
                <= (candidate.ego_interval.end_s or self.prediction_horizon_s)
            )
            preexisting = self._preexisting_zone_occupancy.get(candidate.zone_id, False)
            entry_distance = abs(candidate.route_entry_s_m - ego_front_s)
            t_in = (
                candidate.other_interval.start_s
                if candidate.other_interval is not None
                else self.prediction_horizon_s
            )
            actor_distance = hypot(
                candidate.actor.position_xy[0] - ego.position_xy[0],
                candidate.actor.position_xy[1] - ego.position_xy[1],
            )
            return (
                not (ego_in or other_in),
                not overlap,
                not preexisting,
                entry_distance,
                t_in,
                actor_distance,
                candidate.zone_id,
                candidate.actor.actor_id,
            )

        candidates.sort(key=ranking_key)
        payload = np.zeros((8, 33), dtype=np.float32)
        mask = np.zeros(8, dtype=np.float32)
        for index, candidate in enumerate(candidates[:8]):
            polygon, actor = candidate.polygon, candidate.actor
            centroid = polygon.centroid
            relative = _relative_position(
                (centroid.x, centroid.y), ego.position_xy, ego.heading_rad
            )
            priority = MovementPriority.UNDEFINED
            for record in context.episode_cache.movement_priority_records:
                if (
                    record.ego_movement_key == candidate.ego_movement_key
                    and record.other_movement_key == candidate.other_movement_key
                ):
                    priority = record.relation
            priority_index = {
                MovementPriority.EGO_HAS_PRIORITY: 0,
                MovementPriority.OTHER_HAS_PRIORITY: 1,
                MovementPriority.UNDEFINED: 2,
            }[priority]
            ego_control = self._approach_control_in_horizon(ego, context, ego_front_s)
            other_control = self._approach_control_in_horizon(actor, context, ego_front_s)
            control_index = {
                ApproachControl.NONE: 0,
                ApproachControl.STOP: 1,
                ApproachControl.SIGNAL: 2,
                ApproachControl.UNKNOWN: 3,
            }
            ego_interval, other_interval = candidate.ego_interval, candidate.other_interval
            values = [
                _clip(relative[0], 50.0),
                _clip(relative[1], 50.0),
                _clip(candidate.route_entry_s_m - ego_front_s, 50.0),
                _clip(candidate.route_exit_s_m - ego_front_s, 50.0),
                *_one_hot(candidate.zone_type_index, 4),
                float(ego.footprint.intersects(polygon)),
                float(actor.footprint.intersects(polygon)),
                *_one_hot(_actor_type_index(actor.actor_class), 3),
                _clip(ego_interval.start_s if ego_interval else 0.0, 3.0, lower=0.0),
                _clip(
                    ego_interval.end_s if ego_interval and ego_interval.end_s is not None else 3.0,
                    3.0,
                    lower=0.0,
                ),
                _clip(other_interval.start_s if other_interval else 0.0, 3.0, lower=0.0),
                _clip(
                    other_interval.end_s
                    if other_interval and other_interval.end_s is not None
                    else 3.0,
                    3.0,
                    lower=0.0,
                ),
                float(ego_interval is not None),
                float(ego_interval.is_open_end if ego_interval else False),
                float(other_interval is not None),
                float(other_interval.is_open_end if other_interval else False),
                *_one_hot(control_index[ego_control], 4),
                *_one_hot(control_index[other_control], 4),
                *_one_hot(priority_index, 3),
                float(self._preexisting_zone_occupancy.get(candidate.zone_id, False)),
            ]
            if len(values) != 33:
                raise RuntimeError(f"OBS-V1.3 interaction contract produced {len(values)} values")
            payload[index] = np.asarray(values, dtype=np.float32)
            mask[index] = 1.0
        self._interaction_diagnostics = (len(candidates), min(len(candidates), 8))
        return payload, mask

    def _build_static_v12(
        self, context: CausalSceneContext, ego: ActorSnapshot
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates: list[
            tuple[str, tuple[float, float], float, tuple[float, float], int, float, float]
        ] = []
        for actor in context.snapshot.actors:
            if (
                actor.actor_class != ActorClass.STATIC_COLLIDABLE
                or actor.actor_id not in self._visible_actor_ids
            ):
                continue
            distance = hypot(
                actor.position_xy[0] - ego.position_xy[0],
                actor.position_xy[1] - ego.position_xy[1],
            )
            if distance > self.static_radius_m or (
                abs(actor.position_z - ego.position_z) > self.vertical_tolerance_m
            ):
                continue
            # REQ-023: a static actor the route cannot accept degrades its own
            # token instead of failing the whole observation.
            try:
                projection = self.route.project(actor.position_xy, position_z=actor.position_z)
            except ValueError:
                self._route_incompatible_static_features += 1
                continue
            candidates.append(
                (
                    actor.actor_id,
                    actor.position_xy,
                    actor.heading_rad,
                    _dimensions(actor.footprint),
                    STATIC_SUBCLASS_TYPE_INDEX.get(
                        actor.static_subclass, STATIC_TYPE_OTHER_OBSTACLE
                    ),
                    projection.s_m,
                    projection.lateral_distance_m,
                )
            )
        ego_point = Point(ego.position_xy)
        for feature_id, feature in context.episode_cache.map_feature_catalog.items():
            if feature.feature_class not in {
                MapFeatureClass.OTHER_NON_DRIVABLE,
                MapFeatureClass.ROAD_BOUNDARY,
            }:
                continue
            _source, closest = nearest_points(feature.geometry, ego_point)
            position = (float(closest.x), float(closest.y))
            elevation = feature.elevation_at_xy(position)
            if (
                elevation is not None
                and abs(elevation - ego.position_z) > self.vertical_tolerance_m
            ):
                continue
            distance = hypot(position[0] - ego.position_xy[0], position[1] - ego.position_xy[1])
            if distance > self.static_radius_m:
                continue
            projection_z = elevation if elevation is not None else ego.position_z
            try:
                projection = self.route.project(position, position_z=projection_z)
            except ValueError as error:
                diagnostics = self.route.projection_diagnostics(position, position_z=projection_z)
                if diagnostics.vertically_compatible_segment_count == 0:
                    self._route_incompatible_static_features += 1
                    continue
                raise CausalSemanticObservationError(
                    "Semantic static-map route projection unavailable: "
                    f"scenario_id={context.snapshot.scenario_id}; "
                    f"step={context.snapshot.step_index}; feature_id={feature_id}; "
                    f"feature_class={feature.feature_class.value}; "
                    f"feature_position_xy={position}; feature_elevation_m={elevation}"
                ) from error
            # REQ-005 / REQ-006: describe the geometry locally, and carry its
            # local tangent instead of a constant world-frame zero.
            local = _local_window(feature.geometry, ego.position_xy)
            candidates.append(
                (
                    str(feature_id),
                    position,
                    _local_tangent_rad(feature.geometry, position),
                    _local_extent(local),
                    STATIC_TYPE_ROAD_BOUNDARY
                    if feature.feature_class == MapFeatureClass.ROAD_BOUNDARY
                    else STATIC_TYPE_OTHER_NON_DRIVABLE,
                    projection.s_m,
                    projection.lateral_distance_m,
                )
            )
        ego_front_s = self._front_bumper_s(ego, _mission_s_m(context))
        candidates.sort(
            key=lambda item: (
                not (item[5] >= ego_front_s),
                abs(item[6]),
                hypot(item[1][0] - ego.position_xy[0], item[1][1] - ego.position_xy[1]),
                item[0],
            )
        )
        payload = np.zeros((8, 13), dtype=np.float32)
        mask = np.zeros(8, dtype=np.float32)
        for index, (_id, position, heading, dimensions, type_index, route_s, lateral) in enumerate(
            candidates[:8]
        ):
            relative = _relative_position(position, ego.position_xy, ego.heading_rad)
            payload[index] = np.asarray(
                [
                    _clip(relative[0], 50.0),
                    _clip(relative[1], 50.0),
                    *_heading_sincos(heading - ego.heading_rad),
                    _clip(dimensions[0], 10.0, lower=0.0),
                    _clip(dimensions[1], 5.0, lower=0.0),
                    *_one_hot(type_index, 5),
                    _clip(route_s - ego_front_s, 50.0),
                    _clip(lateral, 50.0),
                ],
                dtype=np.float32,
            )
            mask[index] = 1.0
        self._static_diagnostics = (len(candidates), min(len(candidates), 8))
        return payload, mask

    def _build_controls_v12(
        self,
        context: CausalSceneContext,
        ego: ActorSnapshot,
        current_s: float,
        vehicle: object,
        ego_speed_cap: float,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str | None, ApproachControl | None, str, float, bool]]:
        front_s = self._front_bumper_s(ego, current_s)
        candidates = [
            control
            for control in context.traffic_controls
            if abs(control.route_s_m - front_s) <= self.control_radius_m
            and abs(control.elevation_m - ego.position_z) <= self.vertical_tolerance_m
        ]
        # REQ-016: OBS-V1.1 SS8.3 criteria 1-2. Criterion 3 is permanently
        # dropped because it required a Rulebook latch (ADR-033, DEC-009).
        candidates.sort(
            key=lambda control: (
                ego.live_lane_id not in control.controlled_lane_ids,
                control.route_s_m < front_s,
                abs(control.route_s_m - front_s),
                control.control_group_id,
            )
        )
        signal_ids = tuple(
            physical_id
            for control in candidates
            if control.control_type == ApproachControl.SIGNAL
            for physical_id in control.physical_control_ids
        )
        visible_signals = (
            mapped_signal_visibility(
                vehicle,
                signal_ids,
                range_m=self.signal_range_m,
                horizontal_fov_deg=self.signal_fov_degrees,
                camera_height_m=self.signal_camera_height_m,
            )
            if signal_ids
            else {}
        )
        payload = np.zeros((8, 15), dtype=np.float32)
        mask = np.zeros(8, dtype=np.float32)
        active_trace: tuple[str | None, ApproachControl | None, str, float, bool] = (
            None,
            None,
            "unknown",
            0.0,
            False,
        )
        for index, control in enumerate(candidates[:8]):
            midpoint = control.control_line.centroid
            relative = _relative_position(
                (midpoint.x, midpoint.y), ego.position_xy, ego.heading_rad
            )
            coordinates = (
                list(control.control_line.coords) if hasattr(control.control_line, "coords") else []
            )
            direction = (
                atan2(
                    coordinates[-1][1] - coordinates[0][1],
                    coordinates[-1][0] - coordinates[0][0],
                )
                if len(coordinates) >= 2
                else 0.0
            )
            controls_ego = ego.live_lane_id in control.controlled_lane_ids
            route_distance = control.route_s_m - front_s
            # REQ-010/REQ-011: the observable state never decides the type, and
            # a stop control keeps its own `not-signal` slot.
            if control.control_type == ApproachControl.STOP:
                state, state_valid = "not-signal", True
                state_index: int | None = 4
            else:
                state, state_valid, state_index = "unknown", False, None
                if control.physical_control_ids and all(
                    visible_signals.get(physical_id, False)
                    for physical_id in control.physical_control_ids
                ):
                    observed = next(
                        (
                            str(
                                context.snapshot.signal_states_by_physical_id.get(
                                    physical_id, "UNKNOWN"
                                )
                            ).lower()
                            for physical_id in control.physical_control_ids
                        ),
                        "unknown",
                    )
                    state_index = {
                        "green": 0,
                        "yellow": 1,
                        "red": 2,
                        "flashing-yellow": 3,
                        "flashing_yellow": 3,
                    }.get(observed)
                    if state_index is not None:
                        state, state_valid = observed, True
            signal_state = _one_hot(state_index, 5) if state_valid else [0.0] * 5
            is_active = bool(controls_ego and route_distance >= 0.0 and active_trace[0] is None)
            if is_active:
                active_trace = (
                    control.control_group_id,
                    control.control_type,
                    state,
                    route_distance,
                    controls_ego,
                )
                self._update_yellow_memory(
                    control.control_group_id, state, route_distance, ego, ego_speed_cap
                )
            values = [
                _clip(relative[0], 80.0),
                _clip(relative[1], 80.0),
                _clip(route_distance, 80.0),
                sin(direction - ego.heading_rad),
                cos(direction - ego.heading_rad),
                *_one_hot(0 if control.control_type == ApproachControl.SIGNAL else 1, 2),
                *signal_state,
                float(controls_ego),
                float(is_active),
                float(state_valid),
            ]
            if len(values) != 15:
                raise RuntimeError(f"OBS-V1.3 control contract produced {len(values)} values")
            payload[index] = np.asarray(values, dtype=np.float32)
            mask[index] = 1.0
        if active_trace[0] is None:
            self._clear_yellow_memory()
        self._control_diagnostics = (len(candidates), min(len(candidates), 8))
        return payload, mask, active_trace

    def _update_yellow_memory(
        self,
        control_id: str,
        state: str,
        distance_m: float,
        ego: ActorSnapshot,
        ego_speed_cap: float,
    ) -> None:
        if state == "yellow":
            if self._yellow_control_id != control_id:
                self._yellow_control_id = control_id
                self._yellow_onset_distance_m = distance_m
                self._yellow_onset_speed_mps = min(hypot(*ego.velocity_xy), ego_speed_cap)
            return
        self._clear_yellow_memory()

    def _clear_yellow_memory(self) -> None:
        self._yellow_control_id = None
        self._yellow_onset_distance_m = 0.0
        self._yellow_onset_speed_mps = 0.0

    def _yellow_memory(self, ego_speed_cap: float) -> np.ndarray:
        return np.asarray(
            [
                float(self._yellow_control_id is not None),
                _clip(self._yellow_onset_distance_m, 80.0, lower=0.0),
                _clip(self._yellow_onset_speed_mps, ego_speed_cap, lower=0.0),
            ],
            dtype=np.float32,
        )

    def _nearest_dashed_feature_id(
        self, context: CausalSceneContext, ego: ActorSnapshot
    ) -> str | None:
        """Nearest intersecting dashed marking at the ego's own level."""

        candidates: list[tuple[float, str]] = []
        for feature in context.episode_cache.map_feature_catalog.values():
            if feature.feature_class != MapFeatureClass.LANE_MARKING_DASHED:
                continue
            elevation = feature.elevation_m if feature.elevation_m is not None else ego.position_z
            if abs(elevation - ego.position_z) > self.vertical_tolerance_m:
                continue
            if feature.geometry.intersects(ego.footprint):
                candidates.append(
                    (float(feature.geometry.distance(ego.footprint)), feature.feature_id)
                )
        return min(candidates)[1] if candidates else None

    def _append_timestamped_context_row(
        self,
        context: CausalSceneContext,
        ego: ActorSnapshot,
        lane_road: np.ndarray,
        control_trace: tuple[str | None, ApproachControl | None, str, float, bool],
        ego_speed_cap: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Append one actual-step trace row without reading Rulebook memory."""

        step = context.snapshot.step_index
        dashed_feature_id = self._nearest_dashed_feature_id(context, ego)
        control_id, control_type, state, control_distance, _controls_ego = control_trace
        signal_index = {"red": 0, "yellow": 1, "green": 2, "off": 3}.get(state, 4)
        # REQ-026: continuity is only meaningful against the immediately
        # preceding step.
        contiguous = self._last_context_row_step == step - 1
        row = np.asarray(
            [
                _clip(hypot(*ego.velocity_xy), ego_speed_cap, lower=0.0),
                float(lane_road[1]),
                float(lane_road[2]),
                *lane_road[3:7].tolist(),
                *lane_road[7:11].tolist(),
                float(dashed_feature_id is not None),
                float(
                    contiguous
                    and dashed_feature_id is not None
                    and dashed_feature_id == self._previous_dashed_feature_id
                ),
                float(control_id is not None),
                float(
                    contiguous
                    and control_id is not None
                    and control_id == self._previous_control_id
                ),
                *_one_hot(
                    None
                    if control_type is None
                    else (0 if control_type == ApproachControl.SIGNAL else 1),
                    2,
                ),
                *_one_hot(signal_index, 5),
                _clip(control_distance, 80.0),
            ],
            dtype=np.float32,
        )
        if row.shape != (23,):
            raise RuntimeError("OBS-V1.3 context history row must contain 23 values")
        self._previous_dashed_feature_id = dashed_feature_id
        self._previous_control_id = control_id
        self._last_context_row_step = step
        self._context_rows.append((step, row))
        history = np.zeros((self.context_history_length, 23), dtype=np.float32)
        mask = np.zeros(self.context_history_length, dtype=np.float32)
        by_step = dict(self._context_rows)
        for index, source_step in enumerate(
            range(step - self.context_history_length + 1, step + 1)
        ):
            source = by_step.get(source_step)
            if source is not None:
                history[index] = source
                mask[index] = 1.0
        return history, mask

    def _append_context_row(
        self,
        context: CausalSceneContext,
        ego: ActorSnapshot,
        lane_road: np.ndarray,
        control_trace: tuple[str | None, ApproachControl | None, str, float, bool],
        ego_speed_cap: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._append_timestamped_context_row(
            context, ego, lane_road, control_trace, ego_speed_cap
        )


__all__ = [
    "CausalSemanticBatchBuilder",
    "CausalSemanticObservationError",
    "PerceptionBoundedSemanticBatchBuilder",
    "SemanticOverflowDiagnostics",
]
