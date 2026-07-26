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
from shapely.geometry import Point
from shapely.ops import nearest_points
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
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline, RouteProjection
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    ApproachControl,
    ConflictZoneRecord,
    MapFeatureClass,
    MovementPriority,
    TrafficControlRecord,
)


class CausalSemanticObservationError(ValueError):
    """Raised when a required causal semantic input is unavailable or invalid."""


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


class CausalSemanticBatchBuilder:
    """Build one structured semantic batch from committed causal state.

    ``commit_context`` is idempotent per scenario/step.  The builder retains
    only observations it has actually received, so a newly selected actor
    cannot acquire retroactive history from the scenario's future tracks.
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
        if selected.route_polyline is not None and selected.route_polyline != self.route:
            raise CausalSemanticObservationError(
                "Semantic route differs from committed route geometry"
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
        self._record_ego_frame(vehicle, ego)
        route_projection = self.route.project(ego.position_xy, position_z=ego.position_z)
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
                _clip(route_projection.s_m, max(self.route.length_m, 1.0), lower=0.0),
            ],
            dtype=np.float32,
        )
        route, route_mask = self._build_route(ego, route_projection.s_m)
        dynamic, dynamic_mask = self._build_dynamic(context, ego, ego_speed_cap)
        static, static_mask = self._build_static(context, ego)
        self._last_diagnostics = replace(
            self._last_diagnostics,
            route_incompatible_static_features=self._route_incompatible_static_features,
        )
        lane_road = self._build_lane_road(context, ego, route_projection.s_m, vehicle)
        controls, controls_mask = self._build_controls(context, ego, route_projection.s_m)
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

    def _record_ego_frame(self, vehicle: object, ego: ActorSnapshot) -> None:
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
        projection = self.route.project(ego.position_xy, position_z=ego.position_z)
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
            visibility = mapped_signal_visibility(vehicle, str(physical_id))
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

    def _build_controls_v12(
        self, context: CausalSceneContext, ego: ActorSnapshot, current_s: float, vehicle: object
    ) -> tuple[np.ndarray, np.ndarray, tuple[TrafficControlRecord, str, bool] | None]:
        candidates = self._control_candidates(context, ego, current_s)
        payload = np.zeros((8, 17), dtype=np.float32)
        mask = np.zeros(8, dtype=np.float32)
        active: tuple[TrafficControlRecord, str, bool] | None = None
        for index, control in enumerate(candidates[:8]):
            midpoint = control.control_line.centroid
            line_coords = list(control.control_line.coords) if hasattr(control.control_line, "coords") else []
            direction = (
                atan2(line_coords[-1][1] - line_coords[0][1], line_coords[-1][0] - line_coords[0][0])
                if len(line_coords) >= 2
                else 0.0
            )
            state, valid = self._visible_signal_state(control, context, vehicle)
            state_index = {"green": 0, "yellow": 1, "red": 2, "flashing-yellow": 3}.get(state)
            signal_state = _one_hot(state_index if valid and state_index is not None else 4, 5)
            governs = ego.live_lane_id in control.controlled_lane_ids
            if active is None and governs and control.route_s_m >= current_s:
                active = (control, state, valid)
            relative = _relative_position((midpoint.x, midpoint.y), ego.position_xy, ego.heading_rad)
            payload[index] = np.asarray(
                [
                    _clip(relative[0], 80.0),
                    _clip(relative[1], 80.0),
                    _clip(control.route_s_m - current_s, 80.0),
                    sin(direction - ego.heading_rad),
                    cos(direction - ego.heading_rad),
                    *_one_hot(0 if control.control_type == ApproachControl.SIGNAL else 1, 2),
                    *signal_state,
                    float(governs),
                    float(active is not None and active[0].control_group_id == control.control_group_id),
                    float(valid),
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            mask[index] = 1.0
        return payload, mask, active

    def _active_dashed_key(self, context: CausalSceneContext, ego: ActorSnapshot) -> str | None:
        candidates: list[tuple[float, str]] = []
        for feature in context.episode_cache.map_feature_catalog.values():
            if feature.feature_class != MapFeatureClass.LANE_MARKING_DASHED:
                continue
            elevation = feature.elevation_m if feature.elevation_m is not None else ego.position_z
            if abs(elevation - ego.position_z) > self.vertical_tolerance_m:
                continue
            if feature.geometry.intersects(ego.footprint):
                candidates.append((float(feature.geometry.distance(ego.footprint)), feature.feature_id))
        return min(candidates)[1] if candidates else None

    def _build_compliance_v12(
        self,
        *,
        context: CausalSceneContext,
        ego: ActorSnapshot,
        ego_speed_cap: float,
        lane_road: np.ndarray,
        active_control: tuple[TrafficControlRecord, str, bool] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dashed_key = self._active_dashed_key(context, ego)
        control, state, state_valid = active_control if active_control is not None else (None, "unknown", False)
        control_key = control.control_group_id if control is not None else None
        signal_index = {"red": 0, "yellow": 1, "green": 2, "off": 3}.get(
            state if state_valid else "unknown", 4
        )
        control_distance = (
            _clip(control.route_s_m - self._route_s(ego, ego), 80.0) if control is not None else 0.0
        )
        row = np.asarray(
            [
                _clip(hypot(*ego.velocity_xy), ego_speed_cap, lower=0.0),
                float(lane_road[1]), float(lane_road[2]),
                *lane_road[3:7].tolist(), *lane_road[7:11].tolist(),
                float(dashed_key is not None),
                float(dashed_key is not None and dashed_key == self._previous_dashed_key),
                float(control is not None),
                float(control_key is not None and control_key == self._previous_control_key),
                *_one_hot(0 if control and control.control_type == ApproachControl.SIGNAL else 1 if control else None, 2),
                *_one_hot(signal_index, 5),
                control_distance,
                float(control is not None and ego.live_lane_id in control.controlled_lane_ids),
            ],
            dtype=np.float32,
        )
        if row.shape != (24,):
            raise RuntimeError("OBS-V1.2 compliance history row must contain 24 values")
        self._previous_dashed_key = dashed_key
        self._previous_control_key = control_key
        if control is not None and state == "yellow" and state_valid:
            if self._yellow_group_id != control_key:
                self._yellow_group_id = control_key
                self._yellow_onset_distance_m = control.route_s_m - self._route_s(ego, ego)
                self._yellow_onset_speed_mps = hypot(*ego.velocity_xy)
        else:
            self._yellow_group_id = None
            self._yellow_onset_distance_m = 0.0
            self._yellow_onset_speed_mps = 0.0
        step = context.snapshot.step_index
        self._compliance_history.append(row)
        self._compliance_steps.append(step)
        history = np.zeros((21, 24), dtype=np.float32)
        mask = np.zeros(21, dtype=np.float32)
        expected_steps = range(step - 20, step + 1)
        by_step = dict(zip(self._compliance_steps, self._compliance_history))
        for index, source_step in enumerate(expected_steps):
            source = by_step.get(source_step)
            if source is not None:
                history[index] = source
                mask[index] = 1.0
        yellow_memory = np.asarray(
            [
                float(self._yellow_group_id is not None),
                _clip(self._yellow_onset_distance_m, 80.0),
                _clip(self._yellow_onset_speed_mps, ego_speed_cap, lower=0.0),
            ],
            dtype=np.float32,
        )
        return history, mask, yellow_memory

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

    compliance_history_length = 21

    def __init__(self, *, brake_mps2: float | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if brake_mps2 is not None and brake_mps2 <= 0.0:
            raise ValueError("OBS-V1.2 brake_mps2 must be positive when available")
        self.brake_mps2 = brake_mps2
        self._v12_actor_cache: dict[str, deque[tuple[int, ActorSnapshot]]] = {}
        self._visible_actor_ids: frozenset[str] = frozenset()
        self._compliance_rows: deque[tuple[int, np.ndarray]] = deque(
            maxlen=self.compliance_history_length
        )
        self._yellow_control_id: str | None = None
        self._yellow_onset_distance_m = 0.0
        self._yellow_onset_speed_mps = 0.0
        self._previous_dashed_feature_id: str | None = None
        self._previous_control_id: str | None = None
        self._last_compliance_step: int | None = None

    def reset(self) -> None:
        super().reset()
        self._v12_actor_cache.clear()
        self._visible_actor_ids = frozenset()
        self._compliance_rows.clear()
        self._yellow_control_id = None
        self._yellow_onset_distance_m = 0.0
        self._yellow_onset_speed_mps = 0.0
        self._previous_dashed_feature_id = None
        self._previous_control_id = None
        self._last_compliance_step = None

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
        if selected.route_polyline is not None and selected.route_polyline != self.route:
            raise CausalSemanticObservationError("Semantic route differs from committed route geometry")
        self.commit_context(selected, vehicle)
        return selected

    def build(
        self, vehicle: object, context: CausalSceneContext | None = None
    ) -> SemanticObservationBatchV12:
        context = self._context_v12(context, vehicle)
        ego = context.snapshot.ego
        self._record_ego_frame(vehicle, ego)
        route_projection = self.route.project(ego.position_xy, position_z=ego.position_z)
        ego_speed_cap = self.ego_speed_cap_mps or ego.configured_speed_cap_mps
        if ego_speed_cap is None or ego_speed_cap <= 0.0:
            raise CausalSemanticObservationError("Ego speed cap is required for semantic normalization")

        ego_history, ego_history_mask = self._build_ego_history(ego, ego_speed_cap)
        ego_current = np.asarray(
            [
                _clip(_dimensions(ego.footprint)[0], 10.0, lower=0.0),
                _clip(_dimensions(ego.footprint)[1], 5.0, lower=0.0),
                _clip(route_projection.s_m, max(self.route.length_m, 1.0), lower=0.0),
            ],
            dtype=np.float32,
        )
        route, route_mask = self._build_route_v12(ego, route_projection.s_m)
        dynamic, dynamic_mask = self._build_dynamic_v12(context, ego, ego_speed_cap)
        static, static_mask = self._build_static_v12(context, ego)
        lane_road = self._build_lane_road(context, ego, route_projection.s_m, vehicle)
        controls, controls_mask, control_trace = self._build_controls_v12(
            context, ego, route_projection.s_m, vehicle, ego_speed_cap
        )
        interactions, interactions_mask = self._build_interactions(context, ego)
        compliance_history, compliance_history_mask = self._append_compliance_row(
            context, ego, lane_road, control_trace, ego_speed_cap
        )
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
            compliance_history=compliance_history,
            compliance_history_mask=compliance_history_mask,
            yellow_onset_memory=self._yellow_memory(ego_speed_cap),
        )

    def _dynamic_candidates(
        self, context: CausalSceneContext, ego: ActorSnapshot
    ) -> list[ActorSnapshot]:
        return [
            actor
            for actor in super()._dynamic_candidates(context, ego)
            if actor.actor_id in self._visible_actor_ids
        ]

    def _build_route_v12(self, ego: ActorSnapshot, current_s: float) -> tuple[np.ndarray, np.ndarray]:
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
            if actor_id not in current and step - self._slot_last_seen.get(slot, step) >= self.history_length:
                self._slot_actor.pop(slot, None)
                self._slot_last_seen.pop(slot, None)
        assignments: dict[int, ActorSnapshot] = {}
        used: set[int] = set()
        for actor, is_conflict in selected:
            previous = next((slot for slot, actor_id in self._slot_actor.items() if actor_id == actor.actor_id), None)
            if previous is not None and ((is_conflict and previous < 8) or (not is_conflict and previous >= 8)):
                assignments[previous] = actor
                used.add(previous)
        for actor, is_conflict in selected:
            if actor in assignments.values():
                continue
            preferred = tuple(range(0, 8)) if is_conflict else tuple(range(8, 16))
            fallback = tuple(range(8, 16)) if is_conflict else tuple(range(0, 8))
            slot = next((candidate for candidate in (*preferred, *fallback) if candidate not in used), None)
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
        selected = conflicts[:8] + [actor for actor in ordered if actor not in conflicts[:8]][: 16 - len(conflicts[:8])]
        slots = self._assign_slots_v12(
            [(actor, actor.actor_id in conflict_ids) for actor in selected], context.snapshot.step_index
        )
        payload = np.zeros((16, 5, 22), dtype=np.float32)
        mask = np.zeros((16, 5), dtype=np.float32)
        for slot, actor in slots.items():
            samples = {step: snapshot for step, snapshot in self._v12_actor_cache.get(actor.actor_id, ())}
            for history_index, sample_step in enumerate(
                range(context.snapshot.step_index - self.history_length + 1, context.snapshot.step_index + 1)
            ):
                snapshot = samples.get(sample_step)
                if snapshot is None:
                    continue
                payload[slot, history_index] = self._dynamic_features(
                    snapshot, ego, ego_speed_cap, context
                )
                mask[slot, history_index] = 1.0
        self._last_diagnostics = SemanticOverflowDiagnostics(
            {"dynamic": len(candidates)},
            {"dynamic": len(selected)},
            {"dynamic": max(0, len(candidates) - len(selected))},
            len(conflicts[8:]),
            {"dynamic": self._dynamic_key(selected[-1], ego, conflict_ids) if selected else ()},
            {"dynamic": self._dynamic_key(ordered[len(selected)], ego, conflict_ids) if len(ordered) > len(selected) else ()},
            self._route_incompatible_static_features,
        )
        return payload, mask

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

    def _build_static_v12(
        self, context: CausalSceneContext, ego: ActorSnapshot
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates: list[tuple[str, tuple[float, float], float, tuple[float, float], int, float, float]] = []
        for actor in context.snapshot.actors:
            if actor.actor_class != ActorClass.STATIC_COLLIDABLE or actor.actor_id not in self._visible_actor_ids:
                continue
            distance = hypot(actor.position_xy[0] - ego.position_xy[0], actor.position_xy[1] - ego.position_xy[1])
            if distance <= self.static_radius_m and abs(actor.position_z - ego.position_z) <= self.vertical_tolerance_m:
                candidates.append(
                    (
                        actor.actor_id,
                        actor.position_xy,
                        actor.heading_rad,
                        _dimensions(actor.footprint),
                        4,  # generic: the live source exposes no confirmed finer taxonomy.
                        self._route_s(actor, ego),
                        self._route_lateral(actor),
                    )
                )
        ego_point = Point(ego.position_xy)
        for feature_id, feature in context.episode_cache.map_feature_catalog.items():
            if feature.feature_class not in {MapFeatureClass.OTHER_NON_DRIVABLE, MapFeatureClass.ROAD_BOUNDARY}:
                continue
            _source, closest = nearest_points(feature.geometry, ego_point)
            position = (float(closest.x), float(closest.y))
            elevation = feature.elevation_at_xy(position)
            if elevation is not None and abs(elevation - ego.position_z) > self.vertical_tolerance_m:
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
                    f"scenario_id={context.snapshot.scenario_id}; step={context.snapshot.step_index}; "
                    f"feature_id={feature_id}; feature_class={feature.feature_class.value}; "
                    f"feature_position_xy={position}; feature_elevation_m={elevation}; "
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
                    str(feature_id),
                    position,
                    0.0,
                    (max(bounds[2] - bounds[0], 0.1), max(bounds[3] - bounds[1], 0.1)),
                    4,
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
        for index, (_id, position, heading, dimensions, type_index, route_s, lateral) in enumerate(candidates[:8]):
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

    def _build_controls_v12(
        self,
        context: CausalSceneContext,
        ego: ActorSnapshot,
        current_s: float,
        vehicle: object,
        ego_speed_cap: float,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str | None, str, float, bool]]:
        candidates = [
            control
            for control in context.traffic_controls
            if abs(control.route_s_m - current_s) <= self.control_radius_m
            and abs(control.elevation_m - ego.position_z) <= self.vertical_tolerance_m
        ]
        candidates.sort(
            key=lambda control: (
                control.route_s_m < current_s,
                abs(control.route_s_m - current_s),
                control.control_group_id,
            )
        )
        signal_ids = tuple(
            physical_id
            for control in candidates
            if control.control_type == ApproachControl.SIGNAL
            for physical_id in control.physical_control_ids
        )
        visible_signals = mapped_signal_visibility(vehicle, signal_ids) if signal_ids else {}
        payload = np.zeros((8, 17), dtype=np.float32)
        mask = np.zeros(8, dtype=np.float32)
        active_trace: tuple[str | None, str, float, bool] = (None, "unknown", 0.0, False)
        for index, control in enumerate(candidates[:8]):
            midpoint = control.control_line.centroid
            relative = _relative_position((midpoint.x, midpoint.y), ego.position_xy, ego.heading_rad)
            coordinates = list(control.control_line.coords) if hasattr(control.control_line, "coords") else []
            direction = (
                atan2(coordinates[-1][1] - coordinates[0][1], coordinates[-1][0] - coordinates[0][0])
                if len(coordinates) >= 2
                else 0.0
            )
            controls_ego = ego.live_lane_id in control.controlled_lane_ids
            route_distance = control.route_s_m - current_s
            state = "not-signal"
            state_valid = control.control_type == ApproachControl.STOP
            if control.control_type == ApproachControl.SIGNAL:
                state_valid = bool(control.physical_control_ids) and all(
                    visible_signals.get(physical_id, False) for physical_id in control.physical_control_ids
                )
                if state_valid:
                    state = next(
                        (
                            str(context.snapshot.signal_states_by_physical_id.get(physical_id, "UNKNOWN")).lower()
                            for physical_id in control.physical_control_ids
                        ),
                        "unknown",
                    )
                    state_valid = state in {"green", "yellow", "red", "flashing_yellow", "flashing-yellow"}
            state_index = {"green": 0, "yellow": 1, "red": 2, "flashing-yellow": 3, "flashing_yellow": 3}.get(state)
            signal_state = _one_hot(state_index, 5) if state_valid and state_index is not None else [0.0] * 5
            is_active = bool(controls_ego and route_distance >= 0.0 and active_trace[0] is None)
            if is_active:
                active_trace = (control.control_group_id, state, route_distance, controls_ego)
                self._update_yellow_memory(control.control_group_id, state, route_distance, ego, ego_speed_cap)
            payload[index] = np.asarray(
                [
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
                    _clip(self._yellow_onset_distance_m, 80.0, lower=0.0)
                    if self._yellow_control_id == control.control_group_id
                    else 0.0,
                    _clip(self._yellow_required_stop_distance(), 80.0, lower=0.0)
                    if self._yellow_control_id == control.control_group_id
                    else 0.0,
                ],
                dtype=np.float32,
            )
            mask[index] = 1.0
        if active_trace[0] is None:
            self._clear_yellow_memory()
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

    def _yellow_required_stop_distance(self) -> float:
        if self._yellow_control_id is None or self.brake_mps2 is None:
            return 0.0
        return self._yellow_onset_speed_mps * self.dt + self._yellow_onset_speed_mps**2 / (2.0 * self.brake_mps2)

    def _yellow_memory(self, ego_speed_cap: float) -> np.ndarray:
        return np.asarray(
            [
                float(self._yellow_control_id is not None),
                _clip(self._yellow_onset_distance_m, 80.0, lower=0.0),
                _clip(self._yellow_onset_speed_mps, ego_speed_cap, lower=0.0),
            ],
            dtype=np.float32,
        )

    def _append_timestamped_compliance_row(
        self,
        context: CausalSceneContext,
        ego: ActorSnapshot,
        lane_road: np.ndarray,
        control_trace: tuple[str | None, str, float, bool],
        ego_speed_cap: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Append one actual-step trace row without reading Rulebook memory."""

        dashed_feature_id = next(
            (
                feature.feature_id
                for feature in context.episode_cache.map_feature_catalog.values()
                if feature.feature_class == MapFeatureClass.LANE_MARKING_DASHED
                and feature.geometry.intersects(ego.footprint)
            ),
            None,
        )
        control_id, state, control_distance, controls_ego = control_trace
        signal_index = {"red": 0, "yellow": 1, "green": 2, "off": 3}.get(state, 4)
        active_control = control_id is not None
        row = np.asarray(
            [
                _clip(hypot(*ego.velocity_xy), ego_speed_cap, lower=0.0),
                float(lane_road[1]),
                float(lane_road[2]),
                *lane_road[3:7].tolist(),
                *lane_road[7:11].tolist(),
                float(dashed_feature_id is not None),
                float(
                    dashed_feature_id is not None
                    and dashed_feature_id == self._previous_dashed_feature_id
                ),
                float(active_control),
                float(control_id is not None and control_id == self._previous_control_id),
                *_one_hot(0 if state != "not-signal" and active_control else 1 if active_control else None, 2),
                *_one_hot(signal_index, 5),
                _clip(control_distance, 80.0),
                float(controls_ego),
            ],
            dtype=np.float32,
        )
        if row.shape != (24,):
            raise RuntimeError("OBS-V1.2 compliance history row must contain 24 values")
        self._previous_dashed_feature_id = dashed_feature_id
        self._previous_control_id = control_id
        self._compliance_rows.append((context.snapshot.step_index, row))
        history = np.zeros((self.compliance_history_length, 24), dtype=np.float32)
        mask = np.zeros(self.compliance_history_length, dtype=np.float32)
        by_step = dict(self._compliance_rows)
        for index, source_step in enumerate(
            range(
                context.snapshot.step_index - self.compliance_history_length + 1,
                context.snapshot.step_index + 1,
            )
        ):
            source = by_step.get(source_step)
            if source is not None:
                history[index] = source
                mask[index] = 1.0
        return history, mask

    def _append_compliance_row(
        self,
        context: CausalSceneContext,
        ego: ActorSnapshot,
        lane_road: np.ndarray,
        control_trace: tuple[str | None, str, float, bool],
        ego_speed_cap: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._append_timestamped_compliance_row(
            context, ego, lane_road, control_trace, ego_speed_cap
        )

    def _active_dashed_feature_id(
        self, context: CausalSceneContext, ego: ActorSnapshot
    ) -> str | None:
        candidates: list[tuple[float, str]] = []
        for feature in context.episode_cache.map_feature_catalog.values():
            if feature.feature_class != MapFeatureClass.LANE_MARKING_DASHED:
                continue
            elevation = feature.elevation_m if feature.elevation_m is not None else ego.position_z
            if abs(elevation - ego.position_z) > self.vertical_tolerance_m:
                continue
            if feature.geometry.intersects(ego.footprint):
                candidates.append((feature.geometry.distance(ego.footprint), feature.feature_id))
        return min(candidates)[1] if candidates else None


__all__ = [
    "CausalSemanticBatchBuilder",
    "CausalSemanticObservationError",
    "PerceptionBoundedSemanticBatchBuilder",
    "SemanticOverflowDiagnostics",
]
