from __future__ import annotations

import math
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from metadrive.obs.observation_base import BaseObservation
from metadrive.type import MetaDriveType


class SemanticStateObservation(BaseObservation):
    """Object-centric semantic observation with internal ID-based temporal tracking."""

    _external_config: dict[str, Any] = {}

    EGO_DIM = 13
    ROUTE_DIM = 5
    DYNAMIC_DIM = 24
    STATIC_DIM = 14
    CONTROL_DIM = 16
    LANE_DIM = 12

    DYN_IDX_SELECTED_CONFLICT = 21
    DYN_IDX_SELECTED_CONTEXT = 22
    DYN_IDX_TRACK_AGE = 23

    @classmethod
    def set_external_config(cls, config: dict[str, Any] | None) -> None:
        cls._external_config = dict(config or {})

    def __init__(self, config: dict[str, Any]):
        self._obs_cfg = dict(self.__class__._external_config)
        self._obs_cfg.update(self._read_semantic_cfg(config))

        self._history_len = self._cfg_int("history_length", 5, minimum=1)
        self._max_route_points = self._cfg_int("max_route_points", 5, minimum=1)
        self._max_dynamic_objects = self._cfg_int("max_dynamic_objects", 8, minimum=0)
        self._max_static_objects = self._cfg_int("max_static_objects", 8, minimum=0)
        self._max_control_objects = self._cfg_int("max_control_objects", 8, minimum=0)

        self._dynamic_radius = self._cfg_float("dynamic_radius_m", 50.0, minimum=1e-6)
        self._static_radius = self._cfg_float("static_radius_m", 50.0, minimum=1e-6)
        self._control_radius = self._cfg_float("control_radius_m", 80.0, minimum=1e-6)

        self._speed_norm_kmh = self._cfg_float("speed_norm_kmh", 120.0, minimum=1.0)
        self._speed_norm_mps = max(self._speed_norm_kmh / 3.6, 1e-6)
        self._accel_norm_mps2 = self._cfg_float("accel_norm_mps2", 8.0, minimum=1e-6)
        self._yaw_rate_norm_rad_s = self._cfg_float("yaw_rate_norm_rad_s", 2.0, minimum=1e-6)
        self._length_norm_m = self._cfg_float("length_norm_m", 8.0, minimum=1e-6)
        self._width_norm_m = self._cfg_float("width_norm_m", 4.0, minimum=1e-6)
        self._lane_width_norm_m = self._cfg_float("lane_width_norm_m", 6.0, minimum=1e-6)
        self._ttc_horizon_s = self._cfg_float("ttc_horizon_s", 5.0, minimum=1e-6)
        self._cpa_time_norm_s = self._cfg_float("cpa_time_norm_s", self._ttc_horizon_s, minimum=1e-6)

        default_conflict_slots = self._max_dynamic_objects // 2
        default_context_slots = self._max_dynamic_objects - default_conflict_slots
        self._dynamic_conflict_slots = self._cfg_int(
            "dynamic_conflict_slots",
            default_conflict_slots,
            minimum=0,
        )
        self._dynamic_context_slots = self._cfg_int(
            "dynamic_context_slots",
            default_context_slots,
            minimum=0,
        )
        self._normalize_dynamic_quota()

        physics_step = self._safe_float(config.get("physics_world_step_size"), default=0.02)
        decision_repeat = self._safe_float(config.get("decision_repeat"), default=5.0)
        self._dt = max(physics_step * decision_repeat, 1e-3)

        self._dynamic_history: dict[str, deque[np.ndarray | None]] = {}
        self._dynamic_last_seen_step: dict[str, int] = {}
        self._ego_history: deque[np.ndarray] = deque(maxlen=self._history_len)
        self._prev_ego_speed_mps: float | None = None
        self._prev_ego_heading: float | None = None
        self._step_count = 0

        self._obs_dim = self._build_obs_dim()
        self._observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        super().__init__(config)

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self, env, vehicle=None):
        del env, vehicle
        self._dynamic_history.clear()
        self._dynamic_last_seen_step.clear()
        self._ego_history.clear()
        self._prev_ego_speed_mps = None
        self._prev_ego_heading = None
        self._step_count = 0

    def observe(self, vehicle):
        self._step_count += 1

        ego_features = self._extract_ego_features(vehicle)
        self._ego_history.append(ego_features)

        dynamic_candidates = self._collect_dynamic_candidates(vehicle)
        selected_dynamic = self._select_dynamic_candidates(dynamic_candidates)
        static_candidates = self._collect_static_candidates(vehicle)
        control_candidates = self._collect_control_candidates(vehicle)

        self._advance_dynamic_tracks()
        for entry in dynamic_candidates:
            track_id = entry["id"]
            features = np.asarray(entry["features"], dtype=np.float32)

            history = self._dynamic_history.get(track_id)
            if history is None:
                history = deque([None] * self._history_len, maxlen=self._history_len)
                self._dynamic_history[track_id] = history
            history[-1] = features

            age = self._track_age(history)
            features = features.copy()
            features[self.DYN_IDX_TRACK_AGE] = np.float32(age / float(self._history_len))
            history[-1] = features
            self._dynamic_last_seen_step[track_id] = self._step_count

        self._prune_stale_tracks()

        route_tokens, route_mask = self._extract_route_tokens(vehicle)
        dynamic_tokens, dynamic_mask = self._pack_dynamic_tokens(selected_dynamic)
        static_tokens, static_mask = self._pack_static_tokens(static_candidates)
        control_tokens, control_mask = self._pack_control_tokens(control_candidates)
        lane_state = self._extract_lane_state(vehicle)

        ego_history = self._stack_ego_history()

        flattened = np.concatenate(
            [
                ego_history.reshape(-1),
                route_tokens.reshape(-1),
                route_mask.reshape(-1),
                dynamic_tokens.reshape(-1),
                dynamic_mask.reshape(-1),
                static_tokens.reshape(-1),
                static_mask.reshape(-1),
                control_tokens.reshape(-1),
                control_mask.reshape(-1),
                lane_state.reshape(-1),
            ]
        ).astype(np.float32)

        self.current_observation = np.clip(flattened, -1.0, 1.0)
        return self.current_observation

    def _build_obs_dim(self) -> int:
        return (
            self._history_len * self.EGO_DIM
            + self._max_route_points * self.ROUTE_DIM
            + self._max_route_points
            + self._max_dynamic_objects * self._history_len * self.DYNAMIC_DIM
            + self._max_dynamic_objects * self._history_len
            + self._max_static_objects * self.STATIC_DIM
            + self._max_static_objects
            + self._max_control_objects * self.CONTROL_DIM
            + self._max_control_objects
            + self.LANE_DIM
        )

    def _extract_ego_features(self, vehicle) -> np.ndarray:
        speed_mps = self._safe_float(getattr(vehicle, "speed", None), default=0.0)
        speed_norm = self._sym_norm(speed_mps, self._speed_norm_mps)

        if self._prev_ego_speed_mps is None:
            accel_mps2 = 0.0
        else:
            accel_mps2 = (speed_mps - self._prev_ego_speed_mps) / self._dt
        accel_norm = self._sym_norm(accel_mps2, self._accel_norm_mps2)
        self._prev_ego_speed_mps = speed_mps

        heading = self._safe_float(getattr(vehicle, "heading_theta", None), default=0.0)
        heading_sin = math.sin(heading)
        heading_cos = math.cos(heading)
        if self._prev_ego_heading is None:
            yaw_rate = 0.0
        else:
            yaw_rate = self._wrap_to_pi(heading - self._prev_ego_heading) / self._dt
        yaw_rate_norm = self._sym_norm(yaw_rate, self._yaw_rate_norm_rad_s)
        self._prev_ego_heading = heading

        steering = self._safe_float(getattr(vehicle, "steering", None), default=0.0)
        max_steering = self._safe_float(getattr(vehicle, "MAX_STEERING", None), default=1.0)
        steering_norm = self._sym_norm(steering, max(max_steering, 1e-6))

        last_action = list(getattr(vehicle, "last_current_action", []))
        prev_action = last_action[-2] if len(last_action) >= 2 else (0.0, 0.0)
        curr_action = last_action[-1] if len(last_action) >= 1 else (0.0, 0.0)
        last_steering_norm = self._sym_norm(float(prev_action[0]), 1.0)
        accel_cmd_norm = self._sym_norm(float(curr_action[1]), 1.0)

        lane = getattr(vehicle, "lane", None)
        lane_width = self._lane_width(lane)
        lane_offset_norm = 0.0
        heading_error_norm = 0.0
        if lane is not None:
            try:
                _, lateral = lane.local_coordinates(np.asarray(vehicle.position))
                lane_offset_norm = self._sym_norm(float(lateral), max(lane_width / 2.0, 1e-6))
            except Exception:
                lane_offset_norm = 0.0
            try:
                heading_error_norm = float(np.clip(vehicle.heading_diff(lane), -1.0, 1.0))
            except Exception:
                heading_error_norm = 0.0

        left_dist = self._safe_float(getattr(vehicle, "dist_to_left_side", None), default=0.0)
        right_dist = self._safe_float(getattr(vehicle, "dist_to_right_side", None), default=0.0)
        lane_range = self._lane_total_width(vehicle, lane_width)
        left_dist_norm = self._sym_norm(left_dist, lane_range)
        right_dist_norm = self._sym_norm(right_dist, lane_range)

        route_completion = self._route_completion(vehicle)

        return np.asarray(
            [
                speed_norm,
                accel_norm,
                heading_sin,
                heading_cos,
                yaw_rate_norm,
                steering_norm,
                last_steering_norm,
                accel_cmd_norm,
                lane_offset_norm,
                heading_error_norm,
                left_dist_norm,
                right_dist_norm,
                route_completion,
            ],
            dtype=np.float32,
        )

    def _extract_route_tokens(self, vehicle) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.zeros((self._max_route_points, self.ROUTE_DIM), dtype=np.float32)
        mask = np.zeros((self._max_route_points,), dtype=np.float32)

        navigation = getattr(vehicle, "navigation", None)
        if navigation is None:
            return tokens, mask
        get_navi_info = getattr(navigation, "get_navi_info", None)
        if not callable(get_navi_info):
            return tokens, mask

        navi_info = np.asarray(get_navi_info(), dtype=np.float32).reshape(-1)
        if navi_info.size == 0:
            return tokens, mask

        chunk_size = self.ROUTE_DIM
        n_chunks = min(self._max_route_points, navi_info.size // chunk_size)
        for idx in range(n_chunks):
            chunk = navi_info[idx * chunk_size : (idx + 1) * chunk_size]
            tokens[idx] = np.clip(chunk, -1.0, 1.0)
            mask[idx] = 1.0
        return tokens, mask

    def _extract_lane_state(self, vehicle) -> np.ndarray:
        lane = getattr(vehicle, "lane", None)
        navigation = getattr(vehicle, "navigation", None)
        if lane is None:
            return np.zeros((self.LANE_DIM,), dtype=np.float32)

        lane_width = self._lane_width(lane)
        lane_width_norm = self._sym_norm(lane_width, self._lane_width_norm_m)

        lane_offset_norm = 0.0
        heading_error_norm = 0.0
        curvature_norm = 0.0
        try:
            _, lateral = lane.local_coordinates(np.asarray(vehicle.position))
            lane_offset_norm = self._sym_norm(float(lateral), max(lane_width / 2.0, 1e-6))
        except Exception:
            lane_offset_norm = 0.0
        try:
            heading_error_norm = float(np.clip(vehicle.heading_diff(lane), -1.0, 1.0))
        except Exception:
            heading_error_norm = 0.0

        radius = self._safe_float(getattr(lane, "radius", None), default=0.0)
        if radius > 1e-6:
            sign = -1.0 if bool(getattr(lane, "is_clockwise", lambda: False)()) else 1.0
            curvature_norm = float(np.clip(sign * 20.0 / radius, -1.0, 1.0))

        left_dist = self._safe_float(getattr(vehicle, "dist_to_left_side", None), default=0.0)
        right_dist = self._safe_float(getattr(vehicle, "dist_to_right_side", None), default=0.0)
        lane_range = self._lane_total_width(vehicle, lane_width)
        left_dist_norm = self._sym_norm(left_dist, lane_range)
        right_dist_norm = self._sym_norm(right_dist, lane_range)

        left_solid = 0.0
        left_broken = 0.0
        right_solid = 0.0
        right_broken = 0.0
        line_types = getattr(lane, "line_types", None)
        if isinstance(line_types, (list, tuple)) and len(line_types) >= 2:
            left_type = line_types[0]
            right_type = line_types[1]
            left_solid = 1.0 if MetaDriveType.is_solid_line(left_type) else 0.0
            left_broken = 1.0 if MetaDriveType.is_broken_line(left_type) else 0.0
            right_solid = 1.0 if MetaDriveType.is_solid_line(right_type) else 0.0
            right_broken = 1.0 if MetaDriveType.is_broken_line(right_type) else 0.0

        lane_idx = self._lane_index_of(vehicle)
        lane_id = lane_idx[2] if lane_idx is not None else 0
        lane_count = self._current_lane_count(navigation)
        left_available = 1.0 if lane_count > 1 and lane_id < (lane_count - 1) else 0.0
        right_available = 1.0 if lane_count > 1 and lane_id > 0 else 0.0

        return np.asarray(
            [
                lane_width_norm,
                lane_offset_norm,
                heading_error_norm,
                left_dist_norm,
                right_dist_norm,
                left_solid,
                left_broken,
                right_solid,
                right_broken,
                left_available,
                right_available,
                curvature_norm,
            ],
            dtype=np.float32,
        )

    def _collect_dynamic_candidates(self, vehicle) -> list[dict[str, Any]]:
        ego_vel = np.asarray(getattr(vehicle, "velocity", (0.0, 0.0)), dtype=np.float32)
        ego_lane = self._lane_index_of(vehicle)
        route_pairs = self._route_pairs(vehicle)
        lane_width = self._lane_width(getattr(vehicle, "lane", None))

        candidates: list[dict[str, Any]] = []
        for obj_id, obj in self._iter_world_objects():
            if str(obj_id) == str(getattr(vehicle, "id", None)):
                continue

            obj_type = self._object_type(obj)
            if obj_type not in {
                MetaDriveType.VEHICLE,
                MetaDriveType.PEDESTRIAN,
                MetaDriveType.CYCLIST,
                MetaDriveType.OTHER,
                MetaDriveType.UNSET,
            }:
                continue

            pos = self._object_position(obj)
            if pos is None:
                continue
            rel_local = self._relative_local(vehicle, pos)
            dist = float(np.linalg.norm(rel_local))
            if dist > self._dynamic_radius:
                continue

            obj_vel = np.asarray(getattr(obj, "velocity", (0.0, 0.0)), dtype=np.float32)
            rel_vel_world = obj_vel - ego_vel
            rel_vel_local = self._vector_to_local(vehicle, rel_vel_world)

            heading = self._safe_float(getattr(obj, "heading_theta", None), default=0.0)
            ego_heading = self._safe_float(getattr(vehicle, "heading_theta", None), default=0.0)
            rel_heading = self._wrap_to_pi(heading - ego_heading)

            speed_kmh = self._safe_float(getattr(obj, "speed_km_h", None), default=0.0)
            length = self._safe_float(getattr(obj, "LENGTH", None), default=0.0)
            width = self._safe_float(getattr(obj, "WIDTH", None), default=0.0)

            lane_idx = self._lane_index_of(obj)
            same_lane = (
                1.0
                if (lane_idx is not None and ego_lane is not None and lane_idx == ego_lane)
                else 0.0
            )
            adjacent_lane = (
                1.0
                if (
                    lane_idx is not None
                    and ego_lane is not None
                    and lane_idx[:2] == ego_lane[:2]
                    and abs(int(lane_idx[2]) - int(ego_lane[2])) == 1
                )
                else 0.0
            )
            ahead = 1.0 if rel_local[0] > 0.0 else 0.0
            on_route = 1.0 if self._is_on_route(lane_idx, route_pairs) else 0.0
            intersects_path = 1.0 if (rel_local[0] > 0.0 and abs(rel_local[1]) <= lane_width) else 0.0
            ahead_or_crossing = max(ahead, intersects_path)
            same_or_adjacent = max(same_lane, adjacent_lane)

            is_approaching, t_cpa_sort, d_cpa_sort = self._closest_approach_metrics(
                rel_pos=rel_local,
                rel_vel=rel_vel_local,
                distance=dist,
            )
            t_cpa_norm = self._pos_norm(t_cpa_sort, self._cpa_time_norm_s)
            d_cpa_norm = self._pos_norm(d_cpa_sort, self._dynamic_radius)
            dist_norm = self._pos_norm(dist, self._dynamic_radius)

            type_vehicle = 1.0 if obj_type == MetaDriveType.VEHICLE else 0.0
            type_ped = 1.0 if obj_type == MetaDriveType.PEDESTRIAN else 0.0
            type_cyc = 1.0 if obj_type == MetaDriveType.CYCLIST else 0.0
            type_other = 1.0 if obj_type in {MetaDriveType.OTHER, MetaDriveType.UNSET} else 0.0

            features = np.asarray(
                [
                    self._sym_norm(rel_local[0], self._dynamic_radius),
                    self._sym_norm(rel_local[1], self._dynamic_radius),
                    self._sym_norm(rel_vel_local[0], self._speed_norm_mps),
                    self._sym_norm(rel_vel_local[1], self._speed_norm_mps),
                    math.sin(rel_heading),
                    math.cos(rel_heading),
                    self._sym_norm(speed_kmh, self._speed_norm_kmh),
                    self._sym_norm(length, self._length_norm_m),
                    self._sym_norm(width, self._width_norm_m),
                    type_vehicle,
                    type_ped,
                    type_cyc,
                    type_other,
                    same_lane,
                    adjacent_lane,
                    ahead,
                    on_route,
                    is_approaching,
                    t_cpa_norm,
                    d_cpa_norm,
                    dist_norm,
                    0.0,  # selected_as_conflict
                    0.0,  # selected_as_context
                    0.0,  # track_age (filled when writing track)
                ],
                dtype=np.float32,
            )

            candidates.append(
                {
                    "id": str(obj_id),
                    "features": features,
                    "same_or_adjacent": same_or_adjacent,
                    "ahead": ahead,
                    "ahead_or_crossing": ahead_or_crossing,
                    "on_route": on_route,
                    "is_approaching": is_approaching,
                    "t_cpa_sort": t_cpa_sort,
                    "d_cpa_sort": d_cpa_sort,
                    "dist_sort": dist_norm,
                    "selected_conflict": 0.0,
                    "selected_context": 0.0,
                }
            )
        return candidates

    def _collect_static_candidates(self, vehicle) -> list[dict[str, Any]]:
        static_types = {
            MetaDriveType.TRAFFIC_CONE,
            MetaDriveType.TRAFFIC_BARRIER,
            MetaDriveType.TRAFFIC_OBJECT,
            MetaDriveType.BUILDING,
            MetaDriveType.INVISIBLE_WALL,
        }
        route_pairs = self._route_pairs(vehicle)
        lane_width = self._lane_width(getattr(vehicle, "lane", None))
        candidates: list[dict[str, Any]] = []
        for obj_id, obj in self._iter_world_objects():
            if str(obj_id) == str(getattr(vehicle, "id", None)):
                continue
            obj_type = self._object_type(obj)
            if obj_type not in static_types:
                continue

            pos = self._object_position(obj)
            if pos is None:
                continue
            rel_local = self._relative_local(vehicle, pos)
            dist = float(np.linalg.norm(rel_local))
            if dist > self._static_radius:
                continue

            heading = self._safe_float(getattr(obj, "heading_theta", None), default=0.0)
            ego_heading = self._safe_float(getattr(vehicle, "heading_theta", None), default=0.0)
            rel_heading = self._wrap_to_pi(heading - ego_heading)
            length = self._safe_float(getattr(obj, "LENGTH", None), default=0.0)
            width = self._safe_float(getattr(obj, "WIDTH", None), default=0.0)
            lane_idx = self._lane_index_of(obj)
            on_route = 1.0 if self._is_on_route(lane_idx, route_pairs) else 0.0
            intersects_path = 1.0 if (rel_local[0] > 0.0 and abs(rel_local[1]) <= lane_width) else 0.0

            type_cone = 1.0 if obj_type == MetaDriveType.TRAFFIC_CONE else 0.0
            type_barrier = 1.0 if obj_type == MetaDriveType.TRAFFIC_BARRIER else 0.0
            type_traffic_object = 1.0 if obj_type == MetaDriveType.TRAFFIC_OBJECT else 0.0
            type_building = 1.0 if obj_type == MetaDriveType.BUILDING else 0.0
            type_wall = 1.0 if obj_type == MetaDriveType.INVISIBLE_WALL else 0.0

            features = np.asarray(
                [
                    self._sym_norm(rel_local[0], self._static_radius),
                    self._sym_norm(rel_local[1], self._static_radius),
                    self._sym_norm(dist, self._static_radius),
                    math.sin(rel_heading),
                    math.cos(rel_heading),
                    self._sym_norm(length, self._length_norm_m),
                    self._sym_norm(width, self._width_norm_m),
                    type_cone,
                    type_barrier,
                    type_traffic_object,
                    type_building,
                    type_wall,
                    on_route,
                    intersects_path,
                ],
                dtype=np.float32,
            )

            dist_norm = min(dist / self._static_radius, 1.0)
            rank_key = (-intersects_path, -on_route, dist_norm)
            candidates.append({"features": features, "rank_key": rank_key})
        return candidates

    def _collect_control_candidates(self, vehicle) -> list[dict[str, Any]]:
        control_types = {
            MetaDriveType.TRAFFIC_LIGHT,
            MetaDriveType.STOP_SIGN,
            MetaDriveType.CROSSWALK,
            MetaDriveType.SPEED_BUMP,
        }
        route_pairs = self._route_pairs(vehicle)
        lane_width = self._lane_width(getattr(vehicle, "lane", None))
        candidates: list[dict[str, Any]] = []
        for obj_id, obj in self._iter_world_objects():
            if str(obj_id) == str(getattr(vehicle, "id", None)):
                continue
            obj_type = self._object_type(obj)
            if obj_type not in control_types:
                continue

            pos = self._object_position(obj)
            if pos is None:
                continue
            rel_local = self._relative_local(vehicle, pos)
            dist = float(np.linalg.norm(rel_local))
            if dist > self._control_radius:
                continue

            lane_idx = self._lane_index_of(obj)
            on_route = 1.0 if self._is_on_route(lane_idx, route_pairs) else 0.0

            associated_lane = 1.0 if lane_idx is not None else 0.0
            intersects_path = 1.0 if (rel_local[0] > 0 and abs(rel_local[1]) <= lane_width) else 0.0
            dist_along_route_norm = self._distance_along_route_norm(rel_local[0], self._control_radius)

            is_light = 1.0 if obj_type == MetaDriveType.TRAFFIC_LIGHT else 0.0
            is_stop = 1.0 if obj_type == MetaDriveType.STOP_SIGN else 0.0
            is_crosswalk = 1.0 if obj_type == MetaDriveType.CROSSWALK else 0.0
            is_speed_bump = 1.0 if obj_type == MetaDriveType.SPEED_BUMP else 0.0

            light_status = self._light_status(obj)
            status_red = 1.0 if light_status == MetaDriveType.LIGHT_RED else 0.0
            status_yellow = 1.0 if light_status == MetaDriveType.LIGHT_YELLOW else 0.0
            status_green = 1.0 if light_status == MetaDriveType.LIGHT_GREEN else 0.0
            status_unknown = 1.0 if light_status == MetaDriveType.LIGHT_UNKNOWN else 0.0

            requires_stop_or_yield = float(
                (is_stop > 0)
                or (is_crosswalk > 0)
                or (status_red > 0)
                or (status_yellow > 0)
            )
            occupied_vru = 0.0

            features = np.asarray(
                [
                    self._sym_norm(rel_local[0], self._control_radius),
                    self._sym_norm(rel_local[1], self._control_radius),
                    self._sym_norm(dist, self._control_radius),
                    dist_along_route_norm,
                    on_route,
                    associated_lane,
                    is_light,
                    is_stop,
                    is_crosswalk,
                    is_speed_bump,
                    status_red,
                    status_yellow,
                    status_green,
                    status_unknown,
                    intersects_path,
                    max(requires_stop_or_yield, occupied_vru),
                ],
                dtype=np.float32,
            )

            dist_norm = min(dist / self._control_radius, 1.0)
            rank_key = (-intersects_path, -on_route, dist_along_route_norm, dist_norm)
            candidates.append({"features": features, "rank_key": rank_key})
        return candidates

    def _select_dynamic_candidates(
        self,
        dynamic_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._max_dynamic_objects <= 0 or not dynamic_candidates:
            return []

        conflict_slots = self._dynamic_conflict_slots
        context_slots = self._dynamic_context_slots

        conflict_sorted = sorted(
            dynamic_candidates,
            key=lambda d: (
                -d["is_approaching"],
                d["d_cpa_sort"],
                d["t_cpa_sort"],
                -d["same_or_adjacent"],
                -d["ahead_or_crossing"],
                d["dist_sort"],
            ),
        )
        selected_conflict = conflict_sorted[:conflict_slots]
        selected_ids = {d["id"] for d in selected_conflict}

        context_pool = [d for d in dynamic_candidates if d["id"] not in selected_ids]
        context_sorted = sorted(
            context_pool,
            key=lambda d: (
                -d["same_or_adjacent"],
                -d["ahead"],
                -d["on_route"],
                d["dist_sort"],
            ),
        )
        selected_context = context_sorted[:context_slots]

        for entry in selected_conflict:
            entry["selected_conflict"] = 1.0
            entry["features"][self.DYN_IDX_SELECTED_CONFLICT] = 1.0
        for entry in selected_context:
            entry["selected_context"] = 1.0
            entry["features"][self.DYN_IDX_SELECTED_CONTEXT] = 1.0

        selected = selected_conflict + selected_context
        if len(selected) < self._max_dynamic_objects:
            remaining_pool = [d for d in dynamic_candidates if d["id"] not in {e["id"] for e in selected}]
            remaining_sorted = sorted(
                remaining_pool,
                key=lambda d: (
                    -d["same_or_adjacent"],
                    -d["ahead"],
                    -d["on_route"],
                    d["dist_sort"],
                ),
            )
            selected.extend(remaining_sorted[: (self._max_dynamic_objects - len(selected))])
        return selected[: self._max_dynamic_objects]

    def _pack_dynamic_tokens(
        self,
        selected_dynamic: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.zeros(
            (self._max_dynamic_objects, self._history_len, self.DYNAMIC_DIM),
            dtype=np.float32,
        )
        mask = np.zeros((self._max_dynamic_objects, self._history_len), dtype=np.float32)
        if self._max_dynamic_objects <= 0:
            return tokens, mask

        for slot, candidate in enumerate(selected_dynamic[: self._max_dynamic_objects]):
            track_id = str(candidate["id"])
            history = self._dynamic_history.get(track_id)
            if history is None:
                continue
            history_list = list(history)
            if len(history_list) < self._history_len:
                history_list = ([None] * (self._history_len - len(history_list))) + history_list
            for t, frame in enumerate(history_list[-self._history_len :]):
                if frame is None:
                    continue
                tokens[slot, t] = np.asarray(frame, dtype=np.float32)
                mask[slot, t] = 1.0
        return tokens, mask

    def _pack_static_tokens(
        self,
        static_candidates: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.zeros((self._max_static_objects, self.STATIC_DIM), dtype=np.float32)
        mask = np.zeros((self._max_static_objects,), dtype=np.float32)
        if self._max_static_objects <= 0:
            return tokens, mask

        selected = sorted(static_candidates, key=lambda d: d["rank_key"])[: self._max_static_objects]
        for slot, candidate in enumerate(selected):
            tokens[slot] = np.asarray(candidate["features"], dtype=np.float32)
            mask[slot] = 1.0
        return tokens, mask

    def _pack_control_tokens(
        self,
        control_candidates: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.zeros((self._max_control_objects, self.CONTROL_DIM), dtype=np.float32)
        mask = np.zeros((self._max_control_objects,), dtype=np.float32)
        if self._max_control_objects <= 0:
            return tokens, mask

        selected = sorted(control_candidates, key=lambda d: d["rank_key"])[: self._max_control_objects]
        for slot, candidate in enumerate(selected):
            tokens[slot] = np.asarray(candidate["features"], dtype=np.float32)
            mask[slot] = 1.0
        return tokens, mask

    def _advance_dynamic_tracks(self) -> None:
        for history in self._dynamic_history.values():
            history.append(None)

    def _prune_stale_tracks(self) -> None:
        stale_after = self._history_len * 4
        to_remove = [
            track_id
            for track_id, last_seen in self._dynamic_last_seen_step.items()
            if (self._step_count - last_seen) > stale_after
        ]
        for track_id in to_remove:
            self._dynamic_last_seen_step.pop(track_id, None)
            self._dynamic_history.pop(track_id, None)

    def _track_age(self, history: deque[np.ndarray | None]) -> int:
        age = 0
        for frame in reversed(history):
            if frame is None:
                break
            age += 1
        return age

    def _stack_ego_history(self) -> np.ndarray:
        stacked = np.zeros((self._history_len, self.EGO_DIM), dtype=np.float32)
        history = list(self._ego_history)
        n = min(len(history), self._history_len)
        if n > 0:
            stacked[-n:] = np.asarray(history[-n:], dtype=np.float32)
        return stacked

    def _iter_world_objects(self):
        get_objects = getattr(self.engine, "get_objects", None)
        if not callable(get_objects):
            return []
        try:
            objects = get_objects()
        except Exception:
            return []
        if not isinstance(objects, dict):
            return []
        return objects.items()

    @staticmethod
    def _read_semantic_cfg(config: dict[str, Any]) -> dict[str, Any]:
        raw = config.get("semantic_observation", {})
        return dict(raw) if isinstance(raw, dict) else {}

    def _cfg_int(self, key: str, default: int, minimum: int = 0) -> int:
        value = self._obs_cfg.get(key, default)
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return max(parsed, minimum)

    def _cfg_float(self, key: str, default: float, minimum: float = 0.0) -> float:
        value = self._obs_cfg.get(key, default)
        return max(self._safe_float(value, default=default), minimum)

    def _normalize_dynamic_quota(self) -> None:
        if self._max_dynamic_objects <= 0:
            self._dynamic_conflict_slots = 0
            self._dynamic_context_slots = 0
            return

        conflict = int(max(self._dynamic_conflict_slots, 0))
        context = int(max(self._dynamic_context_slots, 0))
        total = conflict + context

        if total <= 0:
            self._dynamic_conflict_slots = 0
            self._dynamic_context_slots = self._max_dynamic_objects
            return

        if total > self._max_dynamic_objects:
            overflow = total - self._max_dynamic_objects
            reduce_context = min(context, overflow)
            context -= reduce_context
            overflow -= reduce_context
            if overflow > 0:
                conflict = max(conflict - overflow, 0)
        elif total < self._max_dynamic_objects:
            context += self._max_dynamic_objects - total

        self._dynamic_conflict_slots = conflict
        self._dynamic_context_slots = context

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _sym_norm(value: float, scale: float) -> float:
        if scale <= 1e-6:
            return 0.0
        return float(np.clip(value / scale, -1.0, 1.0))

    @staticmethod
    def _pos_norm(value: float, scale: float) -> float:
        if not np.isfinite(value):
            return 1.0
        if scale <= 1e-6:
            return 0.0
        return float(np.clip(value / scale, 0.0, 1.0))

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _object_type(obj: Any) -> str:
        obj_type = getattr(obj, "metadrive_type", None)
        if isinstance(obj_type, str):
            return obj_type
        return str(obj_type) if obj_type is not None else ""

    @staticmethod
    def _object_position(obj: Any) -> np.ndarray | None:
        pos = getattr(obj, "position", None)
        if pos is None:
            return None
        arr = np.asarray(pos, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            return None
        return arr[:2]

    @staticmethod
    def _lane_index_of(obj: Any) -> tuple[Any, Any, int] | None:
        lane_index = getattr(obj, "lane_index", None)
        if not isinstance(lane_index, tuple) or len(lane_index) < 3:
            return None
        try:
            return (lane_index[0], lane_index[1], int(lane_index[2]))
        except Exception:
            return None

    @staticmethod
    def _lane_width(lane: Any) -> float:
        if lane is None:
            return 3.5
        width = getattr(lane, "width", None)
        if width is None:
            width_fn = getattr(lane, "width_at", None)
            if callable(width_fn):
                try:
                    width = width_fn(0.0)
                except Exception:
                    width = None
        if width is None:
            return 3.5
        try:
            return max(float(width), 1e-3)
        except Exception:
            return 3.5

    @staticmethod
    def _current_lane_count(navigation: Any) -> int:
        if navigation is None:
            return 1
        fn = getattr(navigation, "get_current_lane_num", None)
        if callable(fn):
            try:
                return max(int(fn()), 1)
            except Exception:
                return 1
        lanes = getattr(navigation, "current_ref_lanes", None)
        if isinstance(lanes, list) and lanes:
            return max(int(len(lanes)), 1)
        return 1

    @staticmethod
    def _route_completion(vehicle: Any) -> float:
        navigation = getattr(vehicle, "navigation", None)
        if navigation is None:
            return 0.0
        value = getattr(navigation, "route_completion", None)
        try:
            if value is None:
                return 0.0
            return float(np.clip(float(value), 0.0, 1.0))
        except Exception:
            return 0.0

    def _lane_total_width(self, vehicle: Any, lane_width: float) -> float:
        navigation = getattr(vehicle, "navigation", None)
        lane_count = self._current_lane_count(navigation)
        return max(float(lane_count) * lane_width, lane_width)

    def _relative_local(self, vehicle: Any, target_pos: np.ndarray) -> np.ndarray:
        return np.asarray(
            vehicle.convert_to_local_coordinates(target_pos, np.asarray(vehicle.position)),
            dtype=np.float32,
        )[:2]

    def _vector_to_local(self, vehicle: Any, world_vec: np.ndarray) -> np.ndarray:
        origin = np.asarray(vehicle.position, dtype=np.float32)
        local = vehicle.convert_to_local_coordinates(origin + world_vec, origin)
        return np.asarray(local, dtype=np.float32)[:2]

    @staticmethod
    def _closest_approach_metrics(
        rel_pos: np.ndarray,
        rel_vel: np.ndarray,
        distance: float,
    ) -> tuple[float, float, float]:
        p = np.asarray(rel_pos, dtype=np.float32)[:2]
        v = np.asarray(rel_vel, dtype=np.float32)[:2]
        pv = float(np.dot(p, v))
        vv = float(np.dot(v, v))
        eps = 1e-6
        inf = 1e9

        if vv <= eps:
            return 0.0, inf, max(distance, 0.0)

        t_cpa = -pv / vv
        if t_cpa < 0.0:
            return 0.0, inf, max(distance, 0.0)

        cpa_vec = p + v * float(t_cpa)
        d_cpa = float(np.linalg.norm(cpa_vec))
        is_approaching = 1.0 if pv < 0.0 else 0.0
        return is_approaching, float(t_cpa), max(d_cpa, 0.0)

    @staticmethod
    def _distance_along_route_norm(rel_x: float, radius: float) -> float:
        if radius <= 1e-6:
            return 1.0
        if rel_x <= 0.0:
            return 1.0
        return float(np.clip(rel_x / radius, 0.0, 1.0))

    @staticmethod
    def _hashable_route_token(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return tuple(np.asarray(value).tolist())
        if isinstance(value, list):
            return tuple(
                SemanticStateObservation._hashable_route_token(item) for item in value
            )
        if isinstance(value, tuple):
            return tuple(
                SemanticStateObservation._hashable_route_token(item) for item in value
            )
        return value

    @staticmethod
    def _route_pairs(vehicle: Any) -> set[tuple[Any, Any]]:
        navigation = getattr(vehicle, "navigation", None)
        checkpoints = getattr(navigation, "checkpoints", None) if navigation is not None else None
        if not isinstance(checkpoints, list) or len(checkpoints) < 2:
            return set()
        return {
            (
                SemanticStateObservation._hashable_route_token(start),
                SemanticStateObservation._hashable_route_token(end),
            )
            for start, end in zip(checkpoints[:-1], checkpoints[1:])
        }

    @staticmethod
    def _is_on_route(
        lane_idx: tuple[Any, Any, int] | None,
        route_pairs: set[tuple[Any, Any]],
    ) -> bool:
        if lane_idx is None:
            return False
        if not route_pairs:
            return False
        return (
            SemanticStateObservation._hashable_route_token(lane_idx[0]),
            SemanticStateObservation._hashable_route_token(lane_idx[1]),
        ) in route_pairs

    @staticmethod
    def _light_status(obj: Any) -> str:
        status = getattr(obj, "status", None)
        try:
            return MetaDriveType.simplify_light_status(status)
        except Exception:
            return MetaDriveType.LIGHT_UNKNOWN
