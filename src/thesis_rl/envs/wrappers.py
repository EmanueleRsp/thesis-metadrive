from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from thesis_rl.common.paths import default_output_path_str
from thesis_rl.reward.interfaces.base import BaseRewardManager


class RuleRewardWrapper(gym.Wrapper):
    """Apply rulebook-based hybrid reward and enrich step info."""

    def __init__(
        self,
        env: gym.Env,
        reward_manager: BaseRewardManager,
        reward_mode: str = "monitor_only",
        attach_info: bool = True,
        rule_margin_log_path: str | None = None,
        runtime_info_debug_enabled: bool = False,
        runtime_info_debug_path: str | None = None,
        logger_level: int | str | None = None,
    ) -> None:
        super().__init__(env)
        self.reward_manager = reward_manager
        self.reward_mode = str(reward_mode).lower()
        self.attach_info = bool(attach_info)
        self._rule_margin_log_path = Path(rule_margin_log_path) if rule_margin_log_path else None
        if self._rule_margin_log_path is not None:
            self._rule_margin_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._runtime_info_debug_enabled = bool(runtime_info_debug_enabled)
        self._runtime_info_debug_path = (
            Path(runtime_info_debug_path)
            if runtime_info_debug_path
            else Path(default_output_path_str("runtime_info_debug.jsonl"))
        )
        if self._runtime_info_debug_enabled:
            self._runtime_info_debug_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)
        if logger_level is not None:
            if isinstance(logger_level, int):
                self._logger.setLevel(logger_level)
            elif isinstance(logger_level, str):
                parsed = getattr(logging, logger_level.strip().upper(), None)
                if isinstance(parsed, int):
                    self._logger.setLevel(parsed)
        self._diagnostics_emitted = False
        self._warned_fallbacks: set[str] = set()
        self._cached_map_obj_id: int | None = None
        self._cached_drivable_area: Any | None = None
        self._cached_opposite_carriageway_by_road: dict[tuple[str, str], Any | None] = {}
        self._cached_target_region_by_lane_id: dict[int, Any | None] = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_manager.reset()
        self._diagnostics_emitted = False
        self._cached_target_region_by_lane_id = {}
        return obs, info

    def step(self, action: Any):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        info_dict = dict(info)
        self._enrich_runtime_info(info_dict)
        self._log_rulebook_input_diagnostics(info_dict)

        result = self.reward_manager.compute(float(env_reward), info_dict)
        selected_reward = self._select_reward(float(env_reward), result)
        self._append_rule_margin_log(info_dict=info_dict, env_reward=float(env_reward), result=result)

        if self.attach_info:
            info_dict["env_reward"] = float(env_reward)
            info_dict["hybrid_reward"] = float(result.final_reward)
            info_dict["selected_reward"] = float(selected_reward)
            info_dict["reward_behavior"] = self.reward_mode
            info_dict["rule_reward_vector"] = result.rule_reward_vector
            info_dict["rule_bounded_vector"] = result.rule_bounded_vector
            info_dict["rule_components"] = result.rule_components
            info_dict["rule_metadata"] = result.rule_metadata
            rulebook_rules = result.rule_metadata.get("rules", {})
            diagnostics = result.rule_metadata.get("diagnostics", {})
            if isinstance(rulebook_rules, Mapping):
                info_dict["rulebook"] = {
                    **dict(rulebook_rules),
                    "diagnostics": dict(diagnostics) if isinstance(diagnostics, Mapping) else {},
                }
            info_dict["scalar_rule_reward"] = result.scalar_rule_reward
            if result.rule_violation_vector is not None:
                info_dict["rule_violation_vector"] = result.rule_violation_vector

        return obs, selected_reward, terminated, truncated, info_dict

    def _select_reward(self, env_reward: float, result: Any) -> float:
        if self.reward_mode == "monitor_only":
            return float(env_reward)
        if self.reward_mode == "scalar_reward":
            return float(result.scalar_rule_reward)
        if self.reward_mode in {"hybrid", "lexicographic"}:
            return float(result.final_reward)
        raise ValueError(
            "Unsupported reward mode. "
            f"Got mode='{self.reward_mode}', expected one of: "
            "monitor_only, scalar_reward, hybrid, lexicographic."
        )

    def _append_rule_margin_log(
        self,
        *,
        info_dict: dict[str, Any],
        env_reward: float,
        result: Any,
    ) -> None:
        if self._rule_margin_log_path is None:
            return

        payload = {
            "step": info_dict.get("episode_length"),
            "env_reward": env_reward,
            "scalar_rule_reward": float(result.scalar_rule_reward),
            "final_reward": float(result.final_reward),
            "rule_components": dict(result.rule_components),
            "rulebook": dict(result.rule_metadata.get("rules", {})),
            "diagnostics": dict(result.rule_metadata.get("diagnostics", {})),
        }
        with self._rule_margin_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")

    def _enrich_runtime_info(self, info_dict: dict[str, Any]) -> None:
        '''Enrich step info dict with derived inputs for rule evaluation.
        This method attempts to extract and compute various pieces of information about the ego vehicle, 
        its surroundings, and the environment, which can be used as inputs for rule evaluation. 
        '''

        # Extract env
        base_env = getattr(self.env, "unwrapped", self.env)
        self._sync_map_cache(base_env)
        rule_input_sources: dict[str, str] = {}
        rule_input_available: dict[str, bool] = {}

        ####### EGO VEHICLE #######
        ego_vehicle = self._extract_ego_vehicle(base_env)
        if ego_vehicle is None:
            return
        # Ego state
        ego_state = info_dict.get("ego_state")
        if not isinstance(ego_state, Mapping):
            ego_state = {}
        else:
            ego_state = dict(ego_state)

        # Position
        position = getattr(ego_vehicle, "position", None)
        if position is not None:
            ego_state.setdefault("position", position)

        # Velocity
        velocity = getattr(ego_vehicle, "velocity", None)
        if velocity is not None:
            ego_state.setdefault("velocity", velocity)

        # Prefer km/h for consistency with MetaDrive step info and speed limit keys.
        speed_kmh = self._safe_float(getattr(ego_vehicle, "speed_km_h", None))
        if speed_kmh is not None:
            ego_state.setdefault("speed", speed_kmh)
        else:
            self._warn_once(
                "missing_speed_kmh",
                "RuleRewardWrapper did not find `vehicle.speed_km_h`; speed-limit rule may become neutral.",
            )

        # Keep optional raw speed in m/s if available for downstream diagnostics.
        speed_m_s = self._safe_float(getattr(ego_vehicle, "speed", None))
        if speed_m_s is not None:
            ego_state.setdefault("speed_m_s", speed_m_s)

        yaw = self._safe_float(getattr(ego_vehicle, "heading_theta", None))
        if yaw is not None:
            ego_state.setdefault("yaw", yaw)

        steer = self._safe_float(getattr(ego_vehicle, "steering", None))
        if steer is not None:
            ego_state.setdefault("steer", steer)

        # Distinguish control command from physical acceleration semantics.
        accel_cmd = self._safe_float(getattr(ego_vehicle, "throttle_brake", None))
        if accel_cmd is not None:
            ego_state.setdefault("accel_cmd", accel_cmd)

        acceleration = self._extract_physical_acceleration(base_env, ego_vehicle)
        if acceleration is not None:
            ego_state.setdefault("acceleration", acceleration)

        length = self._safe_float(getattr(ego_vehicle, "LENGTH", None))
        if length is not None:
            ego_state.setdefault("length", length)

        width = self._safe_float(getattr(ego_vehicle, "WIDTH", None))
        if width is not None:
            ego_state.setdefault("width", width)

        polygon = getattr(ego_vehicle, "bounding_box", None)
        if polygon is not None:
            ego_state.setdefault("polygon", polygon)

        if ego_state:
            info_dict["ego_state"] = ego_state
            rule_input_available["ego_state"] = True
            rule_input_sources["ego_state"] = "base_env.vehicle | base_env.agents"
        else:
            rule_input_available["ego_state"] = False

        lane_centerline = self._extract_lane_centerline(ego_vehicle)
        if lane_centerline is not None:
            info_dict.setdefault("lane_centerline", lane_centerline)
            rule_input_sources["lane_centerline"] = "ego_vehicle.lane | ego_vehicle.navigation.current_ref_lanes[0]"
            rule_input_available["lane_centerline"] = True
        else:
            rule_input_available["lane_centerline"] = False

        target_region = self._extract_target_region(ego_vehicle)
        if target_region is not None:
            info_dict.setdefault("target_region", target_region)
            rule_input_sources["target_region"] = "ego_vehicle.navigation.final_lane.shapely_polygon | ego_vehicle.navigation.final_lane.polygon"
            rule_input_available["target_region"] = True
        else:
            rule_input_available["target_region"] = False

        target_point = self._extract_target_point(ego_vehicle)
        if target_point is not None:
            info_dict.setdefault("target_point", target_point)
            rule_input_sources["target_point"] = "ego_vehicle.navigation.final_lane.position(length, 0.0) | ego_vehicle.navigation.current_checkpoint"
            rule_input_available["target_point"] = True
        else:
            rule_input_available["target_point"] = False

        speed_limit = self._extract_speed_limit(ego_vehicle, lane_centerline)
        if speed_limit is not None:
            info_dict.setdefault("speed_limit", speed_limit)
            rule_input_sources["speed_limit"] = "lane_centerline.speed_limit | ego_vehicle.max_speed_km_h"
            rule_input_available["speed_limit"] = True
        else:
            rule_input_available["speed_limit"] = False

        drivable_area = self._extract_drivable_area(base_env)
        if drivable_area is not None:
            info_dict.setdefault("drivable_area", drivable_area)
            rule_input_sources["drivable_area"] = "current_map.road_network.get_all_lanes() -> lane.shapely_polygon union"
            rule_input_available["drivable_area"] = True
        else:
            rule_input_available["drivable_area"] = False

        opposite_carriageway = self._extract_opposite_carriageway(base_env, ego_vehicle)
        if opposite_carriageway is not None:
            info_dict.setdefault("opposite_carriageway", opposite_carriageway)
            rule_input_sources["opposite_carriageway"] = "ego_vehicle.navigation.current_road -> opposite road lane union"
            rule_input_available["opposite_carriageway"] = True
        else:
            rule_input_available["opposite_carriageway"] = False

        allowed_driving_area, allowed_area_source = self._extract_allowed_driving_area(
            ego_vehicle,
            drivable_area,
            opposite_carriageway,
        )
        if allowed_driving_area is not None:
            info_dict.setdefault("allowed_driving_area", allowed_driving_area)
            rule_input_sources["allowed_driving_area"] = allowed_area_source
            rule_input_available["allowed_driving_area"] = True
        else:
            rule_input_available["allowed_driving_area"] = False

        lane_markings = self._extract_lane_markings(base_env, lane_centerline)
        if lane_markings is not None:
            solid_markings, dashed_markings, lane_marking_source = lane_markings
            info_dict.setdefault("solid_lane_markings", solid_markings)
            info_dict.setdefault("dashed_lane_markings", dashed_markings)
            rule_input_sources["lane_markings"] = lane_marking_source
            rule_input_available["lane_markings"] = True
        else:
            rule_input_available["lane_markings"] = False

        route_progress, route_progress_source = self._extract_route_progress(
            ego_vehicle,
            info_dict,
        )
        if route_progress is not None:
            info_dict.setdefault("route_progress", route_progress)
            rule_input_sources["route_progress"] = route_progress_source
            rule_input_available["route_progress"] = True
        else:
            rule_input_available["route_progress"] = False

        if not isinstance(info_dict.get("neighbors"), list):
            neighbors, neighbors_source = self._extract_neighbors(base_env, ego_vehicle)
            info_dict["neighbors"] = neighbors
            rule_input_sources["neighbors"] = neighbors_source or "unavailable"
            rule_input_available["neighbors"] = bool(neighbors)
        else:
            rule_input_available["neighbors"] = True
            rule_input_sources["neighbors"] = "info_dict.neighbors"

        info_dict["rule_input_sources"] = rule_input_sources
        info_dict["rule_input_available"] = rule_input_available

        if not self._runtime_info_debug_enabled:
            return

        # --- Runtime diagnostics: write a compact JSON line per step indicating
        # presence of key inputs and small numeric samples to help debugging.
        try:
            debug_path = self._runtime_info_debug_path
            debug_path.parent.mkdir(parents=True, exist_ok=True)

            top_level = [
                "ego_state",
                "neighbors",
                "drivable_area",
                "opposite_carriageway",
                "lane_centerline",
                "target_region",
                "target_point",
                "speed_limit",
            ]

            ego_keys = [
                "position",
                "velocity",
                "speed",
                "speed_m_s",
                "yaw",
                "steer",
                "accel_cmd",
                "acceleration",
                "length",
                "width",
                "polygon",
            ]

            ego_state_local = info_dict.get("ego_state") if isinstance(info_dict.get("ego_state"), Mapping) else {}
            # sample numeric values where available
            pos_sample = None
            pos = ego_state_local.get("position")
            if pos is not None:
                try:
                    arr = list(pos)
                    if len(arr) >= 2:
                        pos_sample = [float(arr[0]), float(arr[1])]
                except Exception:
                    pos_sample = None

            speed_kmh_sample = None
            try:
                if ego_state_local.get("speed") is not None:
                    speed_kmh_sample = float(ego_state_local.get("speed"))
            except Exception:
                speed_kmh_sample = None

            speed_ms_sample = None
            try:
                if ego_state_local.get("speed_m_s") is not None:
                    speed_ms_sample = float(ego_state_local.get("speed_m_s"))
            except Exception:
                speed_ms_sample = None

            neighbors = info_dict.get("neighbors")
            neighbor_count = len(neighbors) if isinstance(neighbors, list) else 0

            payload = {
                "step": info_dict.get("episode_length") or info_dict.get("step"),
                "top_presence": {name: (info_dict.get(name) is not None) for name in top_level},
                "ego_presence": {name: (ego_state_local.get(name) is not None) for name in ego_keys},
                "position": pos_sample,
                "speed_kmh": speed_kmh_sample,
                "speed_m_s": speed_ms_sample,
                "neighbors_count": neighbor_count,
                "navigation_present": (getattr(ego_vehicle, "navigation", None) is not None),
            }

            with debug_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True))
                handle.write("\n")
        except Exception:
            # Never raise from diagnostics path; log for later inspection.
            self._logger.debug("Failed to write runtime_info_debug.jsonl diagnostics", exc_info=True)

    @staticmethod
    def _extract_ego_vehicle(base_env: Any) -> Any | None:
        agents = getattr(base_env, "agents", None)
        if isinstance(agents, Mapping) and agents:
            for vehicle in agents.values():
                if vehicle is not None:
                    return vehicle
        return getattr(base_env, "vehicle", None)

    def _extract_lane_centerline(self, ego_vehicle: Any) -> Any | None:
        lane = getattr(ego_vehicle, "lane", None)
        if lane is not None:
            return lane

        navigation = getattr(ego_vehicle, "navigation", None)
        if navigation is None:
            return None

        ref_lanes = getattr(navigation, "current_ref_lanes", None)
        if isinstance(ref_lanes, (list, tuple)) and ref_lanes:
            return ref_lanes[0]
        return None

    def _extract_target_point(self, ego_vehicle: Any) -> Any | None:
        navigation = getattr(ego_vehicle, "navigation", None)
        if navigation is None:
            return None

        final_lane = getattr(navigation, "final_lane", None)
        if final_lane is not None and hasattr(final_lane, "position"):
            lane_length = self._safe_float(getattr(final_lane, "length", None))
            longitudinal = lane_length if lane_length is not None else 0.0
            try:
                return final_lane.position(longitudinal, 0.0)
            except Exception:
                return None

        checkpoint = getattr(navigation, "current_checkpoint", None)
        if checkpoint is not None:
            return checkpoint
        return None

    def _extract_target_region(self, ego_vehicle: Any) -> Any | None:
        navigation = getattr(ego_vehicle, "navigation", None)
        if navigation is None:
            return None

        final_lane = getattr(navigation, "final_lane", None)
        if final_lane is None:
            return None
        final_lane_id = id(final_lane)
        if final_lane_id in self._cached_target_region_by_lane_id:
            return self._cached_target_region_by_lane_id[final_lane_id]

        shapely_poly = getattr(final_lane, "shapely_polygon", None)
        if shapely_poly is not None:
            self._cached_target_region_by_lane_id[final_lane_id] = shapely_poly
            return shapely_poly

        polygon = getattr(final_lane, "polygon", None)
        if polygon is not None:
            self._cached_target_region_by_lane_id[final_lane_id] = polygon
            return polygon
        self._cached_target_region_by_lane_id[final_lane_id] = None
        return None

    def _extract_speed_limit(self, ego_vehicle: Any, lane_centerline: Any | None) -> float | None:
        lane_speed_limit = self._safe_float(getattr(lane_centerline, "speed_limit", None))
        if lane_speed_limit is not None:
            return lane_speed_limit

        self._warn_once(
            "speed_limit_fallback",
            "Lane speed limit is unavailable; falling back to `vehicle.max_speed_km_h` for speed-limit rule.",
        )
        return self._safe_float(getattr(ego_vehicle, "max_speed_km_h", None))

    @staticmethod
    def _extract_allowed_driving_area(
        ego_vehicle: Any,
        drivable_area: Any | None,
        opposite_carriageway: Any | None,
    ) -> tuple[Any | None, str]:
        """Build the route-compatible area before considering road-wide geometry.

        ScenarioEnv does not consistently expose an opposite carriageway.  Its
        navigation reference lanes are the stronger semantic source: they
        already identify the lanes compatible with the current route.
        """
        navigation = getattr(ego_vehicle, "navigation", None)
        if navigation is not None:
            lanes: list[Any] = []
            for attr_name in ("current_ref_lanes", "next_ref_lanes"):
                candidates = getattr(navigation, attr_name, None)
                if isinstance(candidates, (list, tuple)):
                    lanes.extend(candidates)
            polygons = [getattr(lane, "shapely_polygon", None) for lane in lanes]
            polygons = [polygon for polygon in polygons if polygon is not None]
            if polygons:
                try:
                    from shapely.ops import unary_union

                    return unary_union(polygons), "navigation.current_ref_lanes + next_ref_lanes"
                except Exception:
                    pass

        if drivable_area is None or opposite_carriageway is None:
            return None, "unavailable"
        try:
            return drivable_area.difference(opposite_carriageway), "drivable_area - opposite_carriageway"
        except Exception:
            return None, "unavailable"

    @staticmethod
    def _extract_lane_markings(
        base_env: Any,
        lane: Any | None,
    ) -> tuple[list[Any], list[Any], str] | None:
        """Extract physical road markings, preferring ScenarioMap features.

        ScenarioEnv's route is a ``PointLane`` whose ``line_types`` are always
        ``NONE``.  Its actual solid and broken lines live in the loaded
        ``ScenarioMap`` feature set, so using the route lane would silently
        erase R3's signal.  The map feature vector is the authoritative source
        for ScenarioEnv; the lane representation remains the direct source for
        non-scenario environments that do not expose it.
        """
        current_map = getattr(base_env, "current_map", None)
        boundary_line_vector = getattr(current_map, "get_boundary_line_vector", None)
        if callable(boundary_line_vector):
            try:
                features = boundary_line_vector(interval=0.5)
            except Exception:
                features = None
            if isinstance(features, Mapping):
                try:
                    from shapely.geometry import LineString

                    solid: list[Any] = []
                    dashed: list[Any] = []
                    for feature in features.values():
                        if not isinstance(feature, Mapping):
                            continue
                        polyline = feature.get("polyline")
                        line_type = str(feature.get("type", "")).lower()
                        if polyline is None:
                            continue
                        geometry = LineString(polyline).buffer(0.10)
                        if "broken" in line_type or "dash" in line_type:
                            dashed.append(geometry)
                        else:
                            # Continuous road lines and road-edge boundaries
                            # are both forbidden boundaries for R3.
                            solid.append(geometry)
                    return solid, dashed, "current_map.get_boundary_line_vector"
                except Exception:
                    # Do not return partial geometry.  The direct lane adapter
                    # below remains valid for non-ScenarioEnv environments.
                    pass

        if lane is None or not hasattr(lane, "get_polyline"):
            return None
        line_types = getattr(lane, "line_types", None)
        width = RuleRewardWrapper._safe_float(getattr(lane, "width", None))
        if not isinstance(line_types, (list, tuple)) or len(line_types) != 2 or width is None:
            return None
        try:
            from shapely.geometry import LineString

            solid: list[Any] = []
            dashed: list[Any] = []
            for side, line_type in enumerate(line_types):
                kind = str(getattr(line_type, "name", line_type)).lower()
                if "none" in kind:
                    continue
                lateral = -width / 2.0 if side == 0 else width / 2.0
                geometry = LineString(lane.get_polyline(interval=0.5, lateral=lateral)).buffer(0.10)
                if "broken" in kind or "dash" in kind:
                    dashed.append(geometry)
                else:
                    solid.append(geometry)
            return solid, dashed, "current_lane.line_types + current_lane.get_polyline"
        except Exception:
            return None

    @staticmethod
    def _extract_route_progress(
        ego_vehicle: Any,
        info_dict: Mapping[str, Any],
    ) -> tuple[float | None, str]:
        """Return a monotonic route coordinate from the active navigation module.

        ``ScenarioEnv`` uses ``TrajectoryNavigation``.  Unlike the road-network
        navigation modules, it reports per-step ``route_completion`` and
        ``track_length`` in its info dictionary, while retaining the absolute
        coordinate as ``current_longitude`` internally.  Converting the two
        public info fields yields the same metric coordinate and keeps the
        rule input tied to ScenarioEnv's documented step interface.
        """
        route_completion = RuleRewardWrapper._safe_float(info_dict.get("route_completion"))
        track_length = RuleRewardWrapper._safe_float(info_dict.get("track_length"))
        if route_completion is not None and track_length is not None and track_length > 0.0:
            return route_completion * track_length, "info.route_completion * info.track_length"

        navigation = getattr(ego_vehicle, "navigation", None)
        if navigation is None:
            return None, "unavailable"

        # This is a direct navigation coordinate, not a derived fallback.  It
        # covers navigation modules that do not publish ScenarioEnv's info
        # fields, while preserving strict-mode guarantees.
        for name in (
            "current_longitude",
            "travelled_length",
            "route_progress",
            "current_route_progress",
            "progress",
        ):
            value = RuleRewardWrapper._safe_float(getattr(navigation, name, None))
            if value is not None:
                return value, f"ego_vehicle.navigation.{name}"
        return None, "unavailable"

    def _extract_physical_acceleration(self, base_env: Any, ego_vehicle: Any) -> dict[str, float] | None:
        current_velocity = self._to_xy_array(getattr(ego_vehicle, "velocity", None))
        previous_velocity = self._to_xy_array(getattr(ego_vehicle, "last_velocity", None))
        if current_velocity is None or previous_velocity is None:
            return None

        dt = self._extract_dt(base_env)
        if dt is None or dt <= 0.0:
            return None

        accel_vec = (current_velocity - previous_velocity) / float(dt)
        ax = float(accel_vec[0])
        ay = float(accel_vec[1])

        yaw = self._safe_float(getattr(ego_vehicle, "heading_theta", None))
        if yaw is None:
            return {"x": ax, "y": ay}

        heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        lateral_axis = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        a_long = float(np.dot(accel_vec, heading))
        a_lat = float(np.dot(accel_vec, lateral_axis))
        return {
            "x": ax,
            "y": ay,
            "longitudinal": a_long,
            "lateral": a_lat,
        }

    def _extract_dt(self, base_env: Any) -> float | None:
        def _dt_from_cfg(cfg_obj: Any) -> float | None:
            if cfg_obj is None:
                return None
            step_size = None
            decision_repeat = None
            policy_frequency = None
            if isinstance(cfg_obj, Mapping):
                step_size = self._safe_float(cfg_obj.get("physics_world_step_size"))
                decision_repeat = self._safe_float(cfg_obj.get("decision_repeat"))
                policy_frequency = self._safe_float(cfg_obj.get("policy_frequency"))
            else:
                step_size = self._safe_float(getattr(cfg_obj, "physics_world_step_size", None))
                decision_repeat = self._safe_float(getattr(cfg_obj, "decision_repeat", None))
                policy_frequency = self._safe_float(getattr(cfg_obj, "policy_frequency", None))

            if step_size is not None:
                if decision_repeat is not None and decision_repeat > 0:
                    return float(step_size * decision_repeat)
                return float(step_size)
            if policy_frequency is not None and policy_frequency > 0:
                return float(1.0 / policy_frequency)
            return None

        cfg = getattr(base_env, "config", None)
        dt = _dt_from_cfg(cfg)
        if dt is not None:
            return dt

        engine = getattr(base_env, "engine", None)
        if engine is not None:
            dt = _dt_from_cfg(getattr(engine, "global_config", None))
            if dt is not None:
                return dt

        self._warn_once(
            "missing_dt_for_acceleration",
            "Could not infer integration dt from env config; physical acceleration will be unavailable.",
        )
        return None

    def _extract_drivable_area(self, base_env: Any) -> Any | None:
        if self._cached_drivable_area is not None:
            return self._cached_drivable_area

        current_map = getattr(base_env, "current_map", None)
        if current_map is None:
            return None

        road_network = getattr(current_map, "road_network", None)
        if road_network is None or not hasattr(road_network, "get_all_lanes"):
            return None

        lanes = road_network.get_all_lanes()
        polygons = []
        for lane in lanes:
            poly = getattr(lane, "shapely_polygon", None)
            if poly is not None:
                polygons.append(poly)

        if not polygons:
            return None

        try:
            self._cached_drivable_area = self._union_lane_polygons(polygons)
            return self._cached_drivable_area
        except Exception:
            self._warn_once(
                "drivable_area_union_failed",
                "Failed to compute drivable area union from lane polygons.",
            )
            return None

    @staticmethod
    def _union_lane_polygons(polygons: list[Any]) -> Any:
        """Union lane polygons while tolerating invalid converted-map geometry.

        Waymo map-feature polygons can contain self-intersections after their
        coordinate conversion.  GEOS rejects a direct unary union of those
        shapes, which previously disabled all drivable-area rules for the
        episode.  Repair each individual polygon before the union so one bad
        lane cannot discard the whole map.
        """

        from shapely import make_valid
        from shapely.ops import unary_union

        repaired: list[Any] = []
        for polygon in polygons:
            if bool(getattr(polygon, "is_empty", False)):
                continue
            candidate = polygon
            if not bool(getattr(candidate, "is_valid", True)):
                candidate = make_valid(candidate)
            if bool(getattr(candidate, "is_empty", False)):
                continue
            repaired.append(candidate)
        if not repaired:
            raise ValueError("no usable lane polygons")
        union = unary_union(repaired)
        if bool(getattr(union, "is_empty", False)):
            raise ValueError("lane polygon union is empty")
        return union

    def _extract_opposite_carriageway(self, base_env: Any, ego_vehicle: Any) -> Any | None:
        navigation = getattr(ego_vehicle, "navigation", None)
        if navigation is None:
            return None

        current_road = getattr(navigation, "current_road", None)
        if current_road is None:
            return None
        road_key = (str(current_road.start_node), str(current_road.end_node))
        if road_key in self._cached_opposite_carriageway_by_road:
            return self._cached_opposite_carriageway_by_road[road_key]

        current_map = getattr(base_env, "current_map", None)
        if current_map is None:
            return None
        road_network = getattr(current_map, "road_network", None)
        if road_network is None:
            return None

        try:
            opposite_road = -current_road
            opposite_lanes = opposite_road.get_lanes(road_network)
        except Exception:
            self._cached_opposite_carriageway_by_road[road_key] = None
            return None

        polygons = []
        for lane in opposite_lanes:
            poly = getattr(lane, "shapely_polygon", None)
            if poly is not None:
                polygons.append(poly)

        if not polygons:
            self._cached_opposite_carriageway_by_road[road_key] = None
            return None

        try:
            union_poly = self._union_lane_polygons(polygons)
            self._cached_opposite_carriageway_by_road[road_key] = union_poly
            return union_poly
        except Exception:
            self._warn_once(
                "opposite_carriageway_union_failed",
                "Failed to compute opposite carriageway union from lane polygons.",
            )
            self._cached_opposite_carriageway_by_road[road_key] = None
            return None

    def _sync_map_cache(self, base_env: Any) -> None:
        current_map = getattr(base_env, "current_map", None)
        current_map_id = id(current_map) if current_map is not None else None
        if current_map_id == self._cached_map_obj_id:
            return

        self._cached_map_obj_id = current_map_id
        self._cached_drivable_area = None
        self._cached_opposite_carriageway_by_road = {}
        self._cached_target_region_by_lane_id = {}

    def _extract_neighbors(self, base_env: Any, ego_vehicle: Any) -> tuple[list[dict[str, Any]], str | None]:
        neighbors: list[dict[str, Any]] = []
        seen_ids: set[int] = set()

        agents = getattr(base_env, "agents", None)
        if isinstance(agents, Mapping):
            for vehicle in agents.values():
                state = self._vehicle_to_state(vehicle, ego_vehicle)
                if state is not None:
                    neighbors.append(state)
                    seen_ids.add(id(vehicle))
            if neighbors:
                return neighbors, "base_env.agents"

        engine = getattr(base_env, "engine", None)
        traffic_manager = getattr(engine, "traffic_manager", None)
        if traffic_manager is None:
            traffic_manager = getattr(base_env, "traffic_manager", None)

        iterator, source_name = self._iter_traffic_vehicles(traffic_manager)

        for vehicle in iterator:
            if id(vehicle) in seen_ids:
                continue
            state = self._vehicle_to_state(vehicle, ego_vehicle)
            if state is not None:
                neighbors.append(state)
                seen_ids.add(id(vehicle))

        # Scenario environments can include static collidable objects that are
        # not owned by the traffic manager. Include them when the engine offers
        # the public object registry, while preserving the traffic fast path.
        get_objects = getattr(engine, "get_objects", None)
        if callable(get_objects):
            try:
                engine_objects = get_objects()
            except Exception:
                engine_objects = {}
            values = engine_objects.values() if isinstance(engine_objects, Mapping) else ()
            for obj in values:
                if id(obj) in seen_ids or obj is ego_vehicle:
                    continue
                state = self._vehicle_to_state(obj, ego_vehicle)
                if state is not None:
                    neighbors.append(state)
                    seen_ids.add(id(obj))

        return neighbors, source_name

    def _iter_traffic_vehicles(self, traffic_manager: Any):
        if traffic_manager is None:
            return (), None

        for attr_name in ("traffic_vehicles", "vehicles", "_traffic_vehicles"):
            traffic_vehicles = getattr(traffic_manager, attr_name, None)
            if attr_name == "_traffic_vehicles" and traffic_vehicles is not None:
                self._warn_once(
                    "private_traffic_vehicles",
                    "RuleRewardWrapper is using private traffic manager field `_traffic_vehicles` as fallback.",
                    level="info",
                )
            if isinstance(traffic_vehicles, Mapping):
                if traffic_vehicles:
                    if attr_name != "traffic_vehicles":
                        self._warn_once(
                            f"traffic_manager_{attr_name}",
                            f"RuleRewardWrapper is using traffic manager field `{attr_name}` for neighbors fallback.",
                            level="info",
                        )
                    return traffic_vehicles.values(), f"traffic_manager.{attr_name}"
                continue
            if isinstance(traffic_vehicles, (list, tuple)):
                if traffic_vehicles:
                    if attr_name != "traffic_vehicles":
                        self._warn_once(
                            f"traffic_manager_{attr_name}",
                            f"RuleRewardWrapper is using traffic manager field `{attr_name}` for neighbors fallback.",
                            level="info",
                        )
                    return traffic_vehicles, f"traffic_manager.{attr_name}"
                continue
        return (), None

    def _warn_once(self, key: str, message: str, *, level: str = "warning") -> None:
        if key in self._warned_fallbacks:
            return
        self._warned_fallbacks.add(key)
        if level == "info":
            self._logger.info(message)
            return
        self._logger.warning(message)

    def _log_rulebook_input_diagnostics(self, info_dict: dict[str, Any]) -> None:
        if self._diagnostics_emitted:
            return

        top_level_fields = (
            "ego_state",
            "neighbors",
            "drivable_area",
            "opposite_carriageway",
            "lane_centerline",
            "target_region",
            "target_point",
            "speed_limit",
        )
        top_available = [name for name in top_level_fields if info_dict.get(name) is not None]
        top_missing = [name for name in top_level_fields if info_dict.get(name) is None]

        ego_state = info_dict.get("ego_state")
        if not isinstance(ego_state, Mapping):
            ego_state = {}

        ego_fields = (
            "position",
            "velocity",
            "speed",
            "speed_m_s",
            "yaw",
            "steer",
            "accel_cmd",
            "acceleration",
            "length",
            "width",
            "polygon",
        )
        ego_available = [name for name in ego_fields if ego_state.get(name) is not None]
        ego_missing = [name for name in ego_fields if ego_state.get(name) is None]

        neighbors = info_dict.get("neighbors")
        neighbor_count = len(neighbors) if isinstance(neighbors, list) else 0

        self._logger.debug(
            "Rulebook input availability | top available=%s missing=%s | ego available=%s missing=%s | neighbors=%d",
            top_available,
            top_missing,
            ego_available,
            ego_missing,
            neighbor_count,
        )

        if info_dict.get("drivable_area") is None:
            self._warn_once(
                "missing_drivable_area",
                "Rulebook input `drivable_area` is unavailable; related rules may become neutral.",
            )
        active_rule_names = self._active_rule_names()
        if "wrong_way" in active_rule_names and info_dict.get("opposite_carriageway") is None:
            self._warn_once(
                "missing_opposite_carriageway",
                "Rulebook input `opposite_carriageway` is unavailable; wrong-way rule may become neutral.",
            )
        if "goal_progress" in active_rule_names and info_dict.get("target_region") is None:
            self._warn_once(
                "missing_target_region",
                "Rulebook input `target_region` is unavailable; goal-progress rule may rely only on `target_point`.",
            )

        self._diagnostics_emitted = True

    def _active_rule_names(self) -> set[str]:
        """Return configured rule names when the manager exposes an evaluator.

        Runtime diagnostics must describe the active rulebook.  In particular,
        ScenarioEnv legitimately lacks inputs needed only by legacy rules, and
        those should not look like a degraded v1 evaluation.
        """
        evaluator = getattr(self.reward_manager, "evaluator", None)
        specs = getattr(evaluator, "rules", ())
        return {
            str(name)
            for spec in specs
            for name in (getattr(spec, "name", None),)
            if isinstance(name, str)
        }

    @classmethod
    def _vehicle_to_state(cls, vehicle: Any, ego_vehicle: Any) -> dict[str, Any] | None:
        if vehicle is None or vehicle is ego_vehicle:
            return None

        state: dict[str, Any] = {}
        state["entity_id"] = cls._extract_entity_id(vehicle)

        position = getattr(vehicle, "position", None)
        if position is not None:
            state["position"] = position

        velocity = getattr(vehicle, "velocity", None)
        if velocity is not None:
            state["velocity"] = velocity

        speed_m_s = cls._safe_float(getattr(vehicle, "speed", None))
        if speed_m_s is not None:
            state["speed_m_s"] = speed_m_s

        length = cls._safe_float(getattr(vehicle, "LENGTH", None))
        if length is not None:
            state["length"] = length

        width = cls._safe_float(getattr(vehicle, "WIDTH", None))
        if width is not None:
            state["width"] = width

        heading = cls._safe_float(getattr(vehicle, "heading_theta", None))
        if heading is not None:
            state["yaw"] = heading

        polygon = getattr(vehicle, "bounding_box", None)
        if polygon is not None:
            state["polygon"] = polygon

        class_name = getattr(vehicle, "__class__", None)
        if class_name is not None and hasattr(class_name, "__name__"):
            state["type"] = str(class_name.__name__).lower()

        return state if state else None

    @staticmethod
    def _extract_entity_id(vehicle: Any) -> str:
        for attr_name in ("name", "id"):
            value = getattr(vehicle, attr_name, None)
            if value is not None:
                return str(value)
        return f"pyid:{id(vehicle)}"

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        return None

    @staticmethod
    def _to_xy_array(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size < 2:
            return None
        return arr[:2]
