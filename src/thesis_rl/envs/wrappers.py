from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from thesis_rl.reward.base import BaseRewardManager


class RuleRewardWrapper(gym.Wrapper):
    """Apply rulebook-based hybrid reward and enrich step info."""

    def __init__(
        self,
        env: gym.Env,
        reward_manager: BaseRewardManager,
        reward_mode: str = "hybrid",
        attach_info: bool = True,
        rule_margin_log_path: str | None = None,
    ) -> None:
        super().__init__(env)
        self.reward_manager = reward_manager
        self.reward_mode = str(reward_mode).lower()
        self.attach_info = bool(attach_info)
        self._rule_margin_log_path = Path(rule_margin_log_path) if rule_margin_log_path else None
        if self._rule_margin_log_path is not None:
            self._rule_margin_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)
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
            info_dict["reward_mode"] = self.reward_mode
            info_dict["rule_reward_vector"] = result.rule_reward_vector
            info_dict["rule_bounded_vector"] = result.rule_bounded_vector
            info_dict["rule_components"] = result.rule_components
            info_dict["rule_metadata"] = result.rule_metadata
            info_dict["scalar_rule_reward"] = result.scalar_rule_reward
            if result.rule_violation_vector is not None:
                info_dict["rule_violation_vector"] = result.rule_violation_vector

        return obs, selected_reward, terminated, truncated, info_dict

    def _select_reward(self, env_reward: float, result: Any) -> float:
        if self.reward_mode == "scalar_default":
            return float(env_reward)
        if self.reward_mode in {"rulebook", "scalar_rulebook"}:
            return float(result.scalar_rule_reward)
        if self.reward_mode in {"hybrid", "lexicographic"}:
            return float(result.final_reward)
        raise ValueError(
            "Unsupported reward mode. "
            f"Got mode='{self.reward_mode}', expected one of: "
            "scalar_default, rulebook, scalar_rulebook, hybrid, lexicographic."
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

        lane_centerline = self._extract_lane_centerline(ego_vehicle)
        if lane_centerline is not None:
            info_dict.setdefault("lane_centerline", lane_centerline)

        target_region = self._extract_target_region(ego_vehicle)
        if target_region is not None:
            info_dict.setdefault("target_region", target_region)

        target_point = self._extract_target_point(ego_vehicle)
        if target_point is not None:
            info_dict.setdefault("target_point", target_point)

        speed_limit = self._extract_speed_limit(ego_vehicle, lane_centerline)
        if speed_limit is not None:
            info_dict.setdefault("speed_limit", speed_limit)

        drivable_area = self._extract_drivable_area(base_env)
        if drivable_area is not None:
            info_dict.setdefault("drivable_area", drivable_area)

        opposite_carriageway = self._extract_opposite_carriageway(base_env, ego_vehicle)
        if opposite_carriageway is not None:
            info_dict.setdefault("opposite_carriageway", opposite_carriageway)

        if not isinstance(info_dict.get("neighbors"), list):
            info_dict["neighbors"] = self._extract_neighbors(base_env, ego_vehicle)

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
            "missing_lane_speed_limit",
            "Lane speed limit is unavailable; falling back to `vehicle.max_speed_km_h` for speed-limit rule.",
        )
        return self._safe_float(getattr(ego_vehicle, "max_speed_km_h", None))

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
        cfg = getattr(base_env, "config", None)
        if isinstance(cfg, Mapping):
            step_size = self._safe_float(cfg.get("physics_world_step_size"))
            decision_repeat = self._safe_float(cfg.get("decision_repeat"))
            if step_size is not None:
                if decision_repeat is not None and decision_repeat > 0:
                    return float(step_size * decision_repeat)
                return float(step_size)

            policy_frequency = self._safe_float(cfg.get("policy_frequency"))
            if policy_frequency is not None and policy_frequency > 0:
                return float(1.0 / policy_frequency)

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
            from shapely.ops import unary_union

            self._cached_drivable_area = unary_union(polygons)
            return self._cached_drivable_area
        except Exception:
            self._warn_once(
                "drivable_area_union_failed",
                "Failed to compute drivable area union from lane polygons.",
            )
            return None

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
            from shapely.ops import unary_union

            union_poly = unary_union(polygons)
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

    def _extract_neighbors(self, base_env: Any, ego_vehicle: Any) -> list[dict[str, Any]]:
        neighbors: list[dict[str, Any]] = []
        seen_ids: set[int] = set()

        agents = getattr(base_env, "agents", None)
        if isinstance(agents, Mapping):
            for vehicle in agents.values():
                state = self._vehicle_to_state(vehicle, ego_vehicle)
                if state is not None:
                    neighbors.append(state)
                    seen_ids.add(id(vehicle))

        engine = getattr(base_env, "engine", None)
        traffic_manager = getattr(engine, "traffic_manager", None)
        if traffic_manager is None:
            traffic_manager = getattr(base_env, "traffic_manager", None)

        iterator = self._iter_traffic_vehicles(traffic_manager)

        for vehicle in iterator:
            if id(vehicle) in seen_ids:
                continue
            state = self._vehicle_to_state(vehicle, ego_vehicle)
            if state is not None:
                neighbors.append(state)

        return neighbors

    def _iter_traffic_vehicles(self, traffic_manager: Any):
        if traffic_manager is None:
            return ()

        for attr_name in ("traffic_vehicles", "vehicles", "_traffic_vehicles"):
            traffic_vehicles = getattr(traffic_manager, attr_name, None)
            if attr_name == "_traffic_vehicles" and traffic_vehicles is not None:
                self._warn_once(
                    "private_traffic_vehicles",
                    "RuleRewardWrapper is using private traffic manager field `_traffic_vehicles` as fallback.",
                )
            if isinstance(traffic_vehicles, Mapping):
                return traffic_vehicles.values()
            if isinstance(traffic_vehicles, (list, tuple)):
                return traffic_vehicles
        return ()

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_fallbacks:
            return
        self._warned_fallbacks.add(key)
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

        self._logger.info(
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
        if info_dict.get("opposite_carriageway") is None:
            self._warn_once(
                "missing_opposite_carriageway",
                "Rulebook input `opposite_carriageway` is unavailable; wrong-way rule may become neutral.",
            )
        if info_dict.get("target_region") is None:
            self._warn_once(
                "missing_target_region",
                "Rulebook input `target_region` is unavailable; goal-progress rule may rely only on `target_point`.",
            )

        self._diagnostics_emitted = True

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
