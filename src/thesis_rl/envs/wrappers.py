from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from thesis_rl.reward.base import BaseRewardManager


class RuleRewardWrapper(gym.Wrapper):
    """Apply rulebook-based hybrid reward and enrich step info."""

    def __init__(self, env: gym.Env, reward_manager: BaseRewardManager, attach_info: bool = True) -> None:
        super().__init__(env)
        self.reward_manager = reward_manager
        self.attach_info = bool(attach_info)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_manager.reset()
        return obs, info

    def step(self, action: Any):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        info_dict = dict(info)
        self._enrich_runtime_info(info_dict)

        result = self.reward_manager.compute(float(env_reward), info_dict)

        if self.attach_info:
            info_dict["rule_reward_vector"] = result.rule_reward_vector
            info_dict["rule_bounded_vector"] = result.rule_bounded_vector
            info_dict["rule_components"] = result.rule_components
            info_dict["rule_metadata"] = result.rule_metadata
            info_dict["scalar_rule_reward"] = result.scalar_rule_reward
            if result.rule_violation_vector is not None:
                info_dict["rule_violation_vector"] = result.rule_violation_vector

        return obs, result.final_reward, terminated, truncated, info_dict

    def _enrich_runtime_info(self, info_dict: dict[str, Any]) -> None:
        base_env = getattr(self.env, "unwrapped", self.env)
        ego_vehicle = self._extract_ego_vehicle(base_env)
        if ego_vehicle is None:
            return

        ego_state = info_dict.get("ego_state")
        if not isinstance(ego_state, Mapping):
            ego_state = {}
        else:
            ego_state = dict(ego_state)

        position = getattr(ego_vehicle, "position", None)
        if position is not None:
            ego_state.setdefault("position", position)

        velocity = getattr(ego_vehicle, "velocity", None)
        if velocity is not None:
            ego_state.setdefault("velocity", velocity)

        speed = self._safe_float(getattr(ego_vehicle, "speed", None))
        if speed is not None:
            ego_state.setdefault("speed", speed)

        yaw = self._safe_float(getattr(ego_vehicle, "heading_theta", None))
        if yaw is not None:
            ego_state.setdefault("yaw", yaw)

        steer = self._safe_float(getattr(ego_vehicle, "steering", None))
        if steer is not None:
            ego_state.setdefault("steer", steer)

        accel = self._safe_float(getattr(ego_vehicle, "throttle_brake", None))
        if accel is not None:
            ego_state.setdefault("accel", accel)

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

        target_point = self._extract_target_point(ego_vehicle)
        if target_point is not None:
            info_dict.setdefault("target_point", target_point)

        speed_limit = self._safe_float(getattr(ego_vehicle, "max_speed_km_h", None))
        if speed_limit is not None:
            info_dict.setdefault("speed_limit", speed_limit)

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

        traffic_vehicles = getattr(traffic_manager, "_traffic_vehicles", None)
        if isinstance(traffic_vehicles, Mapping):
            iterator = traffic_vehicles.values()
        elif isinstance(traffic_vehicles, (list, tuple)):
            iterator = traffic_vehicles
        else:
            iterator = ()

        for vehicle in iterator:
            if id(vehicle) in seen_ids:
                continue
            state = self._vehicle_to_state(vehicle, ego_vehicle)
            if state is not None:
                neighbors.append(state)

        return neighbors

    @classmethod
    def _vehicle_to_state(cls, vehicle: Any, ego_vehicle: Any) -> dict[str, Any] | None:
        if vehicle is None or vehicle is ego_vehicle:
            return None

        state: dict[str, Any] = {}

        position = getattr(vehicle, "position", None)
        if position is not None:
            state["position"] = position

        velocity = getattr(vehicle, "velocity", None)
        if velocity is not None:
            state["velocity"] = velocity

        speed = cls._safe_float(getattr(vehicle, "speed", None))
        if speed is not None:
            state["speed"] = speed

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
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        return None
