from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from thesis_rl.reward.base import BaseRewardManager, RewardComputationResult
from thesis_rl.rulebook import RuleEvalInput, ScenicRulesEvaluator


class HybridRulebookRewardManager(BaseRewardManager):
    """Compute hybrid reward from env scalar reward and rulebook margins."""

    def __init__(
        self,
        *,
        evaluator: ScenicRulesEvaluator,
        a: float,
        c: float,
        lambda_env: float,
        lambda_rule: float,
        scales: Mapping[str, float],
        include_violation_vector: bool = False,
    ) -> None:
        self.evaluator = evaluator
        self.a = float(a)
        self.c = float(c)
        self.lambda_env = float(lambda_env)
        self.lambda_rule = float(lambda_rule)
        self.scales = {str(k): float(v) for k, v in scales.items()}
        self.include_violation_vector = bool(include_violation_vector)

        self._step = 0
        self._prev_ego_state: dict[str, Any] | None = None
        self._prev_neighbors: list[dict[str, Any]] | None = None
        self._rule_eval_counts: dict[str, int] = {}
        self._rule_saturation_counts: dict[str, int] = {}

    @classmethod
    def from_configs(
        cls,
        cfg_reward: DictConfig,
        cfg_rulebook: DictConfig,
    ) -> "HybridRulebookRewardManager":
        rulebook_cfg = OmegaConf.to_container(cfg_rulebook, resolve=True)
        if not isinstance(rulebook_cfg, dict):
            raise ValueError("rulebook config must resolve to a mapping")

        evaluator = ScenicRulesEvaluator.from_config(rulebook_cfg)
        scales_cfg = dict(cfg_reward.get("scales", {}))

        return cls(
            evaluator=evaluator,
            a=float(cfg_reward.get("a", 2.01)),
            c=float(cfg_reward.get("c", 30.0)),
            lambda_env=float(cfg_reward.get("lambda_env", 1.0)),
            lambda_rule=float(cfg_reward.get("lambda_rule", 0.2)),
            scales=scales_cfg,
            include_violation_vector=bool(cfg_reward.get("include_violation_vector", False)),
        )

    def reset(self) -> None:
        self._step = 0
        self._prev_ego_state = None
        self._prev_neighbors = None
        self._rule_eval_counts = {}
        self._rule_saturation_counts = {}

    def compute(self, env_reward: float, info: dict[str, Any]) -> RewardComputationResult:
        self._step += 1
        rule_eval_input = self._build_rule_eval_input(info)
        rule_vector = self.evaluator.evaluate(rule_eval_input)

        margins = rule_vector.values.astype(np.float32)
        n_rules = max(len(margins), 1)

        bounded_values: list[float] = []
        step_saturated_rules: list[str] = []
        for idx, name in enumerate(rule_vector.names):
            margin = float(margins[idx])
            scale = max(float(self.scales.get(name, 1.0)), 1e-6)
            rho = float(np.tanh(margin / scale))
            bounded_values.append(rho)

            self._rule_eval_counts[name] = int(self._rule_eval_counts.get(name, 0) + 1)
            if abs(rho) > 0.95:
                self._rule_saturation_counts[name] = int(
                    self._rule_saturation_counts.get(name, 0) + 1
                )
                step_saturated_rules.append(name)

        bounded_arr = np.asarray(bounded_values, dtype=np.float64)
        exponents = np.arange(n_rules, 0, -1, dtype=np.float64)
        sigmoid = 1.0 / (1.0 + np.exp(-self.c * bounded_arr))
        scalar_rule_reward = float(np.sum((self.a**exponents) * sigmoid + (bounded_arr / n_rules)))

        final_reward = float(self.lambda_env * float(env_reward) + self.lambda_rule * scalar_rule_reward)

        rule_components = {
            name: float(margins[idx])
            for idx, name in enumerate(rule_vector.names)
        }

        violation_vector = None
        if self.include_violation_vector:
            violation_vector = [float(max(0.0, -m)) for m in margins.tolist()]

        saturation_ratio_by_rule = {
            name: float(self._rule_saturation_counts.get(name, 0))
            / max(float(self._rule_eval_counts.get(name, 0)), 1.0)
            for name in rule_vector.names
        }

        self._prev_ego_state = dict(rule_eval_input.ego_state)
        self._prev_neighbors = [dict(item) for item in rule_eval_input.neighbors]

        return RewardComputationResult(
            final_reward=final_reward,
            scalar_rule_reward=scalar_rule_reward,
            rule_reward_vector=[float(x) for x in margins.tolist()],
            rule_bounded_vector=[float(x) for x in bounded_arr.tolist()],
            rule_components=rule_components,
            rule_metadata={
                **dict(rule_vector.metadata),
                "rule_names": list(rule_vector.names),
                "priorities": list(rule_vector.priorities),
                "step": self._step,
                "step_saturated_rules": step_saturated_rules,
                "saturation_ratio_by_rule": saturation_ratio_by_rule,
            },
            rule_violation_vector=violation_vector,
        )

    def _build_rule_eval_input(self, info: dict[str, Any]) -> RuleEvalInput:
        ego_state = self._extract_ego_state(info)
        neighbors = self._extract_neighbors(info)

        return RuleEvalInput(
            ego_state=ego_state,
            neighbors=neighbors,
            drivable_area=info.get("drivable_area"),
            opposite_carriageway=info.get("opposite_carriageway"),
            lane_centerline=info.get("lane_centerline"),
            target_region=info.get("target_region"),
            target_point=info.get("target_point"),
            speed_limit=self._extract_speed_limit(info),
            prev_ego_state=self._prev_ego_state,
            prev_neighbors=self._prev_neighbors,
            metadata={"timestamp": info.get("episode_length", self._step)},
        )

    @staticmethod
    def _extract_ego_state(info: dict[str, Any]) -> dict[str, Any]:
        if isinstance(info.get("ego_state"), dict):
            return dict(info["ego_state"])

        ego_state: dict[str, Any] = {}

        for key in ("position", "vehicle_position", "current_position"):
            if key in info:
                ego_state["position"] = info[key]
                break

        velocity_value = info.get("velocity")
        if velocity_value is not None:
            if isinstance(velocity_value, (int, float)):
                ego_state["speed"] = float(velocity_value)
            else:
                ego_state["velocity"] = velocity_value
        elif "velocity_km_h" in info:
            ego_state["speed"] = float(info["velocity_km_h"])

        acceleration_value = info.get("acceleration")
        if acceleration_value is not None:
            if isinstance(acceleration_value, (int, float)):
                ego_state["accel"] = float(acceleration_value)
            else:
                ego_state["acceleration"] = acceleration_value

        for key in ("steering", "steer"):
            if key in info:
                ego_state["steer"] = float(info[key])
                break
        if "heading" in info:
            ego_state["heading"] = float(info["heading"])
        if "yaw" in info:
            ego_state["yaw"] = float(info["yaw"])

        if "raw_action" in info and isinstance(info["raw_action"], (list, tuple)):
            raw_action = info["raw_action"]
            if len(raw_action) >= 1 and "steer" not in ego_state:
                ego_state["steer"] = float(raw_action[0])
            if len(raw_action) >= 2 and "accel" not in ego_state:
                ego_state["accel"] = float(raw_action[1])

        carsize = info.get("carsize")
        if isinstance(carsize, (list, tuple)) and len(carsize) >= 2:
            ego_state["width"] = float(carsize[0])
            ego_state["length"] = float(carsize[1])

        return ego_state

    @staticmethod
    def _extract_neighbors(info: dict[str, Any]) -> list[dict[str, Any]]:
        neighbors = info.get("neighbors")
        if isinstance(neighbors, list):
            return [dict(n) for n in neighbors if isinstance(n, Mapping)]

        for key in ("neighbor_states", "surrounding_vehicles", "vehicles"):
            candidate = info.get(key)
            if isinstance(candidate, list):
                return [dict(n) for n in candidate if isinstance(n, Mapping)]

        return []

    @staticmethod
    def _extract_speed_limit(info: dict[str, Any]) -> float | None:
        speed_limit = info.get("speed_limit")
        if speed_limit is not None:
            return float(speed_limit)

        speed_limit_kmh = info.get("speed_limit_km_h")
        if speed_limit_kmh is not None:
            return float(speed_limit_kmh)

        max_speed = info.get("max_speed_km_h")
        if max_speed is not None:
            return float(max_speed)

        max_speed_alt = info.get("max_speed")
        if max_speed_alt is not None:
            return float(max_speed_alt)
        return None
