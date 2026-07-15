from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class OffPolicyDebugLogger:
    def __init__(self, cfg: dict[str, Any] | None, *, algorithm: str, action_dim: int) -> None:
        cfg_dict = dict(cfg or {})
        self.enabled = bool(cfg_dict.get("enabled", False))
        self.algorithm = str(algorithm)
        self.action_dim = int(action_dim)
        self.log_interval_steps = max(int(cfg_dict.get("log_interval_steps", 20000)), 1)
        self.log_interval_update_calls = max(int(cfg_dict.get("log_interval_update_calls", 250)), 1)
        self.near_zero_eps = float(cfg_dict.get("near_zero_eps", 0.05))
        self.path = Path(str(cfg_dict.get("log_path", ""))) if self.enabled and cfg_dict.get("log_path") else None
        self._last_collect_flush_step = 0
        self._last_update_flush_call = 0
        self._update_calls = 0
        self._collect = self._new_collect_accumulator()
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write(
                {
                    "event": "metadata",
                    "algorithm": self.algorithm,
                    "action_dim": self.action_dim,
                    "log_interval_steps": self.log_interval_steps,
                    "log_interval_update_calls": self.log_interval_update_calls,
                    "near_zero_eps": self.near_zero_eps,
                }
            )

    def _new_collect_accumulator(self) -> dict[str, Any]:
        return {
            "transition_count": 0,
            "action_sum": np.zeros((self.action_dim,), dtype=np.float64),
            "action_sq_sum": np.zeros((self.action_dim,), dtype=np.float64),
            "action_abs_sum": np.zeros((self.action_dim,), dtype=np.float64),
            "reward_sum": 0.0,
            "reward_sq_sum": 0.0,
            "done_count": 0,
            "timeout_count": 0,
            "collision_count": 0,
            "out_of_road_count": 0,
            "success_count": 0,
            "route_completion_sum": 0.0,
            "warmup_transitions": 0,
        }

    def _write(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")

    def record_collect(
        self,
        *,
        total_steps: int,
        replay_size: int,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        warmup_active: bool,
    ) -> None:
        if not self.enabled:
            return
        act = np.asarray(actions, dtype=np.float32).reshape(-1, self.action_dim)
        rew = np.asarray(rewards, dtype=np.float32).reshape(-1)
        done_arr = np.asarray(dones, dtype=bool).reshape(-1)
        acc = self._collect
        acc["transition_count"] += int(act.shape[0])
        acc["action_sum"] += act.sum(axis=0, dtype=np.float64)
        acc["action_sq_sum"] += np.square(act, dtype=np.float64).sum(axis=0)
        acc["action_abs_sum"] += np.abs(act, dtype=np.float64).sum(axis=0)
        acc["reward_sum"] += float(rew.sum(dtype=np.float64))
        acc["reward_sq_sum"] += float(np.square(rew, dtype=np.float64).sum(dtype=np.float64))
        acc["done_count"] += int(done_arr.sum())
        if warmup_active:
            acc["warmup_transitions"] += int(act.shape[0])
        for idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            if bool(info.get("TimeLimit.truncated", False)):
                acc["timeout_count"] += 1
            if bool(info.get("collision", False) or info.get("crash_vehicle", False)):
                acc["collision_count"] += 1
            if bool(info.get("out_of_road", False)):
                acc["out_of_road_count"] += 1
            if bool(info.get("success", False) or info.get("arrive_dest", False)):
                acc["success_count"] += 1
            route_completion = info.get("route_completion", info.get("route_completion_ratio", 0.0))
            try:
                acc["route_completion_sum"] += float(route_completion)
            except Exception:
                acc["route_completion_sum"] += 0.0
            _ = idx
        if total_steps - self._last_collect_flush_step >= self.log_interval_steps:
            self.flush_collect(total_steps=total_steps, replay_size=replay_size)

    def flush_collect(self, *, total_steps: int, replay_size: int) -> None:
        if not self.enabled:
            return
        acc = self._collect
        n = int(acc["transition_count"])
        if n <= 0:
            return
        action_mean = acc["action_sum"] / max(n, 1)
        action_var = np.maximum(acc["action_sq_sum"] / max(n, 1) - np.square(action_mean), 0.0)
        action_std = np.sqrt(action_var)
        action_abs_mean = acc["action_abs_sum"] / max(n, 1)
        near_zero_frac = float(np.sum(action_abs_mean <= self.near_zero_eps) / max(self.action_dim, 1))
        reward_mean = acc["reward_sum"] / max(n, 1)
        reward_var = max(acc["reward_sq_sum"] / max(n, 1) - reward_mean**2, 0.0)
        payload = {
            "event": "collect",
            "algorithm": self.algorithm,
            "total_steps": int(total_steps),
            "replay_size": int(replay_size),
            "transition_count": n,
            "warmup_fraction": float(acc["warmup_transitions"] / max(n, 1)),
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
            "action_abs_mean": action_abs_mean.tolist(),
            "action_frac_near_zero_dimwise": near_zero_frac,
            "reward_mean": float(reward_mean),
            "reward_std": float(np.sqrt(reward_var)),
            "done_rate": float(acc["done_count"] / max(n, 1)),
            "timeout_rate": float(acc["timeout_count"] / max(n, 1)),
            "collision_rate": float(acc["collision_count"] / max(n, 1)),
            "out_of_road_rate": float(acc["out_of_road_count"] / max(n, 1)),
            "success_rate": float(acc["success_count"] / max(n, 1)),
            "route_completion_mean": float(acc["route_completion_sum"] / max(n, 1)),
        }
        self._write(payload)
        self._collect = self._new_collect_accumulator()
        self._last_collect_flush_step = int(total_steps)

    def record_update(self, *, total_steps: int, replay_size: int, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._update_calls += 1
        if (self._update_calls - self._last_update_flush_call) < self.log_interval_update_calls:
            return
        payload = {
            "event": "update",
            "algorithm": self.algorithm,
            "total_steps": int(total_steps),
            "replay_size": int(replay_size),
            "update_call": int(self._update_calls),
        }
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                payload[key] = value.item()
            else:
                payload[key] = value
        self._write(payload)
        self._last_update_flush_call = int(self._update_calls)

    def close(self, *, total_steps: int, replay_size: int) -> None:
        if not self.enabled:
            return
        self.flush_collect(total_steps=total_steps, replay_size=replay_size)
