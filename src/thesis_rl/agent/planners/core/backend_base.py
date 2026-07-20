from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from thesis_rl.agent.planners.core.utils import count_envs, resolve_device, to_plain_dict
from thesis_rl.agent.planners.interfaces.backend import PlannerBackendProtocol
from thesis_rl.agent.planners.core.types import TrainState
from thesis_rl.agent.planners.core.lifecycle import BasePlannerLifecycle


class BasePlannerBackend(PlannerBackendProtocol):
    lifecycle_cls: type[BasePlannerLifecycle]

    def __init__(self, env: Any, cfg_planner: Any, device: str = "auto") -> None:
        self.env = env
        self.cfg_planner = to_plain_dict(cfg_planner)
        self.device = resolve_device(device)
        self.state = TrainState()
        self.n_envs = count_envs(env)
        self._acl_collection_residuals: dict[tuple[int, int], list[float]] = {}

    def get_lifecycle(self) -> BasePlannerLifecycle:
        return self.lifecycle_cls(self)

    def set_env(self, env: Any) -> None:
        self.env = env
        self.n_envs = count_envs(env)

    def begin_training(
        self, chunk_timesteps: int, global_total_timesteps: int | None, global_steps_done: int
    ) -> None:
        _ = (chunk_timesteps, global_total_timesteps, global_steps_done)

    def end_training(self) -> None:
        return None

    def on_episode_end(self, indices: list[int] | np.ndarray | None = None) -> None:
        _ = indices

    def collection_learning_potential_batch(
        self,
        observations: np.ndarray,
        buffer_actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_observations: np.ndarray,
        infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> np.ndarray | None:
        """Return collection-time residuals, never residuals sampled from replay.

        Backends that do not implement a collection-time Bellman/value snapshot
        return ``None``.  The ACL driver must not substitute an aggregate replay
        metric in that case.
        """

        del observations, buffer_actions, rewards, dones, next_observations, infos
        return None

    def acl_ready_learning_potentials(self) -> dict[tuple[int, int], float]:
        """Return finalized episode LPs retained by a PPO-style backend."""

        return {}

    def save_replay_buffer(self, path: str | Path) -> bool:
        _ = path
        return False

    def load_replay_buffer(self, path: str | Path) -> bool:
        _ = path
        return False

    def replay_buffer_n_envs(self) -> int:
        replay_buffer = getattr(self, "replay_buffer", None)
        if replay_buffer is not None and hasattr(replay_buffer, "n_envs"):
            return int(getattr(replay_buffer, "n_envs"))
        return int(self.n_envs)
