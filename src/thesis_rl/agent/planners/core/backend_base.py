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

    def get_lifecycle(self) -> BasePlannerLifecycle:
        return self.lifecycle_cls(self)

    def set_env(self, env: Any) -> None:
        self.env = env
        self.n_envs = count_envs(env)

    def begin_training(self, chunk_timesteps: int, global_total_timesteps: int | None, global_steps_done: int) -> None:
        _ = (chunk_timesteps, global_total_timesteps, global_steps_done)

    def end_training(self) -> None:
        return None

    def on_episode_end(self, indices: list[int] | np.ndarray | None = None) -> None:
        _ = indices

    def save_replay_buffer(self, path: str | Path) -> bool:
        _ = path
        return False

    def load_replay_buffer(self, path: str | Path) -> bool:
        _ = path
        return False

    def replay_buffer_n_envs(self) -> int:
        return int(self.n_envs)
