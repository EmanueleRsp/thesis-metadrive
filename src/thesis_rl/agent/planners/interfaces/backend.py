from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from thesis_rl.agent.types import Transition

UpdateMetrics = dict[str, float | int]


class PlannerBackendProtocol(Protocol):
    n_envs: int

    def begin_training(
        self, chunk_timesteps: int, global_total_timesteps: int | None, global_steps_done: int
    ) -> None: ...

    def end_training(self) -> None: ...

    def act_train(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray: ...

    def act_train_batch(
        self, observations: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def observe_transition(self, transition: Transition) -> None: ...

    def observe_transition_batch(
        self,
        observations: np.ndarray,
        buffer_actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_observations: np.ndarray,
        infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        terminated: np.ndarray | None = None,
        truncated: np.ndarray | None = None,
    ) -> None: ...

    def maybe_update(
        self,
        collected_steps: int,
        step_count: int,
        global_total_timesteps: int | None,
        global_steps_done: int,
    ) -> UpdateMetrics: ...

    def to_buffer_action(self, env_action: np.ndarray) -> np.ndarray: ...

    def on_episode_end(self, indices: list[int] | np.ndarray | None = None) -> None: ...

    def predict(self, observation: Any, deterministic: bool = False): ...

    def save(self, checkpoint_path: str | Path) -> None: ...

    def set_env(self, env: Any) -> None: ...

    def save_replay_buffer(self, path: str | Path) -> bool: ...

    def load_replay_buffer(self, path: str | Path) -> bool: ...

    def replay_buffer_n_envs(self) -> int: ...
