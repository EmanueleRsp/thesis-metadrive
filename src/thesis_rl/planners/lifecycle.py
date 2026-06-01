"""Training lifecycle protocol and delegating implementations for planner backends."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from thesis_rl.agents.types import Transition


class BasePlannerLifecycle(Protocol):
    def begin_training(
        self,
        chunk_timesteps: int,
        global_total_timesteps: int | None = None,
        global_steps_done: int = 0,
    ) -> None: ...

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray: ...

    def observe_transition(self, transition: Transition) -> None: ...

    def maybe_update(self, collected_steps: int = 1) -> None: ...

    def to_buffer_action(self, env_action: np.ndarray) -> np.ndarray: ...

    def on_episode_end(self, indices: list[int] | np.ndarray | None = None) -> None: ...

    def end_training(self) -> None: ...


class _DelegatingLifecycle:
    def __init__(self, planner_backend: Any) -> None:
        self.backend = planner_backend
        self.step_count = 0
        self.update_count = 0
        self.gradient_step_count = 0
        self.last_actor_loss = float("nan")
        self.last_critic_loss = float("nan")
        self.last_learning_rate = float("nan")
        self.chunk_timesteps: int | None = None
        self.global_total_timesteps: int | None = None
        self.global_steps_done: int = 0

    def begin_training(
        self,
        chunk_timesteps: int,
        global_total_timesteps: int | None = None,
        global_steps_done: int = 0,
    ) -> None:
        self.step_count = 0
        self.update_count = 0
        self.gradient_step_count = 0
        self.last_actor_loss = float("nan")
        self.last_critic_loss = float("nan")
        self.last_learning_rate = float("nan")
        self.chunk_timesteps = int(chunk_timesteps)
        self.global_total_timesteps = global_total_timesteps
        self.global_steps_done = int(global_steps_done)
        self.backend.begin_training(
            chunk_timesteps=int(chunk_timesteps),
            global_total_timesteps=global_total_timesteps,
            global_steps_done=int(global_steps_done),
        )

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.backend.act_train(np.asarray(observation, dtype=np.float32), deterministic=deterministic)

    def act_batch(self, observations: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        return self.backend.act_train_batch(np.asarray(observations, dtype=np.float32), deterministic=deterministic)

    def observe_transition(self, transition: Transition) -> None:
        self.backend.observe_transition(transition)

    def observe_transition_batch(
        self,
        observations: np.ndarray,
        buffer_actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_observations: np.ndarray,
        infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> None:
        self.backend.observe_transition_batch(
            observations=np.asarray(observations, dtype=np.float32),
            buffer_actions=np.asarray(buffer_actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=bool),
            next_observations=np.asarray(next_observations, dtype=np.float32),
            infos=infos,
        )

    def maybe_update(self, collected_steps: int = 1) -> None:
        self.step_count += int(collected_steps)
        metrics = self.backend.maybe_update(
            collected_steps=int(collected_steps),
            step_count=int(self.step_count),
            global_total_timesteps=self.global_total_timesteps,
            global_steps_done=self.global_steps_done,
        )
        if not isinstance(metrics, dict):
            return
        self.last_actor_loss = float(metrics.get("actor_loss", self.last_actor_loss))
        self.last_critic_loss = float(metrics.get("critic_loss", self.last_critic_loss))
        self.last_learning_rate = float(metrics.get("learning_rate", self.last_learning_rate))
        self.update_count += int(metrics.get("update_calls", 0))
        self.gradient_step_count += int(metrics.get("gradient_steps", 0))

    def to_buffer_action(self, env_action: np.ndarray) -> np.ndarray:
        return self.backend.to_buffer_action(np.asarray(env_action, dtype=np.float32))

    def on_episode_end(self, indices: list[int] | np.ndarray | None = None) -> None:
        self.backend.on_episode_end(indices=indices)

    def end_training(self) -> None:
        self.backend.end_training()


class Td3Lifecycle(_DelegatingLifecycle):
    pass


class SacLifecycle(_DelegatingLifecycle):
    pass


class PpoLifecycle(_DelegatingLifecycle):
    pass
