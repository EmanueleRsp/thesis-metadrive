"""Training lifecycle protocol and delegating implementations for planner backends."""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol

import numpy as np

from thesis_rl.agent.types import Transition


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
        self.last_learning_potential = float("nan")
        self.learning_potential_values: list[float] = []
        self.update_timing_seconds: Counter[str] = Counter()
        self.chunk_timesteps: int | None = None
        self.global_total_timesteps: int | None = None
        self.global_steps_done: int = 0
        self._acl_collection_residuals: dict[tuple[int, int], list[float]] = dict(
            getattr(planner_backend, "_acl_collection_residuals", {})
        )
        if hasattr(planner_backend, "_acl_collection_residuals"):
            planner_backend._acl_collection_residuals = {}

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
        self.last_learning_potential = float("nan")
        self.learning_potential_values = []
        self.update_timing_seconds = Counter()
        self.chunk_timesteps = int(chunk_timesteps)
        self.global_total_timesteps = global_total_timesteps
        self.global_steps_done = int(global_steps_done)
        self.backend.begin_training(
            chunk_timesteps=int(chunk_timesteps),
            global_total_timesteps=global_total_timesteps,
            global_steps_done=int(global_steps_done),
        )

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.backend.act_train(
            np.asarray(observation, dtype=np.float32), deterministic=deterministic
        )

    def act_batch(
        self, observations: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.backend.act_train_batch(
            np.asarray(observations, dtype=np.float32), deterministic=deterministic
        )

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
        terminated: np.ndarray | None = None,
        truncated: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        observations_array = np.asarray(observations, dtype=np.float32)
        actions_array = np.asarray(buffer_actions, dtype=np.float32)
        rewards_array = np.asarray(rewards, dtype=np.float32)
        dones_array = np.asarray(dones, dtype=bool)
        next_observations_array = np.asarray(next_observations, dtype=np.float32)
        self.backend.observe_transition_batch(
            observations=observations_array,
            buffer_actions=actions_array,
            rewards=rewards_array,
            dones=dones_array,
            next_observations=next_observations_array,
            infos=infos,
            terminated=None if terminated is None else np.asarray(terminated, dtype=bool),
            truncated=None if truncated is None else np.asarray(truncated, dtype=bool),
            valid_mask=None if valid_mask is None else np.asarray(valid_mask, dtype=bool),
        )
        if valid_mask is not None and not np.all(valid_mask):
            return
        collection_residuals = self.backend.collection_learning_potential_batch(
            observations=observations_array,
            buffer_actions=actions_array,
            rewards=rewards_array,
            dones=dones_array,
            next_observations=next_observations_array,
            infos=infos,
        )
        if collection_residuals is not None:
            residual_array = np.asarray(collection_residuals, dtype=np.float64).reshape(-1)
            if residual_array.shape != rewards_array.shape:
                raise ValueError("Collection LP residuals must have one value per transition.")
            for index, info in enumerate(infos):
                slot_id = int(info.get("acl_slot_id", index))
                episode_id = info.get("acl_episode_id")
                if episode_id is None:
                    continue
                value = float(residual_array[index])
                if not np.isfinite(value):
                    raise ValueError("Collection LP residuals must be finite.")
                self._acl_collection_residuals.setdefault((slot_id, int(episode_id)), []).append(
                    value
                )

    def acl_learning_potential(self, slot_id: int, episode_id: int) -> float | None:
        """Return the mean positive-part collection residual for one ACL episode.

        DEC-006: `max(delta, 0)` instead of `|delta|`, restoring for TD3/SAC the
        ZPD hopelessness filter PPO already has via `max(GAE, 0)` (worse-than-
        expected outcomes contribute zero rather than inflating LP).
        """

        values = self._acl_collection_residuals.pop((int(slot_id), int(episode_id)), None)
        if values:
            return float(np.maximum(np.asarray(values, dtype=np.float64), 0.0).mean())
        backend_value = getattr(self.backend, "pop_acl_episode_learning_potential", None)
        if callable(backend_value):
            return backend_value(int(slot_id), int(episode_id))
        return None

    def acl_ready_learning_potentials(self) -> dict[tuple[int, int], float]:
        """Expose completed collection-time LP values without replay attribution."""

        ready: dict[tuple[int, int], float] = {
            key: float(np.maximum(np.asarray(values, dtype=np.float64), 0.0).mean())
            for key, values in self._acl_collection_residuals.items()
            if values
        }
        backend_ready = getattr(self.backend, "acl_ready_learning_potentials", None)
        if callable(backend_ready):
            ready.update(
                {(int(key[0]), int(key[1])): float(value) for key, value in backend_ready().items()}
            )
        return ready

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
        if metrics.get("learning_potential") is not None:
            self.last_learning_potential = float(metrics["learning_potential"])
            self.learning_potential_values.append(self.last_learning_potential)
        self.update_count += int(metrics.get("update_calls", 0))
        self.gradient_step_count += int(metrics.get("gradient_steps", 0))
        for metric_name, metric_value in metrics.items():
            if not metric_name.startswith("timing_") or not metric_name.endswith("_seconds"):
                continue
            value = float(metric_value)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Lifecycle timing metric {metric_name} must be finite and non-negative."
                )
            self.update_timing_seconds[metric_name.removeprefix("timing_")] += value

    def to_buffer_action(self, env_action: np.ndarray) -> np.ndarray:
        return self.backend.to_buffer_action(np.asarray(env_action, dtype=np.float32))

    def on_episode_end(self, indices: list[int] | np.ndarray | None = None) -> None:
        self.backend.on_episode_end(indices=indices)

    def close_previous_transition_as_data_abort(
        self, *, env_index: int, final_observation: np.ndarray
    ) -> None:
        close = getattr(self.backend, "close_previous_transition_as_data_abort", None)
        if not callable(close):
            raise RuntimeError("Planner backend does not support runtime data-abort boundaries.")
        close(env_index=int(env_index), final_observation=np.asarray(final_observation))

    def end_training(self) -> None:
        if self._acl_collection_residuals:
            pending = getattr(self.backend, "_acl_collection_residuals", {})
            for key, values in self._acl_collection_residuals.items():
                pending.setdefault(key, []).extend(values)
            self.backend._acl_collection_residuals = pending
        self.backend.end_training()


class Td3Lifecycle(_DelegatingLifecycle):
    pass


class SacLifecycle(_DelegatingLifecycle):
    pass


class PpoLifecycle(_DelegatingLifecycle):
    pass
