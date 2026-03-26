"""Training lifecycle protocol and implementations for planner backends.

This module defines the contract for granular training loops that generalize
beyond Stable-Baselines3 to support diverse backends (D4PG, SAC, custom RL, etc.).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from stable_baselines3.common.logger import configure


class BasePlannerLifecycle(Protocol):
    """Protocol for granular training lifecycle.

    Exposes the training loop as discrete steps:
    begin_training() → (act → observe → maybe_update)* → end_training()

    This allows external training loops to compose algorithms with curriculum,
    rule-based reward shaping, and other interventions without modifying the
    planner internals.
    """

    def begin_training(self, total_timesteps: int | None = None) -> None:
        """Initialize training state (buffers, counters, logging).

        Called once at the start of a training session before any steps.
        """
        ...

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Compute action from observation during training.

        Args:
            observation: Environmental state from env.step() or env.reset().
            deterministic: If True, disable exploration noise (typical for eval/inference).

        Returns:
            Action array, typically shape (action_dim,)
        """
        ...

    def observe_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_observation: np.ndarray,
    ) -> None:
        """Record transition to replay buffer or internal state.

        Args:
            observation: Starting state (before action).
            action: Action taken.
            reward: Immediate scalar reward (may be shaped externally).
            done: Episode termination indicator.
            next_observation: Resulting state (after action).
        """
        ...

    def maybe_update(self) -> None:
        """Conditionally update policy (gradient steps, buffer sampling, etc.).

        Called after each transition. Implementation decides whether to perform
        an actual update (e.g., TD3 updates every 2 steps, SAC updates every step).
        """
        ...

    def end_training(self) -> None:
        """Finalize training (cleanup, logging, final checkpoint).

        Called once at the end of a training session after all steps.
        """
        ...


class Td3Lifecycle:
    """Adapter that implements BasePlannerLifecycle for Stable-Baselines3 TD3.

    This wraps a Td3PlannerBackend's underlying SB3 model to expose the granular
    lifecycle interface. Allows external training loops to compose curriculum,
    rule-based rewards, and logging while maintaining SB3's internal scheduling.
    """

    def __init__(self, planner_backend: Any) -> None:
        """Initialize Td3Lifecycle.

        Args:
            planner_backend: Td3PlannerBackend instance with sb3_model attribute.
        """
        self.backend = planner_backend
        self.sb3_model = planner_backend.sb3_model
        self.replay_buffer = self.sb3_model.replay_buffer
        self.step_count = 0
        self.update_count = 0
        self.total_timesteps: int | None = None
        self.last_actor_loss = float("nan")
        self.last_critic_loss = float("nan")
        self.last_learning_rate = float("nan")

    def begin_training(self, total_timesteps: int | None = None) -> None:
        """Reset counters and initialize SB3 internals for manual loop."""
        self.step_count = 0
        self.update_count = 0
        self.total_timesteps = total_timesteps

        # Manual lifecycle does not go through model.learn(), so initialize
        # runtime state expected by OffPolicyAlgorithm.train().
        if not hasattr(self.sb3_model, "_logger"):
            self.sb3_model._logger = configure(folder=None, format_strings=[])
        if not hasattr(self.sb3_model, "_current_progress_remaining"):
            self.sb3_model._current_progress_remaining = 1.0
        if not hasattr(self.sb3_model, "num_timesteps"):
            self.sb3_model.num_timesteps = 0

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Action from policy, with optional deterministic mode for eval/inference."""
        action, _ = self.sb3_model.predict(observation, deterministic=deterministic)
        return action

    def observe_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_observation: np.ndarray,
    ) -> None:
        """Add transition to SB3 replay buffer."""
        # ReplayBuffer expects batched tensors with leading n_env axis.
        batched_obs = np.expand_dims(np.asarray(observation, dtype=np.float32), axis=0)
        batched_next_obs = np.expand_dims(np.asarray(next_observation, dtype=np.float32), axis=0)
        batched_action = np.expand_dims(np.asarray(action, dtype=np.float32), axis=0)
        batched_reward = np.asarray([reward], dtype=np.float32)
        batched_done = np.asarray([done], dtype=np.float32)

        self.replay_buffer.add(
            obs=batched_obs,
            action=batched_action,
            reward=batched_reward,
            done=batched_done,
            next_obs=batched_next_obs,
            infos=[{}],  # Empty info dict
        )

        # Keep SB3 timestep counter consistent when bypassing learn().
        self.sb3_model.num_timesteps += 1

    def maybe_update(self) -> None:
        """Conditionally update TD3 policy (every 2 steps, after learning_starts)."""
        # TD3 hyperparameters from SB3
        learning_starts = self.sb3_model.learning_starts
        train_freq = self.sb3_model.train_freq

        # Only update if we have enough data
        if self.replay_buffer.size() < learning_starts:
            self.step_count += 1
            return

        # TD3 update frequency (default: every 2 steps)
        if self.step_count % train_freq.frequency == 0:
            if self.total_timesteps and self.total_timesteps > 0:
                self.sb3_model._current_progress_remaining = max(
                    1.0 - (self.step_count / float(self.total_timesteps)),
                    0.0,
                )
            self.sb3_model.train(
                gradient_steps=int(self.sb3_model.gradient_steps),
                batch_size=int(self.sb3_model.batch_size),
            )

            # Extract latest metrics recorded by SB3 train() for external monitor.
            logger_values = self.sb3_model.logger.name_to_value
            self.last_actor_loss = float(logger_values.get("train/actor_loss", float("nan")))
            self.last_critic_loss = float(logger_values.get("train/critic_loss", float("nan")))
            self.last_learning_rate = float(
                logger_values.get("train/learning_rate", float("nan"))
            )
            self.update_count += 1

        self.step_count += 1

    def end_training(self) -> None:
        """Finalize training (SB3 has no explicit cleanup needed)."""
        pass
