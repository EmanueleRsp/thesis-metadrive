"""Training lifecycle protocol and implementations for planner backends.

This module defines the contract for granular training loops that generalize
beyond Stable-Baselines3 to support diverse backends (D4PG, SAC, custom RL, etc.).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from stable_baselines3.common.logger import configure

from thesis_rl.agents.types import Transition


class BasePlannerLifecycle(Protocol):
    """Protocol for granular training lifecycle.

    Exposes the training loop as discrete steps:
    begin_training() → (act → observe → maybe_update)* → end_training()

    This allows external training loops to compose algorithms with curriculum,
    rule-based reward shaping, and other interventions without modifying the
    planner internals.
    """

    def begin_training(self, 
        chunk_timesteps: int,
        global_total_timesteps: int | None = None,
        global_steps_done: int = 0, 
    ) -> None:
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

    def observe_transition(self, transition: Transition) -> None:
        """Record transition to replay buffer or internal state.

        Args:
            transition: Full transition collected from the environment step.
        """
        ...

    def maybe_update(self) -> None:
        """Conditionally update policy (gradient steps, buffer sampling, etc.).

        Called after each transition. Implementation decides whether to perform
        an actual update (e.g., TD3 updates every 2 steps, SAC updates every step).
        """
        ...

    def to_buffer_action(self, env_action: np.ndarray) -> np.ndarray:
        """Convert an environment action into the replay-buffer action representation."""
        ...

    def on_episode_end(self) -> None:
        """Handle end-of-episode lifecycle state, such as resetting exploration noise."""
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
        self.chunk_timesteps: int | None = None
        self.global_total_timesteps: int | None = None
        self.global_steps_done: int = 0
        self.num_timesteps: int = 0
        self.last_actor_loss = float("nan")
        self.last_critic_loss = float("nan")
        self.last_learning_rate = float("nan")
        self._policy: Any | None = None
        self._action_noise: Any | None = None

    def _validate_sb3_components(self) -> None:
        policy = getattr(self.sb3_model, "policy", None)
        if policy is None:
            raise AttributeError("TD3 model is missing `policy`; cannot run lifecycle training.")
        if not hasattr(policy, "scale_action") or not hasattr(policy, "unscale_action"):
            raise AttributeError(
                "TD3 policy must expose `scale_action` and `unscale_action` for replay/env action mapping."
            )

        action_noise = getattr(self.sb3_model, "action_noise", None)
        if action_noise is None:
            raise ValueError(
                "TD3 lifecycle requires configured `action_noise`. "
                "Set `action_noise_type` in planner config."
            )
        if not callable(action_noise):
            raise TypeError("TD3 `action_noise` must be callable.")
        if not hasattr(action_noise, "reset"):
            raise AttributeError("TD3 `action_noise` must expose `reset()`.")

        self._policy = policy
        self._action_noise = action_noise

    def _validate_train_freq(self) -> None:
        train_freq = getattr(self.sb3_model, "train_freq", None)
        if train_freq is None:
            raise AttributeError("TD3 model is missing `train_freq`.")
        if not hasattr(train_freq, "frequency") or not hasattr(train_freq, "unit"):
            raise AttributeError("TD3 `train_freq` must expose `frequency` and `unit`.")
        if int(train_freq.frequency) <= 0:
            raise ValueError("TD3 `train_freq.frequency` must be > 0.")

        unit_name = str(train_freq.unit).lower()
        if "step" not in unit_name:
            raise ValueError(
                "Current TD3 lifecycle supports only step-based `train_freq`. "
                f"Got unit={train_freq.unit}."
            )

    def begin_training(
        self, 
        chunk_timesteps: int,
        global_total_timesteps: int | None = None,
        global_steps_done: int = 0,
    ) -> None:
        """Reset counters and initialize SB3 internals for manual loop."""

        self.step_count = 0
        self.update_count = 0
        self.chunk_timesteps = chunk_timesteps
        self.global_total_timesteps = global_total_timesteps
        self.global_steps_done = global_steps_done

        # Manual lifecycle does not go through model.learn(), so initialize
        # runtime state expected by OffPolicyAlgorithm.train().
        if not hasattr(self.sb3_model, "_logger"):
            self.sb3_model._logger = configure(folder=None, format_strings=[])
        if not hasattr(self.sb3_model, "_current_progress_remaining"):
            self.sb3_model._current_progress_remaining = 1.0
        if not hasattr(self.sb3_model, "num_timesteps"):
            self.sb3_model.num_timesteps = 0
        self.num_timesteps = int(self.sb3_model.num_timesteps)
        self._validate_sb3_components()
        self._validate_train_freq()
        self.on_episode_end()

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Action from policy, with TD3-style exploration during training."""
        action_space = self.sb3_model.action_space
        if self._policy is None or self._action_noise is None:
            raise RuntimeError("TD3 lifecycle was not initialized correctly; call begin_training() first.")
        policy = self._policy
        action_noise = self._action_noise
        policy.set_training_mode(False)

        def _scale(env_action: np.ndarray) -> np.ndarray:
            return np.asarray(policy.scale_action(env_action), dtype=np.float32)

        def _unscale(buffer_action: np.ndarray) -> np.ndarray:
            return np.asarray(policy.unscale_action(buffer_action), dtype=np.float32)

        if deterministic:
            action, _ = self.sb3_model.predict(observation, deterministic=True)
            env_action = np.asarray(action, dtype=np.float32)
            return env_action

        if self.num_timesteps < self.sb3_model.learning_starts:
            env_action = np.asarray(action_space.sample(), dtype=np.float32)
            return env_action

        # Align with SB3 off-policy rollout path:
        # predict -> scale -> noise -> clip scaled -> unscale for env step.
        action, _ = self.sb3_model.predict(observation, deterministic=False)
        buffer_action = np.asarray(_scale(np.asarray(action, dtype=np.float32)), dtype=np.float32)

        noise = np.asarray(action_noise(), dtype=np.float32)
        buffer_action = buffer_action + noise

        buffer_action = np.clip(buffer_action, -1.0, 1.0).astype(np.float32)
        env_action = _unscale(buffer_action)
        env_action = np.clip(env_action, action_space.low, action_space.high).astype(np.float32)
        return env_action

    def observe_transition(self, transition: Transition) -> None:
        """Add transition to SB3 replay buffer."""
        # ReplayBuffer expects batched tensors with leading n_env axis.
        batched_obs = np.expand_dims(np.asarray(transition.observation, dtype=np.float32), axis=0)
        batched_next_obs = np.expand_dims(
            np.asarray(transition.next_observation, dtype=np.float32),
            axis=0,
        )
        batched_action = np.expand_dims(np.asarray(transition.buffer_action, dtype=np.float32), axis=0)
        batched_reward = np.asarray([transition.scalar_reward], dtype=np.float32)
        batched_done = np.asarray([transition.terminated or transition.truncated], dtype=np.float32)
        replay_info = dict(transition.info)
        replay_info["TimeLimit.truncated"] = bool(transition.truncated and not transition.terminated)
        if transition.terminal_observation is not None:
            replay_info["terminal_observation"] = np.asarray(
                transition.terminal_observation,
                dtype=np.float32,
            )

        self.replay_buffer.add(
            obs=batched_obs,
            action=batched_action,
            reward=batched_reward,
            done=batched_done,
            next_obs=batched_next_obs,
            infos=[replay_info],
        )

        # Keep SB3 timestep counter consistent when bypassing learn().
        self.num_timesteps += 1
        self.sb3_model.num_timesteps += 1

    def to_buffer_action(self, env_action: np.ndarray) -> np.ndarray:
        """Map env-space action to SB3 TD3 replay-buffer action representation."""
        if self._policy is None:
            raise RuntimeError("TD3 lifecycle policy is unavailable; call begin_training() first.")

        action = np.asarray(env_action, dtype=np.float32)
        scaled = self._policy.scale_action(action)
        return np.clip(np.asarray(scaled, dtype=np.float32), -1.0, 1.0)

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
            if self.global_total_timesteps and self.global_total_timesteps > 0:
                global_step = self.global_steps_done + self.step_count
                self.sb3_model._current_progress_remaining = max(
                    1.0 - (global_step / float(self.global_total_timesteps)),
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

    def on_episode_end(self) -> None:
        """Reset TD3 action noise between episodes when configured."""
        if self._action_noise is None:
            raise RuntimeError("TD3 lifecycle action noise is unavailable; call begin_training() first.")
        self._action_noise.reset()

    def end_training(self) -> None:
        """Finalize training (SB3 has no explicit cleanup needed)."""
        pass
