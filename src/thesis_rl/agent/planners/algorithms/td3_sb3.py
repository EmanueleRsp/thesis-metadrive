from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from thesis_rl.agent.types import Transition
from thesis_rl.agent.transition_boundary import normalize_vector_transition_boundary
from thesis_rl.agent.planners.core.backend_base import BasePlannerBackend
from thesis_rl.agent.planners.core.utils import normalize_checkpoint_path, to_plain_dict
from thesis_rl.agent.planners.core.lifecycle import Td3Lifecycle
from thesis_rl.sb3_extensions import (
    build_sb3_specs_from_configs,
    resolve_transition_replay_config,
)

if TYPE_CHECKING:
    from stable_baselines3 import TD3


def _require_sb3_td3():
    try:
        from stable_baselines3 import TD3
        from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
        from stable_baselines3.common.vec_env import VecEnv
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "The `td3_sb3` backend requires `stable-baselines3` to be installed. "
            "Run `uv sync` to install project dependencies."
        ) from exc
    return TD3, NormalActionNoise, VectorizedActionNoise, VecEnv


class Sb3Td3PlannerBackend(BasePlannerBackend):
    lifecycle_cls = Td3Lifecycle

    def __init__(
        self,
        env: Any,
        cfg_planner: Any,
        model: "TD3",
        device: str = "auto",
    ) -> None:
        super().__init__(env=env, cfg_planner=cfg_planner, device=device)
        self.model = model
        self.sb3_model = model
        self.replay_buffer = self.model.replay_buffer

        self.num_timesteps = 0
        self.collected_transitions = 0
        self.last_actor_loss = float("nan")
        self.last_critic_loss = float("nan")
        self.last_learning_rate = float("nan")

        self._policy: Any | None = None
        self._action_noise: Any | None = None
        self._last_buffer_actions: np.ndarray | None = None
        self._diagnostic_mixed_precision = bool(
            self.cfg_planner.get("diagnostic_mixed_precision", False)
        )

    def _sync_action_noise(self) -> None:
        self.model.action_noise = self._build_action_noise(self.env, self.cfg_planner)

    @staticmethod
    def _build_action_noise(env: Any, cfg_planner: dict[str, Any]):
        _TD3, NormalActionNoise, VectorizedActionNoise, VecEnv = _require_sb3_td3()
        del _TD3

        noise_type = str(cfg_planner.get("action_noise_type", "none")).lower()
        if noise_type == "none":
            return None
        if noise_type != "normal":
            raise ValueError(
                f"Unsupported action_noise_type '{cfg_planner.get('action_noise_type')}'. "
                "Currently supported: none, normal"
            )

        action_space = getattr(env, "action_space", None)
        if action_space is None or not hasattr(action_space, "shape"):
            raise ValueError(
                "TD3 normal action noise requires an environment with a shaped action space."
            )

        action_dim = int(np.prod(action_space.shape))
        sigma_value = float(cfg_planner.get("action_noise_sigma", 0.1))
        mean_value = float(cfg_planner.get("action_noise_mean", 0.0))

        mean = np.full(action_dim, mean_value, dtype=np.float32)
        sigma = np.full(action_dim, sigma_value, dtype=np.float32)
        noise = NormalActionNoise(mean=mean, sigma=sigma)
        n_envs = int(env.num_envs) if isinstance(env, VecEnv) else 1
        if n_envs > 1:
            return VectorizedActionNoise(noise, n_envs=n_envs)
        return noise

    @staticmethod
    def _resolve_gradient_steps(env: Any, cfg_planner: dict[str, Any]) -> int:
        _TD3, _NormalActionNoise, _VectorizedActionNoise, VecEnv = _require_sb3_td3()
        del _TD3, _NormalActionNoise, _VectorizedActionNoise

        gradient_steps_cfg = cfg_planner.get("gradient_steps", "auto")
        if isinstance(gradient_steps_cfg, str) and gradient_steps_cfg.strip().lower() == "auto":
            train_freq = int(cfg_planner.get("train_freq", 1))
            n_envs = int(env.num_envs) if isinstance(env, VecEnv) else 1
            return max(train_freq * n_envs, 1)
        return int(gradient_steps_cfg)

    def _validate_sb3_components(self) -> None:
        policy = getattr(self.model, "policy", None)
        if policy is None:
            raise AttributeError(
                "SB3 TD3 model is missing `policy`; cannot run lifecycle training."
            )
        if not hasattr(policy, "scale_action") or not hasattr(policy, "unscale_action"):
            raise AttributeError(
                "SB3 TD3 policy must expose `scale_action` and `unscale_action` "
                "for replay/env action mapping."
            )

        action_noise = getattr(self.model, "action_noise", None)
        if action_noise is None:
            raise ValueError(
                "SB3 TD3 lifecycle requires configured `action_noise`. "
                "Set `action_noise_type=normal` in planner config."
            )
        if not callable(action_noise):
            raise TypeError("SB3 TD3 `action_noise` must be callable.")
        if not hasattr(action_noise, "reset"):
            raise AttributeError("SB3 TD3 `action_noise` must expose `reset()`.")

        self._policy = policy
        self._action_noise = action_noise

    def _validate_train_freq(self) -> None:
        train_freq = getattr(self.model, "train_freq", None)
        if train_freq is None:
            raise AttributeError("SB3 TD3 model is missing `train_freq`.")
        if not hasattr(train_freq, "frequency") or not hasattr(train_freq, "unit"):
            raise AttributeError("SB3 TD3 `train_freq` must expose `frequency` and `unit`.")
        if int(train_freq.frequency) <= 0:
            raise ValueError("SB3 TD3 `train_freq.frequency` must be > 0.")

        unit_name = str(train_freq.unit).lower()
        if "step" not in unit_name:
            raise ValueError(
                "Current SB3 TD3 lifecycle supports only step-based `train_freq`. "
                f"Got unit={train_freq.unit}."
            )

    @classmethod
    def build(
        cls,
        env: Any,
        cfg_planner: Any,
        cfg_encoder: Any | None = None,
        cfg_decoder: Any | None = None,
        cfg_obs: Any | None = None,
        device: str = "auto",
        seed: int | None = None,
    ) -> "Sb3Td3PlannerBackend":
        TD3, _NormalActionNoise, _VectorizedActionNoise, _VecEnv = _require_sb3_td3()
        del _NormalActionNoise, _VectorizedActionNoise, _VecEnv

        planner_cfg = to_plain_dict(cfg_planner)
        policy_spec, algorithm_spec = build_sb3_specs_from_configs(
            "td3_sb3",
            planner_cfg,
            encoder_cfg=to_plain_dict(cfg_encoder),
            decoder_cfg=to_plain_dict(cfg_decoder),
            obs_cfg=to_plain_dict(cfg_obs),
        )
        model_kwargs = algorithm_spec.merged_algorithm_kwargs()
        transition_replay = resolve_transition_replay_config(
            planner_cfg.get("transition_replay"),
            algorithm_name="td3_sb3",
        )
        replay_buffer_kwargs = (
            dict(algorithm_spec.replay_buffer_kwargs)
            if algorithm_spec.replay_buffer_kwargs
            else None
        )
        replay_buffer_class = algorithm_spec.replay_buffer_class
        if transition_replay.prioritized:
            from thesis_rl.sb3_extensions.replay.prioritized import PrioritizedNStepReplayBuffer

            replay_buffer_class = PrioritizedNStepReplayBuffer
            per_raw = dict(planner_cfg.get("transition_replay", {}).get("per", {}))
            per_cfg = {
                key: per_raw[key]
                for key in ("alpha", "beta_initial", "beta_final", "beta_anneal_steps", "epsilon")
                if key in per_raw
            }
            replay_buffer_kwargs = {
                **(replay_buffer_kwargs or {}),
                **per_cfg,
                "n_steps": transition_replay.n_steps,
                "gamma": float(planner_cfg.get("gamma", 0.99)),
                "seed": seed,
            }

        model = TD3(
            policy=policy_spec.policy,
            env=env,
            learning_starts=int(planner_cfg.get("learning_starts", 10000)),
            batch_size=int(planner_cfg.get("batch_size", 2048)),
            buffer_size=int(planner_cfg.get("buffer_size", 300000)),
            train_freq=int(planner_cfg.get("train_freq", 1)),
            gradient_steps=cls._resolve_gradient_steps(env, planner_cfg),
            learning_rate=float(planner_cfg.get("learning_rate", 3e-4)),
            gamma=float(planner_cfg.get("gamma", 0.99)),
            n_steps=transition_replay.n_steps if transition_replay.enabled else 1,
            optimize_memory_usage=transition_replay.optimize_memory_usage,
            tau=float(planner_cfg.get("tau", 0.005)),
            action_noise=cls._build_action_noise(env, planner_cfg),
            policy_kwargs=policy_spec.policy_kwargs,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=int(planner_cfg.get("verbose", 0)),
            device=device,
            seed=seed,
            **model_kwargs,
        )
        return cls(env=env, cfg_planner=planner_cfg, model=model, device=device)

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        env: Any,
        device: str = "auto",
        cfg_planner: Any | None = None,
        cfg_encoder: Any | None = None,
        cfg_decoder: Any | None = None,
        cfg_obs: Any | None = None,
    ) -> "Sb3Td3PlannerBackend":
        del cfg_encoder, cfg_decoder, cfg_obs
        TD3, _NormalActionNoise, _VectorizedActionNoise, _VecEnv = _require_sb3_td3()
        del _NormalActionNoise, _VectorizedActionNoise, _VecEnv

        model = TD3.load(str(normalize_checkpoint_path(checkpoint_path)), env=env, device=device)
        resolved_cfg = {} if cfg_planner is None else to_plain_dict(cfg_planner)
        return cls(env=env, cfg_planner=resolved_cfg, model=model, device=device)

    def begin_training(
        self,
        chunk_timesteps: int,
        global_total_timesteps: int | None,
        global_steps_done: int,
    ) -> None:
        del chunk_timesteps
        _require_sb3_td3()
        from stable_baselines3.common.logger import configure

        if not hasattr(self.model, "_logger"):
            self.model._logger = configure(folder=None, format_strings=[])
        if not hasattr(self.model, "_current_progress_remaining"):
            self.model._current_progress_remaining = 1.0
        if not hasattr(self.model, "num_timesteps"):
            self.model.num_timesteps = 0

        self.num_timesteps = int(self.model.num_timesteps)
        self.collected_transitions = 0
        self._sync_action_noise()
        self._validate_sb3_components()
        self._validate_train_freq()
        self.on_episode_end()
        self._global_total_timesteps = global_total_timesteps
        self._global_steps_done = int(global_steps_done)

    def end_training(self) -> None:
        return None

    def act_train(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self._policy is None or self._action_noise is None:
            raise RuntimeError("SB3 TD3 backend not initialized; call begin_training() first.")

        action_space = self.model.action_space
        if deterministic:
            action, _ = self.model.predict(observation, deterministic=True)
            env_action = np.asarray(action, dtype=np.float32)
            self._last_buffer_actions = np.asarray(
                self._policy.scale_action(env_action),
                dtype=np.float32,
            )[None, ...]
            return env_action

        if self.num_timesteps < int(self.model.learning_starts):
            env_action = np.asarray(action_space.sample(), dtype=np.float32)
            self._last_buffer_actions = np.asarray(
                self._policy.scale_action(env_action),
                dtype=np.float32,
            )[None, ...]
            return np.clip(env_action, action_space.low, action_space.high).astype(np.float32)

        obs_batch = np.expand_dims(np.asarray(observation, dtype=np.float32), axis=0)
        self.model._last_obs = obs_batch
        env_actions, buffer_actions = self.model._sample_action(
            int(self.model.learning_starts),
            self._action_noise,
            1,
        )
        self._last_buffer_actions = np.asarray(buffer_actions, dtype=np.float32)
        return np.asarray(env_actions[0], dtype=np.float32)

    def act_train_batch(
        self,
        observations: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._policy is None or self._action_noise is None:
            raise RuntimeError("SB3 TD3 backend not initialized; call begin_training() first.")

        obs_batch = np.asarray(observations, dtype=np.float32)
        n_envs = int(obs_batch.shape[0])
        self._policy.set_training_mode(False)

        if deterministic:
            env_actions, _ = self.model.predict(obs_batch, deterministic=True)
            env_actions = np.asarray(env_actions, dtype=np.float32)
            buffer_actions = np.asarray(self._policy.scale_action(env_actions), dtype=np.float32)
            buffer_actions = np.clip(buffer_actions, -1.0, 1.0).astype(np.float32)
            self._last_buffer_actions = buffer_actions
            return env_actions, buffer_actions

        self.model._last_obs = obs_batch
        env_actions, buffer_actions = self.model._sample_action(
            int(self.model.learning_starts),
            self._action_noise,
            n_envs,
        )
        self._last_buffer_actions = np.asarray(buffer_actions, dtype=np.float32)
        return (
            np.asarray(env_actions, dtype=np.float32),
            np.asarray(buffer_actions, dtype=np.float32),
        )

    def observe_transition(self, transition: Transition) -> None:
        batched_obs = np.expand_dims(np.asarray(transition.observation, dtype=np.float32), axis=0)
        resolved_next_obs = np.asarray(
            transition.terminal_observation
            if transition.terminal_observation is not None
            else transition.next_observation,
            dtype=np.float32,
        )
        batched_next_obs = np.expand_dims(resolved_next_obs, axis=0)
        batched_action = np.expand_dims(
            np.asarray(transition.buffer_action, dtype=np.float32),
            axis=0,
        )
        batched_reward = np.asarray([transition.scalar_reward], dtype=np.float32)
        batched_done = np.asarray([transition.terminated or transition.truncated], dtype=np.float32)
        replay_info = dict(transition.info)
        replay_info["TimeLimit.truncated"] = bool(
            transition.truncated and not transition.terminated
        )
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
        self.num_timesteps += 1
        self.model.num_timesteps += 1
        self.collected_transitions += 1

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
        obs_batch = np.asarray(observations, dtype=np.float32)
        terminated_batch, truncated_batch, next_obs_batch = normalize_vector_transition_boundary(
            dones=dones,
            infos=infos,
            next_observations=next_observations,
            terminated=terminated,
            truncated=truncated,
        )
        action_batch = np.asarray(buffer_actions, dtype=np.float32)
        reward_batch = np.asarray(rewards, dtype=np.float32)
        done_batch = np.asarray(terminated_batch | truncated_batch, dtype=np.float32)
        replay_infos: list[dict[str, Any]] = []

        for idx, info in enumerate(infos):
            replay_info = dict(info)
            replay_info["TimeLimit.truncated"] = bool(
                truncated_batch[idx] and not terminated_batch[idx]
            )
            terminal_observation = replay_info.get("terminal_observation")
            if bool(done_batch[idx]) and terminal_observation is not None:
                terminal_obs = np.asarray(terminal_observation, dtype=np.float32)
                replay_info["terminal_observation"] = terminal_obs
                next_obs_batch[idx] = terminal_obs
            replay_infos.append(replay_info)

        add_kwargs: dict[str, Any] = {}
        if valid_mask is not None and not np.all(valid_mask):
            if not hasattr(self.replay_buffer, "valid_transitions"):
                raise RuntimeError(
                    "Sparse runtime data-abort collection requires the owned sparse replay buffer."
                )
            add_kwargs["valid_mask"] = np.asarray(valid_mask, dtype=bool)
        self.replay_buffer.add(
            obs=obs_batch,
            action=action_batch,
            reward=reward_batch,
            done=done_batch,
            next_obs=next_obs_batch,
            infos=replay_infos,
            **add_kwargs,
        )

        collected = int(obs_batch.shape[0])
        self.num_timesteps += collected
        self.model.num_timesteps += collected
        self.collected_transitions += collected

    def close_previous_transition_as_data_abort(
        self, *, env_index: int, final_observation: np.ndarray) -> None:
        close = getattr(self.replay_buffer, "close_previous_transition_as_data_abort", None)
        if not callable(close):
            raise RuntimeError("Replay buffer does not support runtime data-abort boundaries.")
        close(env_index=int(env_index), final_observation=np.asarray(final_observation))

    def collection_learning_potential_batch(
        self,
        observations: np.ndarray,
        buffer_actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_observations: np.ndarray,
        infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> np.ndarray:
        """Compute TD3 residuals on collected transitions, not replay samples."""

        del infos
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.model.device)
        actions = torch.as_tensor(buffer_actions, dtype=torch.float32, device=self.model.device)
        rewards_tensor = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.model.device
        ).reshape(-1, 1)
        dones_tensor = torch.as_tensor(
            dones, dtype=torch.float32, device=self.model.device
        ).reshape(-1, 1)
        next_obs = torch.as_tensor(next_observations, dtype=torch.float32, device=self.model.device)
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.model.target_policy_noise
            noise = noise.clamp(-self.model.target_noise_clip, self.model.target_noise_clip)
            next_actions = (self.model.actor_target(next_obs) + noise).clamp(-1, 1)
            target_q = (
                torch.cat(self.model.critic_target(next_obs, next_actions), dim=1)
                .min(dim=1, keepdim=True)
                .values
            )
            current_q = (
                torch.cat(self.model.critic(obs, actions), dim=1).min(dim=1, keepdim=True).values
            )
            residuals = (
                rewards_tensor + (1.0 - dones_tensor) * self.model.gamma * target_q - current_q
            )
        return residuals.detach().cpu().numpy().reshape(-1)

    def maybe_update(
        self,
        collected_steps: int,
        step_count: int,
        global_total_timesteps: int | None,
        global_steps_done: int,
    ) -> dict[str, float | int]:
        del collected_steps
        learning_starts = int(self.model.learning_starts)
        train_freq = self.model.train_freq

        if self.num_timesteps < learning_starts:
            return {}

        if step_count % int(train_freq.frequency) != 0:
            return {}

        if global_total_timesteps and global_total_timesteps > 0:
            global_step = int(global_steps_done) + int(self.collected_transitions)
            self.model._current_progress_remaining = max(
                1.0 - (global_step / float(global_total_timesteps)),
                0.0,
            )

        if hasattr(self.replay_buffer, "set_beta_progress"):
            self.replay_buffer.set_beta_progress(
                int(global_steps_done) + int(self.collected_transitions)
            )

        grad_steps = int(self.model.gradient_steps)
        train_replay_sample_seconds = 0.0
        train_per_priority_seconds = 0.0
        replay_sample = self.replay_buffer.sample
        update_priorities = getattr(self.replay_buffer, "update_priorities", None)

        def timed_sample(*args, **kwargs):
            nonlocal train_replay_sample_seconds
            started = time.perf_counter()
            try:
                return replay_sample(*args, **kwargs)
            finally:
                train_replay_sample_seconds += time.perf_counter() - started

        def timed_update_priorities(*args, **kwargs):
            nonlocal train_per_priority_seconds
            started = time.perf_counter()
            try:
                return update_priorities(*args, **kwargs)
            finally:
                train_per_priority_seconds += time.perf_counter() - started

        self.replay_buffer.sample = timed_sample
        if callable(update_priorities):
            self.replay_buffer.update_priorities = timed_update_priorities
        train_started = time.perf_counter()
        try:
            if self._diagnostic_mixed_precision:
                if torch.device(self.model.device).type != "cuda":
                    raise ValueError("TD3 mixed-precision diagnostic requires a CUDA device.")
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    self.model.train(
                        gradient_steps=grad_steps,
                        batch_size=int(self.model.batch_size),
                    )
            else:
                self.model.train(
                    gradient_steps=grad_steps,
                    batch_size=int(self.model.batch_size),
                )
        finally:
            self.replay_buffer.sample = replay_sample
            if callable(update_priorities):
                self.replay_buffer.update_priorities = update_priorities
        train_seconds = time.perf_counter() - train_started

        learning_potential_started = time.perf_counter()
        replay_data = self.replay_buffer.sample(
            int(self.model.batch_size), env=self.model._vec_normalize_env
        )
        with torch.no_grad():
            noise = replay_data.actions.clone().data.normal_(0, self.model.target_policy_noise)
            noise = noise.clamp(-self.model.target_noise_clip, self.model.target_noise_clip)
            next_actions = (self.model.actor_target(replay_data.next_observations) + noise).clamp(
                -1, 1
            )
            next_q_values = (
                torch.cat(
                    self.model.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                .min(dim=1, keepdim=True)
                .values
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else float(self.model.gamma)
            )
            target_q_values = (
                replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
            )
            current_q_values = (
                torch.cat(self.model.critic(replay_data.observations, replay_data.actions), dim=1)
                .min(dim=1, keepdim=True)
                .values
            )
            td_residuals = (target_q_values - current_q_values).detach().cpu().numpy().reshape(-1)
        from thesis_rl.curriculum.scenario_acl.usefulness import compute_td3_learning_potential

        learning_potential = compute_td3_learning_potential(td_residuals)
        learning_potential_seconds = time.perf_counter() - learning_potential_started

        logger_values = self.model.logger.name_to_value
        self.last_actor_loss = float(logger_values.get("train/actor_loss", float("nan")))
        self.last_critic_loss = float(logger_values.get("train/critic_loss", float("nan")))
        self.last_learning_rate = float(logger_values.get("train/learning_rate", float("nan")))
        return {
            "actor_loss": self.last_actor_loss,
            "critic_loss": self.last_critic_loss,
            "learning_rate": self.last_learning_rate,
            "update_calls": 1,
            "gradient_steps": max(grad_steps, 0),
            "learning_potential": learning_potential,
            "timing_td3_train_seconds": train_seconds,
            "timing_td3_train_replay_sample_seconds": train_replay_sample_seconds,
            "timing_td3_train_per_priority_seconds": train_per_priority_seconds,
            "timing_acl_replay_learning_potential_seconds": learning_potential_seconds,
        }

    def to_buffer_action(self, env_action: np.ndarray) -> np.ndarray:
        if self._last_buffer_actions is not None:
            cached = np.asarray(self._last_buffer_actions, dtype=np.float32)
            if cached.ndim == 2 and cached.shape[0] > 0:
                return cached[0]
        if self._policy is None:
            raise RuntimeError("SB3 TD3 backend policy unavailable; call begin_training() first.")
        action = np.asarray(env_action, dtype=np.float32)
        scaled = self._policy.scale_action(action)
        return np.clip(np.asarray(scaled, dtype=np.float32), -1.0, 1.0)

    def on_episode_end(self, indices: list[int] | np.ndarray | None = None) -> None:
        if self._action_noise is None:
            return
        if indices is None:
            self._action_noise.reset()
            return
        try:
            self._action_noise.reset(indices=list(indices))
        except TypeError:
            self._action_noise.reset()

    def predict(self, observation: Any, deterministic: bool = False):
        return self.model.predict(observation, deterministic=deterministic)

    def set_env(self, env: Any) -> None:
        self.model.set_env(env)
        super().set_env(env)
        self._sync_action_noise()
        self.replay_buffer = self.model.replay_buffer

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint = normalize_checkpoint_path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(checkpoint))

    def save_replay_buffer(self, path: str | Path) -> bool:
        replay_path = Path(path)
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_replay_buffer(str(replay_path))
        return True

    def load_replay_buffer(self, path: str | Path) -> bool:
        replay_path = Path(path)
        if not replay_path.exists():
            return False
        self.model.load_replay_buffer(str(replay_path))
        self.replay_buffer = self.model.replay_buffer
        return True

    def replay_buffer_n_envs(self) -> int:
        replay_buffer = getattr(self.model, "replay_buffer", None)
        if replay_buffer is not None and hasattr(replay_buffer, "n_envs"):
            return int(getattr(replay_buffer, "n_envs"))
        return int(self.n_envs)
