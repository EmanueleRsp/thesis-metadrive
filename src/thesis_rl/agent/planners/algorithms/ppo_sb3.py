from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from thesis_rl.agent.types import Transition
from thesis_rl.agent.transition_boundary import normalize_vector_transition_boundary
from thesis_rl.agent.planners.core.backend_base import BasePlannerBackend
from thesis_rl.agent.planners.core.lifecycle import PpoLifecycle
from thesis_rl.agent.planners.core.utils import normalize_checkpoint_path, to_plain_dict
from thesis_rl.sb3_extensions import (
    build_sb3_specs_from_configs,
    resolve_transition_replay_config,
)

if TYPE_CHECKING:
    from stable_baselines3 import PPO


def _require_sb3_ppo():
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "The `ppo_sb3` backend requires `stable-baselines3` to be installed. "
            "Run `uv sync` to install project dependencies."
        ) from exc
    return PPO


class Sb3PpoPlannerBackend(BasePlannerBackend):
    lifecycle_cls = PpoLifecycle

    def __init__(
        self,
        env: Any,
        cfg_planner: Any,
        model: "PPO",
        device: str = "auto",
    ) -> None:
        super().__init__(env=env, cfg_planner=cfg_planner, device=device)
        self.model = model
        self.sb3_model = model

        self.last_actor_loss = float("nan")
        self.last_critic_loss = float("nan")
        self.last_learning_rate = float("nan")
        self.collected_transitions = 0

        self._last_values: Any | None = None
        self._last_log_probs: Any | None = None
        self._last_buffer_actions: np.ndarray | None = None
        self._last_next_obs: np.ndarray | None = None
        self._acl_episode_advantages: dict[tuple[int, int], list[float]] = {}
        self._acl_rollout_slot_ids = np.full(
            (int(self.model.rollout_buffer.buffer_size), self.n_envs), -1, dtype=np.int64
        )
        self._acl_rollout_episode_ids = np.full(
            (int(self.model.rollout_buffer.buffer_size), self.n_envs), -1, dtype=np.int64
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
    ) -> "Sb3PpoPlannerBackend":
        PPO = _require_sb3_ppo()

        planner_cfg = to_plain_dict(cfg_planner)
        resolve_transition_replay_config(
            planner_cfg.get("transition_replay"),
            algorithm_name="ppo_sb3",
        )
        policy_spec, algorithm_spec = build_sb3_specs_from_configs(
            "ppo_sb3",
            planner_cfg,
            encoder_cfg=to_plain_dict(cfg_encoder),
            decoder_cfg=to_plain_dict(cfg_decoder),
            obs_cfg=to_plain_dict(cfg_obs),
        )
        model_kwargs = algorithm_spec.merged_algorithm_kwargs()

        model = PPO(
            policy=policy_spec.policy,
            env=env,
            n_steps=int(planner_cfg.get("n_steps", 2048)),
            batch_size=int(planner_cfg.get("batch_size", 64)),
            n_epochs=int(planner_cfg.get("n_epochs", 10)),
            learning_rate=float(planner_cfg.get("learning_rate", 3e-4)),
            gamma=float(planner_cfg.get("gamma", 0.99)),
            gae_lambda=float(planner_cfg.get("gae_lambda", 0.95)),
            clip_range=float(planner_cfg.get("clip_range", 0.2)),
            clip_range_vf=planner_cfg.get("clip_range_vf", None),
            normalize_advantage=bool(planner_cfg.get("normalize_advantage", True)),
            ent_coef=float(planner_cfg.get("ent_coef", 0.0)),
            vf_coef=float(planner_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(planner_cfg.get("max_grad_norm", 0.5)),
            use_sde=bool(planner_cfg.get("use_sde", False)),
            sde_sample_freq=int(planner_cfg.get("sde_sample_freq", -1)),
            target_kl=planner_cfg.get("target_kl", None),
            policy_kwargs=policy_spec.policy_kwargs,
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
    ) -> "Sb3PpoPlannerBackend":
        del cfg_encoder, cfg_decoder, cfg_obs
        PPO = _require_sb3_ppo()

        model = PPO.load(str(normalize_checkpoint_path(checkpoint_path)), env=env, device=device)
        resolved_cfg = {} if cfg_planner is None else to_plain_dict(cfg_planner)
        return cls(env=env, cfg_planner=resolved_cfg, model=model, device=device)

    def begin_training(
        self,
        chunk_timesteps: int,
        global_total_timesteps: int | None,
        global_steps_done: int,
    ) -> None:
        del chunk_timesteps
        _require_sb3_ppo()
        from stable_baselines3.common.logger import configure

        if not hasattr(self.model, "_logger"):
            self.model._logger = configure(folder=None, format_strings=[])
        if not hasattr(self.model, "_current_progress_remaining"):
            self.model._current_progress_remaining = 1.0
        if not hasattr(self.model, "num_timesteps"):
            self.model.num_timesteps = 0
        if getattr(self.model, "_last_episode_starts", None) is None:
            self.model._last_episode_starts = np.ones((self.n_envs,), dtype=bool)

        self.collected_transitions = 0
        self._global_total_timesteps = global_total_timesteps
        self._global_steps_done = int(global_steps_done)
        if not hasattr(self, "_acl_episode_advantages"):
            self._acl_episode_advantages = {}
        shape = (int(self.model.rollout_buffer.buffer_size), self.n_envs)
        if not hasattr(self, "_acl_rollout_slot_ids") or self._acl_rollout_slot_ids.shape != shape:
            self._acl_rollout_slot_ids = np.full(shape, -1, dtype=np.int64)
            self._acl_rollout_episode_ids = np.full(shape, -1, dtype=np.int64)

    def end_training(self) -> None:
        return None

    def _sample_actions(
        self,
        observations: np.ndarray,
        deterministic: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        obs_batch = np.asarray(observations, dtype=np.float32)
        if self.model.use_sde:
            self.model.policy.reset_noise(obs_batch.shape[0])

        obs_tensor = self.model.policy.obs_to_tensor(obs_batch)[0]
        with torch.no_grad():
            actions_t, values_t, log_probs_t = self.model.policy(
                obs_tensor,
                deterministic=deterministic,
            )
        raw_actions = np.asarray(actions_t.cpu().numpy(), dtype=np.float32)

        env_actions = raw_actions
        if self.model.policy.squash_output:
            env_actions = np.asarray(
                self.model.policy.unscale_action(raw_actions), dtype=np.float32
            )
        else:
            env_actions = np.clip(
                raw_actions,
                self.model.action_space.low,
                self.model.action_space.high,
            ).astype(np.float32)

        self._last_values = values_t
        self._last_log_probs = log_probs_t
        self._last_buffer_actions = raw_actions
        return env_actions, raw_actions

    def act_train(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        env_actions, _buffer_actions = self._sample_actions(
            np.expand_dims(np.asarray(observation, dtype=np.float32), axis=0),
            deterministic=deterministic,
        )
        return np.asarray(env_actions[0], dtype=np.float32)

    def act_train_batch(
        self,
        observations: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._sample_actions(
            np.asarray(observations, dtype=np.float32),
            deterministic=deterministic,
        )

    def to_buffer_action(self, env_action: np.ndarray) -> np.ndarray:
        if self._last_buffer_actions is not None:
            cached = np.asarray(self._last_buffer_actions, dtype=np.float32)
            if cached.ndim == 2 and cached.shape[0] > 0:
                return cached[0]
        return np.asarray(env_action, dtype=np.float32)

    def _bootstrap_timeout_rewards(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> np.ndarray:
        adjusted_rewards = np.asarray(rewards, dtype=np.float32).copy()
        for idx, info in enumerate(infos):
            if not bool(dones[idx]):
                continue
            if not isinstance(info, dict):
                continue
            terminal_observation = info.get("terminal_observation")
            if terminal_observation is None or not bool(info.get("TimeLimit.truncated", False)):
                continue
            terminal_obs_tensor = self.model.policy.obs_to_tensor(terminal_observation)[0]
            with torch.no_grad():
                terminal_value = self.model.policy.predict_values(terminal_obs_tensor)[0]
            adjusted_rewards[idx] += float(self.model.gamma * terminal_value.cpu().item())
        return adjusted_rewards

    def observe_transition(self, transition: Transition) -> None:
        if self._last_values is None or self._last_log_probs is None:
            raise RuntimeError("PPO transition observed before action evaluation cache is set.")

        transition_info = dict(transition.info)
        transition_info["TimeLimit.truncated"] = bool(
            transition.truncated and not transition.terminated
        )
        rewards = self._bootstrap_timeout_rewards(
            rewards=np.asarray([transition.scalar_reward], dtype=np.float32),
            dones=np.asarray([transition.terminated or transition.truncated], dtype=bool),
            infos=[transition_info | {"terminal_observation": transition.terminal_observation}],
        )
        obs = np.expand_dims(np.asarray(transition.observation, dtype=np.float32), axis=0)
        actions = np.expand_dims(np.asarray(transition.buffer_action, dtype=np.float32), axis=0)
        episode_starts = np.asarray(self.model._last_episode_starts, dtype=np.float32)

        self.model.rollout_buffer.add(
            obs=obs,
            action=actions,
            reward=rewards,
            episode_start=episode_starts,
            value=self._last_values,
            log_prob=self._last_log_probs,
        )
        self.model._last_episode_starts = np.asarray(
            [transition.terminated or transition.truncated],
            dtype=bool,
        )
        self._last_next_obs = np.asarray(transition.next_observation, dtype=np.float32)[None, :]
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
    ) -> None:
        if self._last_values is None or self._last_log_probs is None:
            raise RuntimeError(
                "PPO batch transition observed before action evaluation cache is set."
            )

        obs_batch = np.asarray(observations, dtype=np.float32)
        action_batch = np.asarray(buffer_actions, dtype=np.float32)
        terminated_batch, truncated_batch, resolved_next_obs = normalize_vector_transition_boundary(
            dones=dones,
            infos=infos,
            next_observations=next_observations,
            terminated=terminated,
            truncated=truncated,
        )
        done_batch = terminated_batch | truncated_batch
        adjusted_rewards = self._bootstrap_timeout_rewards(
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=done_batch,
            infos=infos,
        )
        episode_starts = np.asarray(self.model._last_episode_starts, dtype=np.float32)

        position = int(self.model.rollout_buffer.pos)
        if position >= int(self.model.rollout_buffer.buffer_size):
            raise RuntimeError("PPO rollout provenance position exceeded the rollout buffer.")
        self._acl_rollout_slot_ids[position] = np.asarray(
            [int(info.get("acl_slot_id", -1)) for info in infos], dtype=np.int64
        )
        self._acl_rollout_episode_ids[position] = np.asarray(
            [int(info.get("acl_episode_id", -1)) for info in infos], dtype=np.int64
        )

        self.model.rollout_buffer.add(
            obs=obs_batch,
            action=action_batch,
            reward=adjusted_rewards,
            episode_start=episode_starts,
            value=self._last_values,
            log_prob=self._last_log_probs,
        )
        self.model._last_episode_starts = done_batch.copy()
        self._last_next_obs = resolved_next_obs
        collected = int(obs_batch.shape[0])
        self.model.num_timesteps += collected
        self.collected_transitions += collected

    def maybe_update(
        self,
        collected_steps: int,
        step_count: int,
        global_total_timesteps: int | None,
        global_steps_done: int,
    ) -> dict[str, float | int]:
        del collected_steps, step_count
        if not self.model.rollout_buffer.full:
            return {}

        if self._last_next_obs is None:
            raise RuntimeError("PPO rollout is full but next-observation cache is missing.")

        if global_total_timesteps and global_total_timesteps > 0:
            global_step = int(global_steps_done) + int(self.collected_transitions)
            self.model._current_progress_remaining = max(
                1.0 - (global_step / float(global_total_timesteps)),
                0.0,
            )

        with torch.no_grad():
            last_values = self.model.policy.predict_values(
                self.model.policy.obs_to_tensor(self._last_next_obs)[0]
            )
        self.model.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=np.asarray(self.model._last_episode_starts, dtype=bool),
        )
        from thesis_rl.curriculum.scenario_acl.usefulness import compute_ppo_learning_potential

        values = np.asarray(self.model.rollout_buffer.values)
        rewards = np.asarray(self.model.rollout_buffer.rewards)
        episode_starts = np.asarray(self.model.rollout_buffer.episode_starts, dtype=bool)
        next_values = np.empty_like(values)
        next_values[:-1] = values[1:]
        next_values[-1] = np.asarray(last_values.detach().cpu()).reshape(-1)
        dones = np.empty_like(episode_starts)
        dones[:-1] = episode_starts[1:]
        dones[-1] = np.asarray(self.model._last_episode_starts, dtype=bool)
        for slot_id, episode_id, advantage in zip(
            self._acl_rollout_slot_ids.reshape(-1),
            self._acl_rollout_episode_ids.reshape(-1),
            np.asarray(self.model.rollout_buffer.advantages).reshape(-1),
            strict=True,
        ):
            if int(slot_id) >= 0 and int(episode_id) >= 0:
                self._acl_episode_advantages.setdefault((int(slot_id), int(episode_id)), []).append(
                    float(max(float(advantage), 0.0))
                )
        learning_potential = compute_ppo_learning_potential(
            rewards=rewards.reshape(-1),
            values=values.reshape(-1),
            next_values=next_values.reshape(-1),
            dones=dones.reshape(-1),
            gamma=float(self.cfg_planner.get("learning_potential_gamma", 0.99)),
            gae_lambda=float(self.cfg_planner.get("learning_potential_gae_lambda", 0.9)),
        )

        prev_updates = int(getattr(self.model, "_n_updates", 0))
        self.model.train()
        self.model.rollout_buffer.reset()

        logger_values = self.model.logger.name_to_value
        self.last_actor_loss = float(logger_values.get("train/policy_gradient_loss", float("nan")))
        self.last_critic_loss = float(logger_values.get("train/value_loss", float("nan")))
        self.last_learning_rate = float(logger_values.get("train/learning_rate", float("nan")))
        update_delta = int(getattr(self.model, "_n_updates", 0)) - prev_updates

        self._last_buffer_actions = None
        self._last_next_obs = None
        return {
            "actor_loss": self.last_actor_loss,
            "critic_loss": self.last_critic_loss,
            "learning_rate": self.last_learning_rate,
            "update_calls": 1,
            "gradient_steps": max(update_delta, 0),
            "learning_potential": learning_potential,
        }

    def pop_acl_episode_learning_potential(self, slot_id: int, episode_id: int) -> float | None:
        values = self._acl_episode_advantages.pop((int(slot_id), int(episode_id)), None)
        if not values:
            return None
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    def acl_ready_learning_potentials(self) -> dict[tuple[int, int], float]:
        return {
            key: float(np.mean(np.asarray(values, dtype=np.float64)))
            for key, values in self._acl_episode_advantages.items()
            if values
        }

    def predict(self, observation: Any, deterministic: bool = False):
        return self.model.predict(observation, deterministic=deterministic)

    def set_env(self, env: Any) -> None:
        self.model.set_env(env)
        super().set_env(env)

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint = normalize_checkpoint_path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(checkpoint))
