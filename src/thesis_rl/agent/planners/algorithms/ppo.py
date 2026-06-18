from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from thesis_rl.agent.types import Transition
from thesis_rl.agent.planners.decoders.factory import build_decoder
from thesis_rl.agent.planners.core.backend_base import BasePlannerBackend
from thesis_rl.agent.planners.core.utils import (
    assert_box_spaces,
    build_encoder_for_env,
    normalize_checkpoint_path,
    to_batch_obs,
    to_plain_dict,
)
from thesis_rl.agent.planners.core.types import TrainState
from thesis_rl.agent.planners.core.lifecycle import PpoLifecycle
from thesis_rl.agent.planners.core.buffers import RolloutBuffer


class PpoPlannerBackend(BasePlannerBackend):
    lifecycle_cls = PpoLifecycle
    _SB3_PPO_HIDDEN_LAYERS = [64, 64]

    def __init__(
        self,
        env: Any,
        cfg_planner: Any,
        cfg_encoder: Any | None,
        cfg_decoder: Any | None,
        cfg_obs: Any | None,
        device: str = "auto",
        seed: int | None = None,
    ) -> None:
        super().__init__(env=env, cfg_planner=cfg_planner, device=device)
        self.cfg_encoder = to_plain_dict(cfg_encoder)
        self.cfg_decoder = to_plain_dict(cfg_decoder)
        self.cfg_obs = to_plain_dict(cfg_obs)
        self._validate_network_config()
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        obs_space, action_space = assert_box_spaces(env)
        self.obs_dim = int(np.prod(obs_space.shape))
        self.action_dim = int(np.prod(action_space.shape))
        self.action_low = np.asarray(action_space.low, dtype=np.float32)
        self.action_high = np.asarray(action_space.high, dtype=np.float32)
        self.action_low_t = torch.as_tensor(self.action_low, dtype=torch.float32, device=self.device)
        self.action_high_t = torch.as_tensor(self.action_high, dtype=torch.float32, device=self.device)

        share_encoder = bool(self.cfg_planner.get("share_encoder", True))
        enc_actor = build_encoder_for_env(self.cfg_encoder, self.cfg_obs, self.obs_dim).to(self.device)
        enc_value = enc_actor if share_encoder else build_encoder_for_env(self.cfg_encoder, self.cfg_obs, self.obs_dim).to(self.device)

        decoder_cfg = dict(self.cfg_decoder)
        self.actor_decoder = build_decoder(cfg_decoder=decoder_cfg, input_dim=int(enc_actor.output_dim)).to(self.device)
        self.value_decoder = build_decoder(cfg_decoder=decoder_cfg, input_dim=int(enc_value.output_dim)).to(self.device)
        self.actor_encoder = enc_actor
        self.value_encoder = enc_value

        hidden_actor = int(self.actor_decoder.output_dim)
        self.mu_head = nn.Linear(hidden_actor, self.action_dim).to(self.device)
        log_std_init = float(self.cfg_planner.get("log_std_init", 0.0))
        self.log_std = nn.Parameter(torch.full((self.action_dim,), log_std_init, device=self.device))

        hidden_value = int(self.value_decoder.output_dim)
        self.value_head = nn.Linear(hidden_value, 1).to(self.device)

        params = list(self.actor_encoder.parameters()) + list(self.actor_decoder.parameters()) + list(self.mu_head.parameters()) + [self.log_std]
        if not share_encoder:
            params += list(self.value_encoder.parameters())
        params += list(self.value_decoder.parameters()) + list(self.value_head.parameters())
        optimizer_kwargs: dict[str, float] = {"lr": float(self.cfg_planner.get("learning_rate", 3e-4))}
        if self._uses_sb3_policy_setup():
            optimizer_kwargs["eps"] = float(self.cfg_planner.get("optimizer_eps", 1e-5))
            self._apply_sb3_parameter_initialization()
        self.optimizer = torch.optim.Adam(params, **optimizer_kwargs)

        self.gamma = float(self.cfg_planner.get("gamma", 0.99))
        self.gae_lambda = float(self.cfg_planner.get("gae_lambda", 0.95))
        self.n_steps = int(self.cfg_planner.get("n_steps", 2048))
        self.batch_size = int(self.cfg_planner.get("batch_size", 256))
        self.n_epochs = int(self.cfg_planner.get("n_epochs", 10))
        self.clip_range = float(self.cfg_planner.get("clip_range", 0.2))
        clip_range_vf_cfg = self.cfg_planner.get("clip_range_vf", None)
        self.clip_range_vf = (
            None
            if clip_range_vf_cfg in (None, "none", "null")
            else float(clip_range_vf_cfg)
        )
        self.ent_coef = float(self.cfg_planner.get("ent_coef", 0.0))
        self.vf_coef = float(self.cfg_planner.get("vf_coef", 0.5))
        self.max_grad_norm = float(self.cfg_planner.get("max_grad_norm", 0.5))
        self.target_kl = (
            None
            if self.cfg_planner.get("target_kl", None) in (None, "none", "null")
            else float(self.cfg_planner.get("target_kl"))
        )
        self.normalize_advantage = bool(self.cfg_planner.get("normalize_advantage", True))
        self._validate_training_config()

        self.rollout = RolloutBuffer(
            n_steps=self.n_steps,
            n_envs=max(self.n_envs, 1),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        self._last_values: np.ndarray | None = None
        self._last_log_probs: np.ndarray | None = None
        self._last_buffer_actions: np.ndarray | None = None
        self._last_next_obs: np.ndarray | None = None
        self._last_dones: np.ndarray | None = None
        self._current_episode_starts = np.ones((max(self.n_envs, 1),), dtype=np.float32)

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
    ) -> "PpoPlannerBackend":
        return cls(env, cfg_planner, cfg_encoder, cfg_decoder, cfg_obs, device=device, seed=seed)

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
    ) -> "PpoPlannerBackend":
        payload = torch.load(str(normalize_checkpoint_path(checkpoint_path)), map_location="cpu")
        resolved_planner = cfg_planner if cfg_planner is not None else payload.get("cfg_planner", {})
        resolved_encoder = cfg_encoder if cfg_encoder is not None else payload.get("cfg_encoder", {})
        resolved_decoder = cfg_decoder if cfg_decoder is not None else payload.get("cfg_decoder", {})
        resolved_obs = cfg_obs if cfg_obs is not None else payload.get("cfg_obs", {})
        backend = cls(env, resolved_planner, resolved_encoder, resolved_decoder, resolved_obs, device=device, seed=None)
        backend._load_payload(payload)
        return backend

    def _state_dict_modules(self) -> dict[str, Any]:
        return {
            "actor_encoder": self.actor_encoder.state_dict(),
            "value_encoder": self.value_encoder.state_dict(),
            "actor_decoder": self.actor_decoder.state_dict(),
            "value_decoder": self.value_decoder.state_dict(),
            "mu_head": self.mu_head.state_dict(),
            "value_head": self.value_head.state_dict(),
        }

    def _load_payload(self, payload: dict[str, Any]) -> None:
        self.actor_encoder.load_state_dict(payload["actor_encoder"])
        self.value_encoder.load_state_dict(payload["value_encoder"])
        self.actor_decoder.load_state_dict(payload["actor_decoder"])
        self.value_decoder.load_state_dict(payload["value_decoder"])
        self.mu_head.load_state_dict(payload["mu_head"])
        self.value_head.load_state_dict(payload["value_head"])
        self.log_std.data.copy_(torch.as_tensor(payload.get("log_std", self.log_std.detach().cpu()), device=self.device))
        self.optimizer.load_state_dict(payload["optimizer"])
        self.state = TrainState(**dict(payload.get("train_state", {})))

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint = normalize_checkpoint_path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = self._state_dict_modules()
        payload.update(
            {
                "algorithm": "ppo",
                "log_std": self.log_std.detach().cpu(),
                "optimizer": self.optimizer.state_dict(),
                "train_state": {"total_steps": self.state.total_steps, "update_steps": self.state.update_steps},
                "cfg_planner": dict(self.cfg_planner),
                "cfg_encoder": dict(self.cfg_encoder),
                "cfg_decoder": dict(self.cfg_decoder),
                "cfg_obs": dict(self.cfg_obs),
            }
        )
        torch.save(payload, str(checkpoint))

    def begin_training(
        self,
        chunk_timesteps: int,
        global_total_timesteps: int | None,
        global_steps_done: int,
    ) -> None:
        super().begin_training(
            chunk_timesteps=chunk_timesteps,
            global_total_timesteps=global_total_timesteps,
            global_steps_done=global_steps_done,
        )
        self._current_episode_starts = np.ones((max(self.n_envs, 1),), dtype=np.float32)

    def _uses_sb3_policy_setup(self) -> bool:
        return str(self.cfg_decoder.get("name", "")).strip().lower() == "ppo_sb3"

    def _validate_network_config(self) -> None:
        if not self._uses_sb3_policy_setup():
            return

        encoder_type = str(self.cfg_encoder.get("type", "none")).strip().lower()
        decoder_type = str(self.cfg_decoder.get("type", "mlp")).strip().lower()
        hidden_layers = [int(width) for width in self.cfg_decoder.get("hidden_layers", [])]
        activation = str(self.cfg_decoder.get("activation", "tanh")).strip().lower()
        layer_norm = bool(self.cfg_decoder.get("layer_norm", False))
        dropout = float(self.cfg_decoder.get("dropout", 0.0))

        if encoder_type != "none":
            raise ValueError("`decoder=ppo_sb3` requires `encoder=none` for SB3-faithful PPO.")
        if decoder_type != "mlp":
            raise ValueError("`decoder=ppo_sb3` must use an MLP decoder.")
        if hidden_layers != self._SB3_PPO_HIDDEN_LAYERS:
            raise ValueError(
                "`decoder=ppo_sb3` must keep hidden_layers=[64, 64] for SB3-faithful PPO."
            )
        if activation != "tanh":
            raise ValueError("`decoder=ppo_sb3` must use Tanh activations for SB3-faithful PPO.")
        if layer_norm:
            raise ValueError("`decoder=ppo_sb3` must keep layer_norm=false for SB3-faithful PPO.")
        if dropout != 0.0:
            raise ValueError("`decoder=ppo_sb3` must keep dropout=0.0 for SB3-faithful PPO.")

    @staticmethod
    def _init_linear_weights(module: nn.Module, gain: float) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _apply_sb3_parameter_initialization(self) -> None:
        gain = math.sqrt(2.0)
        self.actor_encoder.apply(lambda module: self._init_linear_weights(module, gain))
        if self.value_encoder is not self.actor_encoder:
            self.value_encoder.apply(lambda module: self._init_linear_weights(module, gain))
        self.actor_decoder.apply(lambda module: self._init_linear_weights(module, gain))
        self.value_decoder.apply(lambda module: self._init_linear_weights(module, gain))
        self._init_linear_weights(self.mu_head, 0.01)
        self._init_linear_weights(self.value_head, 1.0)

    def _validate_training_config(self) -> None:
        if self.n_steps <= 0:
            raise ValueError("`n_steps` must be positive for PPO.")
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive for PPO.")
        if self.n_epochs <= 0:
            raise ValueError("`n_epochs` must be positive for PPO.")
        if self.normalize_advantage and self.batch_size <= 1:
            raise ValueError(
                "`batch_size` must be greater than 1 when `normalize_advantage=true`."
            )
        rollout_size = max(self.n_envs, 1) * self.n_steps
        if self.normalize_advantage and rollout_size <= 1:
            raise ValueError(
                "`n_steps * n_envs` must be greater than 1 when `normalize_advantage=true`."
            )
        if self.clip_range_vf is not None and self.clip_range_vf <= 0.0:
            raise ValueError("`clip_range_vf` must be positive or null for PPO.")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("`target_kl` must be positive or null for PPO.")
        if rollout_size % self.batch_size != 0:
            warnings.warn(
                "PPO rollout size `n_steps * n_envs` is not divisible by `batch_size`; "
                "the last minibatch of each epoch will be truncated.",
                UserWarning,
                stacklevel=2,
            )

    def _policy_params(self, obs_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.actor_encoder(obs_t)
        h = self.actor_decoder(z)
        mu = self.mu_head(h)
        log_std = self.log_std.unsqueeze(0).expand_as(mu)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        return mu, log_std

    def _value(self, obs_t: torch.Tensor) -> torch.Tensor:
        z = self.value_encoder(obs_t)
        h = self.value_decoder(z)
        return self.value_head(h)

    def _sample_action(self, obs_t: torch.Tensor, deterministic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self._policy_params(obs_t)
        std = torch.exp(log_std)
        normal = Normal(mu, std)
        if deterministic:
            raw_action = mu
        else:
            raw_action = normal.rsample()
        action = torch.clamp(raw_action, self.action_low_t, self.action_high_t)
        log_prob = normal.log_prob(raw_action).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return action, raw_action, log_prob, entropy

    def _evaluate_action(self, obs_t: torch.Tensor, actions_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self._policy_params(obs_t)
        std = torch.exp(log_std)
        normal = Normal(mu, std)
        log_prob = normal.log_prob(actions_t).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        values = self._value(obs_t)
        return log_prob, entropy, values

    def _timeout_bootstrap_reward(
        self,
        reward: float,
        info: dict[str, Any] | None,
        terminal_observation: np.ndarray | None,
    ) -> float:
        if not isinstance(info, dict) or not bool(info.get("TimeLimit.truncated", False)):
            return float(reward)
        if terminal_observation is None:
            return float(reward)
        terminal_obs_t = torch.as_tensor(
            np.asarray(terminal_observation, dtype=np.float32)[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            terminal_value = float(self._value(terminal_obs_t).squeeze().cpu().item())
        return float(reward) + self.gamma * terminal_value

    def predict(self, observation: Any, deterministic: bool = False):
        obs = to_batch_obs(np.asarray(observation, dtype=np.float32))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_t, _raw_action_t, _log_prob, _entropy = self._sample_action(obs_t, deterministic=deterministic)
        action_np = action_t.cpu().numpy()
        action_np = np.clip(action_np, self.action_low, self.action_high).astype(np.float32)
        if np.asarray(observation).ndim == 1:
            return action_np[0], None
        return action_np, None

    def act_train(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = to_batch_obs(observation)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions_t, raw_actions_t, log_prob_t, _entropy_t = self._sample_action(obs_t, deterministic=deterministic)
            values_t = self._value(obs_t)
        self._last_values = values_t.squeeze(-1).cpu().numpy()
        self._last_log_probs = log_prob_t.squeeze(-1).cpu().numpy()
        self._last_buffer_actions = np.asarray(raw_actions_t.cpu().numpy(), dtype=np.float32)
        actions = np.asarray(actions_t.cpu().numpy(), dtype=np.float32)
        return actions[0]

    def act_train_batch(self, observations: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions_t, raw_actions_t, log_prob_t, _entropy_t = self._sample_action(obs_t, deterministic=deterministic)
            values_t = self._value(obs_t)
        self._last_values = values_t.squeeze(-1).cpu().numpy()
        self._last_log_probs = log_prob_t.squeeze(-1).cpu().numpy()
        self._last_buffer_actions = np.asarray(raw_actions_t.cpu().numpy(), dtype=np.float32)
        actions = np.asarray(actions_t.cpu().numpy(), dtype=np.float32)
        buffer_actions = np.asarray(raw_actions_t.cpu().numpy(), dtype=np.float32)
        return actions, buffer_actions

    def to_buffer_action(self, env_action: np.ndarray) -> np.ndarray:
        if self._last_buffer_actions is not None:
            cached = np.asarray(self._last_buffer_actions, dtype=np.float32)
            if cached.ndim == 2 and cached.shape[0] > 0:
                return cached[0]
        return np.asarray(env_action, dtype=np.float32)

    def observe_transition(self, transition: Transition) -> None:
        if self._last_values is None or self._last_log_probs is None:
            raise RuntimeError("PPO transition observed before action evaluation cache is set.")
        obs = np.expand_dims(np.asarray(transition.observation, dtype=np.float32), axis=0)
        action = np.expand_dims(np.asarray(transition.buffer_action, dtype=np.float32), axis=0)
        reward = np.asarray(
            [
                self._timeout_bootstrap_reward(
                    float(transition.scalar_reward),
                    transition.info,
                    transition.terminal_observation,
                )
            ],
            dtype=np.float32,
        )
        done = np.asarray([float(transition.terminated or transition.truncated)], dtype=np.float32)
        self.rollout.add(
            obs=obs,
            actions=action,
            rewards=reward,
            episode_starts=np.asarray(self._current_episode_starts, dtype=np.float32),
            values=np.asarray(self._last_values, dtype=np.float32),
            log_probs=np.asarray(self._last_log_probs, dtype=np.float32),
        )
        self._last_next_obs = np.asarray(transition.next_observation, dtype=np.float32)[None, :]
        self._last_dones = np.asarray([float(transition.terminated or transition.truncated)], dtype=np.float32)
        self._current_episode_starts = done
        self.state.total_steps += 1

    def observe_transition_batch(
        self,
        observations: np.ndarray,
        buffer_actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_observations: np.ndarray,
        infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> None:
        if self._last_values is None or self._last_log_probs is None:
            raise RuntimeError("PPO batch transition observed before action evaluation cache is set.")
        adjusted_rewards = np.asarray(rewards, dtype=np.float32).copy()
        for idx, info in enumerate(infos):
            terminal_observation = info.get("terminal_observation") if isinstance(info, dict) else None
            adjusted_rewards[idx] = self._timeout_bootstrap_reward(
                float(adjusted_rewards[idx]),
                info if isinstance(info, dict) else None,
                terminal_observation,
            )
        self.rollout.add(
            obs=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(buffer_actions, dtype=np.float32),
            rewards=adjusted_rewards,
            episode_starts=np.asarray(self._current_episode_starts, dtype=np.float32),
            values=np.asarray(self._last_values, dtype=np.float32),
            log_probs=np.asarray(self._last_log_probs, dtype=np.float32),
        )
        self._last_next_obs = np.asarray(next_observations, dtype=np.float32)
        self._last_dones = np.asarray(dones, dtype=np.float32)
        self._current_episode_starts = np.asarray(dones, dtype=np.float32)
        self.state.total_steps += int(observations.shape[0])

    def maybe_update(
        self,
        collected_steps: int,
        step_count: int,
        global_total_timesteps: int | None,
        global_steps_done: int,
    ) -> dict[str, float | int]:
        del collected_steps, step_count, global_total_timesteps, global_steps_done
        if not self.rollout.full:
            return {}

        if self._last_next_obs is None or self._last_dones is None:
            raise RuntimeError("PPO rollout full but bootstrap cache is missing.")

        with torch.no_grad():
            last_values_t = self._value(
                torch.as_tensor(self._last_next_obs, dtype=torch.float32, device=self.device)
            )
            last_values = last_values_t.squeeze(-1).cpu().numpy()

        self.rollout.compute_returns_and_advantages(last_values=last_values, last_dones=self._last_dones)

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        approx_kl_divs: list[float] = []
        clip_fractions: list[float] = []
        optimizer_steps = 0
        continue_training = True

        for _ in range(max(self.n_epochs, 1)):
            for batch in self.rollout.iter_minibatches(batch_size=self.batch_size, device=self.device):
                adv = batch.advantages
                if self.normalize_advantage and len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                log_prob, entropy, values = self._evaluate_action(batch.obs, batch.actions)
                ratio = torch.exp(log_prob.squeeze(-1) - batch.old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_range).float()).cpu().item()
                clip_fractions.append(float(clip_fraction))

                value_pred = values.squeeze(-1)
                if self.clip_range_vf is not None:
                    value_pred = batch.old_values + torch.clamp(
                        value_pred - batch.old_values,
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                value_loss = torch.mean((value_pred - batch.returns) ** 2)
                entropy_bonus = entropy.mean()
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus

                with torch.no_grad():
                    log_ratio = log_prob.squeeze(-1) - batch.old_log_probs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio).cpu().item()
                    approx_kl_divs.append(float(approx_kl))
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.max_grad_norm)
                self.optimizer.step()
                optimizer_steps += 1

                actor_losses.append(float(actor_loss.detach().cpu().item()))
                critic_losses.append(float(value_loss.detach().cpu().item()))
                self.state.update_steps += 1
            if not continue_training:
                break

        self.rollout.reset()
        self._last_buffer_actions = None
        self._last_next_obs = None
        self._last_dones = None

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else float("nan"),
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else float("nan"),
            "approx_kl": float(np.mean(approx_kl_divs)) if approx_kl_divs else float("nan"),
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else float("nan"),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "update_calls": 1,
            "gradient_steps": int(optimizer_steps),
        }
