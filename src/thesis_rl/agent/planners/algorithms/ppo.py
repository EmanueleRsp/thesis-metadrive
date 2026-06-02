from __future__ import annotations

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
    safe_atanh,
    to_batch_obs,
    to_plain_dict,
)
from thesis_rl.agent.planners.core.types import TrainState
from thesis_rl.agent.planners.core.lifecycle import PpoLifecycle
from thesis_rl.agent.planners.core.buffers import RolloutBuffer


class PpoPlannerBackend(BasePlannerBackend):
    lifecycle_cls = PpoLifecycle

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
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        obs_space, action_space = assert_box_spaces(env)
        self.obs_dim = int(np.prod(obs_space.shape))
        self.action_dim = int(np.prod(action_space.shape))
        self.action_low = np.asarray(action_space.low, dtype=np.float32)
        self.action_high = np.asarray(action_space.high, dtype=np.float32)

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
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, device=self.device))

        hidden_value = int(self.value_decoder.output_dim)
        self.value_head = nn.Linear(hidden_value, 1).to(self.device)

        params = list(self.actor_encoder.parameters()) + list(self.actor_decoder.parameters()) + list(self.mu_head.parameters()) + [self.log_std]
        if not share_encoder:
            params += list(self.value_encoder.parameters())
        params += list(self.value_decoder.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=float(self.cfg_planner.get("learning_rate", 3e-4)))

        self.gamma = float(self.cfg_planner.get("gamma", 0.99))
        self.gae_lambda = float(self.cfg_planner.get("gae_lambda", 0.95))
        self.n_steps = int(self.cfg_planner.get("n_steps", 2048))
        self.batch_size = int(self.cfg_planner.get("batch_size", 256))
        self.n_epochs = int(self.cfg_planner.get("n_epochs", 10))
        self.clip_range = float(self.cfg_planner.get("clip_range", 0.2))
        self.ent_coef = float(self.cfg_planner.get("ent_coef", 0.0))
        self.vf_coef = float(self.cfg_planner.get("vf_coef", 0.5))
        self.max_grad_norm = float(self.cfg_planner.get("max_grad_norm", 0.5))

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
        self._last_next_obs: np.ndarray | None = None
        self._last_dones: np.ndarray | None = None

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

    def _sample_action(self, obs_t: torch.Tensor, deterministic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self._policy_params(obs_t)
        std = torch.exp(log_std)
        normal = Normal(mu, std)
        if deterministic:
            u = mu
        else:
            u = normal.rsample()
        action = torch.tanh(u)
        log_prob = normal.log_prob(u).sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(torch.clamp(1.0 - action.pow(2), min=1e-6)).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy

    def _evaluate_action(self, obs_t: torch.Tensor, actions_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self._policy_params(obs_t)
        std = torch.exp(log_std)
        normal = Normal(mu, std)
        u = safe_atanh(actions_t)
        log_prob = normal.log_prob(u).sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(torch.clamp(1.0 - actions_t.pow(2), min=1e-6)).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        values = self._value(obs_t)
        return log_prob, entropy, values

    def predict(self, observation: Any, deterministic: bool = False):
        obs = to_batch_obs(np.asarray(observation, dtype=np.float32))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_t, _log_prob, _entropy = self._sample_action(obs_t, deterministic=deterministic)
        action_np = action_t.cpu().numpy()
        action_np = np.clip(action_np, self.action_low, self.action_high).astype(np.float32)
        if np.asarray(observation).ndim == 1:
            return action_np[0], None
        return action_np, None

    def act_train(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs = to_batch_obs(observation)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions_t, log_prob_t, _entropy_t = self._sample_action(obs_t, deterministic=deterministic)
            values_t = self._value(obs_t)
        self._last_values = values_t.squeeze(-1).cpu().numpy()
        self._last_log_probs = log_prob_t.squeeze(-1).cpu().numpy()
        actions = np.asarray(actions_t.cpu().numpy(), dtype=np.float32)
        return actions[0]

    def act_train_batch(self, observations: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions_t, log_prob_t, _entropy_t = self._sample_action(obs_t, deterministic=deterministic)
            values_t = self._value(obs_t)
        self._last_values = values_t.squeeze(-1).cpu().numpy()
        self._last_log_probs = log_prob_t.squeeze(-1).cpu().numpy()
        actions = np.asarray(actions_t.cpu().numpy(), dtype=np.float32)
        return actions, actions

    def to_buffer_action(self, env_action: np.ndarray) -> np.ndarray:
        return np.asarray(env_action, dtype=np.float32)

    def observe_transition(self, transition: Transition) -> None:
        if self._last_values is None or self._last_log_probs is None:
            raise RuntimeError("PPO transition observed before action evaluation cache is set.")
        obs = np.expand_dims(np.asarray(transition.observation, dtype=np.float32), axis=0)
        action = np.expand_dims(np.asarray(transition.buffer_action, dtype=np.float32), axis=0)
        reward = np.asarray([float(transition.scalar_reward)], dtype=np.float32)
        done = np.asarray([float(transition.terminated or transition.truncated)], dtype=np.float32)
        self.rollout.add(
            obs=obs,
            actions=action,
            rewards=reward,
            dones=done,
            values=np.asarray(self._last_values, dtype=np.float32),
            log_probs=np.asarray(self._last_log_probs, dtype=np.float32),
        )
        self._last_next_obs = np.asarray(transition.next_observation, dtype=np.float32)[None, :]
        self._last_dones = np.asarray([float(transition.terminated or transition.truncated)], dtype=np.float32)
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
        del infos
        if self._last_values is None or self._last_log_probs is None:
            raise RuntimeError("PPO batch transition observed before action evaluation cache is set.")
        self.rollout.add(
            obs=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(buffer_actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
            values=np.asarray(self._last_values, dtype=np.float32),
            log_probs=np.asarray(self._last_log_probs, dtype=np.float32),
        )
        self._last_next_obs = np.asarray(next_observations, dtype=np.float32)
        self._last_dones = np.asarray(dones, dtype=np.float32)
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

        for _ in range(max(self.n_epochs, 1)):
            for batch in self.rollout.iter_minibatches(batch_size=self.batch_size, device=self.device):
                adv = batch.advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                log_prob, entropy, values = self._evaluate_action(batch.obs, batch.actions)
                ratio = torch.exp(log_prob.squeeze(-1) - batch.old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = torch.mean((values.squeeze(-1) - batch.returns) ** 2)
                entropy_bonus = entropy.mean()
                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.max_grad_norm)
                self.optimizer.step()

                actor_losses.append(float(actor_loss.detach().cpu().item()))
                critic_losses.append(float(value_loss.detach().cpu().item()))
                self.state.update_steps += 1

        self.rollout.reset()
        self._last_next_obs = None
        self._last_dones = None

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else float("nan"),
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else float("nan"),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "update_calls": 1,
            "gradient_steps": int(max(self.n_epochs, 1)),
        }
