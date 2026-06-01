from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from thesis_rl.agents.types import Transition
from thesis_rl.networks.decoders.factory import build_decoder
from thesis_rl.planners.base_backend import BasePlannerBackend
from thesis_rl.planners.common.networks import SquashedGaussianActor, TwinQCritic
from thesis_rl.planners.common.utils import assert_box_spaces, build_encoder_for_env, soft_update, to_batch_obs, to_plain_dict
from thesis_rl.planners.interfaces.types import TrainState
from thesis_rl.planners.lifecycle import SacLifecycle
from thesis_rl.rl_core.buffers import ReplayBuffer


class SacPlannerBackend(BasePlannerBackend):
    lifecycle_cls = SacLifecycle

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

        enc_actor = build_encoder_for_env(self.cfg_encoder, self.cfg_obs, self.obs_dim).to(self.device)
        share_encoder = bool(self.cfg_planner.get("share_encoder", False))
        enc_critic = enc_actor if share_encoder else build_encoder_for_env(self.cfg_encoder, self.cfg_obs, self.obs_dim).to(self.device)

        decoder_cfg = dict(self.cfg_decoder)
        actor_decoder = build_decoder(cfg_decoder=decoder_cfg, input_dim=int(enc_actor.output_dim)).to(self.device)
        critic_decoder_1 = build_decoder(
            cfg_decoder=decoder_cfg,
            input_dim=int(enc_critic.output_dim) + self.action_dim,
        ).to(self.device)
        critic_decoder_2 = build_decoder(
            cfg_decoder=decoder_cfg,
            input_dim=int(enc_critic.output_dim) + self.action_dim,
        ).to(self.device)

        log_std_bounds = tuple(self.cfg_planner.get("log_std_bounds", [-20.0, 2.0]))
        self.actor = SquashedGaussianActor(
            encoder=enc_actor,
            decoder=actor_decoder,
            action_dim=self.action_dim,
            log_std_bounds=(float(log_std_bounds[0]), float(log_std_bounds[1])),
            state_dependent_std=True,
        ).to(self.device)
        self.critic = TwinQCritic(enc_critic, critic_decoder_1, critic_decoder_2, self.action_dim).to(self.device)
        self.critic_target = TwinQCritic(
            build_encoder_for_env(self.cfg_encoder, self.cfg_obs, self.obs_dim).to(self.device),
            build_decoder(cfg_decoder=decoder_cfg, input_dim=int(enc_critic.output_dim) + self.action_dim).to(self.device),
            build_decoder(cfg_decoder=decoder_cfg, input_dim=int(enc_critic.output_dim) + self.action_dim).to(self.device),
            self.action_dim,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        lr = float(self.cfg_planner.get("learning_rate", 3e-4))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = float(self.cfg_planner.get("gamma", 0.99))
        self.tau = float(self.cfg_planner.get("tau", 0.005))
        self.learning_starts = int(self.cfg_planner.get("learning_starts", 10000))
        self.batch_size = int(self.cfg_planner.get("batch_size", 256))
        self.train_freq = int(self.cfg_planner.get("train_freq", 1))
        gradient_steps_cfg = self.cfg_planner.get("gradient_steps", "auto")
        self.gradient_steps = int(self.train_freq * max(self.n_envs, 1)) if str(gradient_steps_cfg).lower() == "auto" else int(gradient_steps_cfg)

        ent_coef_cfg = self.cfg_planner.get("ent_coef", "auto")
        self.auto_alpha = isinstance(ent_coef_cfg, str) and str(ent_coef_cfg).lower().startswith("auto")
        if self.auto_alpha:
            init = 1.0
            if isinstance(ent_coef_cfg, str) and "_" in ent_coef_cfg:
                try:
                    init = float(str(ent_coef_cfg).split("_", 1)[1])
                except Exception:
                    init = 1.0
            self.log_alpha = torch.tensor(np.log(max(init, 1e-6)), dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
            self.target_entropy = float(self.cfg_planner.get("target_entropy", -self.action_dim))
        else:
            self.log_alpha = torch.tensor(np.log(float(ent_coef_cfg)), dtype=torch.float32, device=self.device)
            self.alpha_opt = None
            self.target_entropy = float(self.cfg_planner.get("target_entropy", -self.action_dim))

        self.replay_buffer = ReplayBuffer(
            capacity=int(self.cfg_planner.get("buffer_size", 300000)),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha.detach())

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
    ) -> "SacPlannerBackend":
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
    ) -> "SacPlannerBackend":
        payload = torch.load(str(checkpoint_path), map_location="cpu")
        resolved_planner = cfg_planner if cfg_planner is not None else payload.get("cfg_planner", {})
        resolved_encoder = cfg_encoder if cfg_encoder is not None else payload.get("cfg_encoder", {})
        resolved_decoder = cfg_decoder if cfg_decoder is not None else payload.get("cfg_decoder", {})
        resolved_obs = cfg_obs if cfg_obs is not None else payload.get("cfg_obs", {})
        backend = cls(env, resolved_planner, resolved_encoder, resolved_decoder, resolved_obs, device=device, seed=None)
        backend._load_payload(payload)
        return backend

    def _payload(self) -> dict[str, Any]:
        return {
            "algorithm": "sac",
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": self.alpha_opt.state_dict() if self.alpha_opt is not None else None,
            "train_state": {"total_steps": self.state.total_steps, "update_steps": self.state.update_steps},
            "cfg_planner": dict(self.cfg_planner),
            "cfg_encoder": dict(self.cfg_encoder),
            "cfg_decoder": dict(self.cfg_decoder),
            "cfg_obs": dict(self.cfg_obs),
        }

    def _load_payload(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        loaded_log_alpha = torch.as_tensor(payload.get("log_alpha", self.log_alpha), device=self.device).float()
        self.log_alpha.data.copy_(loaded_log_alpha.data)
        self.log_alpha.requires_grad_(bool(self.auto_alpha))
        alpha_opt_state = payload.get("alpha_opt")
        if self.alpha_opt is not None and alpha_opt_state is not None:
            self.alpha_opt.load_state_dict(alpha_opt_state)
        self.state = TrainState(**dict(payload.get("train_state", {})))

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._payload(), str(checkpoint))

    def save_replay_buffer(self, path: str | Path) -> bool:
        replay_path = Path(path)
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.replay_buffer.state_dict(), str(replay_path))
        return True

    def load_replay_buffer(self, path: str | Path) -> bool:
        replay_path = Path(path)
        if not replay_path.exists():
            return False
        payload = torch.load(str(replay_path), map_location="cpu")
        self.replay_buffer.load_state_dict(payload)
        return True

    def predict(self, observation: Any, deterministic: bool = False):
        obs = to_batch_obs(np.asarray(observation, dtype=np.float32))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _log_prob, _entropy = self.actor.sample(obs_t, deterministic=deterministic)
        action_np = action.cpu().numpy()
        action_np = np.clip(action_np, self.action_low, self.action_high).astype(np.float32)
        if np.asarray(observation).ndim == 1:
            return action_np[0], None
        return action_np, None

    def act_train(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action, _ = self.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def act_train_batch(self, observations: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        actions, _ = self.predict(observations, deterministic=deterministic)
        action_batch = np.asarray(actions, dtype=np.float32)
        return action_batch, action_batch

    def to_buffer_action(self, env_action: np.ndarray) -> np.ndarray:
        return np.asarray(env_action, dtype=np.float32)

    def observe_transition(self, transition: Transition) -> None:
        done = bool(transition.terminated or transition.truncated)
        self.replay_buffer.add(
            obs=np.asarray(transition.observation, dtype=np.float32),
            action=np.asarray(transition.buffer_action, dtype=np.float32),
            reward=float(transition.scalar_reward),
            done=done,
            next_obs=np.asarray(transition.next_observation, dtype=np.float32),
        )
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
        self.replay_buffer.add_batch(
            obs=observations,
            actions=buffer_actions,
            rewards=rewards,
            dones=dones,
            next_obs=next_observations,
        )
        self.state.total_steps += int(observations.shape[0])

    def maybe_update(
        self,
        collected_steps: int,
        step_count: int,
        global_total_timesteps: int | None,
        global_steps_done: int,
    ) -> dict[str, float | int]:
        del collected_steps, global_total_timesteps, global_steps_done
        if self.state.total_steps < self.learning_starts:
            return {}
        if step_count % self.train_freq != 0:
            return {}

        actor_losses: list[float] = []
        critic_losses: list[float] = []

        for _ in range(max(self.gradient_steps, 1)):
            batch = self.replay_buffer.sample(self.batch_size, device=self.device)
            obs = batch["obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"]
            next_obs = batch["next_obs"]

            with torch.no_grad():
                next_actions, next_log_prob, _ = self.actor.sample(next_obs, deterministic=False)
                q1_t, q2_t = self.critic_target(next_obs, next_actions)
                min_q_t = torch.min(q1_t, q2_t) - self.alpha * next_log_prob
                q_target = rewards + (1.0 - dones) * self.gamma * min_q_t

            q1, q2 = self.critic(obs, actions)
            critic_loss = 0.5 * (torch.mean((q1 - q_target) ** 2) + torch.mean((q2 - q_target) ** 2))
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()
            critic_losses.append(float(critic_loss.detach().cpu().item()))

            new_actions, log_prob, _ = self.actor.sample(obs, deterministic=False)
            q1_pi, q2_pi = self.critic(obs, new_actions)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * log_prob - min_q_pi).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            actor_losses.append(float(actor_loss.detach().cpu().item()))

            if self.auto_alpha and self.alpha_opt is not None:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()

            soft_update(self.critic, self.critic_target, self.tau)
            self.state.update_steps += 1

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else float("nan"),
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else float("nan"),
            "learning_rate": float(self.actor_opt.param_groups[0]["lr"]),
            "update_calls": 1,
            "gradient_steps": int(max(self.gradient_steps, 1)),
        }
