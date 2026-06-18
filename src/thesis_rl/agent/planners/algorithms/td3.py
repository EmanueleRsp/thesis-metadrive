from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from thesis_rl.agent.types import Transition
from thesis_rl.agent.planners.decoders.factory import build_decoder
from thesis_rl.agent.planners.core.action_noise import build_action_noise
from thesis_rl.agent.planners.core.backend_base import BasePlannerBackend
from thesis_rl.agent.planners.modules.actor_critic import DeterministicActor, TwinQCritic
from thesis_rl.agent.planners.core.utils import (
    assert_box_spaces,
    build_encoder_for_env,
    normalize_checkpoint_path,
    soft_update,
    to_batch_obs,
    to_plain_dict,
)
from thesis_rl.agent.planners.core.types import TrainState
from thesis_rl.agent.planners.core.lifecycle import Td3Lifecycle
from thesis_rl.agent.planners.core.buffers import ReplayBuffer
from thesis_rl.agent.planners.core.offpolicy_debug import OffPolicyDebugLogger


class Td3PlannerBackend(BasePlannerBackend):
    lifecycle_cls = Td3Lifecycle
    _SB3_TD3_HIDDEN_LAYERS = [400, 300]

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
        self._validate_network_config()

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

        self.actor = DeterministicActor(enc_actor, actor_decoder, self.action_dim).to(self.device)
        self.critic = TwinQCritic(enc_critic, critic_decoder_1, critic_decoder_2, self.action_dim).to(self.device)
        self.actor_target = DeterministicActor(
            build_encoder_for_env(self.cfg_encoder, self.cfg_obs, self.obs_dim).to(self.device),
            build_decoder(cfg_decoder=decoder_cfg, input_dim=int(enc_actor.output_dim)).to(self.device),
            self.action_dim,
        ).to(self.device)
        self.critic_target = TwinQCritic(
            build_encoder_for_env(self.cfg_encoder, self.cfg_obs, self.obs_dim).to(self.device),
            build_decoder(cfg_decoder=decoder_cfg, input_dim=int(enc_critic.output_dim) + self.action_dim).to(self.device),
            build_decoder(cfg_decoder=decoder_cfg, input_dim=int(enc_critic.output_dim) + self.action_dim).to(self.device),
            self.action_dim,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        lr = float(self.cfg_planner.get("learning_rate", 3e-4))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = float(self.cfg_planner.get("gamma", 0.99))
        self.tau = float(self.cfg_planner.get("tau", 0.005))
        self.learning_starts = int(self.cfg_planner.get("learning_starts", 10000))
        self.batch_size = int(self.cfg_planner.get("batch_size", 256))
        self.train_freq = int(self.cfg_planner.get("train_freq", 1))
        if self.train_freq <= 0:
            raise ValueError("`train_freq` must be >= 1 for TD3.")
        self.gradient_steps_cfg = self.cfg_planner.get("gradient_steps", "auto")
        self.policy_delay = int(self.cfg_planner.get("policy_delay", 2))
        self.target_policy_noise = float(self.cfg_planner.get("target_policy_noise", 0.2))
        self.target_noise_clip = float(self.cfg_planner.get("target_noise_clip", 0.5))
        self.action_noise = build_action_noise(self.cfg_planner, action_dim=self.action_dim)
        self._rollout_steps_since_update = 0
        self._rollout_transitions_since_update = 0

        self.replay_buffer = ReplayBuffer(
            capacity=int(self.cfg_planner.get("buffer_size", 300000)),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_envs=self.n_envs,
        )
        self.debug_logger = OffPolicyDebugLogger(
            self.cfg_planner.get("offpolicy_debug"),
            algorithm="td3",
            action_dim=self.action_dim,
        )

    def _validate_network_config(self) -> None:
        decoder_name = str(self.cfg_decoder.get("name", "")).strip().lower()
        if decoder_name != "td3_sb3":
            return

        encoder_type = str(self.cfg_encoder.get("type", "none")).strip().lower()
        if encoder_type != "none":
            raise ValueError("`decoder=td3_sb3` requires `encoder=none` for SB3-faithful TD3.")

        decoder_type = str(self.cfg_decoder.get("type", "mlp")).strip().lower()
        hidden_layers = [int(width) for width in self.cfg_decoder.get("hidden_layers", [])]
        activation = str(self.cfg_decoder.get("activation", "relu")).strip().lower()
        layer_norm = bool(self.cfg_decoder.get("layer_norm", False))
        dropout = float(self.cfg_decoder.get("dropout", 0.0))

        if decoder_type != "mlp":
            raise ValueError("`decoder=td3_sb3` must use an MLP decoder.")
        if hidden_layers != self._SB3_TD3_HIDDEN_LAYERS:
            raise ValueError(
                "`decoder=td3_sb3` must keep hidden_layers=[400, 300] for SB3-faithful TD3."
            )
        if activation != "relu":
            raise ValueError("`decoder=td3_sb3` must use ReLU activations for SB3-faithful TD3.")
        if layer_norm:
            raise ValueError("`decoder=td3_sb3` must keep layer_norm=false for SB3-faithful TD3.")
        if dropout != 0.0:
            raise ValueError("`decoder=td3_sb3` must keep dropout=0.0 for SB3-faithful TD3.")

    def _sample_random_actions(self, batch_size: int) -> np.ndarray:
        low = np.broadcast_to(self.action_low, (int(batch_size), self.action_dim))
        high = np.broadcast_to(self.action_high, (int(batch_size), self.action_dim))
        return np.random.uniform(low=low, high=high).astype(np.float32)

    def _policy_actions(self, observation: Any) -> np.ndarray:
        obs = to_batch_obs(np.asarray(observation, dtype=np.float32))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(obs_t)
        action_np = action.cpu().numpy()
        return np.clip(action_np, self.action_low, self.action_high).astype(np.float32)

    def _apply_action_noise(self, actions: np.ndarray) -> np.ndarray:
        action_batch = np.asarray(actions, dtype=np.float32)
        if self.action_noise is None:
            return np.clip(action_batch, self.action_low, self.action_high).astype(np.float32)
        noisy_actions = action_batch + self.action_noise.sample(int(action_batch.shape[0]))
        return np.clip(noisy_actions, self.action_low, self.action_high).astype(np.float32)

    def _resolve_gradient_steps(self, rollout_transitions: int) -> int:
        cfg_value = self.gradient_steps_cfg
        if isinstance(cfg_value, str):
            key = cfg_value.strip().lower()
            if key == "auto":
                return max(int(rollout_transitions), 1)
        resolved = int(cfg_value)
        if resolved == -1:
            return max(int(rollout_transitions), 1)
        return max(resolved, 1)

    @staticmethod
    def _is_timeout(info: dict[str, Any] | None) -> bool:
        if not isinstance(info, dict):
            return False
        return bool(info.get("TimeLimit.truncated", False))

    @staticmethod
    def _resolve_next_observation(
        next_observation: np.ndarray,
        terminal_observation: np.ndarray | None,
    ) -> np.ndarray:
        if terminal_observation is not None:
            return np.asarray(terminal_observation, dtype=np.float32)
        return np.asarray(next_observation, dtype=np.float32)

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
    ) -> "Td3PlannerBackend":
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
    ) -> "Td3PlannerBackend":
        payload = torch.load(str(normalize_checkpoint_path(checkpoint_path)), map_location="cpu")
        resolved_planner = cfg_planner if cfg_planner is not None else payload.get("cfg_planner", {})
        resolved_encoder = cfg_encoder if cfg_encoder is not None else payload.get("cfg_encoder", {})
        resolved_decoder = cfg_decoder if cfg_decoder is not None else payload.get("cfg_decoder", {})
        resolved_obs = cfg_obs if cfg_obs is not None else payload.get("cfg_obs", {})
        backend = cls(env, resolved_planner, resolved_encoder, resolved_decoder, resolved_obs, device=device, seed=None)
        backend._load_payload(payload)
        return backend

    def _load_payload(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.state = TrainState(**dict(payload.get("train_state", {})))

    def _payload(self) -> dict[str, Any]:
        return {
            "algorithm": "td3",
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "train_state": {"total_steps": self.state.total_steps, "update_steps": self.state.update_steps},
            "cfg_planner": dict(self.cfg_planner),
            "cfg_encoder": dict(self.cfg_encoder),
            "cfg_decoder": dict(self.cfg_decoder),
            "cfg_obs": dict(self.cfg_obs),
        }

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint = normalize_checkpoint_path(checkpoint_path)
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
        del deterministic
        action_np = self._policy_actions(observation)
        if np.asarray(observation).ndim == 1:
            return action_np[0], None
        return action_np, None

    def act_train(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not deterministic and self.state.total_steps < self.learning_starts:
            return self._sample_random_actions(1)[0]
        action, _ = self.predict(observation, deterministic=True)
        action_batch = to_batch_obs(np.asarray(action, dtype=np.float32))
        if deterministic:
            return action_batch[0]
        return self._apply_action_noise(action_batch)[0]

    def act_train_batch(self, observations: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not deterministic and self.state.total_steps < self.learning_starts:
            action_batch = self._sample_random_actions(int(observations.shape[0]))
            return action_batch, action_batch
        actions, _ = self.predict(observations, deterministic=True)
        action_batch = np.asarray(actions, dtype=np.float32)
        if not deterministic:
            action_batch = self._apply_action_noise(action_batch)
        return action_batch, action_batch

    def to_buffer_action(self, env_action: np.ndarray) -> np.ndarray:
        return np.asarray(env_action, dtype=np.float32)

    def observe_transition(self, transition: Transition) -> None:
        is_timeout = self._is_timeout(transition.info) or bool(transition.truncated)
        done = bool(transition.terminated or transition.truncated)
        self.replay_buffer.add(
            obs=np.asarray(transition.observation, dtype=np.float32),
            action=np.asarray(transition.buffer_action, dtype=np.float32),
            reward=float(transition.scalar_reward),
            done=done,
            timeout=is_timeout,
            next_obs=self._resolve_next_observation(
                transition.next_observation,
                transition.terminal_observation,
            ),
        )
        self.state.total_steps += 1
        self._rollout_transitions_since_update += 1

    def observe_transition_batch(
        self,
        observations: np.ndarray,
        buffer_actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_observations: np.ndarray,
        infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> None:
        resolved_next_observations = np.asarray(next_observations, dtype=np.float32).copy()
        stored_dones = np.asarray(dones, dtype=bool).copy()
        timeouts = np.zeros_like(stored_dones, dtype=bool)
        for idx, info in enumerate(infos):
            if self._is_timeout(info):
                timeouts[idx] = True
            terminal_observation = info.get("terminal_observation") if isinstance(info, dict) else None
            if terminal_observation is not None:
                resolved_next_observations[idx] = np.asarray(terminal_observation, dtype=np.float32)
        self.replay_buffer.add_batch(
            obs=observations,
            actions=buffer_actions,
            rewards=rewards,
            dones=stored_dones,
            timeouts=timeouts,
            next_obs=resolved_next_observations,
        )
        self.state.total_steps += int(observations.shape[0])
        self._rollout_transitions_since_update += int(observations.shape[0])
        self.debug_logger.record_collect(
            total_steps=int(self.state.total_steps),
            replay_size=int(self.replay_buffer.size),
            actions=np.asarray(buffer_actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=stored_dones * (~timeouts),
            infos=infos,
            warmup_active=bool(self.state.total_steps <= self.learning_starts),
        )

    def maybe_update(
        self,
        collected_steps: int,
        step_count: int,
        global_total_timesteps: int | None,
        global_steps_done: int,
    ) -> dict[str, float | int]:
        del step_count, global_total_timesteps, global_steps_done
        self._rollout_steps_since_update += int(collected_steps)
        if self.state.total_steps < self.learning_starts:
            self._rollout_steps_since_update = 0
            self._rollout_transitions_since_update = 0
            return {}
        if self._rollout_steps_since_update < self.train_freq:
            return {}

        gradient_steps = self._resolve_gradient_steps(self._rollout_transitions_since_update)
        self._rollout_steps_since_update = 0
        self._rollout_transitions_since_update = 0

        critic_losses: list[float] = []
        actor_losses: list[float] = []

        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(self.batch_size, device=self.device)
            obs = batch["obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"]
            next_obs = batch["next_obs"]

            with torch.no_grad():
                noise = torch.randn_like(actions) * self.target_policy_noise
                noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
                next_actions = torch.clamp(self.actor_target(next_obs) + noise, -1.0, 1.0)
                q1_t, q2_t = self.critic_target(next_obs, next_actions)
                q_target = rewards + (1.0 - dones) * self.gamma * torch.min(q1_t, q2_t)

            q1, q2 = self.critic(obs, actions)
            critic_loss = torch.mean((q1 - q_target) ** 2) + torch.mean((q2 - q_target) ** 2)

            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()
            critic_losses.append(float(critic_loss.detach().cpu().item()))

            self.state.update_steps += 1
            if self.state.update_steps % self.policy_delay == 0:
                actor_actions = self.actor(obs)
                actor_loss = -self.critic.q1(obs, actor_actions).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_opt.step()
                actor_losses.append(float(actor_loss.detach().cpu().item()))

                soft_update(self.actor, self.actor_target, self.tau)
                soft_update(self.critic, self.critic_target, self.tau)

            with torch.no_grad():
                policy_actions = self.actor(obs)
                debug_metrics = {
                    "batch_reward_mean": float(rewards.mean().detach().cpu().item()),
                    "batch_reward_std": float(rewards.std(unbiased=False).detach().cpu().item()),
                    "batch_done_rate": float(dones.mean().detach().cpu().item()),
                    "batch_action_abs_mean": float(actions.abs().mean().detach().cpu().item()),
                    "policy_action_abs_mean": float(policy_actions.abs().mean().detach().cpu().item()),
                    "target_action_abs_mean": float(next_actions.abs().mean().detach().cpu().item()),
                    "q_target_mean": float(q_target.mean().detach().cpu().item()),
                    "q1_mean": float(q1.mean().detach().cpu().item()),
                    "q2_mean": float(q2.mean().detach().cpu().item()),
                    "critic_loss": float(critic_losses[-1]) if critic_losses else float("nan"),
                    "actor_loss": float(actor_losses[-1]) if actor_losses else float("nan"),
                    "random_warmup_active": False,
                }
                self.debug_logger.record_update(
                    total_steps=int(self.state.total_steps),
                    replay_size=int(self.replay_buffer.size),
                    metrics=debug_metrics,
                )

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else float("nan"),
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else float("nan"),
            "learning_rate": float(self.actor_opt.param_groups[0]["lr"]),
            "update_calls": 1,
            "gradient_steps": int(gradient_steps),
        }

    def end_training(self) -> None:
        self.debug_logger.close(total_steps=int(self.state.total_steps), replay_size=int(self.replay_buffer.size))

    def on_episode_end(self, indices: list[int] | np.ndarray | None = None) -> None:
        if self.action_noise is not None:
            self.action_noise.reset(indices=indices)
