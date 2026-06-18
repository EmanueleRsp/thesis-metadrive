from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from thesis_rl.agent.types import Transition
from thesis_rl.agent.planners.decoders.factory import build_decoder
from thesis_rl.agent.planners.core.backend_base import BasePlannerBackend
from thesis_rl.agent.planners.modules.actor_critic import SquashedGaussianActor, TwinQCritic
from thesis_rl.agent.planners.core.utils import (
    assert_box_spaces,
    build_encoder_for_env,
    normalize_checkpoint_path,
    soft_update,
    to_batch_obs,
    to_plain_dict,
)
from thesis_rl.agent.planners.core.types import TrainState
from thesis_rl.agent.planners.core.lifecycle import SacLifecycle
from thesis_rl.agent.planners.core.buffers import ReplayBuffer
from thesis_rl.agent.planners.core.offpolicy_debug import OffPolicyDebugLogger


class SacPlannerBackend(BasePlannerBackend):
    lifecycle_cls = SacLifecycle
    _SB3_SAC_HIDDEN_LAYERS = [256, 256]

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

        log_std_bounds = tuple(self.cfg_planner.get("log_std_bounds", [-20.0, 2.0]))
        self.actor = SquashedGaussianActor(
            encoder=enc_actor,
            decoder=actor_decoder,
            action_dim=self.action_dim,
            log_std_bounds=(float(log_std_bounds[0]), float(log_std_bounds[1])),
            log_std_init=float(self.cfg_planner.get("log_std_init", -3.0)),
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
        if self.train_freq <= 0:
            raise ValueError("`train_freq` must be >= 1 for SAC.")
        self.gradient_steps_cfg = self.cfg_planner.get("gradient_steps", "auto")
        self.target_update_interval = int(self.cfg_planner.get("target_update_interval", 1))
        if self.target_update_interval <= 0:
            raise ValueError("`target_update_interval` must be >= 1 for SAC.")
        self._rollout_steps_since_update = 0
        self._rollout_transitions_since_update = 0

        ent_coef_cfg = self.cfg_planner.get("ent_coef", "auto")
        ent_coef_str = str(ent_coef_cfg).strip().lower() if isinstance(ent_coef_cfg, str) else None
        self.auto_alpha = ent_coef_str == "auto" or (ent_coef_str is not None and ent_coef_str.startswith("auto_"))
        if self.auto_alpha:
            init = self._resolve_auto_ent_coef_init(ent_coef_cfg)
            self.log_alpha = torch.tensor(np.log(max(init, 1e-6)), dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
            self.fixed_alpha = None
        else:
            fixed_alpha = float(ent_coef_cfg)
            if fixed_alpha < 0.0:
                raise ValueError("SAC `ent_coef` must be >= 0 when specified as a float.")
            self.log_alpha = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            self.alpha_opt = None
            self.fixed_alpha = torch.tensor(fixed_alpha, dtype=torch.float32, device=self.device)
        self.target_entropy = self._resolve_target_entropy(self.cfg_planner.get("target_entropy", "auto"))

        self.replay_buffer = ReplayBuffer(
            capacity=int(self.cfg_planner.get("buffer_size", 300000)),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_envs=self.n_envs,
        )
        self.debug_logger = OffPolicyDebugLogger(
            self.cfg_planner.get("offpolicy_debug"),
            algorithm="sac",
            action_dim=self.action_dim,
        )

    def _validate_network_config(self) -> None:
        decoder_name = str(self.cfg_decoder.get("name", "")).strip().lower()
        if decoder_name != "sac_sb3":
            return

        encoder_type = str(self.cfg_encoder.get("type", "none")).strip().lower()
        if encoder_type != "none":
            raise ValueError("`decoder=sac_sb3` requires `encoder=none` for SB3-faithful SAC.")

        decoder_type = str(self.cfg_decoder.get("type", "mlp")).strip().lower()
        hidden_layers = [int(width) for width in self.cfg_decoder.get("hidden_layers", [])]
        activation = str(self.cfg_decoder.get("activation", "relu")).strip().lower()
        layer_norm = bool(self.cfg_decoder.get("layer_norm", False))
        dropout = float(self.cfg_decoder.get("dropout", 0.0))
        log_std_bounds = tuple(float(v) for v in self.cfg_planner.get("log_std_bounds", [-20.0, 2.0]))

        if decoder_type != "mlp":
            raise ValueError("`decoder=sac_sb3` must use an MLP decoder.")
        if hidden_layers != self._SB3_SAC_HIDDEN_LAYERS:
            raise ValueError(
                "`decoder=sac_sb3` must keep hidden_layers=[256, 256] for SB3-faithful SAC."
            )
        if activation != "relu":
            raise ValueError("`decoder=sac_sb3` must use ReLU activations for SB3-faithful SAC.")
        if layer_norm:
            raise ValueError("`decoder=sac_sb3` must keep layer_norm=false for SB3-faithful SAC.")
        if dropout != 0.0:
            raise ValueError("`decoder=sac_sb3` must keep dropout=0.0 for SB3-faithful SAC.")
        if log_std_bounds != (-20.0, 2.0):
            raise ValueError(
                "`decoder=sac_sb3` must keep log_std_bounds=[-20, 2] for SB3-faithful SAC."
            )

    def _sample_random_actions(self, batch_size: int) -> np.ndarray:
        low = np.broadcast_to(self.action_low, (int(batch_size), self.action_dim))
        high = np.broadcast_to(self.action_high, (int(batch_size), self.action_dim))
        return np.random.uniform(low=low, high=high).astype(np.float32)

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

    @property
    def alpha(self) -> torch.Tensor:
        if self.fixed_alpha is not None:
            return self.fixed_alpha
        return torch.exp(self.log_alpha.detach())

    @staticmethod
    def _resolve_auto_ent_coef_init(ent_coef_cfg: Any) -> float:
        ent_coef_str = str(ent_coef_cfg).strip().lower()
        if ent_coef_str == "auto":
            return 1.0
        if not ent_coef_str.startswith("auto_"):
            raise ValueError(
                "Invalid SAC `ent_coef`: expected a float, 'auto', or 'auto_<positive_init>', "
                f"got {ent_coef_cfg!r}"
            )
        init = float(ent_coef_str.split("_", 1)[1])
        if init <= 0.0:
            raise ValueError("The initial value of SAC `ent_coef` must be greater than 0.")
        return init

    def _resolve_target_entropy(self, target_entropy_cfg: Any) -> float:
        if isinstance(target_entropy_cfg, str) and target_entropy_cfg.lower() == "auto":
            return -float(self.action_dim)
        try:
            return float(target_entropy_cfg)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid SAC `target_entropy`: expected a float or 'auto', "
                f"got {target_entropy_cfg!r}"
            ) from exc

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
        payload = torch.load(str(normalize_checkpoint_path(checkpoint_path)), map_location="cpu")
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
        if not deterministic and self.state.total_steps < self.learning_starts:
            return self._sample_random_actions(1)[0]
        action, _ = self.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def act_train_batch(self, observations: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not deterministic and self.state.total_steps < self.learning_starts:
            action_batch = self._sample_random_actions(int(observations.shape[0]))
            return action_batch, action_batch
        actions, _ = self.predict(observations, deterministic=deterministic)
        action_batch = np.asarray(actions, dtype=np.float32)
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

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        alpha_losses: list[float] = []

        for gradient_step in range(gradient_steps):
            batch = self.replay_buffer.sample(self.batch_size, device=self.device)
            obs = batch["obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"]
            next_obs = batch["next_obs"]

            actions_pi, log_prob, _ = self.actor.sample(obs, deterministic=False)
            if self.auto_alpha and self.alpha_opt is not None:
                alpha = torch.exp(self.log_alpha.detach())
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()
                alpha_losses.append(float(alpha_loss.detach().cpu().item()))
            else:
                alpha = self.alpha

            with torch.no_grad():
                next_actions, next_log_prob, _ = self.actor.sample(next_obs, deterministic=False)
                q1_t, q2_t = self.critic_target(next_obs, next_actions)
                min_q_t = torch.min(q1_t, q2_t) - alpha * next_log_prob
                q_target = rewards + (1.0 - dones) * self.gamma * min_q_t

            q1, q2 = self.critic(obs, actions)
            critic_loss = 0.5 * (torch.mean((q1 - q_target) ** 2) + torch.mean((q2 - q_target) ** 2))
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()
            critic_losses.append(float(critic_loss.detach().cpu().item()))

            q1_pi, q2_pi = self.critic(obs, actions_pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (alpha * log_prob - min_q_pi).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            actor_losses.append(float(actor_loss.detach().cpu().item()))

            if gradient_step % self.target_update_interval == 0:
                soft_update(self.critic, self.critic_target, self.tau)
            self.state.update_steps += 1
            with torch.no_grad():
                debug_metrics = {
                    "batch_reward_mean": float(rewards.mean().detach().cpu().item()),
                    "batch_reward_std": float(rewards.std(unbiased=False).detach().cpu().item()),
                    "batch_done_rate": float(dones.mean().detach().cpu().item()),
                    "batch_action_abs_mean": float(actions.abs().mean().detach().cpu().item()),
                    "policy_action_abs_mean": float(actions_pi.abs().mean().detach().cpu().item()),
                    "policy_log_prob_mean": float(log_prob.mean().detach().cpu().item()),
                    "alpha": float(alpha.detach().cpu().item()),
                    "q_target_mean": float(q_target.mean().detach().cpu().item()),
                    "q1_mean": float(q1.mean().detach().cpu().item()),
                    "q2_mean": float(q2.mean().detach().cpu().item()),
                    "critic_loss": float(critic_losses[-1]) if critic_losses else float("nan"),
                    "actor_loss": float(actor_losses[-1]) if actor_losses else float("nan"),
                    "alpha_loss": float(alpha_losses[-1]) if alpha_losses else float("nan"),
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
