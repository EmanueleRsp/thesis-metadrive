from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import VecEnv

from thesis_rl.agents.planner_lifecycle import BasePlannerLifecycle, Td3Lifecycle


class Td3PlannerBackend:
    """Thin TD3 wrapper exposing a stable planner-facing interface."""

    def __init__(self, model: TD3) -> None:
        self.model = model

    @property
    def sb3_model(self) -> TD3:
        """Expose underlying Stable-Baselines3 model."""
        return self.model

    def get_lifecycle(self) -> BasePlannerLifecycle:
        """Get granular training lifecycle interface.

        Returns:
            Td3Lifecycle instance implementing BasePlannerLifecycle protocol.
        """
        return Td3Lifecycle(self)

    @staticmethod
    def _build_action_noise(env: Any, cfg_planner: Any):
        noise_type = str(cfg_planner.get("action_noise_type", "none")).lower()
        if noise_type == "none":
            return None

        if noise_type != "normal":
            raise ValueError(
                f"Unsupported action_noise_type '{cfg_planner.action_noise_type}'. "
                "Currently supported: none, normal"
            )

        action_space = getattr(env, "action_space", None)
        if action_space is None or not hasattr(action_space, "shape"):
            raise ValueError("TD3 normal action noise requires an environment with a shaped action space")

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

    @classmethod
    def build(
        cls,
        env: Any,
        cfg_planner: Any,
        device: str = "auto",
        seed: int | None = None,
    ) -> "Td3PlannerBackend":
        planner_cfg = cfg_planner
        model = TD3(
            policy=planner_cfg.policy,
            env=env,
            learning_starts=int(planner_cfg.learning_starts),
            batch_size=int(planner_cfg.batch_size),
            buffer_size=int(planner_cfg.buffer_size),
            train_freq=int(planner_cfg.train_freq),
            gradient_steps=int(planner_cfg.gradient_steps),
            learning_rate=float(planner_cfg.learning_rate),
            gamma=float(planner_cfg.gamma),
            tau=float(planner_cfg.tau),
            action_noise=cls._build_action_noise(env, planner_cfg),
            policy_kwargs={"net_arch": list(planner_cfg.policy_kwargs.net_arch)},
            verbose=int(planner_cfg.verbose),
            device=device,
            seed=seed,
        )
        return cls(model=model)

    @classmethod
    def load(cls, checkpoint_path: str | Path, env: Any, device: str = "auto") -> "Td3PlannerBackend":
        model = TD3.load(str(checkpoint_path), env=env, device=device)
        return cls(model=model)

    def predict(self, observation: Any, deterministic: bool = False):
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state

    # def evaluate(self, env: Any, n_eval_episodes: int, deterministic: bool = False) -> dict[str, float]:
    #     episode_returns: list[float] = []
    #     for _ in range(n_eval_episodes):
    #         obs, _ = env.reset()
    #         done = False
    #         truncated = False
    #         ep_return = 0.0

    #         while not (done or truncated):
    #             action, _ = self.model.predict(obs, deterministic=deterministic)
    #             obs, reward, done, truncated, _ = env.step(action)
    #             ep_return += float(reward)

    #         episode_returns.append(ep_return)

    #     mean_reward = float(np.mean(episode_returns))
    #     std_reward = float(np.std(episode_returns))
    #    return {"mean_reward": float(mean_reward), "std_reward": float(std_reward)}

    def set_env(self, env: Any) -> None:
        """Attach a new environment while keeping policy and replay state."""
        self.model.set_env(env)

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(checkpoint))
