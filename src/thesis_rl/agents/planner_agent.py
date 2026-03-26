from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import TD3

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

    @classmethod
    def build(cls, env: Any, cfg_agent: Any, cfg_planner: Any, device: str = "auto") -> "Td3PlannerBackend":
        sb3_cfg = cfg_agent.sb3
        model = TD3(
            policy=sb3_cfg.policy,
            env=env,
            learning_starts=int(sb3_cfg.learning_starts),
            batch_size=int(sb3_cfg.batch_size),
            buffer_size=int(sb3_cfg.buffer_size),
            train_freq=int(sb3_cfg.train_freq),
            gradient_steps=int(sb3_cfg.gradient_steps),
            learning_rate=float(sb3_cfg.learning_rate),
            gamma=float(sb3_cfg.gamma),
            tau=float(sb3_cfg.tau),
            policy_kwargs={"net_arch": list(cfg_planner.policy_kwargs.net_arch)},
            verbose=int(sb3_cfg.verbose),
            device=device,
        )
        return cls(model=model)

    @classmethod
    def load(cls, checkpoint_path: str | Path, env: Any, device: str = "auto") -> "Td3PlannerBackend":
        model = TD3.load(str(checkpoint_path), env=env, device=device)
        return cls(model=model)

    def predict(self, observation: Any, deterministic: bool = False):
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state

    def evaluate(self, env: Any, n_eval_episodes: int, deterministic: bool = False) -> dict[str, float]:
        episode_returns: list[float] = []
        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            ep_return = 0.0

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, _ = env.step(action)
                ep_return += float(reward)

            episode_returns.append(ep_return)

        mean_reward = float(np.mean(episode_returns))
        std_reward = float(np.std(episode_returns))
        return {"mean_reward": float(mean_reward), "std_reward": float(std_reward)}

    def set_env(self, env: Any) -> None:
        """Attach a new environment while keeping policy and replay state."""
        self.model.set_env(env)

    def save(self, checkpoint_path: str | Path) -> None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(checkpoint))
