from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf

from thesis_rl.agent.planners.algorithms import Sb3PpoPlannerBackend, Sb3SacPlannerBackend
from thesis_rl.agent.types import Transition


@pytest.fixture
def env():
    env_instance = gym.make("Pendulum-v1")
    try:
        yield env_instance
    finally:
        env_instance.close()


def test_sac_sb3_builds_collects_and_updates(env) -> None:
    pytest.importorskip("stable_baselines3")
    planner = Sb3SacPlannerBackend.build(
        env=env,
        cfg_planner=OmegaConf.create(
            {
                "policy": "MlpPolicy",
                "learning_starts": 0,
                "batch_size": 8,
                "buffer_size": 128,
                "train_freq": 1,
                "gradient_steps": 1,
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "tau": 0.005,
                "ent_coef": "auto",
                "target_update_interval": 1,
                "target_entropy": "auto",
            }
        ),
        device="cpu",
        seed=123,
    )
    lifecycle = planner.get_lifecycle()
    lifecycle.begin_training(chunk_timesteps=8, global_total_timesteps=8, global_steps_done=0)

    obs, _ = env.reset(seed=123)
    action = lifecycle.act(np.asarray(obs, dtype=np.float32), deterministic=False)
    next_obs, reward, terminated, truncated, info = env.step(action)
    lifecycle.observe_transition(
        Transition(
            observation=np.asarray(obs, dtype=np.float32),
            env_action=np.asarray(action, dtype=np.float32),
            buffer_action=lifecycle.to_buffer_action(np.asarray(action, dtype=np.float32)),
            scalar_reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            next_observation=np.asarray(next_obs, dtype=np.float32),
            terminal_observation=np.asarray(next_obs, dtype=np.float32)
            if (terminated or truncated)
            else None,
            info=dict(info),
        )
    )
    lifecycle.maybe_update()

    assert planner.model.replay_buffer.size() >= 1
    assert planner.model.num_timesteps >= 1


def test_ppo_sb3_builds_collects_and_updates(env) -> None:
    pytest.importorskip("stable_baselines3")
    planner = Sb3PpoPlannerBackend.build(
        env=env,
        cfg_planner=OmegaConf.create(
            {
                "policy": "MlpPolicy",
                "n_steps": 2,
                "batch_size": 2,
                "n_epochs": 1,
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "clip_range": 0.2,
                "normalize_advantage": True,
                "max_grad_norm": 0.5,
            }
        ),
        device="cpu",
        seed=123,
    )
    lifecycle = planner.get_lifecycle()
    lifecycle.begin_training(chunk_timesteps=4, global_total_timesteps=4, global_steps_done=0)

    obs, _ = env.reset(seed=123)
    for _ in range(2):
        action = lifecycle.act(np.asarray(obs, dtype=np.float32), deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        lifecycle.observe_transition(
            Transition(
                observation=np.asarray(obs, dtype=np.float32),
                env_action=np.asarray(action, dtype=np.float32),
                buffer_action=lifecycle.to_buffer_action(np.asarray(action, dtype=np.float32)),
                scalar_reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                next_observation=np.asarray(next_obs, dtype=np.float32),
                terminal_observation=np.asarray(next_obs, dtype=np.float32)
                if (terminated or truncated)
                else None,
                info=dict(info),
            )
        )
        lifecycle.maybe_update()
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            lifecycle.on_episode_end()

    assert planner.model.num_timesteps >= 2
    assert planner.model.rollout_buffer.pos in (0, 1)
