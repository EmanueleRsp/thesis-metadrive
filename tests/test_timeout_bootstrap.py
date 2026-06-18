from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from thesis_rl.agent.planners.algorithms import PpoPlannerBackend, Td3PlannerBackend
from thesis_rl.agent.types import Transition


@pytest.fixture
def env():
    env_instance = gym.make("Pendulum-v1")
    try:
        yield env_instance
    finally:
        env_instance.close()


def test_td3_timeout_transition_uses_terminal_obs_and_non_terminal_done(env) -> None:
    cfg_planner = OmegaConf.create(
        {
            "learning_starts": 5,
            "batch_size": 8,
            "buffer_size": 128,
            "train_freq": 1,
            "gradient_steps": 1,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "tau": 0.005,
            "action_noise_sigma": 0.1,
        }
    )
    planner = Td3PlannerBackend.build(
        env,
        cfg_planner,
        cfg_encoder={"type": "none"},
        cfg_decoder={"type": "mlp", "hidden_layers": [32, 32]},
        cfg_obs={},
        device="cpu",
        seed=123,
    )

    transition = Transition(
        observation=np.zeros(env.observation_space.shape, dtype=np.float32),
        env_action=np.zeros(env.action_space.shape, dtype=np.float32),
        buffer_action=np.zeros(env.action_space.shape, dtype=np.float32),
        scalar_reward=1.0,
        terminated=False,
        truncated=True,
        next_observation=np.full(env.observation_space.shape, 99.0, dtype=np.float32),
        terminal_observation=np.full(env.observation_space.shape, 7.0, dtype=np.float32),
        info={"TimeLimit.truncated": True},
    )

    planner.observe_transition(transition)

    assert planner.replay_buffer.dones[0, 0] == 1.0
    assert planner.replay_buffer.timeouts[0, 0] == 1.0
    assert np.allclose(planner.replay_buffer.next_obs[0], 7.0)
    batch = planner.replay_buffer.sample(batch_size=1, device=torch.device("cpu"))
    assert float(batch["dones"][0, 0].item()) == pytest.approx(0.0)


def test_ppo_timeout_bootstraps_reward_from_terminal_value(env) -> None:
    cfg_planner = OmegaConf.create(
        {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    )
    planner = PpoPlannerBackend.build(
        env,
        cfg_planner,
        cfg_encoder={"type": "none"},
        cfg_decoder={"type": "mlp", "hidden_layers": [32, 32]},
        cfg_obs={},
        device="cpu",
        seed=123,
    )

    obs = np.zeros(env.observation_space.shape, dtype=np.float32)
    planner.act_train(obs, deterministic=False)
    terminal_obs = np.full(env.observation_space.shape, 0.5, dtype=np.float32)
    with torch.no_grad():
        terminal_value = float(
            planner._value(torch.as_tensor(terminal_obs[None, :], dtype=torch.float32, device=planner.device))
            .squeeze()
            .cpu()
            .item()
        )

    transition = Transition(
        observation=obs,
        env_action=np.zeros(env.action_space.shape, dtype=np.float32),
        buffer_action=np.zeros(env.action_space.shape, dtype=np.float32),
        scalar_reward=1.0,
        terminated=False,
        truncated=True,
        next_observation=np.full(env.observation_space.shape, 42.0, dtype=np.float32),
        terminal_observation=terminal_obs,
        info={"TimeLimit.truncated": True},
    )

    planner.observe_transition(transition)

    expected_reward = 1.0 + planner.gamma * terminal_value
    assert planner.rollout.rewards[0, 0] == pytest.approx(expected_reward)
    assert planner.rollout.episode_starts[0, 0] == pytest.approx(1.0)
    assert planner._current_episode_starts[0] == pytest.approx(1.0)
