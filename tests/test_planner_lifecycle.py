"""Tests for TD3 planner lifecycle integration."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf

from thesis_rl.agents.planner_agent import Td3PlannerBackend
from thesis_rl.agents.planner_lifecycle import Td3Lifecycle
from thesis_rl.agents.types import Transition


@pytest.fixture
def cfg_planner():
    return OmegaConf.create(
        {
            "policy": "MlpPolicy",
            "learning_starts": 5,
            "batch_size": 8,
            "buffer_size": 2000,
            "train_freq": 1,
            "gradient_steps": 1,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "tau": 0.005,
            "verbose": 0,
            "action_noise_type": "normal",
            "action_noise_sigma": 0.1,
            "action_noise_mean": 0.0,
            "policy_kwargs": {"net_arch": [32, 32]},
        }
    )


@pytest.fixture
def env():
    env_instance = gym.make("Pendulum-v1")
    try:
        yield env_instance
    finally:
        env_instance.close()


def _make_transition(
    lifecycle: Td3Lifecycle,
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    next_obs: np.ndarray,
) -> Transition:
    terminal_obs = next_obs if (terminated or truncated) else None
    return Transition(
        observation=np.asarray(obs, dtype=np.float32),
        env_action=np.asarray(action, dtype=np.float32),
        buffer_action=lifecycle.to_buffer_action(np.asarray(action, dtype=np.float32)),
        scalar_reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        next_observation=np.asarray(next_obs, dtype=np.float32),
        terminal_observation=np.asarray(terminal_obs, dtype=np.float32) if terminal_obs is not None else None,
        info={},
    )


def test_get_lifecycle_returns_protocol(cfg_planner, env):
    planner = Td3PlannerBackend.build(env, cfg_planner, device="cpu", seed=123)
    lifecycle = planner.get_lifecycle()

    assert isinstance(lifecycle, Td3Lifecycle)
    assert callable(lifecycle.begin_training)
    assert callable(lifecycle.act)
    assert callable(lifecycle.observe_transition)
    assert callable(lifecycle.maybe_update)
    assert callable(lifecycle.end_training)


def test_lifecycle_training_loop_collects_replay(cfg_planner, env):
    planner = Td3PlannerBackend.build(env, cfg_planner, device="cpu", seed=123)
    lifecycle = planner.get_lifecycle()
    lifecycle.begin_training(chunk_timesteps=40, global_total_timesteps=100, global_steps_done=0)

    obs, _ = env.reset(seed=123)
    for _ in range(40):
        action = lifecycle.act(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        transition = _make_transition(lifecycle, obs, action, reward, terminated, truncated, next_obs)
        lifecycle.observe_transition(transition)
        lifecycle.maybe_update()
        obs = next_obs
        if terminated or truncated:
            lifecycle.on_episode_end()
            obs, _ = env.reset()

    lifecycle.end_training()

    assert lifecycle.replay_buffer.size() > 0


def test_lifecycle_act_returns_valid_action(cfg_planner, env):
    planner = Td3PlannerBackend.build(env, cfg_planner, device="cpu", seed=123)
    lifecycle = planner.get_lifecycle()
    lifecycle.begin_training(chunk_timesteps=1, global_total_timesteps=10, global_steps_done=0)

    obs, _ = env.reset(seed=123)
    action = lifecycle.act(obs, deterministic=False)

    assert isinstance(action, np.ndarray)
    assert action.shape == env.action_space.shape
    assert np.all(action <= env.action_space.high + 1e-6)
    assert np.all(action >= env.action_space.low - 1e-6)


def test_lifecycle_step_and_update_counters_progress(cfg_planner, env):
    planner = Td3PlannerBackend.build(env, cfg_planner, device="cpu", seed=123)
    lifecycle = planner.get_lifecycle()
    lifecycle.begin_training(chunk_timesteps=30, global_total_timesteps=200, global_steps_done=50)
    assert lifecycle.step_count == 0
    assert lifecycle.update_count == 0

    obs, _ = env.reset(seed=123)
    for _ in range(30):
        action = lifecycle.act(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        lifecycle.observe_transition(
            _make_transition(lifecycle, obs, action, reward, terminated, truncated, next_obs)
        )
        lifecycle.maybe_update()
        obs = next_obs
        if terminated or truncated:
            lifecycle.on_episode_end()
            obs, _ = env.reset()

    assert lifecycle.step_count == 30
    assert lifecycle.update_count > 0


def test_lifecycle_save_load_compatibility(cfg_planner, env, tmp_path: Path):
    planner = Td3PlannerBackend.build(env, cfg_planner, device="cpu", seed=123)
    lifecycle = planner.get_lifecycle()
    lifecycle.begin_training(chunk_timesteps=12, global_total_timesteps=50, global_steps_done=0)

    obs, _ = env.reset(seed=123)
    for _ in range(12):
        action = lifecycle.act(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        lifecycle.observe_transition(
            _make_transition(lifecycle, obs, action, reward, terminated, truncated, next_obs)
        )
        lifecycle.maybe_update()
        obs = next_obs
        if terminated or truncated:
            lifecycle.on_episode_end()
            obs, _ = env.reset()

    checkpoint_path = tmp_path / "test_checkpoint.zip"
    planner.save(checkpoint_path)

    loaded_planner = Td3PlannerBackend.load(checkpoint_path, env, device="cpu")
    loaded_lifecycle = loaded_planner.get_lifecycle()

    assert isinstance(loaded_lifecycle, Td3Lifecycle)
    assert loaded_lifecycle.step_count == 0
