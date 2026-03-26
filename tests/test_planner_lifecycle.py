"""Test planner training lifecycle protocol."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from thesis_rl.agents.planner_agent import Td3PlannerBackend
from thesis_rl.agents.planner_lifecycle import Td3Lifecycle
from thesis_rl.envs.factory import make_env


@pytest.fixture
def cfg_agent():
    """Minimal agent config for testing."""
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "sb3": {
            "policy": "MlpPolicy",
            "learning_starts": 100,
            "batch_size": 64,
            "buffer_size": 10000,
            "train_freq": 2,
            "gradient_steps": 1,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "tau": 0.005,
            "verbose": 0,
        }
    })


@pytest.fixture
def cfg_planner():
    """Minimal planner config for testing."""
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "policy_kwargs": {
            "net_arch": [64, 64],
        }
    })


@pytest.fixture
def env():
    """Minimal MetaDrive environment."""
    from omegaconf import OmegaConf

    cfg_env = {
        "num_scenarios": 1,
        "horizon": 50,
        "vehicle_config": {
            "lidar": {
                "num_lasers": 12,
                "distance": 20,
            }
        },
        "use_render": False,
        "image_observation": False,
    }
    env_instance = make_env(OmegaConf.create({"config": cfg_env}))
    try:
        yield env_instance
    finally:
        env_instance.close()


def test_get_lifecycle_returns_protocol(cfg_agent, cfg_planner, env):
    """Test that get_lifecycle() returns a valid lifecycle."""
    planner = Td3PlannerBackend.build(env, cfg_agent, cfg_planner, device="cpu")
    lifecycle = planner.get_lifecycle()

    assert isinstance(lifecycle, Td3Lifecycle)
    assert hasattr(lifecycle, "begin_training")
    assert hasattr(lifecycle, "act")
    assert hasattr(lifecycle, "observe_transition")
    assert hasattr(lifecycle, "maybe_update")
    assert hasattr(lifecycle, "end_training")


def test_lifecycle_protocol_implemented(cfg_agent, cfg_planner, env):
    """Test that Td3Lifecycle fully implements BasePlannerLifecycle protocol."""
    planner = Td3PlannerBackend.build(env, cfg_agent, cfg_planner, device="cpu")
    lifecycle = planner.get_lifecycle()

    # Check protocol conformance
    assert callable(lifecycle.begin_training)
    assert callable(lifecycle.act)
    assert callable(lifecycle.observe_transition)
    assert callable(lifecycle.maybe_update)
    assert callable(lifecycle.end_training)


def test_lifecycle_training_loop(cfg_agent, cfg_planner, env):
    """Test a minimal training loop using lifecycle interface."""
    planner = Td3PlannerBackend.build(env, cfg_agent, cfg_planner, device="cpu")
    lifecycle = planner.get_lifecycle()

    # Begin training
    lifecycle.begin_training()

    obs, _ = env.reset()
    total_steps = 200

    for _ in range(total_steps):
        # Act
        action = lifecycle.act(obs, deterministic=False)
        assert isinstance(action, np.ndarray)

        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)

        # Observe transition
        lifecycle.observe_transition(
            observation=obs,
            action=action,
            reward=float(reward),
            done=bool(done),
            next_observation=next_obs,
        )

        # Maybe update
        lifecycle.maybe_update()

        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()

    # End training
    lifecycle.end_training()

    # Verify that replay buffer accumulated steps
    assert lifecycle.replay_buffer.size() > 0


def test_lifecycle_act_returns_valid_action(cfg_agent, cfg_planner, env):
    """Test that lifecycle.act() returns properly shaped actions."""
    planner = Td3PlannerBackend.build(env, cfg_agent, cfg_planner, device="cpu")
    lifecycle = planner.get_lifecycle()
    lifecycle.begin_training()

    obs, _ = env.reset()
    action = lifecycle.act(obs, deterministic=False)

    # TD3 continuous action space should be 1D array
    assert isinstance(action, np.ndarray)
    assert action.dtype == np.float32
    assert action.shape == env.action_space.shape


def test_lifecycle_step_counters(cfg_agent, cfg_planner, env):
    """Test that lifecycle correctly tracks step and update counts."""
    planner = Td3PlannerBackend.build(env, cfg_agent, cfg_planner, device="cpu")
    lifecycle = planner.get_lifecycle()

    lifecycle.begin_training()
    assert lifecycle.step_count == 0
    assert lifecycle.update_count == 0

    obs, _ = env.reset()

    # Simulate 150 steps (100 learning_starts + 50 updates)
    for _ in range(150):
        action = lifecycle.act(obs, deterministic=False)
        next_obs, reward, done, truncated, _ = env.step(action)

        lifecycle.observe_transition(obs, action, float(reward), bool(done), next_obs)
        lifecycle.maybe_update()

        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()

    # step_count should be 150
    assert lifecycle.step_count == 150
    # update_count depends on learning_starts and train_freq
    # With learning_starts=100 and train_freq=2, we expect ~25 updates
    assert lifecycle.update_count > 0


def test_lifecycle_save_load_compatibility(cfg_agent, cfg_planner, env):
    """Test that lifecycle-trained checkpoints can be loaded."""
    planner = Td3PlannerBackend.build(env, cfg_agent, cfg_planner, device="cpu")
    lifecycle = planner.get_lifecycle()

    # Train with lifecycle
    lifecycle.begin_training()
    obs, _ = env.reset()
    for _ in range(100):
        action = lifecycle.act(obs, deterministic=False)
        next_obs, reward, done, truncated, _ = env.step(action)
        lifecycle.observe_transition(obs, action, float(reward), bool(done), next_obs)
        lifecycle.maybe_update()
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()
    lifecycle.end_training()

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.zip"
        planner.save(checkpoint_path)

        # Load and verify
        loaded_planner = Td3PlannerBackend.load(checkpoint_path, env, device="cpu")
        loaded_lifecycle = loaded_planner.get_lifecycle()

        # Loaded lifecycle should work
        assert isinstance(loaded_lifecycle, Td3Lifecycle)
        assert loaded_lifecycle.step_count == 0  # Reset on load
