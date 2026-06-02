from __future__ import annotations

import gymnasium as gym
import pytest
from omegaconf import OmegaConf

from thesis_rl.agent.planners.algorithms import SacPlannerBackend


@pytest.fixture
def env():
    env_instance = gym.make("Pendulum-v1")
    try:
        yield env_instance
    finally:
        env_instance.close()


def test_sac_target_entropy_auto_uses_negative_action_dim(env) -> None:
    cfg_planner = OmegaConf.create(
        {
            "learning_starts": 5,
            "batch_size": 8,
            "buffer_size": 2000,
            "train_freq": 1,
            "gradient_steps": 1,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": "auto",
            "target_entropy": "auto",
            "log_std_bounds": [-20, 2],
        }
    )

    planner = SacPlannerBackend.build(
        env,
        cfg_planner,
        cfg_encoder={"type": "none"},
        cfg_decoder={"type": "mlp", "hidden_layers": [32, 32]},
        cfg_obs={},
        device="cpu",
        seed=123,
    )

    assert planner.target_entropy == -float(env.action_space.shape[0])
