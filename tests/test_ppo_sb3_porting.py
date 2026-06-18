from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.distributions import Normal

from thesis_rl.agent.planners.algorithms import PpoPlannerBackend
from thesis_rl.agent.planners.core.buffers import RolloutBuffer


@pytest.fixture
def env():
    env_instance = gym.make("Pendulum-v1")
    try:
        yield env_instance
    finally:
        env_instance.close()


def _ppo_sb3_cfg() -> dict[str, object]:
    return {
        "learning_rate": 3e-4,
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


def test_ppo_sb3_decoder_requires_sb3_faithful_network_config(env) -> None:
    cfg_planner = OmegaConf.create(_ppo_sb3_cfg())

    with pytest.raises(ValueError, match="encoder=none"):
        PpoPlannerBackend.build(
            env,
            cfg_planner,
            cfg_encoder={"type": "mlp", "output_dim": 32},
            cfg_decoder={"name": "ppo_sb3", "type": "mlp", "hidden_layers": [64, 64], "activation": "tanh"},
            cfg_obs={},
            device="cpu",
            seed=123,
        )

    with pytest.raises(ValueError, match="Tanh activations"):
        PpoPlannerBackend.build(
            env,
            cfg_planner,
            cfg_encoder={"type": "none"},
            cfg_decoder={"name": "ppo_sb3", "type": "mlp", "hidden_layers": [64, 64], "activation": "relu"},
            cfg_obs={},
            device="cpu",
            seed=123,
        )


def test_ppo_stores_raw_action_and_log_prob_for_clipped_env_action(env) -> None:
    cfg_planner = OmegaConf.create({**_ppo_sb3_cfg(), "log_std_init": -20.0})
    planner = PpoPlannerBackend.build(
        env,
        cfg_planner,
        cfg_encoder={"type": "none"},
        cfg_decoder={"name": "ppo_sb3", "type": "mlp", "hidden_layers": [64, 64], "activation": "tanh"},
        cfg_obs={},
        device="cpu",
        seed=123,
    )

    with torch.no_grad():
        for param in planner.actor_decoder.parameters():
            param.zero_()
        planner.mu_head.weight.zero_()
        planner.mu_head.bias.fill_(10.0)
        planner.log_std.fill_(-20.0)

    obs = np.zeros(env.observation_space.shape, dtype=np.float32)
    action = planner.act_train(obs, deterministic=True)
    buffer_action = planner.to_buffer_action(action)

    expected_raw_action = np.full(env.action_space.shape, 10.0, dtype=np.float32)
    expected_env_action = np.asarray(env.action_space.high, dtype=np.float32)

    assert np.allclose(action, expected_env_action)
    assert np.allclose(buffer_action, expected_raw_action)

    obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=planner.device)
    log_prob, _entropy, _values = planner._evaluate_action(
        obs_t,
        torch.as_tensor(buffer_action[None, :], dtype=torch.float32, device=planner.device),
    )
    expected_log_prob = Normal(
        torch.full((1, planner.action_dim), 10.0, dtype=torch.float32, device=planner.device),
        torch.full((1, planner.action_dim), torch.exp(torch.tensor(-20.0)).item(), dtype=torch.float32, device=planner.device),
    ).log_prob(
        torch.as_tensor(buffer_action[None, :], dtype=torch.float32, device=planner.device)
    ).sum(dim=-1)

    assert planner._last_log_probs is not None
    assert float(planner._last_log_probs[0]) == pytest.approx(float(expected_log_prob.cpu().item()))
    assert float(log_prob.squeeze(-1).cpu().item()) == pytest.approx(float(expected_log_prob.cpu().item()))


def test_rollout_buffer_matches_sb3_episode_start_gae_semantics() -> None:
    buffer = RolloutBuffer(
        n_steps=3,
        n_envs=1,
        obs_dim=2,
        action_dim=1,
        gamma=0.99,
        gae_lambda=0.95,
    )

    buffer.add(
        obs=np.zeros((1, 2), dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.float32),
        rewards=np.asarray([1.0], dtype=np.float32),
        episode_starts=np.asarray([1.0], dtype=np.float32),
        values=np.asarray([0.5], dtype=np.float32),
        log_probs=np.asarray([0.0], dtype=np.float32),
    )
    buffer.add(
        obs=np.ones((1, 2), dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.float32),
        rewards=np.asarray([2.0], dtype=np.float32),
        episode_starts=np.asarray([0.0], dtype=np.float32),
        values=np.asarray([1.0], dtype=np.float32),
        log_probs=np.asarray([0.0], dtype=np.float32),
    )
    buffer.add(
        obs=np.full((1, 2), 2.0, dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.float32),
        rewards=np.asarray([3.0], dtype=np.float32),
        episode_starts=np.asarray([1.0], dtype=np.float32),
        values=np.asarray([1.5], dtype=np.float32),
        log_probs=np.asarray([0.0], dtype=np.float32),
    )

    buffer.compute_returns_and_advantages(
        last_values=np.asarray([2.0], dtype=np.float32),
        last_dones=np.asarray([0.0], dtype=np.float32),
    )

    adv2 = 3.0 + 0.99 * 2.0 - 1.5
    adv1 = 2.0 - 1.0
    adv0 = (1.0 + 0.99 * 1.0 - 0.5) + 0.99 * 0.95 * adv1
    expected_advantages = np.asarray([adv0, adv1, adv2], dtype=np.float32)

    assert np.allclose(buffer.advantages[:, 0], expected_advantages, atol=1e-6)
    assert np.allclose(buffer.returns[:, 0], expected_advantages + np.asarray([0.5, 1.0, 1.5], dtype=np.float32), atol=1e-6)


def test_ppo_update_config_rejects_invalid_sb3_scheduling(env) -> None:
    base_cfg = _ppo_sb3_cfg()

    with pytest.raises(ValueError, match="greater than 1"):
        PpoPlannerBackend.build(
            env,
            OmegaConf.create({**base_cfg, "batch_size": 1, "normalize_advantage": True}),
            cfg_encoder={"type": "none"},
            cfg_decoder={"name": "ppo_sb3", "type": "mlp", "hidden_layers": [64, 64], "activation": "tanh"},
            cfg_obs={},
            device="cpu",
            seed=123,
        )

    with pytest.raises(ValueError, match="clip_range_vf"):
        PpoPlannerBackend.build(
            env,
            OmegaConf.create({**base_cfg, "clip_range_vf": 0.0}),
            cfg_encoder={"type": "none"},
            cfg_decoder={"name": "ppo_sb3", "type": "mlp", "hidden_layers": [64, 64], "activation": "tanh"},
            cfg_obs={},
            device="cpu",
            seed=123,
        )


def test_ppo_warns_when_rollout_size_is_not_multiple_of_batch_size(env) -> None:
    with pytest.warns(UserWarning, match="not divisible by `batch_size`"):
        PpoPlannerBackend.build(
            env,
            OmegaConf.create({**_ppo_sb3_cfg(), "n_steps": 5, "batch_size": 4}),
            cfg_encoder={"type": "none"},
            cfg_decoder={"name": "ppo_sb3", "type": "mlp", "hidden_layers": [64, 64], "activation": "tanh"},
            cfg_obs={},
            device="cpu",
            seed=123,
        )
