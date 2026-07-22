from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf
from torch import nn

from thesis_rl.agent.planners.algorithms import Td3PlannerBackend
from thesis_rl.agent.planners.encoders.none_encoder import NoneEncoder


@pytest.fixture
def env():
    env_instance = gym.make("Pendulum-v1")
    try:
        yield env_instance
    finally:
        env_instance.close()


def _build_planner(
    env: gym.Env,
    *,
    cfg_planner: dict | None = None,
    cfg_encoder: dict | None = None,
    cfg_decoder: dict | None = None,
) -> Td3PlannerBackend:
    planner_cfg = {
        "learning_starts": 5,
        "batch_size": 8,
        "buffer_size": 128,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "action_noise_type": "normal",
        "action_noise_mean": 0.0,
        "action_noise_sigma": 0.1,
    }
    if cfg_planner is not None:
        planner_cfg.update(cfg_planner)

    encoder_cfg = {"type": "none"} if cfg_encoder is None else cfg_encoder
    decoder_cfg = (
        {
            "name": "td3_sb3",
            "type": "mlp",
            "hidden_layers": [400, 300],
            "activation": "relu",
            "dropout": 0.0,
            "layer_norm": False,
        }
        if cfg_decoder is None
        else cfg_decoder
    )

    return Td3PlannerBackend.build(
        env,
        OmegaConf.create(planner_cfg),
        cfg_encoder=OmegaConf.create(encoder_cfg),
        cfg_decoder=OmegaConf.create(decoder_cfg),
        cfg_obs={},
        device="cpu",
        seed=123,
    )


def test_td3_sb3_network_is_flat_mlp_without_layer_norm(env) -> None:
    planner = _build_planner(env)

    assert isinstance(planner.actor.encoder, NoneEncoder)
    assert isinstance(planner.critic.encoder, NoneEncoder)

    actor_linears = [
        module for module in planner.actor.decoder.net if isinstance(module, nn.Linear)
    ]
    critic_linears = [
        module for module in planner.critic.decoder_q1.net if isinstance(module, nn.Linear)
    ]

    assert [(layer.in_features, layer.out_features) for layer in actor_linears] == [
        (planner.obs_dim, 400),
        (400, 300),
    ]
    assert [(layer.in_features, layer.out_features) for layer in critic_linears] == [
        (planner.obs_dim + planner.action_dim, 400),
        (400, 300),
    ]
    assert not any(isinstance(module, nn.LayerNorm) for module in planner.actor.decoder.net)
    assert not any(isinstance(module, nn.LayerNorm) for module in planner.critic.decoder_q1.net)


def test_td3_sb3_decoder_requires_none_encoder(env) -> None:
    with pytest.raises(ValueError, match="requires `encoder=none`"):
        _build_planner(
            env,
            cfg_encoder={"type": "mlp", "hidden_layers": [64, 64], "output_dim": 64},
        )


def test_td3_predict_is_noise_free_even_when_exploration_noise_is_configured(env) -> None:
    planner = _build_planner(
        env,
        cfg_planner={"learning_starts": 0, "action_noise_mean": 0.35, "action_noise_sigma": 0.0},
    )
    obs, _ = env.reset(seed=123)

    action_1, _ = planner.predict(obs, deterministic=False)
    action_2, _ = planner.predict(obs, deterministic=False)

    assert np.allclose(action_1, action_2)


def test_td3_act_train_separates_random_warmup_from_post_warmup_action_noise(
    env, monkeypatch
) -> None:
    planner = _build_planner(
        env,
        cfg_planner={"learning_starts": 5, "action_noise_mean": 0.25, "action_noise_sigma": 0.0},
    )
    obs, _ = env.reset(seed=123)
    warmup_action = np.full(env.action_space.shape, 0.42, dtype=np.float32)

    def _fixed_random_actions(batch_size: int) -> np.ndarray:
        return np.repeat(warmup_action[None, :], int(batch_size), axis=0)

    monkeypatch.setattr(planner, "_sample_random_actions", _fixed_random_actions)

    planner.state.total_steps = 0
    assert np.allclose(planner.act_train(obs, deterministic=False), warmup_action)

    planner.state.total_steps = planner.learning_starts
    policy_action, _ = planner.predict(obs, deterministic=True)
    expected_noisy_action = np.clip(policy_action + 0.25, planner.action_low, planner.action_high)

    assert np.allclose(planner.act_train(obs, deterministic=False), expected_noisy_action)


def test_td3_act_train_with_action_noise_disabled_matches_policy_action(env) -> None:
    planner = _build_planner(
        env,
        cfg_planner={
            "learning_starts": 0,
            "action_noise_type": "none",
            "action_noise_mean": 0.25,
            "action_noise_sigma": 0.5,
        },
    )
    obs, _ = env.reset(seed=123)

    policy_action, _ = planner.predict(obs, deterministic=True)

    assert np.allclose(planner.act_train(obs, deterministic=False), policy_action)


def test_td3_gradient_steps_auto_matches_collected_rollout_transitions(env) -> None:
    planner = _build_planner(
        env,
        cfg_planner={
            "learning_starts": 0,
            "train_freq": 1,
            "gradient_steps": "auto",
            "batch_size": 2,
            "buffer_size": 64,
        },
    )
    batch_size = 4
    obs = np.zeros((batch_size, planner.obs_dim), dtype=np.float32)
    actions = np.zeros((batch_size, planner.action_dim), dtype=np.float32)
    rewards = np.zeros((batch_size,), dtype=np.float32)
    dones = np.zeros((batch_size,), dtype=bool)
    infos = [{} for _ in range(batch_size)]

    planner.observe_transition_batch(
        observations=obs,
        buffer_actions=actions,
        rewards=rewards,
        dones=dones,
        next_observations=obs.copy(),
        infos=infos,
    )
    metrics = planner.maybe_update(
        collected_steps=1,
        step_count=1,
        global_total_timesteps=None,
        global_steps_done=0,
    )

    assert int(metrics["update_calls"]) == 1
    assert int(metrics["gradient_steps"]) == batch_size


def test_td3_train_freq_counts_vecenv_rollout_steps_not_transitions(env) -> None:
    planner = _build_planner(
        env,
        cfg_planner={
            "learning_starts": 0,
            "train_freq": 2,
            "gradient_steps": 1,
            "batch_size": 2,
            "buffer_size": 64,
        },
    )
    batch_size = 4
    obs = np.zeros((batch_size, planner.obs_dim), dtype=np.float32)
    actions = np.zeros((batch_size, planner.action_dim), dtype=np.float32)
    rewards = np.zeros((batch_size,), dtype=np.float32)
    dones = np.zeros((batch_size,), dtype=bool)
    infos = [{} for _ in range(batch_size)]

    planner.observe_transition_batch(
        observations=obs,
        buffer_actions=actions,
        rewards=rewards,
        dones=dones,
        next_observations=obs.copy(),
        infos=infos,
    )
    first_metrics = planner.maybe_update(
        collected_steps=1,
        step_count=1,
        global_total_timesteps=None,
        global_steps_done=0,
    )
    assert first_metrics == {}

    planner.observe_transition_batch(
        observations=obs,
        buffer_actions=actions,
        rewards=rewards,
        dones=dones,
        next_observations=obs.copy(),
        infos=infos,
    )
    second_metrics = planner.maybe_update(
        collected_steps=1,
        step_count=2,
        global_total_timesteps=None,
        global_steps_done=0,
    )

    assert int(second_metrics["update_calls"]) == 1
    assert int(second_metrics["gradient_steps"]) == 1


def test_td3_replay_buffer_tracks_raw_dones_timeouts_and_n_envs(env) -> None:
    planner = _build_planner(
        env,
        cfg_planner={"learning_starts": 0, "batch_size": 2, "buffer_size": 64},
    )
    batch_size = 3
    observations = np.zeros((batch_size, planner.obs_dim), dtype=np.float32)
    actions = np.zeros((batch_size, planner.action_dim), dtype=np.float32)
    rewards = np.zeros((batch_size,), dtype=np.float32)
    dones = np.array([True, True, False], dtype=bool)
    next_observations = np.full((batch_size, planner.obs_dim), 5.0, dtype=np.float32)
    terminal_observation = np.full((planner.obs_dim,), 7.0, dtype=np.float32)
    infos = [
        {},
        {"TimeLimit.truncated": True, "terminal_observation": terminal_observation},
        {},
    ]

    planner.observe_transition_batch(
        observations=observations,
        buffer_actions=actions,
        rewards=rewards,
        dones=dones,
        next_observations=next_observations,
        infos=infos,
    )

    assert planner.replay_buffer.n_envs == 1
    assert planner.replay_buffer.dones[:3, 0].tolist() == [1.0, 1.0, 0.0]
    assert planner.replay_buffer.timeouts[:3, 0].tolist() == [0.0, 1.0, 0.0]
    assert np.allclose(planner.replay_buffer.next_obs[1], terminal_observation)
