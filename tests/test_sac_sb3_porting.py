from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import thesis_rl.agent.planners.algorithms.sac as sac_module
from thesis_rl.agent.planners.algorithms import SacPlannerBackend
from thesis_rl.agent.planners.encoders.none_encoder import NoneEncoder
from thesis_rl.agent.types import Transition


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
) -> SacPlannerBackend:
    planner_cfg = {
        "learning_starts": 5,
        "batch_size": 8,
        "buffer_size": 128,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "tau": 0.005,
        "ent_coef": "auto",
        "target_entropy": "auto",
        "log_std_bounds": [-20, 2],
    }
    if cfg_planner is not None:
        planner_cfg.update(cfg_planner)

    encoder_cfg = {"type": "none"} if cfg_encoder is None else cfg_encoder
    decoder_cfg = (
        {
            "name": "sac_sb3",
            "type": "mlp",
            "hidden_layers": [256, 256],
            "activation": "relu",
            "dropout": 0.0,
            "layer_norm": False,
        }
        if cfg_decoder is None
        else cfg_decoder
    )

    return SacPlannerBackend.build(
        env,
        OmegaConf.create(planner_cfg),
        cfg_encoder=OmegaConf.create(encoder_cfg),
        cfg_decoder=OmegaConf.create(decoder_cfg),
        cfg_obs={},
        device="cpu",
        seed=123,
    )


def test_sac_sb3_network_is_flat_mlp_without_layer_norm(env) -> None:
    planner = _build_planner(env)

    assert isinstance(planner.actor.encoder, NoneEncoder)
    assert isinstance(planner.critic.encoder, NoneEncoder)

    actor_linears = [module for module in planner.actor.decoder.net if isinstance(module, nn.Linear)]
    critic_linears = [module for module in planner.critic.decoder_q1.net if isinstance(module, nn.Linear)]

    assert [(layer.in_features, layer.out_features) for layer in actor_linears] == [
        (planner.obs_dim, 256),
        (256, 256),
    ]
    assert [(layer.in_features, layer.out_features) for layer in critic_linears] == [
        (planner.obs_dim + planner.action_dim, 256),
        (256, 256),
    ]
    assert not any(isinstance(module, nn.LayerNorm) for module in planner.actor.decoder.net)
    assert not any(isinstance(module, nn.LayerNorm) for module in planner.critic.decoder_q1.net)


def test_sac_sb3_decoder_requires_none_encoder(env) -> None:
    with pytest.raises(ValueError, match="requires `encoder=none`"):
        _build_planner(
            env,
            cfg_encoder={"type": "mlp", "hidden_layers": [64, 64], "output_dim": 64},
        )


def test_sac_log_std_head_uses_state_dependent_linear_output_without_constant_init(env) -> None:
    planner = _build_planner(env)

    assert isinstance(planner.actor.log_std_head, nn.Linear)
    assert planner.actor.log_std_param is None
    assert not torch.allclose(
        planner.actor.log_std_head.weight.detach(),
        torch.zeros_like(planner.actor.log_std_head.weight.detach()),
    )
    assert not torch.allclose(
        planner.actor.log_std_head.bias.detach(),
        torch.full_like(planner.actor.log_std_head.bias.detach(), -3.0),
    )


def test_sac_sample_log_prob_matches_evaluate_log_prob(env) -> None:
    planner = _build_planner(env)
    obs, _ = env.reset(seed=123)
    obs_t = torch.as_tensor(np.stack([obs, obs], axis=0), dtype=torch.float32, device=planner.device)

    with torch.no_grad():
        actions, log_prob, _ = planner.actor.sample(obs_t, deterministic=False)
        recomputed_log_prob, _ = planner.actor.evaluate(obs_t, actions)

    assert torch.allclose(log_prob, recomputed_log_prob, atol=1e-5, rtol=1e-5)


def test_sac_target_entropy_auto_uses_negative_action_dim(env) -> None:
    planner = _build_planner(env)

    assert planner.target_entropy == -float(env.action_space.shape[0])


def test_sac_ent_coef_auto_suffix_sets_initial_alpha(env) -> None:
    planner = _build_planner(env, cfg_planner={"ent_coef": "auto_0.37"})

    assert planner.auto_alpha is True
    assert planner.alpha.item() == pytest.approx(0.37, rel=1e-6)


def test_sac_ent_coef_auto_suffix_must_be_positive(env) -> None:
    with pytest.raises(ValueError, match="must be greater than 0"):
        _build_planner(env, cfg_planner={"ent_coef": "auto_0.0"})


def test_sac_fixed_zero_ent_coef_is_supported(env) -> None:
    planner = _build_planner(env, cfg_planner={"ent_coef": 0.0})

    assert planner.auto_alpha is False
    assert planner.alpha.item() == pytest.approx(0.0, abs=1e-9)


def test_sac_gradient_steps_auto_matches_collected_rollout_transitions(env) -> None:
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
    observations = np.zeros((batch_size, planner.obs_dim), dtype=np.float32)
    actions = np.zeros((batch_size, planner.action_dim), dtype=np.float32)
    rewards = np.zeros((batch_size,), dtype=np.float32)
    dones = np.zeros((batch_size,), dtype=bool)
    infos = [{} for _ in range(batch_size)]

    planner.observe_transition_batch(
        observations=observations,
        buffer_actions=actions,
        rewards=rewards,
        dones=dones,
        next_observations=observations.copy(),
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


def test_sac_train_freq_counts_vecenv_rollout_steps_not_transitions(env) -> None:
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
    observations = np.zeros((batch_size, planner.obs_dim), dtype=np.float32)
    actions = np.zeros((batch_size, planner.action_dim), dtype=np.float32)
    rewards = np.zeros((batch_size,), dtype=np.float32)
    dones = np.zeros((batch_size,), dtype=bool)
    infos = [{} for _ in range(batch_size)]

    planner.observe_transition_batch(
        observations=observations,
        buffer_actions=actions,
        rewards=rewards,
        dones=dones,
        next_observations=observations.copy(),
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
        observations=observations,
        buffer_actions=actions,
        rewards=rewards,
        dones=dones,
        next_observations=observations.copy(),
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


def test_sac_target_update_interval_matches_sb3_train_loop_semantics(env, monkeypatch) -> None:
    planner = _build_planner(
        env,
        cfg_planner={
            "learning_starts": 0,
            "train_freq": 1,
            "gradient_steps": 3,
            "target_update_interval": 2,
            "batch_size": 2,
            "buffer_size": 64,
        },
    )
    observations = np.zeros((4, planner.obs_dim), dtype=np.float32)
    actions = np.zeros((4, planner.action_dim), dtype=np.float32)
    rewards = np.zeros((4,), dtype=np.float32)
    dones = np.zeros((4,), dtype=bool)
    infos = [{} for _ in range(4)]
    soft_update_calls = 0

    def _counting_soft_update(source, target, tau) -> None:
        del source, target, tau
        nonlocal soft_update_calls
        soft_update_calls += 1

    monkeypatch.setattr(sac_module, "soft_update", _counting_soft_update)

    planner.observe_transition_batch(
        observations=observations,
        buffer_actions=actions,
        rewards=rewards,
        dones=dones,
        next_observations=observations.copy(),
        infos=infos,
    )
    metrics = planner.maybe_update(
        collected_steps=1,
        step_count=1,
        global_total_timesteps=None,
        global_steps_done=0,
    )

    assert int(metrics["gradient_steps"]) == 3
    assert soft_update_calls == 2


def test_sac_replay_buffer_tracks_raw_dones_timeouts_and_n_envs(env) -> None:
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


def test_sac_timeout_transition_uses_terminal_obs_and_non_terminal_done(env) -> None:
    planner = _build_planner(env, cfg_planner={"learning_starts": 0, "batch_size": 8, "buffer_size": 64})

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


def test_sac_auto_alpha_updates_during_training(env) -> None:
    planner = _build_planner(
        env,
        cfg_planner={
            "learning_starts": 0,
            "batch_size": 16,
            "buffer_size": 64,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto_0.5",
        },
    )
    batch_size = 16
    observations = np.zeros((batch_size, planner.obs_dim), dtype=np.float32)
    actions = np.zeros((batch_size, planner.action_dim), dtype=np.float32)
    rewards = np.zeros((batch_size,), dtype=np.float32)
    dones = np.zeros((batch_size,), dtype=bool)
    infos = [{} for _ in range(batch_size)]
    log_alpha_before = float(planner.log_alpha.detach().cpu().item())

    planner.observe_transition_batch(
        observations=observations,
        buffer_actions=actions,
        rewards=rewards,
        dones=dones,
        next_observations=observations.copy(),
        infos=infos,
    )
    metrics = planner.maybe_update(
        collected_steps=1,
        step_count=1,
        global_total_timesteps=None,
        global_steps_done=0,
    )
    log_alpha_after = float(planner.log_alpha.detach().cpu().item())

    assert int(metrics["update_calls"]) == 1
    assert not np.isclose(log_alpha_after, log_alpha_before)


def test_sac_structural_parity_with_sb3_defaults(env) -> None:
    stable_baselines3 = pytest.importorskip("stable_baselines3")
    del stable_baselines3
    from stable_baselines3 import SAC

    planner = _build_planner(
        env,
        cfg_planner={
            "learning_starts": 100,
            "batch_size": 256,
            "buffer_size": 1_000_000,
            "train_freq": 1,
            "gradient_steps": 1,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "target_update_interval": 1,
            "ent_coef": "auto",
            "target_entropy": "auto",
            "log_std_bounds": [-20, 2],
        },
    )
    sb3_model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        policy_kwargs={"net_arch": [256, 256]},
        device="cpu",
        seed=123,
    )

    assert planner.learning_starts == sb3_model.learning_starts
    assert planner.batch_size == sb3_model.batch_size
    assert planner.gamma == pytest.approx(sb3_model.gamma)
    assert planner.tau == pytest.approx(sb3_model.tau)
    assert planner.target_update_interval == sb3_model.target_update_interval
    assert planner.target_entropy == pytest.approx(float(sb3_model.target_entropy))
