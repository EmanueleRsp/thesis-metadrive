from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf

from thesis_rl.agent.planners.algorithms import Sb3PpoPlannerBackend


@pytest.fixture
def vec_env():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    made = DummyVecEnv([lambda: gym.make("Pendulum-v1"), lambda: gym.make("Pendulum-v1")])
    try:
        yield made
    finally:
        made.close()


def _build_planner(vec_env) -> Sb3PpoPlannerBackend:
    return Sb3PpoPlannerBackend.build(
        env=vec_env,
        cfg_planner=OmegaConf.create(
            {
                "policy": "MlpPolicy",
                "n_steps": 3,
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


def test_ppo_sb3_data_abort_preserves_peer_env_and_bootstraps_boundary(vec_env) -> None:
    planner = _build_planner(vec_env)
    lifecycle = planner.get_lifecycle()
    lifecycle.begin_training(chunk_timesteps=100, global_total_timesteps=100, global_steps_done=0)

    obs = vec_env.reset()

    def step(obs, valid_mask=None):
        actions, buffer_actions = lifecycle.act_batch(obs.astype(np.float32), deterministic=False)
        next_obs, rewards, step_dones, infos = vec_env.step(actions)
        lifecycle.observe_transition_batch(
            observations=obs,
            buffer_actions=buffer_actions,
            rewards=rewards,
            dones=step_dones,
            next_observations=next_obs,
            infos=infos,
            terminated=np.zeros(2, dtype=bool),
            truncated=np.zeros(2, dtype=bool),
            valid_mask=valid_mask,
        )
        return next_obs

    # Step 1: both envs contribute a real, valid transition.
    obs = step(obs)
    buffer = planner.model.rollout_buffer
    env0_row0_obs = np.array(buffer.observations[0, 0], copy=True)
    env0_row0_reward = float(buffer.rewards[0, 0])
    env1_row0_reward_before_abort = float(buffer.rewards[0, 1])

    # Step 2: env 1 hits a typed data-abort. The previous row is retroactively
    # bootstrapped, and this step's row is written only for env 0.
    final_observation = np.asarray(obs[1], dtype=np.float32)
    lifecycle.close_previous_transition_as_data_abort(
        env_index=1, final_observation=final_observation
    )
    assert buffer.rewards[0, 1] != pytest.approx(env1_row0_reward_before_abort)

    obs = step(obs, valid_mask=np.array([True, False]))
    obs[1] = final_observation  # the aborted slot resumes from its valid post-abort state
    assert not buffer.full
    assert bool(buffer.valid_transitions[1, 0])
    assert not bool(buffer.valid_transitions[1, 1])
    # Env 0's first row must be completely untouched by env 1's abort.
    np.testing.assert_array_equal(buffer.observations[0, 0], env0_row0_obs)
    assert buffer.rewards[0, 0] == pytest.approx(env0_row0_reward)

    # Step 3: both envs contribute again; the rollout completes and trains.
    obs = step(obs)
    assert buffer.full

    lifecycle.maybe_update()
    assert not np.isnan(planner.last_actor_loss)
    assert not np.isnan(planner.last_critic_loss)
    assert planner.model.num_timesteps == 3 * 2 - 1
