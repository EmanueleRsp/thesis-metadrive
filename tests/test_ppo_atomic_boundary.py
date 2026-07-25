"""EVAL-PROTOCOL v1.0 REQ-002/DEC-015: PPO's atomic collection unit is the
complete global rollout; ``atomic_boundary_remaining`` must report exactly
how many global transitions are still needed to complete it."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from omegaconf import OmegaConf

from thesis_rl.agent.agent import Agent
from thesis_rl.agent.planners.algorithms import Sb3PpoPlannerBackend


class _IdentityPreprocessor:
    def __call__(self, obs):
        return obs

    def reset(self):
        pass


class _IdentityAdapter:
    requires_training = False

    def __call__(self, planner_output):
        return planner_output


@pytest.fixture
def vec_env():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    made = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    try:
        yield made
    finally:
        made.close()


def _build_planner(vec_env, n_steps: int) -> Sb3PpoPlannerBackend:
    return Sb3PpoPlannerBackend.build(
        env=vec_env,
        cfg_planner=OmegaConf.create(
            {
                "policy": "MlpPolicy",
                "n_steps": n_steps,
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


def test_atomic_boundary_remaining_tracks_rollout_fill(vec_env) -> None:
    n_steps = 4
    planner = _build_planner(vec_env, n_steps=n_steps)
    agent = Agent(preprocessor=_IdentityPreprocessor(), planner=planner, adapter=_IdentityAdapter())

    lifecycle = planner.get_lifecycle()
    lifecycle.begin_training(chunk_timesteps=100, global_total_timesteps=100, global_steps_done=0)

    # Exactly at a boundary before any transition is collected.
    assert agent.atomic_boundary_remaining() == 0

    obs = vec_env.reset()

    def step(obs):
        actions, buffer_actions = lifecycle.act_batch(obs.astype(np.float32), deterministic=False)
        next_obs, rewards, step_dones, infos = vec_env.step(actions)
        lifecycle.observe_transition_batch(
            observations=obs,
            buffer_actions=buffer_actions,
            rewards=rewards,
            dones=step_dones,
            next_observations=next_obs,
            infos=infos,
            terminated=np.zeros(1, dtype=bool),
            truncated=np.zeros(1, dtype=bool),
            valid_mask=None,
        )
        return next_obs

    for expected_remaining in range(n_steps - 1, -1, -1):
        obs = step(obs)
        # After collecting one more transition, exactly `expected_remaining`
        # global transitions are still needed to complete this rollout.
        assert agent.atomic_boundary_remaining() == expected_remaining

    buffer = planner.model.rollout_buffer
    assert buffer.full

    # Completing the update resets the buffer to a fresh boundary (pos == 0).
    lifecycle.maybe_update()
    assert agent.atomic_boundary_remaining() == 0


def test_atomic_boundary_remaining_zero_for_algorithms_without_a_buffer() -> None:
    class _NoBufferPlanner:
        pass

    agent = Agent(
        preprocessor=_IdentityPreprocessor(), planner=_NoBufferPlanner(), adapter=_IdentityAdapter()
    )
    assert agent.atomic_boundary_remaining() == 0
