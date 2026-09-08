"""Audit 2026-09-06, block A4: chunk boundaries continue the slots' episodes.

``train_vectorized`` accepts the previous chunk's last observations; the training
loop must pass them so no worker is reset mid-episode. A reset at the boundary
left the last transition of every slot stored with ``done=0``, which the n-step
sampler chained into the next episode.
"""

from __future__ import annotations

import numpy as np

from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor


class _Lifecycle:
    last_actor_loss = float("nan")
    last_critic_loss = float("nan")
    update_count = 0
    gradient_step_count = 0

    def __init__(self) -> None:
        self.observed: list[dict] = []

    def begin_training(self, **kwargs) -> None:
        _ = kwargs

    def act_batch(self, observations: np.ndarray, deterministic: bool = False):
        _ = deterministic
        action = np.zeros((observations.shape[0], 1), dtype=np.float32)
        return action, action

    def observe_transition_batch(self, **kwargs) -> None:
        self.observed.append(kwargs)

    def maybe_update(self) -> None:
        pass

    def on_episode_end(self, indices=None) -> None:
        _ = indices

    def end_training(self) -> None:
        pass


class _Planner:
    def __init__(self, lifecycle: _Lifecycle) -> None:
        self._lifecycle = lifecycle

    def get_lifecycle(self) -> _Lifecycle:
        return self._lifecycle


class _CountingVectorEnv:
    num_envs = 2

    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_count = 0

    def reset(self):
        self.reset_calls += 1
        return np.zeros((self.num_envs, 1), dtype=np.float32)

    def step_slots(self, actions: dict[int, np.ndarray]):
        _ = actions
        self.step_count += 1
        value = float(self.step_count)
        return {
            slot: (np.full((1,), value, dtype=np.float32), 0.0, False, {}, {})
            for slot in range(self.num_envs)
        }

    def reset_slots(self, slots, *, force: bool = False):
        _ = force
        return {int(slot): (np.zeros((1,), dtype=np.float32), {}) for slot in slots}


def test_initial_observations_skip_the_reset_and_seed_the_first_transition() -> None:
    lifecycle = _Lifecycle()
    agent = Agent(
        preprocessor=IdentityPreprocessor(),
        planner=_Planner(lifecycle),
        adapter=IdentityAdapter(low=-1.0, high=1.0, expected_shape=(1,)),
    )
    env = _CountingVectorEnv()

    first = agent.train_vectorized(
        env=env, chunk_timesteps=2, global_total_timesteps=4, global_steps_done=0
    )
    assert env.reset_calls == 1
    carried = first["last_observations"]

    agent.train_vectorized(
        env=env,
        chunk_timesteps=2,
        global_total_timesteps=4,
        global_steps_done=2,
        initial_observations=carried,
    )

    assert env.reset_calls == 1
    boundary = lifecycle.observed[1]
    assert np.array_equal(boundary["observations"], np.asarray(carried))
