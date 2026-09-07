"""REQ-AF-05 (OBS-AUDIT-FIX-001): a chunk boundary is not an episode boundary.

``Agent.train_vectorized`` resets every slot when it receives no
``initial_observations``.  The non-curriculum training loop never forwarded
the previous chunk's ``last_observations``, so every ``eval_interval`` steps
all in-flight episodes were discarded without a truncation flag.
"""

from __future__ import annotations

import numpy as np

from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor
from thesis_rl.runtime.loops.train_loop import chunk_carry_kwargs


class _Lifecycle:
    last_actor_loss = float("nan")
    last_critic_loss = float("nan")
    update_count = 0
    gradient_step_count = 0

    def __init__(self) -> None:
        self.observed_batches: list[dict] = []

    def begin_training(self, **kwargs) -> None:
        _ = kwargs

    def act_batch(self, observations: np.ndarray, deterministic: bool = False):
        _ = deterministic
        action = np.zeros((observations.shape[0], 1), dtype=np.float32)
        return action, action

    def observe_transition_batch(self, **kwargs) -> None:
        self.observed_batches.append(kwargs)

    def close_previous_transition_as_data_abort(self, *, env_index, final_observation) -> None:
        _ = (env_index, final_observation)

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
    """Two slots whose observation is a running step counter; never terminates."""

    num_envs = 2

    def __init__(self) -> None:
        self.reset_calls = 0
        self._counter = np.zeros(self.num_envs, dtype=np.float32)

    def reset(self):
        self.reset_calls += 1
        self._counter[:] = 0.0
        return self._counter.copy().reshape(self.num_envs, 1)

    def step_slots(self, actions):
        results = {}
        for slot in actions:
            self._counter[slot] += 1.0
            results[int(slot)] = (
                np.asarray([self._counter[slot]], dtype=np.float32),
                0.0,
                False,
                {},
                {},
            )
        return results

    def reset_slots(self, slots, *, force: bool = False):
        _ = force
        return {int(slot): (np.zeros((1,), dtype=np.float32), {}) for slot in slots}

    def env_method(self, method_name: str, *args, **kwargs):
        _ = (method_name, args, kwargs)
        return [None for _ in range(self.num_envs)]


def _agent(lifecycle: _Lifecycle) -> Agent:
    return Agent(
        preprocessor=IdentityPreprocessor(),
        planner=_Planner(lifecycle),
        adapter=IdentityAdapter(low=-1.0, high=1.0, expected_shape=(1,)),
    )


def test_carried_observations_skip_reset_and_continue_episode_lengths() -> None:
    lifecycle = _Lifecycle()
    agent = _agent(lifecycle)
    env = _CountingVectorEnv()

    first = agent.train_vectorized(
        env=env, chunk_timesteps=4, global_total_timesteps=8, global_steps_done=0
    )
    assert env.reset_calls == 1
    assert first["last_observations"].reshape(-1).tolist() == [2.0, 2.0]
    assert first["last_episode_lengths"].tolist() == [2, 2]

    carry = chunk_carry_kwargs(first, previous_env=env, env=env)
    second = agent.train_vectorized(
        env=env, chunk_timesteps=4, global_total_timesteps=8, global_steps_done=4, **carry
    )

    assert env.reset_calls == 1, "the second chunk must continue, not reset"
    first_batch_of_second_chunk = lifecycle.observed_batches[2]
    assert first_batch_of_second_chunk["observations"].reshape(-1).tolist() == [2.0, 2.0]
    assert first_batch_of_second_chunk["next_observations"].reshape(-1).tolist() == [3.0, 3.0]
    assert second["last_episode_lengths"].tolist() == [4, 4]


def test_chunk_carry_is_dropped_when_the_environment_was_rebuilt() -> None:
    env_a = _CountingVectorEnv()
    env_b = _CountingVectorEnv()
    summary = {
        "last_observations": np.zeros((2, 1), dtype=np.float32),
        "last_episode_lengths": np.asarray([3, 1]),
    }

    assert chunk_carry_kwargs(summary, previous_env=env_a, env=env_a).keys() == {
        "initial_observations",
        "initial_episode_lengths",
    }
    assert chunk_carry_kwargs(summary, previous_env=env_a, env=env_b) == {}
    assert chunk_carry_kwargs(None, previous_env=env_a, env=env_a) == {}
    assert chunk_carry_kwargs({"last_observations": None}, previous_env=env_a, env=env_a) == {}


def test_train_loop_forwards_last_observations_between_chunks() -> None:
    """The plain loop must build the carry from the previous chunk summary."""

    import inspect

    from thesis_rl.runtime.loops import train_loop

    source = inspect.getsource(train_loop.run_training)
    assert "chunk_carry_kwargs(" in source
    assert "previous_chunk_summary = chunk_summary" in source
