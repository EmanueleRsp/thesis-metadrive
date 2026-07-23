from __future__ import annotations

import os
import time

import numpy as np
import pytest
from gymnasium import Env, spaces

from thesis_rl.runtime.execution.deterministic_subproc_vec_env import (
    DeterministicSubprocVecEnv,
    SubprocessWorkerError,
)


class _SeedWindowEnv(Env):
    def __init__(self, start_index: int, num_scenarios: int) -> None:
        super().__init__()
        self.start_index = int(start_index)
        self.num_scenarios = int(num_scenarios)
        self.current_seed = self.start_index
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _ = options
        if seed is None:
            seed = int(np.random.randint(self.start_index, self.start_index + self.num_scenarios))
        self.current_seed = int(seed)
        obs = np.array([self.current_seed], dtype=np.float32)
        return obs, {}

    def step(self, action):
        _ = action
        obs = np.array([self.current_seed], dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        info: dict = {}
        return obs, reward, terminated, truncated, info


def _make_env(start_index: int, num_scenarios: int):
    def _factory():
        return _SeedWindowEnv(start_index=start_index, num_scenarios=num_scenarios)

    return _factory


class _FailingStepEnv(_SeedWindowEnv):
    def step(self, action):
        _ = action
        raise ValueError("intentional worker failure")


class _AbruptExitEnv(_SeedWindowEnv):
    def step(self, action):
        _ = action
        os._exit(23)


class _SlowStepEnv(_SeedWindowEnv):
    def step(self, action):
        _ = action
        time.sleep(10.0)
        return super().step(action)


class _NumericThreadEnvironmentEnv(_SeedWindowEnv):
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, _info = super().reset(seed=seed, options=options)
        return observation, {
            "numeric_thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            }
        }


def _make_failing_step_env():
    return _FailingStepEnv(start_index=0, num_scenarios=1)


def _make_abrupt_exit_env():
    return _AbruptExitEnv(start_index=0, num_scenarios=1)


def _make_slow_step_env():
    return _SlowStepEnv(start_index=0, num_scenarios=1)


def _make_numeric_thread_environment_env():
    return _NumericThreadEnvironmentEnv(start_index=0, num_scenarios=1)


def test_deterministic_subproc_vec_env_auto_reset_uses_deterministic_worker_seeds() -> None:
    vec_env = DeterministicSubprocVecEnv(
        [
            _make_env(start_index=10, num_scenarios=3),
            _make_env(start_index=20, num_scenarios=2),
        ],
        start_method="spawn",
    )
    try:
        vec_env._seeds = [10, 20]  # type: ignore[attr-defined]
        obs = vec_env.reset()
        np.testing.assert_array_equal(
            obs[:, 0].astype(np.int64), np.array([10, 20], dtype=np.int64)
        )

        expected = [
            np.array([11, 21], dtype=np.int64),
            np.array([12, 20], dtype=np.int64),
            np.array([10, 21], dtype=np.int64),
            np.array([11, 20], dtype=np.int64),
        ]
        actions = np.zeros((2, 1), dtype=np.float32)
        for expected_obs in expected:
            obs, _rewards, dones, _infos = vec_env.step(actions)
            np.testing.assert_array_equal(dones, np.array([True, True]))
            np.testing.assert_array_equal(obs[:, 0].astype(np.int64), expected_obs)
    finally:
        vec_env.close()


def test_deterministic_subproc_vec_env_supports_selective_manual_resets() -> None:
    vec_env = DeterministicSubprocVecEnv(
        [
            _make_env(start_index=10, num_scenarios=3),
            _make_env(start_index=20, num_scenarios=2),
        ],
        start_method="spawn",
        auto_reset=False,
    )
    try:
        initial = vec_env.reset_slots([0, 1], seeds={0: 10, 1: 20})
        np.testing.assert_array_equal(initial[0][0], np.array([10], dtype=np.float32))
        np.testing.assert_array_equal(initial[1][0], np.array([20], dtype=np.float32))

        results = vec_env.step_slots(
            {0: np.zeros(1, dtype=np.float32), 1: np.zeros(1, dtype=np.float32)}
        )
        assert set(results) == {0, 1}
        assert results[0][2] is True
        assert results[1][2] is True

        reset = vec_env.reset_slots([1], seeds={1: 21})
        np.testing.assert_array_equal(reset[1][0], np.array([21], dtype=np.float32))
        assert vec_env.reset_infos[1]["_thesis_worker_timing_seconds"]["reset"] >= 0.0
        assert set(vec_env.step_slots({1: np.zeros(1, dtype=np.float32)})) == {1}
    finally:
        vec_env.close()


def test_worker_python_exception_reports_remote_traceback() -> None:
    vec_env = DeterministicSubprocVecEnv([_make_failing_step_env], start_method="spawn")
    try:
        vec_env.reset()
        with pytest.raises(SubprocessWorkerError) as error:
            vec_env.step(np.zeros((1, 1), dtype=np.float32))
        message = str(error.value)
        assert "slot=0" in message
        assert "command='step'" in message
        assert "type=ValueError" in message
        assert "intentional worker failure" in message
        assert "Remote traceback:" in message
    finally:
        vec_env.close()


def test_reset_response_with_array_observation_is_not_an_error_envelope() -> None:
    vec_env = DeterministicSubprocVecEnv([_make_env(10, 3)], start_method="spawn")
    try:
        observations = vec_env.reset()
        assert observations.shape == (1, 1)
        assert 10 <= int(observations[0, 0]) < 13
    finally:
        vec_env.close()


def test_worker_eof_reports_slot_and_exit_code() -> None:
    vec_env = DeterministicSubprocVecEnv([_make_abrupt_exit_env], start_method="spawn")
    try:
        vec_env.reset()
        with pytest.raises(SubprocessWorkerError) as error:
            vec_env.step(np.zeros((1, 1), dtype=np.float32))
        message = str(error.value)
        assert "slot=0" in message
        assert "command='step'" in message
        assert "exitcode=" in message
    finally:
        vec_env.close()


def test_later_worker_failure_does_not_wait_for_an_earlier_slow_worker() -> None:
    vec_env = DeterministicSubprocVecEnv(
        [_make_slow_step_env, _make_failing_step_env],
        start_method="spawn",
    )
    try:
        vec_env.reset()
        start = time.monotonic()
        with pytest.raises(SubprocessWorkerError, match="slot=1"):
            vec_env.step(np.zeros((2, 1), dtype=np.float32))
        assert time.monotonic() - start < 5.0
        assert all(not process.is_alive() for process in vec_env.processes)
    finally:
        vec_env.close()


def test_close_reaps_worker_with_an_in_flight_slow_command() -> None:
    vec_env = DeterministicSubprocVecEnv([_make_slow_step_env], start_method="spawn")
    vec_env.reset()
    vec_env.step_async(np.zeros((1, 1), dtype=np.float32))
    start = time.monotonic()
    vec_env.close()
    assert time.monotonic() - start < 5.0
    assert all(not process.is_alive() for process in vec_env.processes)


def test_numeric_library_thread_limit_is_inherited_before_spawn_and_restored(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "11")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "12")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    vec_env = DeterministicSubprocVecEnv(
        [_make_numeric_thread_environment_env],
        start_method="spawn",
        numeric_library_num_threads=3,
    )
    try:
        vec_env.reset()
        inherited = vec_env.reset_infos[0]["numeric_thread_environment"]
        assert inherited == {
            "OMP_NUM_THREADS": "3",
            "OPENBLAS_NUM_THREADS": "3",
            "MKL_NUM_THREADS": "3",
        }
        assert os.environ["OMP_NUM_THREADS"] == "11"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "12"
        assert "MKL_NUM_THREADS" not in os.environ
    finally:
        vec_env.close()
