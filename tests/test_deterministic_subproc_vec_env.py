from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces

from thesis_rl.runtime.execution.deterministic_subproc_vec_env import DeterministicSubprocVecEnv


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
        np.testing.assert_array_equal(obs[:, 0].astype(np.int64), np.array([10, 20], dtype=np.int64))

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
