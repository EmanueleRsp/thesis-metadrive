from __future__ import annotations

import json

import numpy as np

from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor
from thesis_rl.runtime.execution.deterministic_subproc_vec_env import RuntimeScenarioDataAbort


class _FakeLifecycle:
    """Minimal duck-typed lifecycle exercising only what train_vectorized calls."""

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
        n = observations.shape[0]
        action = np.zeros((n, 1), dtype=np.float32)
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
    def __init__(self, lifecycle: _FakeLifecycle) -> None:
        self._lifecycle = lifecycle

    def get_lifecycle(self) -> _FakeLifecycle:
        return self._lifecycle


class _AbortingVectorEnv:
    """Two-worker vector env aborting slot 1 exactly once, on the first step."""

    num_envs = 2

    def __init__(self) -> None:
        self._aborted_once = False
        self.env_method_calls: list[tuple[str, tuple]] = []

    def reset(self):
        return np.zeros((self.num_envs, 1), dtype=np.float32)

    def step_slots(self, actions: dict[int, np.ndarray]):
        _ = actions
        results: dict[int, object] = {
            0: (
                np.zeros((1,), dtype=np.float32),
                0.1,
                False,
                {},
                {},
            )
        }
        if not self._aborted_once:
            self._aborted_once = True
            results[1] = RuntimeScenarioDataAbort(
                slot=1,
                payload={
                    "reason_code": "INVALID_SIGNAL_TRANSITION",
                    "exception_message": "typed runtime-invalid scenario",
                    "traceback": "Traceback (most recent call last):\n...\nValueError",
                    "worker_step_index": 0,
                    "diagnostics": {
                        "scenario_uid": "waymo:fixture-1",
                        "environment_step": 3,
                    },
                    "final_observation": np.array([9.0], dtype=np.float32),
                },
            )
        else:
            results[1] = (
                np.zeros((1,), dtype=np.float32),
                0.1,
                False,
                {},
                {},
            )
        return results

    def reset_slots(self, slots, *, force: bool = False):
        _ = force
        return {int(slot): (np.zeros((1,), dtype=np.float32), {}) for slot in slots}

    def env_method(self, method_name: str, *args, **kwargs):
        _ = kwargs
        self.env_method_calls.append((method_name, args))
        return [None for _ in range(self.num_envs)]


def _build_agent(lifecycle: _FakeLifecycle) -> Agent:
    return Agent(
        preprocessor=IdentityPreprocessor(),
        planner=_Planner(lifecycle),
        adapter=IdentityAdapter(low=-1.0, high=1.0, expected_shape=(1,)),
    )


def test_train_vectorized_writes_forensic_jsonl_and_quarantines_on_data_abort(tmp_path) -> None:
    log_path = tmp_path / "runtime_scenario_data_abort.jsonl"
    env = _AbortingVectorEnv()
    lifecycle = _FakeLifecycle()
    agent = _build_agent(lifecycle)

    summary = agent.train_vectorized(
        env=env,
        chunk_timesteps=2,
        global_total_timesteps=2,
        global_steps_done=0,
        data_abort_log_path=log_path,
        run_id="run-fixture-1",
    )

    assert summary["episodes"] >= 1
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["run_id"] == "run-fixture-1"
    assert record["scenario_uid"] == "waymo:fixture-1"
    assert record["reason_code"] == "INVALID_SIGNAL_TRANSITION"
    assert record["worker_slot"] == 1
    assert "final_observation" not in record
    assert record["last_valid_observation_sha256"]

    assert ("quarantine_scenario_uid", ("waymo:fixture-1",)) in env.env_method_calls
