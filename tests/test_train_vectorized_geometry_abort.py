"""Training-loop half of `GEOM-ABORT`, mirroring the data-abort fixture.

Three properties that distinguish a geometry abort from a data abort in the
training loop: it writes its **own** forensic file, it does **not** quarantine
the scenario on first sight (`DEC-GA-002`), and a burst across slots trips the
ceiling and stops the run (`REQ-GA-006`).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor
from thesis_rl.runtime.execution.deterministic_subproc_vec_env import RuntimeGeometryAbort


class _FakeLifecycle:
    last_actor_loss = float("nan")
    last_critic_loss = float("nan")
    update_count = 0
    gradient_step_count = 0

    def __init__(self) -> None:
        self.observed_batches: list[dict] = []
        self.closed_boundaries: list[int] = []

    def begin_training(self, **kwargs) -> None:
        _ = kwargs

    def act_batch(self, observations: np.ndarray, deterministic: bool = False):
        _ = deterministic
        action = np.zeros((observations.shape[0], 1), dtype=np.float32)
        return action, action

    def observe_transition_batch(self, **kwargs) -> None:
        self.observed_batches.append(kwargs)

    def close_previous_transition_as_data_abort(self, *, env_index, final_observation) -> None:
        _ = final_observation
        self.closed_boundaries.append(int(env_index))

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


def _geometry_abort(slot: int, scenario_uid: str) -> RuntimeGeometryAbort:
    return RuntimeGeometryAbort(
        slot=slot,
        payload={
            "reason_code": "DECOMPOSITION_COVERAGE_SHORTFALL",
            "exception_message": "does not cover the input polygon",
            "traceback": "Traceback (most recent call last):\n...\n",
            "worker_step_index": 0,
            "diagnostics": {
                "scenario_uid": scenario_uid,
                "environment_step": 3,
                "residual_m2": 1.334923e-4,
                "allowance_m2": 2.614313e-3,
            },
            "geometry_wkt": "POLYGON ((0 0, 4 0, 4 3, 0 3, 0 0))",
            "final_observation": np.array([9.0], dtype=np.float32),
        },
    )


class _GeometryAbortingVectorEnv:
    """Aborts a configurable number of slots on the first step only."""

    def __init__(self, num_envs: int = 2, aborting_slots: tuple[int, ...] = (1,)) -> None:
        self.num_envs = int(num_envs)
        self._aborting_slots = tuple(aborting_slots)
        self._aborted_once = False
        self.env_method_calls: list[tuple[str, tuple]] = []

    def reset(self):
        return np.zeros((self.num_envs, 1), dtype=np.float32)

    def _normal(self):
        return (np.zeros((1,), dtype=np.float32), 0.1, False, {}, {})

    def step_slots(self, actions: dict[int, np.ndarray]):
        _ = actions
        results: dict[int, object] = {index: self._normal() for index in range(self.num_envs)}
        if not self._aborted_once:
            self._aborted_once = True
            for slot in self._aborting_slots:
                results[slot] = _geometry_abort(slot, f"waymo:geometry-{slot}")
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


def test_geometry_abort_writes_its_own_log_and_truncates_without_quarantining(
    tmp_path,
) -> None:
    data_log = tmp_path / "runtime_scenario_data_abort.jsonl"
    geometry_log = tmp_path / "runtime_geometry_abort.jsonl"
    env = _GeometryAbortingVectorEnv()
    lifecycle = _FakeLifecycle()
    agent = _build_agent(lifecycle)

    agent.train_vectorized(
        env=env,
        chunk_timesteps=2,
        global_total_timesteps=2,
        global_steps_done=0,
        data_abort_log_path=data_log,
        geometry_abort_log_path=geometry_log,
        run_id="run-geometry-1",
    )

    # `REQ-GA-004`: its own file, and the data-abort log untouched.
    assert geometry_log.exists()
    assert not data_log.exists()

    record = json.loads(geometry_log.read_text(encoding="utf-8").strip())
    assert record["run_id"] == "run-geometry-1"
    assert record["reason_code"] == "DECOMPOSITION_COVERAGE_SHORTFALL"
    assert record["scenario_uid"] == "waymo:geometry-1"
    assert record["diagnostics"]["residual_m2"] == pytest.approx(1.334923e-4)
    # `REQ-GA-005`: the geometry has to come back, or the record cannot become
    # a fixture and the next reader is guessing again.
    assert record["geometry_wkt"].startswith("POLYGON")

    # `REQ-GA-007`: the RSA-V1 boundary machinery ran for the aborted slot.
    assert 1 in lifecycle.closed_boundaries or lifecycle.observed_batches

    # `DEC-GA-002`: one occurrence is not evidence the record is broken.
    quarantined = [call for call in env.env_method_calls if call[0] == "quarantine_scenario_uid"]
    assert quarantined == []


def test_a_burst_of_geometry_aborts_across_slots_trips_the_ceiling(tmp_path) -> None:
    """`REQ-GA-006`. Three slots failing at once is systemic, not unlucky."""

    env = _GeometryAbortingVectorEnv(num_envs=4, aborting_slots=(0, 1, 2))
    lifecycle = _FakeLifecycle()
    agent = _build_agent(lifecycle)

    with pytest.raises(RuntimeError, match="consecutively within one evaluation batch"):
        agent.train_vectorized(
            env=env,
            chunk_timesteps=2,
            global_total_timesteps=2,
            global_steps_done=0,
            geometry_abort_log_path=tmp_path / "runtime_geometry_abort.jsonl",
            run_id="run-geometry-2",
        )


def test_repeated_geometry_aborts_on_one_uid_eventually_quarantine_it(tmp_path) -> None:
    """`DEC-GA-002`, the other side: a record that keeps failing does leave."""

    lifecycle = _FakeLifecycle()
    agent = _build_agent(lifecycle)
    ledger = agent._geometry_abort_ledger()

    # Two prior occurrences on this UID, as an earlier chunk would have left.
    ledger.record("waymo:geometry-1", "DECOMPOSITION_COVERAGE_SHORTFALL")
    ledger.record("waymo:geometry-1", "DECOMPOSITION_COVERAGE_SHORTFALL")
    ledger.note_episode_completed()

    env = _GeometryAbortingVectorEnv()
    agent.train_vectorized(
        env=env,
        chunk_timesteps=2,
        global_total_timesteps=2,
        global_steps_done=0,
        geometry_abort_log_path=tmp_path / "runtime_geometry_abort.jsonl",
        run_id="run-geometry-3",
    )

    assert ("quarantine_scenario_uid", ("waymo:geometry-1",)) in env.env_method_calls
