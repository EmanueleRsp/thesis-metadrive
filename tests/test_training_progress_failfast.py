"""C12: TTY-independent training progress and bounded teardown after run_failed."""

from __future__ import annotations

import logging
import threading

import numpy as np
import pytest

from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor
from thesis_rl.runtime.failfast import (
    close_quietly,
    fail_fast_after_run_failure,
    schedule_forced_exit,
)
from thesis_rl.runtime.training_progress import (
    format_training_progress_line,
    progress_bucket,
    progress_due,
)


# --------------------------------------------------------------------------- pure helpers


def test_progress_bucket_disabled_for_non_positive_interval() -> None:
    assert progress_bucket(5000, 0) == -1
    assert progress_bucket(5000, -1) == -1
    assert progress_bucket(999, 1000) == 0
    assert progress_bucket(1000, 1000) == 1
    assert progress_bucket(2999, 1000) == 2


def test_progress_due_first_at_interval_then_at_chunk_end() -> None:
    interval = 1000
    chunk = 2500
    last = progress_bucket(0, interval)
    due_steps = []
    for collected in range(20, chunk + 1, 20):
        if progress_due(collected, chunk, interval, last):
            due_steps.append(collected)
            last = progress_bucket(collected, interval)
    assert due_steps == [1000, 2000, 2500]


def test_progress_due_is_never_true_when_disabled() -> None:
    assert not progress_due(1000, 1000, 0, -1)
    assert not progress_due(5000, 1000, 0, -1)


def test_format_training_progress_line_is_plain_and_complete() -> None:
    line = format_training_progress_line(
        {
            "run_env_steps": 26000,
            "global_total_timesteps": 350000,
            "chunk_env_steps": 1000,
            "chunk_timesteps": 25000,
            "fps": 4.5678,
            "elapsed_seconds": 219.4,
            "episodes": 12,
            "ep_len_mean": 244.77,
            "ep_rew_mean": -3.14159,
            "ema_actor_loss": float("nan"),
            "ema_critic_loss": 22.5,
        }
    )
    assert line.startswith("Training progress | run_step=26000/350000 | chunk_step=1000/25000")
    assert "fps=4.57" in line
    assert "elapsed_s=219" in line
    assert "episodes=12" in line
    assert "ep_len_mean=244.8" in line
    assert "ep_rew_mean=-3.142" in line
    assert "actor_loss_ema=nan" in line
    assert "critic_loss_ema=22.5" in line
    assert "[" not in line  # no Rich markup, safe for plain consoles and logs


# --------------------------------------------------------------------------- train_vectorized wiring


class _FakeLifecycle:
    last_actor_loss = 0.5
    last_critic_loss = 2.0
    update_count = 3
    gradient_step_count = 6

    def begin_training(self, **kwargs) -> None:
        _ = kwargs

    def act_batch(self, observations: np.ndarray, deterministic: bool = False):
        _ = deterministic
        action = np.zeros((observations.shape[0], 1), dtype=np.float32)
        return action, action

    def observe_transition_batch(self, **kwargs) -> None:
        _ = kwargs

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


class _QuietVectorEnv:
    """Two-slot vector env that never terminates and never aborts."""

    num_envs = 2

    def reset(self):
        return np.zeros((self.num_envs, 1), dtype=np.float32)

    def step_slots(self, actions):
        _ = actions
        return {
            index: (np.zeros((1,), dtype=np.float32), 0.1, False, {}, {})
            for index in range(self.num_envs)
        }

    def reset_slots(self, slots, *, force: bool = False):
        _ = force
        return {int(slot): (np.zeros((1,), dtype=np.float32), {}) for slot in slots}

    def env_method(self, method_name: str, *args, **kwargs):
        _ = (method_name, args, kwargs)
        return [None for _ in range(self.num_envs)]


def _build_agent() -> Agent:
    return Agent(
        preprocessor=IdentityPreprocessor(),
        planner=_Planner(_FakeLifecycle()),
        adapter=IdentityAdapter(low=-1.0, high=1.0, expected_shape=(1,)),
    )


def test_train_vectorized_emits_progress_snapshots_and_plain_lines(capsys) -> None:
    snapshots: list[dict] = []
    agent = _build_agent()

    agent.train_vectorized(
        env=_QuietVectorEnv(),
        chunk_timesteps=10,
        global_total_timesteps=110,
        global_steps_done=100,
        progress_callback=snapshots.append,
        progress_interval=4,
    )

    # Two slots advance collected_steps by 2 per iteration: 2, 4, 6, 8, 10.
    # Records at the interval crossings (4, 8) and at the end of the chunk (10).
    assert [s["chunk_env_steps"] for s in snapshots] == [4, 8, 10]
    assert [s["run_env_steps"] for s in snapshots] == [104, 108, 110]
    first = snapshots[0]
    assert first["chunk_timesteps"] == 10
    assert first["global_total_timesteps"] == 110
    assert first["fps"] > 0.0
    assert first["update_calls"] == 3
    assert first["gradient_steps"] == 6
    assert first["ema_actor_loss"] == pytest.approx(0.5)
    assert first["ema_critic_loss"] == pytest.approx(2.0)

    # Under pytest the console is not a terminal, so the plain line channel is active.
    out = capsys.readouterr().out
    assert out.count("Training progress | run_step=") == 3
    assert "run_step=110/110 | chunk_step=10/10" in out


def test_train_vectorized_progress_channel_can_be_disabled(capsys) -> None:
    snapshots: list[dict] = []
    _build_agent().train_vectorized(
        env=_QuietVectorEnv(),
        chunk_timesteps=6,
        global_total_timesteps=6,
        global_steps_done=0,
        progress_callback=snapshots.append,
        progress_interval=0,
    )
    assert snapshots == []
    assert "Training progress" not in capsys.readouterr().out


# --------------------------------------------------------------------------- fail-fast teardown


class _Resource:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.closed = 0

    def close(self) -> None:
        self.closed += 1
        if self.fail:
            raise RuntimeError("engine still alive")


def test_close_quietly_swallows_and_logs_failures(caplog) -> None:
    logger = logging.getLogger("test.failfast.close")
    good = _Resource()
    bad = _Resource(fail=True)
    with caplog.at_level(logging.INFO, logger=logger.name):
        assert close_quietly(good, "good", logger) is True
        assert close_quietly(bad, "bad", logger) is False
        assert close_quietly(None, "absent", logger) is False
        assert close_quietly(object(), "no-close", logger) is False
    assert good.closed == 1
    assert bad.closed == 1
    assert "closing bad failed" in caplog.text
    assert "engine still alive" in caplog.text


def test_schedule_forced_exit_fires_after_grace_with_hook_first() -> None:
    logger = logging.getLogger("test.failfast.exit")
    calls: list[tuple[str, int | None]] = []
    fired = threading.Event()

    def _exit(code: int) -> None:
        calls.append(("exit", code))
        fired.set()

    timer = schedule_forced_exit(
        0.05,
        7,
        logger,
        on_fire=lambda: calls.append(("hook", None)),
        exit_fn=_exit,
    )
    assert timer.daemon is True
    assert fired.wait(timeout=5.0)
    assert calls == [("hook", None), ("exit", 7)]


def test_schedule_forced_exit_exits_even_when_hook_raises() -> None:
    logger = logging.getLogger("test.failfast.exit.hook")
    fired = threading.Event()
    codes: list[int] = []

    def _exit(code: int) -> None:
        codes.append(code)
        fired.set()

    def _hook() -> None:
        raise RuntimeError("events.jsonl unwritable")

    schedule_forced_exit(0.01, 3, logger, on_fire=_hook, exit_fn=_exit)
    assert fired.wait(timeout=5.0)
    assert codes == [3]


def test_fail_fast_after_run_failure_closes_resources_then_arms_watchdog() -> None:
    logger = logging.getLogger("test.failfast.full")
    env = _Resource()
    manager = _Resource(fail=True)
    fired = threading.Event()
    codes: list[int] = []

    def _exit(code: int) -> None:
        codes.append(code)
        fired.set()

    timer = fail_fast_after_run_failure(
        env=env,
        async_evaluation_manager=manager,
        logger=logger,
        grace_seconds=0.05,
        exit_code=1,
        exit_fn=_exit,
    )
    # Resources are closed synchronously, before the watchdog is even armed.
    assert env.closed == 1
    assert manager.closed == 1
    assert fired.wait(timeout=5.0)
    assert codes == [1]
    timer.join(timeout=1.0)


def test_fail_fast_after_run_failure_tolerates_missing_resources() -> None:
    logger = logging.getLogger("test.failfast.none")
    timer = fail_fast_after_run_failure(
        env=None,
        async_evaluation_manager=None,
        logger=logger,
        grace_seconds=60.0,
        exit_fn=lambda code: None,
    )
    assert timer.is_alive()
    timer.cancel()
