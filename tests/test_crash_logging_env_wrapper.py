from __future__ import annotations

import gymnasium as gym
import pytest

from thesis_rl.rulebook.v2.errors import (
    RuntimeScenarioNotEvaluableError,
    RuntimeScenarioNotEvaluableReason,
)
from thesis_rl.runtime.wiring.builders import _CrashLoggingEnvWrapper


class _RaisingEnv(gym.Env):
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def reset(self, **kwargs):
        raise self._exc

    def step(self, action):
        raise self._exc


def test_data_abort_does_not_write_crash_log(tmp_path) -> None:
    crash_log = tmp_path / "eval_subproc_worker_0_crash.log"
    exc = RuntimeScenarioNotEvaluableError(
        RuntimeScenarioNotEvaluableReason.INVALID_SIGNAL_TRANSITION,
        "Signal transition requires valid, known pre/post states",
    )
    wrapped = _CrashLoggingEnvWrapper(_RaisingEnv(exc), crash_log)

    with pytest.raises(RuntimeScenarioNotEvaluableError):
        wrapped.step(None)

    assert not crash_log.exists()


def test_genuine_failure_still_writes_crash_log(tmp_path) -> None:
    crash_log = tmp_path / "eval_subproc_worker_0_crash.log"
    wrapped = _CrashLoggingEnvWrapper(_RaisingEnv(ValueError("unexpected")), crash_log)

    with pytest.raises(ValueError):
        wrapped.step(None)

    assert crash_log.exists()
    assert "unexpected" in crash_log.read_text(encoding="utf-8")
