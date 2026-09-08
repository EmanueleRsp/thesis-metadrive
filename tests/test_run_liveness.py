"""`C10`: the liveness reader must not repeat the mistakes the habits made.

The two habitual checks fail on a hung run: the `tmux` session is still listed,
and the tail of `train.log` shows the last progress line, which reads exactly
like progress. `scripts/run_liveness.py` reads `logs/events.jsonl` instead,
where every record is on disk when written and carries a timestamp.

These pin the two properties that make it trustworthy: a terminal failure is
reported as failure even while the process lingers, and the staleness verdict is
measured against the run's own cadence rather than a chosen constant.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_liveness.py"
_spec = importlib.util.spec_from_file_location("run_liveness", _SCRIPT)
assert _spec is not None and _spec.loader is not None
run_liveness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_liveness)

_NOW = datetime(2026, 9, 8, 12, 0, 0)


def _progress(seconds_ago: float, step: int) -> dict[str, object]:
    return {
        "event": "training_progress",
        "time": (_NOW - timedelta(seconds=seconds_ago)).isoformat(timespec="seconds"),
        "run_env_steps": step,
    }


def _healthy_cadence(
    gap: float = 60.0, count: int = 5, silence: float = 60.0
) -> list[dict[str, object]]:
    """`count` progress events `gap` apart, the newest `silence` seconds ago.

    `silence` is what the reader judges; `gap` is the cadence it measures to
    judge it against. Keeping them separate is the point of the helper — an
    earlier version mutated the last element instead, which broke the sequence's
    monotonicity and left the cadence unmeasurable rather than the run silent.
    """

    return [
        _progress(silence + gap * (count - 1 - index), 1000 * (index + 1)) for index in range(count)
    ]


def test_a_run_still_emitting_at_its_own_cadence_is_running() -> None:
    status, explanation = run_liveness.classify(_healthy_cadence(), now=_NOW)

    assert status == "running"
    assert "60s" in explanation


def test_silence_well_beyond_the_measured_cadence_is_stalled() -> None:
    """The `C10` case: no terminal event, the process alive, and nothing happening."""

    events = _healthy_cadence(silence=60.0 * 60.0)

    status, explanation = run_liveness.classify(events, now=_NOW)

    assert status == "stalled"
    assert "measured cadence" in explanation


def test_a_slow_chunk_boundary_is_not_called_stalled() -> None:
    """Three cadences of silence is jitter, not death; the tolerance must absorb it."""

    events = _healthy_cadence(silence=180.0)

    status, _ = run_liveness.classify(events, now=_NOW)

    assert status == "running"


@pytest.mark.parametrize("terminal", ["run_failed", "run_forced_exit"])
def test_a_terminal_failure_wins_over_a_recent_event(terminal: str) -> None:
    """This is what the `tmux` and log-tail checks got wrong.

    `C12`'s watchdog gives teardown 120 s and the container can linger far
    longer, so a `run_failed` followed by silence — or even by later records —
    is a dead run, whatever the process table says.
    """

    events = [
        *_healthy_cadence(),
        {"event": terminal, "time": (_NOW - timedelta(seconds=30)).isoformat(timespec="seconds")},
    ]

    status, explanation = run_liveness.classify(events, now=_NOW)

    assert status == "failed"
    assert terminal in explanation


def test_a_completed_run_is_not_reported_as_stalled() -> None:
    """Silence after `run_completed` is the expected state, not a fault."""

    events = [
        *_healthy_cadence(),
        {
            "event": "run_completed",
            "time": (_NOW - timedelta(days=2)).isoformat(timespec="seconds"),
        },
    ]

    status, _ = run_liveness.classify(events, now=_NOW)

    assert status == "completed"


def test_too_few_events_to_measure_a_cadence_is_unknown_not_a_guess() -> None:
    """A false "running" costs an idle GPU; a false "stalled" costs an investigation.

    With no measured cadence there is nothing to judge silence against, so the
    reader declines rather than inventing a threshold.
    """

    status, explanation = run_liveness.classify(_healthy_cadence(count=2), now=_NOW)

    assert status == "unknown"
    assert "cadence" in explanation


def test_a_finished_acl_run_is_completed_even_with_no_terminal_event() -> None:
    """The production path records no terminal event, and this is why it matters.

    A smoke that finished successfully on 2026-09-08 with
    `curriculum=scenario_acl_scenarionet` recorded 81 events and **no**
    `run_completed`: that event comes from the baseline loop, and the ACL driver
    has its own. Judging by events alone, this run would have been called
    `stalled` as soon as its silence outgrew its cadence — a false alarm on a
    successful run, which is the error this tool exists not to make.
    """

    events = _healthy_cadence(silence=60.0 * 60.0)

    assert run_liveness.classify(events, now=_NOW)[0] == "stalled"
    assert run_liveness.classify(events, now=_NOW, metadata_status="completed")[0] == "completed"


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [("completed", "completed"), ("failed", "failed"), ("interrupted", "interrupted")],
)
def test_a_terminal_recorded_status_is_honoured(recorded: str, expected: str) -> None:
    status, explanation = run_liveness.classify(
        _healthy_cadence(silence=60.0 * 60.0), now=_NOW, metadata_status=recorded
    )

    assert status == expected
    assert recorded in explanation


def test_the_initial_running_status_does_not_mask_a_stall() -> None:
    """`run_metadata.yaml` opens at `running`, so it is not a terminal statement."""

    status, _ = run_liveness.classify(
        _healthy_cadence(silence=60.0 * 60.0), now=_NOW, metadata_status="running"
    )

    assert status == "stalled"


def test_a_failure_event_outranks_a_stale_running_status() -> None:
    """`failfast.py` writes `run_failed` on a path where the metadata update may not run."""

    events = [
        *_healthy_cadence(),
        {
            "event": "run_failed",
            "time": (_NOW - timedelta(seconds=30)).isoformat(timespec="seconds"),
        },
    ]

    status, _ = run_liveness.classify(events, now=_NOW, metadata_status="running")

    assert status == "failed"


def test_an_empty_or_torn_log_does_not_raise() -> None:
    assert run_liveness.classify([], now=_NOW)[0] == "unknown"
    assert run_liveness.classify([{"event": "x"}], now=_NOW)[0] == "unknown"


def test_exit_status_separates_the_actionable_from_the_healthy(tmp_path: Path) -> None:
    """So a supervision loop can branch on it without parsing the message."""

    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    events_path = run_dir / "logs" / "events.jsonl"

    events_path.write_text(
        '{"event": "run_failed", "time": "2026-09-08T11:00:00"}\n', encoding="utf-8"
    )
    assert run_liveness.main([str(run_dir)]) == 1

    events_path.write_text(
        '{"event": "run_completed", "time": "2026-09-08T11:00:00"}\n', encoding="utf-8"
    )
    assert run_liveness.main([str(run_dir)]) == 0

    assert run_liveness.main([str(tmp_path / "absent")]) == 2
