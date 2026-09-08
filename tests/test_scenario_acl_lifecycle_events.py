"""`C42`: the ACL driver must state its progress and its own end.

`C12` added a TTY-independent `training_progress` event and a step counter that
survives `| tee`, and wired it into the baseline loop. The scenario-ACL driver
collects through its own loop and never passed the callback, so an ACL run — and
every serious profile is an ACL run — had no step figure between chunk
boundaries and no fps or EMA losses anywhere. Separately, the driver returns to
`train_loop` before the loop's own `run_completed`, so a *finished* ACL run left
no terminal event: a successful smoke on 2026-09-08 recorded 81 events and none
of them said the run had ended. `run_interrupted` was recorded by the driver and
`run_failed` by the loop's exception handler, which wraps the delegation, so
those two outcomes were already covered.

Both defects are a call site forgetting something, so they are pinned
structurally: a behavioural test would need the whole runtime, while these catch
the next edit that drops the argument or adds a fifth completion branch without
its event.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

_DRIVER = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "thesis_rl"
    / "curriculum"
    / "scenario_acl"
    / "driver.py"
)
_TREE = ast.parse(_DRIVER.read_text(encoding="utf-8"))


def _calls(name: str) -> list[ast.Call]:
    found: list[ast.Call] = []
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        attribute = func.attr if isinstance(func, ast.Attribute) else None
        plain = func.id if isinstance(func, ast.Name) else None
        if name in {attribute, plain}:
            found.append(node)
    return found


def _log_event_names() -> list[str]:
    names: list[str] = []
    for call in _calls("log_event"):
        for argument in call.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                names.append(argument.value)
                break
    return names


def test_the_vector_collection_call_passes_the_progress_callback() -> None:
    """The one missing argument that left the production path without a counter."""

    calls = _calls("train_vectorized")

    assert calls, "the driver no longer calls train_vectorized; this test needs rewriting"
    for call in calls:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "progress_callback" in keywords, (
            "agent.train_vectorized must receive progress_callback, or an ACL run has no "
            "step counter between chunk boundaries (C42)"
        )


def test_the_callback_reaches_an_enabled_channel() -> None:
    """The two facts that make passing the callback sufficient, pinned not assumed.

    A structural test on the call site proves the argument is passed, not that it
    is *received* under the right name or that the channel is on. Both are
    properties of `Agent.train_vectorized`'s signature, so they are cheap to
    assert here — which is what makes an end-to-end run unnecessary for this
    change: `tests/test_training_progress_failfast.py` already exercises the
    emission itself through a two-slot fake environment.
    """

    from thesis_rl.agent.agent import Agent
    from thesis_rl.runtime.training_progress import TRAINING_PROGRESS_EVENT_INTERVAL_STEPS

    parameters = inspect.signature(Agent.train_vectorized).parameters

    assert "progress_callback" in parameters, "the kwarg the driver passes must exist"
    assert parameters["progress_interval"].default == TRAINING_PROGRESS_EVENT_INTERVAL_STEPS, (
        "the channel must be enabled by default, or the driver would also have to pass "
        "progress_interval and passing the callback alone would emit nothing"
    )


def test_every_completion_branch_records_that_the_run_ended() -> None:
    """One `run_completed` per `status: completed`, so no branch ends in silence.

    The driver has four completion branches — final-panels and legacy for each of
    the vectorized and non-vectorized paths — and each updates the run metadata.
    Counting them against the events is what catches a fifth branch added later
    without its event.
    """

    completed_metadata_updates = 0
    for call in _calls("update_run_metadata"):
        for argument in call.args:
            if not isinstance(argument, ast.Dict):
                continue
            for key, value in zip(argument.keys, argument.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "status"
                    and isinstance(value, ast.Constant)
                    and value.value == "completed"
                ):
                    completed_metadata_updates += 1

    assert completed_metadata_updates == 4, (
        f"expected the four known completion branches, found {completed_metadata_updates}; "
        "if a branch was added or removed, update this test and its run_completed pairing"
    )
    assert _log_event_names().count("run_completed") == completed_metadata_updates


def test_the_driver_states_every_outcome_it_can_reach() -> None:
    """Completed and interrupted from here; failed from the loop that wraps this.

    `train_loop`'s `try` spans the delegation, so an exception raised in the
    driver reaches its `except Exception` and is recorded there as `run_failed`.
    That is why this driver does not emit it, and asserting the split keeps a
    future reader from "fixing" a gap that is not one.
    """

    emitted = set(_log_event_names())

    assert "run_completed" in emitted
    assert "run_interrupted" in emitted
    assert "run_failed" not in emitted, (
        "run_failed is the wrapping loop's to record; two writers of one fact is the C8 defect"
    )


def test_progress_events_are_emitted_under_the_shared_event_name() -> None:
    """So `C41`'s reader and any log consumer see the same name as the baseline."""

    assert "training_progress" in set(_log_event_names())
