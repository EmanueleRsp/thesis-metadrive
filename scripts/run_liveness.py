#!/usr/bin/env python3
"""Answer "is this run alive?" from the one signal that cannot lie.

`C10`: a run that raised inside the training loop can leave its container, its
`tmux` session and its GPU memory alive for hours. Both habitual checks fail
there — the session is listed, and the tail of `train.log` shows the last
`chunk_started` line, which reads exactly like progress. One run was described
as advancing while it had been dead for ninety minutes.

`logs/events.jsonl` is the signal that holds, because `append_jsonl` reopens and
closes the file per record, so every event is on disk when it is written, and
each carries a timestamp. This reads it and reports one of:

    completed   a terminal `run_completed` was recorded
    failed      a terminal `run_failed` / `run_forced_exit` was recorded
    running     no terminal event, and the last event is recent
    stalled     no terminal event, and the last event is older than this run's
                own observed cadence allows

**The staleness threshold is measured, not chosen.** The gaps between this run's
own recent events are what its healthy cadence looks like, and a run is called
stalled once its silence exceeds a multiple of the longest recent gap. A run
with too few events to measure is reported as `unknown` rather than guessed at,
because a false "stalled" costs a needless investigation and a false "running"
costs hours of an idle GPU.

**Any event counts, deliberately.** An earlier version keyed the cadence on
`C12`'s `training_progress` events and was blind on the production path: `C12`
instrumented `train_vectorized`, but the scenario-ACL driver collects through
its own loop, so an `obs=semantic_v3 + curriculum=scenario_acl_scenarionet` run
emits none of them — verified on a live run, whose 81 events were all
`scenario_acl_episode_ended` and `scenario_acl_reset_timing`. Whatever a run
emits is its heartbeat.

**Two terminal signals, because neither path writes both.** The baseline loop
records `run_completed` / `run_failed` as events; the scenario-ACL driver records
neither — a smoke that finished successfully on 2026-09-08 recorded no terminal
event at all — but every path updates `status` in `artifacts/run_metadata.yaml`
to `completed`, `interrupted` or `failed`. Reading only the events would have
reported a finished production run as `stalled` once its silence grew, which is
the false alarm this tool exists to avoid. Events take precedence when both
speak, because `runtime/failfast.py` writes `run_failed` on a path where the
metadata update may not be reached.

Exit status: 0 completed or running, 1 failed or stalled, 2 unknown or unreadable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# How many times the longest recent healthy gap a run may be silent before the
# silence stops being ordinary jitter. 4 is a tolerance, not a calibration: the
# quantity it multiplies is measured, so this only says "clearly beyond the
# spread", and a slow chunk boundary is well inside it.
STALL_TOLERANCE = 4.0

# Fewer intervals than this and the cadence is not established.
MIN_INTERVALS_FOR_CADENCE = 3

TERMINAL_FAILURE_EVENTS = ("run_failed", "run_forced_exit")
TERMINAL_SUCCESS_EVENTS = ("run_completed",)

# `runtime/io/metadata.py` opens a run at `running` and the loops move it to one
# of these. `interrupted` is a deliberate stop, so it is terminal but not a
# fault: nothing is malfunctioning and nothing should restart it automatically.
TERMINAL_METADATA_STATUSES = ("completed", "failed", "interrupted")


def _read_events(events_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            # A run killed mid-write can leave a torn final line; every earlier
            # record is still intact, which is what matters here.
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _timestamp(event: dict[str, Any]) -> datetime | None:
    raw = event.get("time")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _observed_cadence_seconds(events: list[dict[str, Any]]) -> float | None:
    """The longest gap between this run's recent events, or None if unmeasurable.

    Every event type counts: see the module docstring on why keying this on one
    type left the production path unjudgeable.
    """

    stamps = [stamp for stamp in (_timestamp(event) for event in events) if stamp is not None]
    if len(stamps) < MIN_INTERVALS_FOR_CADENCE + 1:
        return None
    gaps = [
        (later - earlier).total_seconds()
        for earlier, later in zip(stamps, stamps[1:])
        if (later - earlier).total_seconds() >= 0.0
    ]
    if len(gaps) < MIN_INTERVALS_FOR_CADENCE:
        return None
    # The longest gap over the *whole* history, not a recent window, and zero
    # gaps are kept rather than filtered. Both choices make the threshold
    # conservative on purpose: `log_event` stamps to the second and the ACL
    # driver emits bursts inside one second, so a recent-window maximum is
    # routinely 0 s or 1 s, and multiplying that would call a healthy run dead
    # during any ordinary pause. Whole-history means the threshold only grows,
    # so the error this can make is being slow to notice a stall — which costs
    # idle GPU time — rather than reporting a working run as dead, which costs
    # the run.
    longest = max(gaps)
    return longest if longest > 0.0 else None


def classify(
    events: list[dict[str, Any]], *, now: datetime, metadata_status: str | None = None
) -> tuple[str, str]:
    """Return ``(status, explanation)`` for one run's event stream and metadata."""

    names = [str(event.get("event", "")) for event in events]
    for failure in TERMINAL_FAILURE_EVENTS:
        if failure in names:
            return "failed", f"{failure} recorded; the process may still be tearing down"
    for success in TERMINAL_SUCCESS_EVENTS:
        if success in names:
            return "completed", f"{success} recorded"

    # The scenario-ACL driver records no terminal event, so on the production
    # path this is the only statement a finished run makes about itself.
    recorded = (metadata_status or "").strip().lower()
    if recorded in TERMINAL_METADATA_STATUSES:
        return recorded, f"run_metadata.yaml records status={recorded!r}"

    if not events:
        return "unknown", "no events recorded yet"

    last_stamp = next(
        (stamp for stamp in (_timestamp(event) for event in reversed(events)) if stamp is not None),
        None,
    )
    if last_stamp is None:
        return "unknown", "no event carries a readable timestamp"

    silence = (now - last_stamp).total_seconds()
    cadence = _observed_cadence_seconds(events)
    if cadence is None:
        return (
            "unknown",
            f"silent for {silence:.0f}s, but this run has too few timestamped events "
            "to establish a cadence to judge it against",
        )

    budget = cadence * STALL_TOLERANCE
    if silence > budget:
        return (
            "stalled",
            f"silent for {silence:.0f}s against a measured cadence of {cadence:.0f}s "
            f"(tolerance {STALL_TOLERANCE:g}x = {budget:.0f}s)",
        )
    return (
        "running",
        f"last event {silence:.0f}s ago, within the measured cadence of {cadence:.0f}s "
        f"(tolerance {budget:.0f}s)",
    )


def _recorded_status(run_dir: Path) -> str | None:
    """`status:` from `artifacts/run_metadata.yaml`, read without a YAML parser.

    The field is a top-level scalar written by `runtime/io/metadata.py`, so a
    line match is enough and keeps this script runnable outside the project
    environment — which is where someone reaching for a liveness check often is.
    """

    path = run_dir / "artifacts" / "run_metadata.yaml"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("status:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="a run directory, or its logs/events.jsonl")
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    events_path = run_dir
    if events_path.is_dir():
        events_path = events_path / "logs" / "events.jsonl"
    else:
        run_dir = events_path.parent.parent
    if not events_path.is_file():
        print(f"unreadable: no event log at {events_path}")
        return 2

    status, explanation = classify(
        _read_events(events_path),
        now=datetime.now(),
        metadata_status=_recorded_status(run_dir),
    )
    print(f"{status}: {explanation}")
    if status in {"failed", "stalled"}:
        return 1
    if status == "unknown":
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through `main`
    sys.exit(main())
