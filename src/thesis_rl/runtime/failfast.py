"""Fail-fast teardown after an unhandled training failure (open_items C10, C12).

A run that raised inside the training loop used to record ``run_failed`` and then
hang for hours in interpreter teardown with the MetaDrive engine and the vector
workers still alive (C10; the same family as C7's exit 139). The ``tmux`` session
and the container stayed up, the GPU memory stayed allocated, and the tail of
``train.log`` kept reading like progress.

This module makes the failure terminal in bounded time:

1. close the asynchronous evaluation manager and the training environment on a
   best-effort basis, each failure logged and swallowed;
2. flush every logging handler so the traceback and ``run_failed`` reach disk;
3. arm a daemon watchdog that forces the process to exit with ``exit_code`` if
   ordinary teardown has not completed within ``grace_seconds``.

The original exception is still re-raised by the caller, so callers and tests
that observe it keep observing it; the watchdog only bounds how long a hung
teardown can outlive the failure.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import Any

RUN_FAILED_EXIT_GRACE_SECONDS: float = 120.0
"""Seconds ordinary teardown gets after ``run_failed`` before the forced exit."""

RUN_FAILED_EXIT_CODE: int = 1


def close_quietly(resource: Any, name: str, logger: logging.Logger) -> bool:
    """Call ``resource.close()`` if present; log and swallow any failure.

    Returns ``True`` when a close method existed and returned normally.
    """
    if resource is None:
        return False
    close = getattr(resource, "close", None)
    if close is None:
        return False
    try:
        close()
    except Exception as exc:  # noqa: BLE001 - teardown must not mask the run failure
        logger.warning("Fail-fast teardown: closing %s failed | error=%s", name, str(exc))
        return False
    logger.info("Fail-fast teardown: %s closed", name)
    return True


def flush_logging_handlers() -> None:
    """Flush every handler attached to the root logger and to named loggers."""
    loggers: list[logging.Logger] = [logging.getLogger()]
    loggers.extend(
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    for logger in loggers:
        for handler in list(logger.handlers):
            try:
                handler.flush()
            except Exception:  # noqa: BLE001 - a broken handler must not block exit
                continue


def schedule_forced_exit(
    grace_seconds: float,
    exit_code: int,
    logger: logging.Logger,
    *,
    on_fire: Callable[[], None] | None = None,
    exit_fn: Callable[[int], Any] = os._exit,
) -> threading.Thread:
    """Arm a daemon watchdog that calls ``exit_fn(exit_code)`` after ``grace_seconds``.

    The thread is a daemon: if ordinary teardown finishes first, the interpreter
    exits and the watchdog dies with it without firing. ``on_fire`` runs before
    the exit so the caller can record the forced exit in its own artifacts.
    """

    def _fire() -> None:
        try:
            logger.error(
                "Fail-fast teardown: forcing process exit | grace_seconds=%s | exit_code=%d",
                grace_seconds,
                exit_code,
            )
            if on_fire is not None:
                try:
                    on_fire()
                except Exception as exc:  # noqa: BLE001 - the exit must still happen
                    logger.warning(
                        "Fail-fast teardown: forced-exit hook failed | error=%s", str(exc)
                    )
            flush_logging_handlers()
        finally:
            exit_fn(exit_code)

    timer = threading.Timer(max(0.0, float(grace_seconds)), _fire)
    timer.daemon = True
    timer.name = "run-failed-forced-exit"
    timer.start()
    return timer


def fail_fast_after_run_failure(
    *,
    env: Any,
    async_evaluation_manager: Any,
    logger: logging.Logger,
    grace_seconds: float = RUN_FAILED_EXIT_GRACE_SECONDS,
    exit_code: int = RUN_FAILED_EXIT_CODE,
    on_forced_exit: Callable[[], None] | None = None,
    exit_fn: Callable[[int], Any] = os._exit,
) -> threading.Thread:
    """Bounded teardown after ``run_failed``: close resources, flush, arm the watchdog."""
    close_quietly(async_evaluation_manager, "async evaluation manager", logger)
    close_quietly(env, "training environment", logger)
    flush_logging_handlers()
    return schedule_forced_exit(
        grace_seconds,
        exit_code,
        logger,
        on_fire=on_forced_exit,
        exit_fn=exit_fn,
    )
