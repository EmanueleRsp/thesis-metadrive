"""Graceful termination: SIGTERM is handled like Ctrl+C (`DEC-RES-002`).

``docker stop`` and batch schedulers send SIGTERM first and SIGKILL after a
grace period. Routing SIGTERM into ``KeyboardInterrupt`` lets the training
loops run their existing interrupt path (run metadata, events log, exit 130)
instead of dying without a trace.
"""

from __future__ import annotations

import logging
import signal
import threading
from typing import Any

_LOGGER = logging.getLogger(__name__)


def _raise_keyboard_interrupt(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise KeyboardInterrupt(f"signal {signum}")


def install_sigterm_as_keyboard_interrupt(logger: logging.Logger | None = None) -> bool:
    """Install the SIGTERM handler in the main thread; return whether it was installed.

    Only the main thread of the main interpreter may set signal handlers, and
    worker subprocesses keep the default disposition, so this is a no-op (with a
    debug log) anywhere else.
    """

    log = logger or _LOGGER
    if threading.current_thread() is not threading.main_thread():
        log.debug("SIGTERM handler not installed: not in the main thread.")
        return False
    try:
        signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    except (ValueError, OSError) as exc:
        log.debug("SIGTERM handler not installed: %s", exc)
        return False
    return True


__all__ = ["install_sigterm_as_keyboard_interrupt"]
