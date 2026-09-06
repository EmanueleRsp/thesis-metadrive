"""TTY-independent training progress (open_items C12).

The Rich ``Live`` training monitor renders nothing when stdout is not a terminal,
which is the case for every run launched through ``| tee``: between two
evaluations no artifact carried the step counter, the throughput or the EMA
losses. This module holds the pure helpers behind two complementary channels:

* a ``training_progress`` event appended to ``logs/events.jsonl`` every
  ``TRAINING_PROGRESS_EVENT_INTERVAL_STEPS`` collected steps of a chunk and at the
  end of the chunk (the durable, machine-readable channel);
* one plain-text line printed through the console when it is not a terminal (the
  human-readable channel that replaces the silent Live view).

Neither channel touches the learner, the environment or any metric written to the
CSV artifacts; both are observability only.
"""

from __future__ import annotations

import math
from typing import Any

TRAINING_PROGRESS_EVENT_INTERVAL_STEPS: int = 1000
"""Collected steps between two ``training_progress`` events within a chunk."""


def progress_bucket(collected_steps: int, interval_steps: int) -> int:
    """Return the progress bucket of ``collected_steps`` for ``interval_steps``.

    A bucket increments once every ``interval_steps`` collected steps; a
    non-positive interval disables the channel and always returns ``-1``.
    """
    if interval_steps <= 0:
        return -1
    return int(collected_steps) // int(interval_steps)


def progress_due(
    collected_steps: int,
    chunk_timesteps: int,
    interval_steps: int,
    last_bucket: int,
) -> bool:
    """Whether a progress record is due at ``collected_steps``.

    Due when the bucket advanced past ``last_bucket`` or when the chunk is
    complete, so that the last record of a chunk carries the chunk's final
    throughput even when ``chunk_timesteps`` is not a multiple of the interval.
    """
    if interval_steps <= 0:
        return False
    if collected_steps >= chunk_timesteps:
        return True
    return progress_bucket(collected_steps, interval_steps) > last_bucket


def _fmt(value: Any, spec: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    return format(number, spec)


def format_training_progress_line(snapshot: dict[str, Any]) -> str:
    """Render a progress snapshot as one plain-text line for non-terminal consoles."""
    return (
        "Training progress"
        f" | run_step={int(snapshot.get('run_env_steps', 0))}"
        f"/{int(snapshot.get('global_total_timesteps', 0))}"
        f" | chunk_step={int(snapshot.get('chunk_env_steps', 0))}"
        f"/{int(snapshot.get('chunk_timesteps', 0))}"
        f" | fps={_fmt(snapshot.get('fps', 0.0), '.2f')}"
        f" | elapsed_s={int(float(snapshot.get('elapsed_seconds', 0.0)))}"
        f" | episodes={int(snapshot.get('episodes', 0))}"
        f" | ep_len_mean={_fmt(snapshot.get('ep_len_mean', 0.0), '.1f')}"
        f" | ep_rew_mean={_fmt(snapshot.get('ep_rew_mean', 0.0), '.3f')}"
        f" | actor_loss_ema={_fmt(snapshot.get('ema_actor_loss', float('nan')), '.3g')}"
        f" | critic_loss_ema={_fmt(snapshot.get('ema_critic_loss', float('nan')), '.3g')}"
    )
