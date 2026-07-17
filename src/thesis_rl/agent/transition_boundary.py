"""Canonical transition-boundary normalization for vectorized collectors."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_vector_transition_boundary(
    dones: np.ndarray,
    infos: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    next_observations: np.ndarray,
    *,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return separate flags and final observations for a vectorized step.

    ``terminated`` and ``truncated`` are the canonical collector contract. The
    inference branch is retained for older direct backend callers; the agent
    collector always supplies both arrays explicitly.
    """
    done_batch = np.asarray(dones, dtype=bool)
    if done_batch.ndim != 1 or len(infos) != int(done_batch.shape[0]):
        raise ValueError(
            "Vector transition flags, infos, and observations must have matching batch sizes."
        )

    if terminated is None or truncated is None:
        timeout_batch = np.asarray(
            [bool(info.get("TimeLimit.truncated", False)) for info in infos],
            dtype=bool,
        )
        truncated_batch = done_batch & timeout_batch
        terminated_batch = done_batch & ~truncated_batch
    else:
        terminated_batch = np.asarray(terminated, dtype=bool)
        truncated_batch = np.asarray(truncated, dtype=bool)
        if terminated_batch.shape != done_batch.shape or truncated_batch.shape != done_batch.shape:
            raise ValueError("`terminated` and `truncated` must match the `dones` shape.")
        if np.any((terminated_batch | truncated_batch) != done_batch):
            raise ValueError(
                "`terminated` and `truncated` must reproduce the combined `dones` flag."
            )

    final_observations = np.asarray(next_observations, dtype=np.float32).copy()
    for idx, info in enumerate(infos):
        if not isinstance(info, dict):
            raise TypeError(
                f"Expected vector transition info to be dict, got {type(info).__name__}."
            )
        final_observation = info.get("final_observation")
        if final_observation is None:
            final_observation = info.get("terminal_observation")
        if final_observation is not None:
            final_observation = np.asarray(final_observation, dtype=np.float32)
            info["final_observation"] = final_observation
            info["terminal_observation"] = final_observation
            if done_batch[idx]:
                final_observations[idx] = final_observation
        if truncated_batch[idx] and final_observation is None:
            raise ValueError(
                "A truncated vector transition must provide `final_observation` "
                "or `terminal_observation` before auto-reset."
            )
        info["terminated"] = bool(terminated_batch[idx])
        info["truncated"] = bool(truncated_batch[idx])
        info["TimeLimit.truncated"] = bool(truncated_batch[idx] and not terminated_batch[idx])

    return terminated_batch, truncated_batch, final_observations
