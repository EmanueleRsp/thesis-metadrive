"""Offline closest-approach metrics used for curriculum catalog labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TrackConflict:
    is_conflict: bool
    min_dcpa_m: float | None
    min_tcpa_s: float | None


def closest_approach_conflict(
    ego_positions: np.ndarray,
    other_positions: np.ndarray,
    valid: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    horizon_s: float,
    distance_threshold_m: float,
    relevance_radius_m: float,
    vertical_tolerance_m: float,
) -> TrackConflict:
    """Return whether one track creates a predicted planar CPA conflict."""

    if horizon_s <= 0 or distance_threshold_m <= 0:
        raise ValueError("CPA horizon and distance threshold must be positive")
    if (
        ego_positions.ndim != 2
        or other_positions.ndim != 2
        or ego_positions.shape[1] < 3
        or other_positions.shape[1] < 3
        or len(ego_positions) != len(other_positions)
        or valid.ndim != 1
        or len(valid) != len(ego_positions)
    ):
        raise ValueError("CPA inputs must share the scenario length")
    if timestamps_s.shape != (len(valid),):
        raise ValueError("CPA timestamps must match the scenario length")
    pair_valid = valid[1:] & valid[:-1]
    dt = np.diff(timestamps_s)
    pair_valid &= np.isfinite(dt) & (dt > 0)
    if not np.any(pair_valid):
        return TrackConflict(False, None, None)

    relative = other_positions - ego_positions
    current = relative[1:][pair_valid]
    relative_velocity = np.diff(relative, axis=0)[pair_valid] / dt[pair_valid, None]
    finite = np.isfinite(current).all(axis=1) & np.isfinite(relative_velocity).all(axis=1)
    current = current[finite]
    relative_velocity = relative_velocity[finite]
    if not len(current):
        return TrackConflict(False, None, None)

    planar_position = current[:, :2]
    planar_velocity = relative_velocity[:, :2]
    speed_sq = np.einsum("ij,ij->i", planar_velocity, planar_velocity)
    dot = np.einsum("ij,ij->i", planar_position, planar_velocity)
    current_distance = np.linalg.norm(planar_position, axis=1)
    candidates = (
        (speed_sq > 1e-6)
        & (dot < 0)
        & (current_distance <= relevance_radius_m)
        & (np.abs(current[:, 2]) < vertical_tolerance_m)
    )
    if not np.any(candidates):
        return TrackConflict(False, None, None)

    position = planar_position[candidates]
    velocity = planar_velocity[candidates]
    candidate_speed_sq = speed_sq[candidates]
    tcpa = -dot[candidates] / candidate_speed_sq
    within_horizon = (tcpa >= 0) & (tcpa <= horizon_s)
    if not np.any(within_horizon):
        return TrackConflict(False, None, None)
    tcpa = tcpa[within_horizon]
    dcpa = np.linalg.norm(
        position[within_horizon] + velocity[within_horizon] * tcpa[:, None],
        axis=1,
    )
    conflict = dcpa <= distance_threshold_m
    if not np.any(conflict):
        return TrackConflict(False, None, None)
    conflict_dcpa = dcpa[conflict]
    conflict_tcpa = tcpa[conflict]
    minimum_index = int(np.argmin(conflict_dcpa))
    return TrackConflict(
        True,
        float(conflict_dcpa[minimum_index]),
        float(conflict_tcpa[minimum_index]),
    )


__all__ = ["TrackConflict", "closest_approach_conflict"]
