from __future__ import annotations

import numpy as np

from thesis_rl.scenarios.conflicts import closest_approach_conflict


def test_closest_approach_detects_planar_conflict() -> None:
    timestamps = np.arange(6, dtype=np.float64)
    ego = np.column_stack((timestamps, np.zeros(6), np.zeros(6)))
    other = np.column_stack((6.0 - timestamps, np.zeros(6), np.zeros(6)))

    conflict = closest_approach_conflict(
        ego,
        other,
        np.ones(6, dtype=bool),
        timestamps,
        horizon_s=5.0,
        distance_threshold_m=4.0,
        relevance_radius_m=50.0,
        vertical_tolerance_m=3.0,
    )

    assert conflict.is_conflict is True
    assert conflict.min_dcpa_m == 0.0
    assert conflict.min_tcpa_s is not None


def test_closest_approach_rejects_parallel_motion() -> None:
    timestamps = np.arange(6, dtype=np.float64)
    ego = np.column_stack((timestamps, np.zeros(6), np.zeros(6)))
    other = ego + np.asarray([10.0, 0.0, 0.0])

    conflict = closest_approach_conflict(
        ego,
        other,
        np.ones(6, dtype=bool),
        timestamps,
        horizon_s=5.0,
        distance_threshold_m=4.0,
        relevance_radius_m=50.0,
        vertical_tolerance_m=3.0,
    )

    assert conflict.is_conflict is False
    assert conflict.min_dcpa_m is None
