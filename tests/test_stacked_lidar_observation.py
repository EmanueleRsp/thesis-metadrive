from __future__ import annotations

import numpy as np
import pytest

from thesis_rl.envs.observations.stacked_lidar import StackedLidarStateObservation


def test_stacked_lidar_fills_history_and_preserves_oldest_to_current_order() -> None:
    observation = StackedLidarStateObservation({})
    observation.set_frame_builder(lambda vehicle: np.full(308, vehicle, dtype=np.float32))

    first = observation.observe(0.25)
    second = observation.observe(0.5)

    assert first.shape == (1540,)
    np.testing.assert_array_equal(first, np.full(1540, 0.25, dtype=np.float32))
    np.testing.assert_array_equal(second[: 4 * 308], np.full(4 * 308, 0.25, dtype=np.float32))
    np.testing.assert_array_equal(second[-308:], np.full(308, 0.5, dtype=np.float32))


def test_stacked_lidar_rejects_missing_builder_and_shape_drift() -> None:
    observation = StackedLidarStateObservation({})
    with pytest.raises(RuntimeError, match="requires a causal"):
        observation.observe(object())
    observation.set_frame_builder(lambda _vehicle: np.zeros(307, dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        observation.observe(object())
