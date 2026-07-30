from __future__ import annotations

import numpy as np
import pytest

from thesis_rl.envs.observations.ray_noise import RayNoiseWrapper


def test_ray_noise_is_reproducible_from_supplied_rng() -> None:
    wrapper = RayNoiseWrapper(sigma_normalized=0.001)
    values = np.full(8, 0.5, dtype=np.float32)

    first = wrapper.perturb(values, np.random.default_rng(4))
    second = wrapper.perturb(values, np.random.default_rng(4))

    np.testing.assert_array_equal(first, second)
    assert np.all((0.0 <= first) & (first <= 1.0))


def test_ray_noise_rejects_native_noise_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="Native noise"):
        RayNoiseWrapper.validate_native_noise_disabled({"lidar": {"gaussian_noise": 0.01}})
    with pytest.raises(ValueError, match="finite"):
        RayNoiseWrapper().perturb([np.nan], np.random.default_rng(0))


class _MappingLikeSensorConfig:
    """Duck-typed stand-in for `metadrive.utils.config.Config` (not a `dict`).

    Regression for the discovery (2026-07-30) that real MetaDrive per-sensor
    config blocks are `Config` objects, not `dict`, so
    `isinstance(sensor, dict)` rejected every real vehicle at runtime.
    `validate_native_noise_disabled` now duck-types on `.get`.
    """

    def __init__(self, data: dict) -> None:
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


def test_validate_native_noise_disabled_accepts_mapping_like_sensor_config() -> None:
    RayNoiseWrapper.validate_native_noise_disabled(
        {"lidar": _MappingLikeSensorConfig({"gaussian_noise": 0.0, "dropout_prob": 0.0})}
    )


def test_validate_native_noise_disabled_rejects_non_mapping_sensor_config() -> None:
    with pytest.raises(ValueError, match="mapping"):
        RayNoiseWrapper.validate_native_noise_disabled({"lidar": object()})


def test_disabled_ray_noise_is_identity_after_clipping() -> None:
    values = np.asarray([-1.0, 0.5, 2.0], dtype=np.float32)
    result = RayNoiseWrapper(enabled=False).perturb(values, np.random.default_rng(0))
    np.testing.assert_array_equal(result, [0.0, 0.5, 1.0])
