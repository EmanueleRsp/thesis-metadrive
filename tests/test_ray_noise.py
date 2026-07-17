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


def test_disabled_ray_noise_is_identity_after_clipping() -> None:
    values = np.asarray([-1.0, 0.5, 2.0], dtype=np.float32)
    result = RayNoiseWrapper(enabled=False).perturb(values, np.random.default_rng(0))
    np.testing.assert_array_equal(result, [0.0, 0.5, 1.0])
