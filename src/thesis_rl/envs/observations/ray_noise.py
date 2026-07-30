"""Deterministic single-owner noise for normalized ray observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RayNoiseWrapper:
    """Apply the contract's Gaussian perturbation to normalized ray blocks."""

    sigma_normalized: float = 0.001
    dropout_prob: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.sigma_normalized < 0.0 or not np.isfinite(self.sigma_normalized):
            raise ValueError("Ray noise sigma must be finite and non-negative")
        if not 0.0 <= self.dropout_prob <= 1.0 or not np.isfinite(self.dropout_prob):
            raise ValueError("Ray noise dropout probability must be in [0, 1]")

    def perturb(self, values: Any, rng: Any) -> np.ndarray:
        """Perturb one ray block using only the supplied seeded RNG stream."""

        array = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError("Ray values must be finite")
        if not self.enabled:
            return np.clip(array, 0.0, 1.0).copy()
        if not hasattr(rng, "normal") or not hasattr(rng, "random"):
            raise TypeError("Ray noise requires a numpy-compatible RNG")
        noisy = array + np.asarray(
            rng.normal(0.0, self.sigma_normalized, size=array.shape), dtype=np.float32
        )
        if self.dropout_prob > 0.0:
            noisy[np.asarray(rng.random(array.shape)) < self.dropout_prob] = 0.0
        return np.clip(noisy, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def validate_native_noise_disabled(config: dict[str, Any]) -> None:
        """Reject native noise so this wrapper remains the sole noise owner."""

        for sensor_name in ("lidar", "side_detector", "lane_line_detector"):
            sensor = config.get(sensor_name, {})
            if not callable(getattr(sensor, "get", None)):
                raise ValueError(f"{sensor_name} configuration must be a mapping")
            gaussian = float(sensor.get("gaussian_noise", sensor.get("native_gaussian_noise", 0.0)))
            dropout = float(sensor.get("dropout_prob", sensor.get("native_dropout_prob", 0.0)))
            if gaussian != 0.0 or dropout != 0.0:
                raise ValueError(
                    f"Native noise must be disabled for {sensor_name} when RayNoiseWrapper is enabled"
                )
