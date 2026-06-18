from __future__ import annotations

from typing import Any

import numpy as np


class NormalActionNoise:
    def __init__(self, mean: np.ndarray, sigma: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.sigma = np.asarray(sigma, dtype=np.float32).reshape(-1)
        if self.mean.shape != self.sigma.shape:
            raise ValueError(
                "Action noise mean and sigma must have the same shape: "
                f"mean={self.mean.shape}, sigma={self.sigma.shape}"
            )

    def reset(self, indices: list[int] | np.ndarray | None = None) -> None:
        _ = indices

    def sample(self, batch_size: int) -> np.ndarray:
        if int(batch_size) <= 0:
            raise ValueError("`batch_size` must be > 0 for action noise sampling.")
        mean = np.broadcast_to(self.mean, (int(batch_size), self.mean.shape[0]))
        sigma = np.broadcast_to(self.sigma, (int(batch_size), self.sigma.shape[0]))
        return np.random.normal(loc=mean, scale=sigma).astype(np.float32)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def build_action_noise(cfg_planner: Any, action_dim: int) -> NormalActionNoise | None:
    sigma = float(_cfg_get(cfg_planner, "action_noise_sigma", 0.1))
    mean = float(_cfg_get(cfg_planner, "action_noise_mean", 0.0))
    noise_type_raw = _cfg_get(cfg_planner, "action_noise_type", None)
    if noise_type_raw is None:
        noise_type = "normal" if sigma > 0.0 or mean != 0.0 else "none"
    else:
        noise_type = str(noise_type_raw).strip().lower()

    if noise_type in {"", "none", "null", "off"}:
        return None
    if noise_type not in {"normal", "gaussian"}:
        raise ValueError(f"Unsupported TD3 action noise type: {noise_type_raw}")

    mean_vector = np.full((int(action_dim),), mean, dtype=np.float32)
    sigma_vector = np.full((int(action_dim),), sigma, dtype=np.float32)
    return NormalActionNoise(mean=mean_vector, sigma=sigma_vector)
