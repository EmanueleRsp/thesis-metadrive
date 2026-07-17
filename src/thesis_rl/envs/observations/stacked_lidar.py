"""Strict five-frame wrapper for the causal 308D LiDAR observation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from metadrive.obs.observation_base import BaseObservation


class StackedLidarStateObservation(BaseObservation):
    """Stack five causal 308D frames in oldest-to-current order.

    A frame builder must be installed by the runtime wiring. Deliberately no
    native MetaDrive navigation or future-trajectory fallback is attempted.
    """

    FRAME_DIM = 308
    HISTORY_LENGTH = 5
    STACKED_DIM = FRAME_DIM * HISTORY_LENGTH

    def __init__(self, config: dict[str, Any]):
        self._frames: deque[np.ndarray] = deque(maxlen=self.HISTORY_LENGTH)
        self._frame_builder: Callable[[object], np.ndarray] | None = None
        super().__init__(config)
        self._observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.STACKED_DIM,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def set_frame_builder(self, builder: Callable[[object], np.ndarray]) -> None:
        if not callable(builder):
            raise TypeError("Stacked LiDAR frame builder must be callable")
        self._frame_builder = builder

    def reset(self, env: object, vehicle: object | None = None) -> None:
        del env, vehicle
        self._frames.clear()

    def observe(self, vehicle: object) -> np.ndarray:
        if self._frame_builder is None:
            raise RuntimeError("StackedLidarStateObservation requires a causal 308D frame builder")
        frame = np.asarray(self._frame_builder(vehicle), dtype=np.float32).reshape(-1)
        if frame.shape != (self.FRAME_DIM,):
            raise ValueError(
                f"Causal LiDAR frame must have shape ({self.FRAME_DIM},), got {frame.shape}"
            )
        if not np.all(np.isfinite(frame)):
            raise ValueError("Causal LiDAR frame must contain finite values")
        if not np.all((-1.0 <= frame) & (frame <= 1.0)):
            raise ValueError("Causal LiDAR frame values must be in [-1, 1]")
        if not self._frames:
            self._frames.extend(frame.copy() for _ in range(self.HISTORY_LENGTH))
        else:
            self._frames.append(frame.copy())
        stacked = np.concatenate(tuple(self._frames), dtype=np.float32)
        if stacked.shape != (self.STACKED_DIM,):
            raise RuntimeError("Stacked LiDAR observation violated its fixed dimension")
        return stacked
