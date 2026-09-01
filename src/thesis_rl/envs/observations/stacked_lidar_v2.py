"""21-frame masked wrapper for the causal 310D LiDAR observation.

Depth and mask rationale: see `docs/implementation/
lidar_arm_temporal_alignment_and_lq_tokenization_v2.0_exec_plan.md` (`DEC-002`,
`DEC-004`). The window is derived from the Rulebook's longest bounded timer
(`DASHED_TCAP_S = 2.0` s, `src/thesis_rl/rulebook/v2/components/road.py`) at the
10 Hz control period, i.e. 21 samples. Absent frames at episode start are
zero-filled and flagged in `frame_mask` rather than replicated, so a genuine
2.0 s static history stays distinguishable from a warm-up.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from metadrive.obs.observation_base import BaseObservation


class StackedLidarObservationV2(BaseObservation):
    """Stack 21 causal 310D frames, oldest to current, with a validity mask."""

    FRAME_DIM = 310
    HISTORY_LENGTH = 21
    MASK_DIM = HISTORY_LENGTH
    STACKED_DIM = FRAME_DIM * HISTORY_LENGTH + MASK_DIM

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
            raise TypeError("Stacked LiDAR v2 frame builder must be callable")
        self._frame_builder = builder

    def reset(self, env: object, vehicle: object | None = None) -> None:
        del env, vehicle
        self._frames.clear()

    def observe(self, vehicle: object) -> np.ndarray:
        if self._frame_builder is None:
            raise RuntimeError("StackedLidarObservationV2 requires a causal 310D frame builder")
        frame = np.asarray(self._frame_builder(vehicle), dtype=np.float32).reshape(-1)
        if frame.shape != (self.FRAME_DIM,):
            raise ValueError(
                f"Causal LiDAR frame must have shape ({self.FRAME_DIM},), got {frame.shape}"
            )
        if not np.all(np.isfinite(frame)):
            raise ValueError("Causal LiDAR frame must contain finite values")
        if not np.all((-1.0 <= frame) & (frame <= 1.0)):
            raise ValueError("Causal LiDAR frame values must be in [-1, 1]")
        self._frames.append(frame.copy())

        num_present = len(self._frames)
        num_absent = self.HISTORY_LENGTH - num_present
        mask = np.concatenate(
            (
                np.zeros(num_absent, dtype=np.float32),
                np.ones(num_present, dtype=np.float32),
            )
        )
        padded_frames = [np.zeros(self.FRAME_DIM, dtype=np.float32)] * num_absent + list(
            self._frames
        )
        stacked = np.concatenate((*padded_frames, mask), dtype=np.float32)
        if stacked.shape != (self.STACKED_DIM,):
            raise RuntimeError("Stacked LiDAR v2 observation violated its fixed dimension")
        return stacked
