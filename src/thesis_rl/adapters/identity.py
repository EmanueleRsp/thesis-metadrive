from __future__ import annotations

import numpy as np


class IdentityAdapter:
    is_neural = False
    requires_training = False

    def __init__(
        self,
        low: float = -1.0,
        high: float = 1.0,
        expected_shape: tuple[int, ...] = (2,),
    ) -> None:
        self.low = low
        self.high = high
        self.expected_shape = expected_shape

    def __call__(self, planner_output) -> np.ndarray:
        action = np.asarray(planner_output, dtype=np.float32)
        if action.shape != self.expected_shape:
            raise ValueError(f"Expected action shape {self.expected_shape}, got {action.shape}")
        return action

    def begin_training(self) -> None:
        return None

    def maybe_update(self) -> None:
        return None

    def end_training(self) -> None:
        return None

    def save(self, checkpoint_path: str) -> None:
        _ = checkpoint_path
        return None

    def load(self, checkpoint_path: str) -> None:
        _ = checkpoint_path
        return None
