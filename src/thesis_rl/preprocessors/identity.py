from __future__ import annotations

from typing import Any

import numpy as np


class IdentityPreprocessor:
    def __init__(self, cast_to_float32: bool = True) -> None:
        self.cast_to_float32 = cast_to_float32

    def reset(self) -> None:
        return None

    def __call__(self, observation: Any) -> Any:
        if self.cast_to_float32 and isinstance(observation, np.ndarray):
            return observation.astype(np.float32, copy=False)
        return observation
