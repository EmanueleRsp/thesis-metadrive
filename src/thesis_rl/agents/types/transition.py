from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Transition:
    """Single environment transition passed from Agent to planner lifecycles."""

    observation: np.ndarray
    env_action: np.ndarray
    buffer_action: np.ndarray
    scalar_reward: float
    terminated: bool
    truncated: bool
    next_observation: np.ndarray
    terminal_observation: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)
