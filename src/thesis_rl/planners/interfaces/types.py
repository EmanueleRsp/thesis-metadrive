from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainState:
    total_steps: int = 0
    update_steps: int = 0


UpdateMetrics = dict[str, float | int]
