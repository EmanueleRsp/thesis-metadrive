from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from thesis_rl.agents.planner_lifecycle import BasePlannerLifecycle


class BasePlannerBackend(Protocol):
    def get_lifecycle(self) -> BasePlannerLifecycle: ...
    def predict(self, observation: Any, deterministic: bool = False): ...
    def save(self, checkpoint_path: str | Path) -> None: ...
