from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from thesis_rl.agent.planners.core.lifecycle import BasePlannerLifecycle


class BasePlanner(Protocol):
    """Planner backend protocol consumed by Agent runtime."""

    def get_lifecycle(self) -> BasePlannerLifecycle: ...

    def predict(self, observation: Any, deterministic: bool = False): ...

    def save(self, checkpoint_path: str | Path) -> None: ...

    def set_env(self, env: Any) -> None: ...
