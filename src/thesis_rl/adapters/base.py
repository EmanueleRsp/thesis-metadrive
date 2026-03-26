from __future__ import annotations

from typing import Any, Protocol


class BaseAdapter(Protocol):
    is_neural: bool
    requires_training: bool

    def __call__(self, planner_output: Any) -> Any: ...

    def begin_training(self) -> None: ...

    def maybe_update(self) -> None: ...

    def end_training(self) -> None: ...

    def save(self, checkpoint_path: str) -> None: ...

    def load(self, checkpoint_path: str) -> None: ...
