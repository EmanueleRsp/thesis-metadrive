from __future__ import annotations

from typing import Any, Protocol


class BasePreprocessor(Protocol):
    def reset(self) -> None: ...
    def __call__(self, observation: Any) -> Any: ...
