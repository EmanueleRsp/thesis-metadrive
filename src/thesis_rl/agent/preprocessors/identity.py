from __future__ import annotations

from typing import Any


class IdentityPreprocessor:
    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        return None

    def __call__(self, observation: Any) -> Any:
        return observation
