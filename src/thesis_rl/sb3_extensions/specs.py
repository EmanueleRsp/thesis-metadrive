"""Common spec objects for thesis-specific SB3 integration.

These dataclasses give the repo a stable place to represent "what we want SB3
to build" before wiring those requests into custom policy classes and the local
SB3 fork itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Sb3PolicySpec:
    """Describe how a thesis run wants SB3 to construct its policy stack."""

    policy: str | type[Any] = "MlpPolicy"
    policy_kwargs: dict[str, Any] = field(default_factory=dict)

    def merged_policy_kwargs(self, **extra_kwargs: Any) -> dict[str, Any]:
        """Return a copy of policy kwargs with explicit overrides applied last."""

        merged = dict(self.policy_kwargs)
        merged.update(extra_kwargs)
        return merged


@dataclass(frozen=True)
class Sb3AlgorithmSpec:
    """Describe SB3 algorithm-side custom components beyond policy kwargs."""

    replay_buffer_class: type[Any] | None = None
    replay_buffer_kwargs: dict[str, Any] = field(default_factory=dict)
    algorithm_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def has_custom_replay_buffer(self) -> bool:
        """Whether the spec requests a non-default replay buffer path."""

        return self.replay_buffer_class is not None or bool(self.replay_buffer_kwargs)

    def merged_algorithm_kwargs(self, **extra_kwargs: Any) -> dict[str, Any]:
        """Return algorithm kwargs with explicit overrides applied last."""

        merged = dict(self.algorithm_kwargs)
        merged.update(extra_kwargs)
        return merged
