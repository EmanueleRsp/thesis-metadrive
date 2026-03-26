from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RuleEvaluationResult:
    """Container for one rule output in runtime order."""

    name: str
    margin: float
    priority: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RewardComputationResult:
    """Output contract for reward managers used by runtime wrappers."""

    final_reward: float
    scalar_rule_reward: float
    rule_reward_vector: list[float]
    rule_bounded_vector: list[float]
    rule_components: dict[str, float]
    rule_metadata: dict[str, Any] = field(default_factory=dict)
    rule_violation_vector: list[float] | None = None


class BaseRewardManager(ABC):
    """Abstract interface for reward composition modules."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal episodic state if needed."""

    @abstractmethod
    def compute(
        self,
        env_reward: float,
        info: dict[str, Any],
    ) -> RewardComputationResult:
        """Compute final scalar reward and structured rulebook outputs."""
