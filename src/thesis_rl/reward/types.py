from __future__ import annotations

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
