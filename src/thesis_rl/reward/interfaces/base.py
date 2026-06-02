from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from thesis_rl.reward.types import RewardComputationResult


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
