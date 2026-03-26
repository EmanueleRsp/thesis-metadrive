from __future__ import annotations

from abc import ABC, abstractmethod

from thesis_rl.rulebook.types import RuleEvalInput, RuleVector


class RuleEvaluator(ABC):
    """Abstract rule evaluator interface."""

    @abstractmethod
    def evaluate(self, rule_eval_input: RuleEvalInput) -> RuleVector:
        """Evaluate all configured rules and return ordered margins."""
