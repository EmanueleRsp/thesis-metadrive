from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from thesis_rl.rulebook.base import RuleEvaluator
from thesis_rl.rulebook.rulebook_config import load_rulebook_from_config
from thesis_rl.rulebook.types import RuleEvalInput, RuleSpec, RuleVector


logger = logging.getLogger(__name__)


class ScenicRulesEvaluator(RuleEvaluator):
    """Evaluate configured rules and return ordered margin vector."""

    def __init__(self, rules: list[RuleSpec]) -> None:
        self.rules = list(rules)
        self._failed_rule_logged_once: set[str] = set()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ScenicRulesEvaluator":
        return cls(load_rulebook_from_config(config))

    def evaluate(self, rule_eval_input: RuleEvalInput) -> RuleVector:
        names: list[str] = []
        values: list[float] = []
        priorities: list[int] = []
        failed_rules: list[str] = []

        for spec in self.rules:
            try:
                _violated, margin = spec.fn(rule_eval_input, **spec.params)
                margin_value = float(margin)
            except Exception as exc:
                margin_value = 0.0
                failed_rules.append(spec.name)
                if spec.name not in self._failed_rule_logged_once:
                    logger.warning(
                        "Rule evaluation failed for '%s': %s. Using neutral margin=0.0.",
                        spec.name,
                        exc,
                    )
                    self._failed_rule_logged_once.add(spec.name)

            names.append(spec.name)
            values.append(margin_value)
            priorities.append(spec.priority)

        metadata = {
            "evaluator": "scenic_rules",
            "failed_rules": failed_rules,
            "rule_count": len(self.rules),
            "timestamp": rule_eval_input.metadata.get("timestamp"),
        }
        return RuleVector(
            names=names,
            values=np.asarray(values, dtype=np.float32),
            priorities=priorities,
            metadata=metadata,
        )
