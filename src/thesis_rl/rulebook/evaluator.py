from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from thesis_rl.rulebook.interfaces.base import RuleEvaluator
from thesis_rl.rulebook.registry import load_rulebook_from_config
from thesis_rl.rulebook.types import RuleEvalInput, RuleResult, RuleSpec, RuleVector


logger = logging.getLogger(__name__)


class RulebookEvaluationError(RuntimeError):
    """Raised when strict Rulebook v1 evaluation cannot be trusted."""


class ScenicRulesEvaluator(RuleEvaluator):
    """Evaluate configured rules and return ordered margin vector."""

    def __init__(self, rules: list[RuleSpec], *, strict: bool = False) -> None:
        self.rules = list(rules)
        self.strict = bool(strict)
        self._failed_rule_logged_once: set[str] = set()
        self._rule_states: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        self._rule_states = {}

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        strict: bool | None = None,
    ) -> "ScenicRulesEvaluator":
        strict_mode = bool(config.get("strict", False)) if strict is None else bool(strict)
        return cls(load_rulebook_from_config(config), strict=strict_mode)

    def evaluate(self, rule_eval_input: RuleEvalInput) -> RuleVector:
        names: list[str] = []
        values: list[float] = []
        priorities: list[int] = []
        failed_rules: list[str] = []
        results: list[RuleResult] = []

        for spec in self.rules:
            try:
                if spec.name in {"lane_marking_compliance"}:
                    output = spec.fn(
                        rule_eval_input,
                        state=self._rule_states.setdefault(spec.name, {}),
                        **spec.params,
                    )
                else:
                    output = spec.fn(rule_eval_input, **spec.params)
                if isinstance(output, RuleResult):
                    result = output
                else:
                    violated, margin = output
                    result = RuleResult(
                        name=spec.name,
                        margin=float(margin),
                        violated=bool(violated),
                        severity=max(0.0, -float(margin)),
                        available=True,
                        fallback_used=False,
                    )
                margin_value = float(result.margin)
            except Exception as exc:
                if self.strict:
                    raise RulebookEvaluationError(
                        f"Rule '{spec.name}' raised during strict evaluation."
                    ) from exc
                margin_value = 0.0
                result = RuleResult(
                    name=spec.name,
                    margin=0.0,
                    violated=False,
                    severity=0.0,
                    available=False,
                    fallback_used=False,
                    raw={"error": str(exc)},
                )
                failed_rules.append(spec.name)
                if spec.name not in self._failed_rule_logged_once:
                    logger.warning(
                        "Rule evaluation failed for '%s': %s. Using neutral margin=0.0.",
                        spec.name,
                        exc,
                    )
                    self._failed_rule_logged_once.add(spec.name)

            if self.strict and not result.available:
                raise RulebookEvaluationError(
                    f"Rule '{spec.name}' is unavailable in strict mode; "
                    "the required runtime input is missing or invalid."
                )
            if self.strict and result.fallback_used:
                raise RulebookEvaluationError(
                    f"Rule '{spec.name}' requested fallback in strict mode."
                )

            names.append(spec.name)
            values.append(margin_value)
            priorities.append(spec.priority)
            results.append(result)

        rule_payloads = {result.name: result.to_dict() for result in results}
        diagnostics = self._build_diagnostics(rule_eval_input, results)

        metadata = {
            "evaluator": "scenic_rules",
            "failed_rules": failed_rules,
            "rule_count": len(self.rules),
            "timestamp": rule_eval_input.metadata.get("timestamp"),
            "rules": rule_payloads,
            "diagnostics": diagnostics,
        }
        return RuleVector(
            names=names,
            values=np.asarray(values, dtype=np.float32),
            priorities=priorities,
            results=results,
            metadata=metadata,
        )

    def _build_diagnostics(
        self,
        rule_eval_input: RuleEvalInput,
        results: list[RuleResult],
    ) -> dict[str, Any]:
        result_by_name = {result.name: result for result in results}
        allowed = result_by_name.get("allowed_driving_area")
        wrong_way = {}
        if allowed is not None:
            raw = allowed.raw.get("wrong_way_diagnostic")
            if isinstance(raw, dict):
                wrong_way = dict(raw)

        progress = result_by_name.get("local_route_progress")
        progress_margin = float(progress.margin) if progress is not None else 0.0
        speed = None
        ego = rule_eval_input.ego_state
        if "speed_m_s" in ego:
            speed = float(ego["speed_m_s"])
        elif "velocity" in ego:
            try:
                speed = float(np.linalg.norm(np.asarray(ego["velocity"], dtype=np.float64)[:2]))
            except Exception:
                speed = None
        stalled = progress is not None and progress.available and abs(progress_margin) < 0.01 and speed is not None and speed < 0.1
        state = self._rule_states.setdefault("__diagnostics__", {})
        state["stagnation_steps"] = int(state.get("stagnation_steps", 0)) + 1 if stalled else 0
        return {
            "wrong_way": wrong_way,
            "stagnation": {
                "available": progress is not None and progress.available and speed is not None,
                "steps": int(state["stagnation_steps"]),
                "detected": int(state["stagnation_steps"]) >= 20,
            },
            "speed_m_s": speed,
        }
