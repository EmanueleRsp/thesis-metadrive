"""Errors raised by the fail-fast Rulebook v2 monitor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    """Typed context attached to a Rulebook v2 evaluation failure."""

    scenario_id: str
    step_index: int
    component: str
    cause: str


class RulebookEvaluationError(RuntimeError):
    """A required Rulebook v2 invariant could not be evaluated.

    ``NOT_EVALUABLE`` is never converted to a neutral component result.  The
    wrapper will attach the scenario and control-step context before raising.
    """

    def __init__(self, failure: EvaluationFailure) -> None:
        self.failure = failure
        super().__init__(
            "Rulebook v2 evaluation failed: "
            f"scenario_id={failure.scenario_id!r}, step_index={failure.step_index}, "
            f"component={failure.component!r}, cause={failure.cause}"
        )


class RuntimeScenarioNotEvaluableReason(str, Enum):
    """Closed set of runtime data defects eligible for episode data-abort."""

    UNKNOWN_SIGNAL_STATE = "UNKNOWN_SIGNAL_STATE"
    INVALID_SIGNAL_TRANSITION = "INVALID_SIGNAL_TRANSITION"
    MISSING_SIGNAL_MAPPING = "MISSING_SIGNAL_MAPPING"
    INCOMPLETE_SIGNAL_TIMELINE = "INCOMPLETE_SIGNAL_TIMELINE"
    UNRESOLVED_PHYSICAL_SIGNAL = "UNRESOLVED_PHYSICAL_SIGNAL"


class RuntimeScenarioNotEvaluableError(RuntimeError):
    """A live scenario data defect that may abort only its current episode.

    This type is deliberately narrow. It must not wrap numerical, programming,
    serialization, worker, or learner failures.
    """

    def __init__(
        self,
        reason: RuntimeScenarioNotEvaluableReason,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason = RuntimeScenarioNotEvaluableReason(reason)
        self.diagnostics = dict(diagnostics or {})
        super().__init__(message)
