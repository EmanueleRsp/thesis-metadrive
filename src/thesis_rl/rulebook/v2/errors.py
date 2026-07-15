"""Errors raised by the fail-fast Rulebook v2 monitor."""

from __future__ import annotations

from dataclasses import dataclass


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
