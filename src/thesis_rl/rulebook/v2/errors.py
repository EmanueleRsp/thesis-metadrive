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
    ROUTE_PROJECTION_DISCONTINUOUS = "ROUTE_PROJECTION_DISCONTINUOUS"


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


class RuntimeGeometryNotEvaluableReason(str, Enum):
    """Closed set of live geometry failures eligible for a geometry abort.

    Deliberately disjoint from :class:`RuntimeScenarioNotEvaluableReason`. That
    type covers *data* defects in the scenario record, recurs on the record, and
    quarantines it. These are *numerical* failures in a geometric construction:
    whether they fire depends on where the actors are, which depends on the
    policy, so the same record may be evaluable on the next episode.
    """

    DECOMPOSITION_COVERAGE_SHORTFALL = "DECOMPOSITION_COVERAGE_SHORTFALL"
    DECOMPOSITION_NO_VISIBLE_BRIDGE = "DECOMPOSITION_NO_VISIBLE_BRIDGE"
    DEGENERATE_RING = "DEGENERATE_RING"


class RuntimeGeometryNotEvaluableError(RuntimeError):
    """A live geometry construction that could not be completed for this step.

    It ends the current episode at its last valid transition rather than the
    run, but it is **not** absorbed: every occurrence is counted, recorded with
    the geometry needed to rebuild it, and charged against a ceiling that fails
    the run once the condition is systemic. `open_items` `C9` is the reason this
    type exists -- a shortfall of 0.13 mm2 on a 261 m2 polygon ended a
    seven-hour run -- and the ceiling is the reason widening recovery here does
    not reduce to silent absorption.

    This type is deliberately **not** a subclass of
    :class:`RuntimeScenarioNotEvaluableError`. `RSA-V1` documents that type as
    narrow and forbids it from wrapping numerical failures; that contract stays
    intact, and the worker transports the two under different markers so no
    artifact can conflate them.
    """

    def __init__(
        self,
        reason: RuntimeGeometryNotEvaluableReason,
        message: str,
        *,
        diagnostics: Mapping[str, Any],
        geometry_wkt: str | None = None,
    ) -> None:
        self.reason = RuntimeGeometryNotEvaluableReason(reason)
        # `REQ-GA-003`: a recoverable geometry failure without its magnitude is
        # the unactionable error `C6` and `C9` both had to be re-instrumented to
        # diagnose. Refusing to construct one is cheaper than discovering it in
        # a log six hours into a run.
        if not diagnostics:
            raise ValueError(
                "A geometry abort must carry the measured magnitude of the violation "
                f"in its diagnostics; got none for reason {self.reason.value}."
            )
        self.diagnostics = dict(diagnostics)
        self.geometry_wkt = geometry_wkt
        super().__init__(message)
