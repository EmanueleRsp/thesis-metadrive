"""Geometry aborts: a third outcome between fatal and silently absorbed.

`GEOM-ABORT`. `RSA-V1` `REQ-RSA-001` recovers only the six enumerated signal and
route data reasons and declares every other failure fatal, explicitly excluding
numerical ones. That is why `open_items` `C9` -- a 0.13 mm2 coverage shortfall on
a 261 m2 polygon -- ended a seven-hour training run.

These tests pin the three properties that make widening recovery here different
from weakening the guard: the geometry type is **disjoint** from the data-abort
type, an unenumerated geometry failure is **still fatal**, and a systemic run of
geometry aborts **still stops the run**.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import shapely
from gymnasium import Env, spaces
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.errors import (
    RuntimeGeometryNotEvaluableError,
    RuntimeGeometryNotEvaluableReason,
    RuntimeScenarioNotEvaluableError,
    RuntimeScenarioNotEvaluableReason,
)
from thesis_rl.rulebook.v2.geometry.continuous_sat import deterministic_convex_decomposition
from thesis_rl.runtime.data_abort import (
    GeometryAbortLedger,
    RuntimeScenarioQuarantine,
    append_geometry_abort_record,
)
from thesis_rl.runtime.execution.deterministic_subproc_vec_env import (
    DeterministicSubprocVecEnv,
    RuntimeGeometryAbort,
    RuntimeScenarioDataAbort,
    SubprocessWorkerError,
)


# --------------------------------------------------------------------------
# `TEST-GA-003` -- the two recovery classes must never be confusable.
# --------------------------------------------------------------------------


def test_geometry_and_data_abort_types_are_disjoint() -> None:
    geometry = RuntimeGeometryNotEvaluableError(
        RuntimeGeometryNotEvaluableReason.DEGENERATE_RING,
        "degenerate",
        diagnostics={"signed_area_m2": 1.0e-9},
    )
    data = RuntimeScenarioNotEvaluableError(
        RuntimeScenarioNotEvaluableReason.INVALID_SIGNAL_TRANSITION, "signal"
    )
    # `RSA-V1` documents its type as narrow and forbidden from wrapping numerical
    # failures. Subclassing either way would silently widen that contract.
    assert not isinstance(geometry, RuntimeScenarioNotEvaluableError)
    assert not isinstance(data, RuntimeGeometryNotEvaluableError)


def test_geometry_abort_cannot_be_constructed_without_a_magnitude() -> None:
    """`REQ-GA-003`, enforced structurally rather than by convention.

    The original `C9` message carried no number, and both plausible readings
    were excluded only once one was printed. Making the type refuse to exist
    without diagnostics is cheaper than re-instrumenting under a dead run.
    """

    with pytest.raises(ValueError, match="measured magnitude"):
        RuntimeGeometryNotEvaluableError(
            RuntimeGeometryNotEvaluableReason.DEGENERATE_RING, "no numbers", diagnostics={}
        )


# --------------------------------------------------------------------------
# `TEST-GA-001` / `TEST-GA-002` -- enumerated conditions recover, others do not.
# --------------------------------------------------------------------------


def test_degenerate_ring_raises_the_typed_geometry_abort_with_its_magnitude() -> None:
    with pytest.raises(RuntimeGeometryNotEvaluableError) as excinfo:
        deterministic_convex_decomposition(Polygon([(0, 0), (1e-9, 0), (0, 1e-9)]))

    error = excinfo.value
    assert error.reason is RuntimeGeometryNotEvaluableReason.DEGENERATE_RING
    assert isinstance(error.diagnostics["signed_area_m2"], float)
    assert error.diagnostics["vertex_count"] == 3


def test_unenumerated_geometry_failure_stays_fatal() -> None:
    """`REQ-GA-002`. Recovery is a closed set, exactly as `RSA-V1` made it."""

    with pytest.raises(ValueError) as excinfo:
        deterministic_convex_decomposition(shapely.from_wkt("LINESTRING (0 0, 1 1)"))

    assert not isinstance(excinfo.value, RuntimeGeometryNotEvaluableError)


# --------------------------------------------------------------------------
# `TEST-GA-005` / `TEST-GA-007` -- separate counters, and a ceiling that bites.
# --------------------------------------------------------------------------


def test_geometry_counters_never_touch_the_data_abort_quarantine() -> None:
    ledger = GeometryAbortLedger()
    quarantine = RuntimeScenarioQuarantine()

    ledger.record("scenario-1", "DECOMPOSITION_COVERAGE_SHORTFALL")
    quarantine.add("scenario-2", "INVALID_SIGNAL_TRANSITION")

    assert ledger.total == 1
    assert ledger.quarantined_uids == set()
    assert quarantine.scenario_uids == {"scenario-2"}
    assert "DECOMPOSITION_COVERAGE_SHORTFALL" not in quarantine.reason_counts


def test_one_geometry_abort_does_not_quarantine_the_scenario() -> None:
    """`DEC-GA-002`. Whether the failure fires depends on the policy, not only
    on the record, so one occurrence is not evidence the record is broken."""

    ledger = GeometryAbortLedger()
    assert ledger.record("scenario-1", "DECOMPOSITION_COVERAGE_SHORTFALL") is False
    assert ledger.quarantined_uids == set()
    assert ledger.ceiling_breach() is None


def test_repeats_on_one_uid_quarantine_it() -> None:
    ledger = GeometryAbortLedger()
    outcomes = [ledger.record("scenario-1", "DEGENERATE_RING") for _ in range(3)]

    assert outcomes == [False, False, True]
    assert ledger.quarantined_uids == {"scenario-1"}


def test_consecutive_ceiling_fails_the_run_and_names_the_reason() -> None:
    ledger = GeometryAbortLedger()
    for index in range(3):
        ledger.record(f"scenario-{index}", "DECOMPOSITION_COVERAGE_SHORTFALL")

    breach = ledger.ceiling_breach()
    assert breach is not None
    assert "DECOMPOSITION_COVERAGE_SHORTFALL" in breach
    assert "forensic" in breach


def test_a_completed_episode_breaks_the_consecutive_run() -> None:
    """The consecutive ceiling must measure a *run* of failures, not a tally."""

    ledger = GeometryAbortLedger()
    ledger.record("scenario-1", "DEGENERATE_RING")
    ledger.record("scenario-2", "DEGENERATE_RING")
    ledger.note_episode_completed()
    ledger.record("scenario-3", "DEGENERATE_RING")

    assert ledger.total == 3
    assert ledger.consecutive_in_batch == 1
    assert ledger.ceiling_breach() is None


def test_rate_ceiling_fails_a_slow_bleed_the_consecutive_ceiling_would_miss() -> None:
    ledger = GeometryAbortLedger()
    for index in range(300):
        ledger.record(f"scenario-{index}", "DEGENERATE_RING")
        for _ in range(80):
            ledger.note_episode_completed()
        if ledger.ceiling_breach() is not None:
            break

    breach = ledger.ceiling_breach()
    assert breach is not None
    assert "above the ceiling" in breach


def test_rate_ceiling_stays_silent_below_its_minimum_sample() -> None:
    """A rate over three episodes is not a rate. Below the minimum sample the
    consecutive ceiling is the only guard, by construction."""

    ledger = GeometryAbortLedger()
    ledger.record("scenario-1", "DEGENERATE_RING")
    ledger.note_episode_completed()

    assert ledger.run_rate == pytest.approx(0.5)
    assert ledger.ceiling_breach() is None


# --------------------------------------------------------------------------
# `TEST-GA-006` -- the record must rebuild the failure, not merely mention it.
# --------------------------------------------------------------------------


def test_forensic_record_round_trips_the_offending_polygon(tmp_path) -> None:
    polygon = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
    target = tmp_path / "geometry_aborts.jsonl"

    append_geometry_abort_record(
        target,
        {
            "run_id": "run-1",
            "scenario_uid": "scenario-1",
            "reason_code": "DECOMPOSITION_COVERAGE_SHORTFALL",
            "diagnostics": {"residual_m2": 1.33e-4, "allowance_m2": 2.6e-3},
            "geometry_wkt": polygon.wkt,
        },
    )

    record = json.loads(target.read_text().strip())
    rebuilt = shapely.from_wkt(record["geometry_wkt"])
    # The point of keeping the WKT is that the next reader can turn the failure
    # into a fixture, which is exactly how `C9` was diagnosed.
    assert rebuilt.equals(polygon)
    assert record["diagnostics"]["residual_m2"] == pytest.approx(1.33e-4)
    assert "timestamp" in record


# --------------------------------------------------------------------------
# `TEST-GA-004` -- worker transport, under a real subprocess vector env.
# --------------------------------------------------------------------------


class _SeedWindowEnv(Env):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return np.array([0], dtype=np.float32), {}


class _GeometryAbortEnv(_SeedWindowEnv):
    def step(self, action):
        _ = action
        raise RuntimeGeometryNotEvaluableError(
            RuntimeGeometryNotEvaluableReason.DECOMPOSITION_COVERAGE_SHORTFALL,
            "typed geometry failure",
            diagnostics={"residual_m2": 1.33e-4, "allowance_m2": 2.6e-3},
            geometry_wkt="POLYGON ((0 0, 4 0, 4 3, 0 3, 0 0))",
        )


class _UnenumeratedGeometryEnv(_SeedWindowEnv):
    def step(self, action):
        _ = action
        raise ValueError("Convex decomposition produced no non-degenerate triangles")


def _make_geometry_abort_env() -> Env:
    return _GeometryAbortEnv()


def _make_unenumerated_env() -> Env:
    return _UnenumeratedGeometryEnv()


def test_worker_survives_a_geometry_abort_and_reports_it_separately() -> None:
    vec_env = DeterministicSubprocVecEnv(
        [_make_geometry_abort_env], start_method="spawn", auto_reset=False
    )
    try:
        vec_env.reset()
        result = vec_env.step_slots({0: np.zeros(1, dtype=np.float32)})[0]

        assert isinstance(result, RuntimeGeometryAbort)
        # `REQ-GA-004`: it must not arrive as a data abort.
        assert not isinstance(result, RuntimeScenarioDataAbort)
        assert result.payload["reason_code"] == "DECOMPOSITION_COVERAGE_SHORTFALL"
        assert result.payload["diagnostics"]["residual_m2"] == pytest.approx(1.33e-4)
        assert shapely.from_wkt(result.payload["geometry_wkt"]).is_valid
        assert vec_env.processes[0].is_alive()

        reset = vec_env.reset_slots([0], seeds={0: 0})
        np.testing.assert_array_equal(reset[0][0], np.array([0], dtype=np.float32))
    finally:
        vec_env.close()


def test_worker_still_dies_on_an_unenumerated_geometry_failure() -> None:
    """The complement of the test above, and the reason widening is safe."""

    vec_env = DeterministicSubprocVecEnv(
        [_make_unenumerated_env], start_method="spawn", auto_reset=False
    )
    try:
        vec_env.reset()
        with pytest.raises(SubprocessWorkerError):
            vec_env.step_slots({0: np.zeros(1, dtype=np.float32)})
    finally:
        vec_env.close()
