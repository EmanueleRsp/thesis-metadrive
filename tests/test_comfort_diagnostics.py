"""Ride-comfort and jerk evaluation diagnostics (EP-COMFORT-DIAG).

Requirement and test IDs refer to
`docs/implementation/comfort_and_jerk_diagnostics_exec_plan.md`.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.agent import Agent
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor
from thesis_rl.analysis.tables.make_comfort_tables import (
    COMFORT_METRICS,
    DIAGNOSTIC_LABEL,
    build_comfort_tables,
)
from thesis_rl.runtime.comfort_diagnostics import (
    ACCEL_SPEC,
    COMFORT_AGGREGATE_COLUMNS,
    COMFORT_EPISODE_COLUMNS,
    COMFORT_STATISTICS,
    JERK_SPEC,
    NUPLAN_COMFORT_BOUNDS,
    YAW_ACCEL_SPEC,
    YAW_RATE_SPEC,
    ComfortBounds,
    ComfortEpisodeAccumulator,
    aggregate_comfort_episodes,
    comfort_aggregate_fields,
    comfort_episode_fields,
    extract_comfort_step,
    segment_statistics,
)
from thesis_rl.rulebook.v2.wrapper import _ego_kinematics_payload
from thesis_rl.runtime.io.csv_recorder import CSVRecorder

DT = 0.1


def _step(
    *,
    t: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    heading: float = 0.0,
) -> dict[str, Any]:
    """One step-info carrying the kinematics `RulebookV2MonitorWrapper` exports."""
    return {
        "ego_kinematics": {
            "sim_time_s": t,
            "velocity_xy": (vx, vy),
            "heading_rad": heading,
        }
    }


def _series(entries: list[tuple[float, float, float]]) -> list[dict[str, Any]]:
    """Build a step sequence from `(vx, vy, heading)` samples, one `DT` apart."""
    return [
        _step(t=index * DT, vx=vx, vy=vy, heading=heading)
        for index, (vx, vy, heading) in enumerate(entries)
    ]


def _summary(steps: list[Any]) -> dict[str, Any]:
    accumulator = ComfortEpisodeAccumulator()
    for step in steps:
        accumulator.observe(step)
    return accumulator.finalize()


# --- TEST-CMF-15: the devkit parameters are pinned ---


def test_savgol_specs_match_the_nuplan_devkit() -> None:
    """These four specs are transcriptions, not choices (`DEC-CMF-006`).

    `state_extractors.py`: acceleration `poly_order=2, window_length=8`; jerk
    `deriv_order=1, poly_order=2, window_length=15`; yaw rate
    `deriv_order=1, poly_order=2`; yaw acceleration `deriv_order=2,
    poly_order=3`. The yaw window is 5 because `extract_ego_yaw_rate` never
    forwards its `window_length` to `approximate_derivatives`, whose own
    default is 5 -- reproducing the devkit's real behaviour, not its docstring.
    """

    assert (ACCEL_SPEC.window_length, ACCEL_SPEC.poly_order, ACCEL_SPEC.deriv_order) == (8, 2, 1)
    assert (JERK_SPEC.window_length, JERK_SPEC.poly_order, JERK_SPEC.deriv_order) == (15, 2, 1)
    assert (
        YAW_RATE_SPEC.window_length,
        YAW_RATE_SPEC.poly_order,
        YAW_RATE_SPEC.deriv_order,
    ) == (5, 2, 1)
    assert (
        YAW_ACCEL_SPEC.window_length,
        YAW_ACCEL_SPEC.poly_order,
        YAW_ACCEL_SPEC.deriv_order,
    ) == (5, 3, 2)


def test_bounds_match_the_nuplan_devkit_configs() -> None:
    """Transcribed from `simulation_metric/low_level/*.yaml`."""

    assert NUPLAN_COMFORT_BOUNDS == ComfortBounds(
        max_lon_accel=2.40,
        min_lon_accel=-4.05,
        max_abs_lat_accel=4.89,
        max_abs_mag_jerk=8.37,
        max_abs_lon_jerk=4.13,
        max_abs_yaw_rate=0.95,
        max_abs_yaw_accel=1.93,
    )


# --- TEST-CMF-01: closed-form kinematics ---


def test_constant_jerk_trajectory_exact_statistics() -> None:
    """A constant longitudinal jerk of 2 m/s^3: v(t) = t^2 along the heading.

    A Savitzky-Golay derivative of order `p` reproduces a polynomial of degree
    <= `p` exactly, so on this quadratic velocity the filtered acceleration is
    the analytic `2t` and the filtered jerk is exactly 2.0 -- filtering costs
    nothing on data the filter can represent, which is what makes it a fair
    closed-form check.
    """

    steps = _series([(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.04, 0.0, 0.0), (0.09, 0.0, 0.0)])
    summary = _summary(steps)

    assert summary["valid_step_count"] == 4
    assert summary["max_lon_accel"] == pytest.approx(0.6)
    assert summary["min_lon_accel"] == pytest.approx(0.0)
    assert summary["max_abs_lat_accel"] == pytest.approx(0.0, abs=1e-9)
    assert summary["max_abs_lon_jerk"] == pytest.approx(2.0)
    assert summary["max_abs_mag_jerk"] == pytest.approx(2.0)
    assert summary["max_abs_yaw_rate"] == pytest.approx(0.0, abs=1e-9)
    assert summary["max_abs_yaw_accel"] == pytest.approx(0.0, abs=1e-9)
    assert summary["is_comfortable"] is True


def test_acceleration_is_projected_onto_the_ego_heading() -> None:
    """A vehicle heading along +y sees a +y velocity change as longitudinal,
    not lateral: the projection must use the reported heading, not the axes."""

    heading = math.pi / 2.0
    steps = _series([(0.0, value, heading) for value in (0.0, 0.3, 0.6, 0.9)])
    summary = _summary(steps)

    assert summary["max_lon_accel"] == pytest.approx(3.0)
    assert summary["max_abs_lat_accel"] == pytest.approx(0.0, abs=1e-9)


def test_lateral_acceleration_uses_magnitude() -> None:
    """A left turn and a right turn of equal severity score the same."""

    left = _summary(_series([(0.0, value, 0.0) for value in (0.0, 0.3, 0.6, 0.9)]))
    right = _summary(_series([(0.0, -value, 0.0) for value in (0.0, 0.3, 0.6, 0.9)]))

    assert left["max_abs_lat_accel"] == pytest.approx(3.0)
    assert right["max_abs_lat_accel"] == pytest.approx(3.0)


def test_magnitude_jerk_differentiates_the_acceleration_magnitude() -> None:
    """nuPlan's `max_abs_mag_jerk` uses `acceleration_coordinate='magnitude'`,
    so it is the derivative of `|a|`, not the norm of the jerk vector. On a
    purely lateral jerk the two coincide in size, which is what pins the
    channel to a number here."""

    # a_y = 9t  =>  v_y = 4.5 t^2 ; a_x = 0, so |a| = 9t and d|a|/dt = 9.
    steps = _series([(0.0, 4.5 * (index * DT) ** 2, 0.0) for index in range(4)])
    summary = _summary(steps)

    assert summary["max_abs_mag_jerk"] == pytest.approx(9.0)
    assert summary["max_abs_lon_jerk"] == pytest.approx(0.0, abs=1e-9)


# --- TEST-CMF-02: every bound flips the verdict on its own ---


def _compliant_steps() -> list[dict[str, Any]]:
    """Steady 1 m/s^2 forward acceleration: inside every nuPlan bound."""
    return _series([(value, 0.0, 0.0) for value in (0.0, 0.1, 0.2, 0.3)])


def test_compliant_trajectory_is_comfortable() -> None:
    summary = _summary(_compliant_steps())

    assert summary["max_lon_accel"] == pytest.approx(1.0)
    assert summary["is_comfortable"] is True


@pytest.mark.parametrize(
    ("channel", "entries"),
    [
        # Steady 3 m/s^2, above nuPlan's 2.40 upper longitudinal bound.
        ("max_lon_accel", [(0.3 * i, 0.0, 0.0) for i in range(4)]),
        # Steady -5 m/s^2, below the -4.05 lower bound (hard braking).
        ("min_lon_accel", [(-0.5 * i, 0.0, 0.0) for i in range(4)]),
        # Steady 5 m/s^2 sideways at heading 0, above the 4.89 lateral bound.
        ("max_abs_lat_accel", [(0.0, 0.5 * i, 0.0) for i in range(4)]),
        # a = 5t  =>  v = 2.5 t^2: a 5 m/s^3 longitudinal jerk, above the 4.13
        # bound, while the acceleration peaks at 1.5 and the magnitude jerk is
        # the same 5.0, inside its wider 8.37 bound.
        ("max_abs_lon_jerk", [(2.5 * (i * DT) ** 2, 0.0, 0.0) for i in range(4)]),
        # The same construction on the lateral axis: 9 m/s^3 magnitude jerk
        # with zero longitudinal jerk.
        ("max_abs_mag_jerk", [(0.0, 4.5 * (i * DT) ** 2, 0.0) for i in range(4)]),
        # 0.2 rad per 0.1 s = 2 rad/s, above the 0.95 rad/s bound.
        ("max_abs_yaw_rate", [(0.0, 0.0, 0.2 * i) for i in range(4)]),
        # heading = 1.5 t^2  =>  yaw rate 3t (peaking at 0.9, inside its own
        # bound) and yaw acceleration 3.0, above the 1.93 bound.
        ("max_abs_yaw_accel", [(0.0, 0.0, 1.5 * (i * DT) ** 2) for i in range(4)]),
    ],
)
def test_each_bound_flips_is_comfortable(
    channel: str, entries: list[tuple[float, float, float]]
) -> None:
    summary = _summary(_series(entries))

    assert summary["is_comfortable"] is False
    value = summary[channel]
    bound = getattr(NUPLAN_COMFORT_BOUNDS, channel)
    if channel == "min_lon_accel":
        assert value < bound
    else:
        assert value > bound

    # Only the targeted channel is out of bounds: every other one complies.
    for other in COMFORT_STATISTICS:
        if other == channel or summary[other] is None:
            continue
        other_bound = getattr(NUPLAN_COMFORT_BOUNDS, other)
        if other == "min_lon_accel":
            assert summary[other] >= other_bound, other
        else:
            assert summary[other] <= other_bound, other


# --- TEST-CMF-03: yaw unwrapping ---


def test_yaw_rate_unwraps_across_pi() -> None:
    """Crossing +-pi must not fabricate a yaw rate of order 2*pi/delta.

    A steady rotation carried across the branch cut: unwrapped, the second
    step's heading difference is -6.20 rad and would score a yaw rate of 62
    rad/s, far outside every bound. `np.unwrap` -- the devkit's `phase_unwrap`
    -- turns the sequence back into the linear ramp it physically is.
    """

    increment = math.remainder(-3.10 - 3.10, 2.0 * math.pi)
    headings = [3.10 + index * increment for index in range(4)]
    wrapped = [math.remainder(heading, 2.0 * math.pi) for heading in headings]
    summary = _summary(_series([(0.0, 0.0, heading) for heading in wrapped]))

    assert summary["max_abs_yaw_rate"] == pytest.approx(abs(increment) / DT)
    assert summary["max_abs_yaw_rate"] < NUPLAN_COMFORT_BOUNDS.max_abs_yaw_rate
    assert summary["max_abs_yaw_accel"] == pytest.approx(0.0, abs=1e-9)
    assert summary["is_comfortable"] is True


# --- TEST-CMF-04: gaps split the series, no window spans one ---


def test_no_filter_window_spans_a_gap() -> None:
    """A step without usable kinematics splits the series; neither one-sample
    fragment can carry a derivative, so nothing is invented across the jump."""

    steps = [_step(t=0.0, vx=0.0), {"ego_kinematics": None}, _step(t=0.2, vx=10.0)]
    summary = _summary(steps)

    assert summary["valid_step_count"] == 2
    assert all(summary[name] is None for name in COMFORT_STATISTICS)
    assert summary["is_comfortable"] is None


def test_a_gap_costs_only_the_segment_it_breaks() -> None:
    """Steps after a gap resume measuring, they are not discarded."""

    steps = [_step(t=0.0, vx=0.0), {}] + [
        _step(t=0.2 + index * DT, vx=0.3 * index) for index in range(4)
    ]
    summary = _summary(steps)

    assert summary["valid_step_count"] == 5
    # The post-gap segment of four steps is a steady 3 m/s^2.
    assert summary["max_lon_accel"] == pytest.approx(3.0)
    assert summary["is_comfortable"] is False


def test_the_episode_statistic_is_the_extremum_across_segments() -> None:
    """Two gap-separated segments: the harsher one must win."""

    gentle = [_step(t=index * DT, vx=0.1 * index) for index in range(4)]
    harsh = [_step(t=1.0 + index * DT, vx=0.3 * index) for index in range(4)]
    summary = _summary([*gentle, {}, *harsh])

    assert summary["max_lon_accel"] == pytest.approx(3.0)
    assert summary["is_comfortable"] is False


def test_simulation_time_that_does_not_advance_splits_the_segment() -> None:
    """A repeated timestamp gives the filter no spacing to work with."""

    steps = [_step(t=0.0, vx=0.0), _step(t=0.0, vx=1.0), _step(t=0.0, vx=2.0)]
    summary = _summary(steps)

    assert summary["valid_step_count"] == 3
    assert all(summary[name] is None for name in COMFORT_STATISTICS)
    assert summary["is_comfortable"] is None


# --- TEST-CMF-05: malformed input is absent, not fatal ---


@pytest.mark.parametrize(
    "step_info",
    [
        None,
        {},
        {"ego_kinematics": {}},
        {"ego_kinematics": None},
        {"ego_kinematics": {"sim_time_s": 0.0, "heading_rad": 0.0}},
        {"ego_kinematics": {"sim_time_s": 0.0, "velocity_xy": (1.0,), "heading_rad": 0.0}},
        {"ego_kinematics": {"sim_time_s": 0.0, "velocity_xy": "xy", "heading_rad": 0.0}},
        {"ego_kinematics": {"sim_time_s": None, "velocity_xy": (0.0, 0.0), "heading_rad": 0.0}},
        {"ego_kinematics": {"sim_time_s": 0.0, "velocity_xy": (0.0, 0.0), "heading_rad": None}},
        _step(vx=float("nan")),
        _step(vx=float("inf")),
        _step(heading=float("nan")),
    ],
)
def test_malformed_steps_are_treated_as_absent(step_info: Any) -> None:
    assert extract_comfort_step(step_info) is None

    summary = _summary([step_info])
    assert summary["valid_step_count"] == 0
    assert summary["is_comfortable"] is None
    assert all(summary[name] is None for name in COMFORT_STATISTICS)


def test_observe_does_not_mutate_the_step_info() -> None:
    """TEST-CMF-13: the diagnostic reads evaluation state, never writes it."""

    step_info = _step(vx=1.0, heading=0.2)
    before = repr(step_info)

    accumulator = ComfortEpisodeAccumulator()
    accumulator.observe(step_info)
    accumulator.finalize()

    assert repr(step_info) == before


def test_module_does_not_depend_on_reward_or_rulebook() -> None:
    """`REQ-CMF-01`: comfort is excluded from the rulebook and the reward, so
    the diagnostic must not import either."""

    source = Path("src/thesis_rl/runtime/comfort_diagnostics.py").read_text(encoding="utf-8")
    import_lines = [line for line in source.splitlines() if line.startswith(("import ", "from "))]

    assert not any("thesis_rl.reward" in line for line in import_lines), import_lines
    assert not any("thesis_rl.rulebook" in line for line in import_lines), import_lines


# --- TEST-CMF-14: the producer/consumer contract (regression, BUG-CMF-002) ---


class _FakeEgo:
    def __init__(self) -> None:
        self.velocity_xy = (1.5, -0.5)
        self.heading_rad = 0.25


class _FakeSnapshot:
    def __init__(self) -> None:
        self.sim_time_s = 3.4
        self.ego = _FakeEgo()


def test_monitor_payload_is_exactly_what_the_extractor_consumes() -> None:
    """Regression: the diagnostic was first built against a key
    (`info["ego_state"]`) that the live evaluation stack never publishes, so
    every comfort column came out empty while every unit test passed. Producer
    and consumer are now pinned to each other in one test."""

    payload = _ego_kinematics_payload(_FakeSnapshot())
    assert payload is not None

    step = extract_comfort_step({"ego_kinematics": payload})
    assert step is not None
    assert step.sim_time_s == pytest.approx(3.4)
    assert step.velocity_x == pytest.approx(1.5)
    assert step.velocity_y == pytest.approx(-0.5)
    assert step.heading_rad == pytest.approx(0.25)


@pytest.mark.parametrize("snapshot", [0, None, object(), _FakeEgo()])
def test_monitor_payload_tolerates_a_snapshot_without_ego_state(snapshot: Any) -> None:
    """The snapshotter is injected, so a caller may supply a stand-in. A
    diagnostic must never be able to fail a rulebook step."""

    assert _ego_kinematics_payload(snapshot) is None


# --- TEST-CMF-06: short episodes leave channels undefined ---


@pytest.mark.parametrize("length", [1, 2])
def test_episodes_too_short_to_filter_define_nothing(length: int) -> None:
    """`approximate_derivatives` clamps its window to the series length and
    then needs `poly_order < window_length`, so two samples cannot carry even
    the order-2 channels."""

    summary = _summary(_series([(0.1 * index, 0.0, 0.0) for index in range(length)]))

    assert summary["valid_step_count"] == length
    assert all(summary[name] is None for name in COMFORT_STATISTICS)
    assert summary["is_comfortable"] is None


def test_three_steps_define_everything_but_the_yaw_acceleration() -> None:
    """Yaw acceleration is the only order-3 channel, so it alone needs a
    fourth sample -- and without it there is no verdict."""

    summary = _summary(_series([(0.1 * index, 0.0, 0.0) for index in range(3)]))

    assert summary["max_lon_accel"] == pytest.approx(1.0)
    assert summary["max_abs_lon_jerk"] is not None
    assert summary["max_abs_yaw_rate"] is not None
    assert summary["max_abs_yaw_accel"] is None
    # Undefined is not "comfortable" (`REQ-CMF-07`).
    assert summary["is_comfortable"] is None


def test_four_valid_steps_are_enough_for_a_verdict() -> None:
    summary = _summary(_compliant_steps())

    assert all(summary[name] is not None for name in COMFORT_STATISTICS)
    assert summary["is_comfortable"] is True


def test_segment_statistics_is_reusable_on_a_bare_segment() -> None:
    """The numerical core is public so it can be checked without an
    accumulator, and it returns all-`None` rather than raising on a stub."""

    segment = [extract_comfort_step(step) for step in _compliant_steps()]
    stats = segment_statistics([step for step in segment if step is not None])

    assert stats["max_lon_accel"] == pytest.approx(1.0)
    assert all(value is None for value in segment_statistics([]).values())


# --- TEST-CMF-07: aggregation excludes and counts undefined episodes ---


def test_undefined_episodes_excluded_and_counted() -> None:
    comfortable = _summary(_compliant_steps())
    uncomfortable = _summary(_series([(0.3 * index, 0.0, 0.0) for index in range(4)]))
    undefined = _summary([_step(vx=0.0)])

    metrics = aggregate_comfort_episodes([comfortable, uncomfortable, undefined])

    assert metrics["comfort_rate"] == pytest.approx(0.5)
    assert metrics["comfort_episode_count"] == 2
    assert metrics["comfort_excluded_episode_count"] == 1
    # The mean is over the episodes where the channel is defined: the
    # single-sample episode measures nothing and contributes to neither.
    assert metrics["mean_comfort_max_lon_accel"] == pytest.approx((1.0 + 3.0) / 2.0)
    assert metrics["mean_comfort_max_abs_yaw_accel"] == pytest.approx(0.0)


def test_aggregate_with_no_defined_episode_reports_none_not_zero() -> None:
    metrics = aggregate_comfort_episodes([_summary([_step(vx=0.0)])])

    assert metrics["comfort_rate"] is None
    assert metrics["comfort_episode_count"] == 0
    assert metrics["comfort_excluded_episode_count"] == 1
    assert metrics["mean_comfort_max_abs_lon_jerk"] is None


def test_undefined_channel_wins_over_a_breached_one() -> None:
    """Regression: `ComfortBounds.satisfied_by` short-circuited on the first
    breached bound, so an episode that also had an *undefined* channel was
    reported as uncomfortable instead of unmeasured. Definedness must be
    settled across every channel before any bound is tested."""

    # Three steps: enough for the order-2 channels, one short of the order-3
    # yaw acceleration. The longitudinal bound is breached and the verdict is
    # still "not measured", not "uncomfortable".
    summary = _summary(_series([(9.9 * index, 0.0, 0.0) for index in range(3)]))

    assert summary["max_lon_accel"] > NUPLAN_COMFORT_BOUNDS.max_lon_accel
    assert summary["max_abs_yaw_accel"] is None
    assert summary["is_comfortable"] is None


def test_aggregate_of_no_episodes_is_empty_not_an_error() -> None:
    metrics = aggregate_comfort_episodes([])

    assert metrics["comfort_rate"] is None
    assert metrics["comfort_episode_count"] == 0
    assert metrics["comfort_excluded_episode_count"] == 0


# --- TEST-CMF-08: CSV schemas ---


def test_eval_episodes_schema_declares_comfort_columns() -> None:
    schema = CSVRecorder.SCHEMAS["eval_episodes.csv"]

    for column in COMFORT_EPISODE_COLUMNS:
        assert column in schema, column
    assert len(schema) == len(set(schema))


@pytest.mark.parametrize("filename", ["evals.csv", "final_eval.csv"])
def test_eval_and_final_schemas_declare_comfort_columns(filename: str) -> None:
    schema = CSVRecorder.SCHEMAS[filename]

    for column in COMFORT_AGGREGATE_COLUMNS:
        assert column in schema, column
    assert len(schema) == len(set(schema))


def test_recorder_writes_comfort_columns(tmp_path: Path) -> None:
    recorder = CSVRecorder(tmp_path)
    summary = _summary(_compliant_steps())
    recorder.append_row(
        "eval_episodes.csv",
        {"episode_id": 1, **comfort_episode_fields({"comfort": [summary]}, 0)},
    )

    rows = list(csv.DictReader((tmp_path / "eval_episodes.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["comfort_is_comfortable"] == "1.0"
    assert float(rows[0]["comfort_max_lon_accel"]) == pytest.approx(1.0)


# --- TEST-CMF-09: row helpers ---


def test_comfort_episode_fields_indexes_per_episode_vector() -> None:
    first = _summary(_compliant_steps())
    second = _summary(_series([(0.3 * index, 0.0, 0.0) for index in range(4)]))
    per_episode = {"comfort": [first, second]}

    assert comfort_episode_fields(per_episode, 0)["comfort_is_comfortable"] == 1.0
    assert comfort_episode_fields(per_episode, 1)["comfort_is_comfortable"] == 0.0
    assert comfort_episode_fields(per_episode, 1)["comfort_max_lon_accel"] == pytest.approx(3.0)


@pytest.mark.parametrize("per_episode", [{}, None, {"comfort": []}, {"comfort": "nonsense"}])
def test_comfort_episode_fields_tolerates_absent_data(per_episode: Any) -> None:
    fields = comfort_episode_fields(per_episode, 0)

    assert set(fields) == set(COMFORT_EPISODE_COLUMNS)
    assert all(value is None for value in fields.values())


def test_comfort_episode_fields_out_of_range_index_is_empty() -> None:
    fields = comfort_episode_fields({"comfort": [_summary(_compliant_steps())]}, 7)

    assert all(value is None for value in fields.values())


def test_comfort_aggregate_fields_projects_metrics() -> None:
    metrics = aggregate_comfort_episodes([_summary(_compliant_steps())])
    fields = comfort_aggregate_fields(metrics)

    assert set(fields) == set(COMFORT_AGGREGATE_COLUMNS)
    assert fields["comfort_rate"] == pytest.approx(1.0)
    assert comfort_aggregate_fields(None)["comfort_rate"] is None


# --- TEST-CMF-10: serial and parallel evaluation agree ---


class _ComfortEnv:
    """Two fixed 4-step episodes with distinct, deterministic kinematics.

    Four samples because the order-3 yaw-acceleration channel needs a fourth
    one before the episode has a verdict at all.
    """

    # (vx, heading) samples, one DT apart. Episode 0 accelerates gently at
    # 1 m/s^2 and turns steadily; episode 1 accelerates at 3 m/s^2, past the
    # 2.40 bound.
    EPISODES: tuple[tuple[tuple[float, float], ...], ...] = (
        ((0.0, 0.00), (0.1, 0.01), (0.2, 0.02), (0.3, 0.03)),
        ((0.0, 0.00), (0.3, 0.00), (0.6, 0.00), (0.9, 0.00)),
    )

    def __init__(self, *, fixed_pattern: tuple[tuple[float, float], ...] | None = None) -> None:
        self._fixed_pattern = fixed_pattern
        self._episode_idx = -1
        self._step = 0

    def reset(self, **kwargs):
        _ = kwargs
        self._episode_idx += 1
        self._step = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        pattern = (
            self._fixed_pattern
            if self._fixed_pattern is not None
            else self.EPISODES[self._episode_idx]
        )
        vx, heading = pattern[self._step]
        info = _step(t=self._step * DT, vx=vx, heading=heading)
        self._step += 1
        done = self._step >= len(pattern)
        if done:
            info.update({"arrive_dest": True, "termination_reason": "success"})
        obs = np.array([self._step, self._step], dtype=np.float32)
        return obs, 0.1, done, False, info


class _Planner:
    def predict(self, observation, deterministic: bool = False):
        _ = (observation, deterministic)
        return np.array([0.5, -0.5], dtype=np.float32), None


def _build_agent() -> Agent:
    return Agent(
        preprocessor=IdentityPreprocessor(),
        planner=_Planner(),
        adapter=IdentityAdapter(low=-1.0, high=1.0, expected_shape=(2,)),
    )


class _Slot:
    def __init__(self, env: _ComfortEnv) -> None:
        self.env = env

    def set_rendered_frame(self, frame) -> None:
        _ = frame


class _Vector:
    num_envs = 2

    def __init__(self) -> None:
        self.envs = [_ComfortEnv(fixed_pattern=pattern) for pattern in _ComfortEnv.EPISODES]

    def get_slot_proxy(self, slot: int) -> _Slot:
        return _Slot(self.envs[int(slot)])

    def reset_slots(self, slots, *, seeds=None, options=None):
        _ = (seeds, options)
        return {int(slot): self.envs[int(slot)].reset() for slot in slots}

    def step_slots(self, actions):
        results = {}
        for slot, action in actions.items():
            observation, reward, done, truncated, info = self.envs[int(slot)].step(action)
            info = dict(info)
            info["terminated"] = bool(done)
            info["truncated"] = bool(truncated)
            results[int(slot)] = (observation, reward, done or truncated, info, {})
        return results

    def render_slots(self, render_kwargs=None, *, slots=None):
        _ = render_kwargs
        selected = tuple(sorted(slots or range(self.num_envs)))
        return {slot: np.zeros((2, 2, 3), dtype=np.uint8) for slot in selected}


def _comfort_metrics(metrics: dict) -> dict:
    return {column: metrics[column] for column in COMFORT_AGGREGATE_COLUMNS}


def test_serial_and_parallel_paths_agree_on_identical_episodes() -> None:
    sequential = _build_agent().evaluate(
        _ComfortEnv(), n_eval_episodes=2, deterministic=True, show_progress=False
    )
    parallel = _build_agent().evaluate(
        _Vector(), n_eval_episodes=2, deterministic=True, show_progress=False
    )

    assert _comfort_metrics(parallel) == _comfort_metrics(sequential)
    # One comfortable episode and one breaching the longitudinal bound.
    assert sequential["comfort_rate"] == pytest.approx(0.5)
    assert sequential["comfort_episode_count"] == 2


def test_evaluate_exposes_per_episode_comfort_summaries() -> None:
    metrics = _build_agent().evaluate(
        _ComfortEnv(),
        n_eval_episodes=2,
        deterministic=True,
        show_progress=False,
        return_episode_metrics=True,
    )

    summaries = metrics["per_episode"]["comfort"]
    assert [summary["is_comfortable"] for summary in summaries] == [True, False]
    assert comfort_episode_fields(metrics["per_episode"], 1)["comfort_is_comfortable"] == 0.0


# --- TEST-CMF-11 / TEST-CMF-12: analysis table ---


def _write_final_eval(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _final_eval_row(condition_id: str, seed: int, comfort_rate: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "condition_id": condition_id,
        "algorithm": "ppo",
        "reward_type": "rulebook",
        "reward_behavior": "scalarized",
        "curriculum_name": "none",
        "rulebook_config": "v5.1",
        "seed": seed,
        "comfort_rate": comfort_rate,
        "comfort_episode_count": 10,
        "comfort_excluded_episode_count": 0,
    }
    for name in COMFORT_STATISTICS:
        row[f"mean_comfort_{name}"] = 1.0
    return row


def test_build_comfort_tables_groups_by_condition(tmp_path: Path) -> None:
    aggregated = tmp_path / "aggregated"
    tables = tmp_path / "tables"
    _write_final_eval(
        aggregated / "final_eval_all_runs.csv",
        [
            _final_eval_row("cond_a", 0, 0.8),
            _final_eval_row("cond_a", 1, 0.6),
            _final_eval_row("cond_b", 0, 0.2),
        ],
    )

    build_comfort_tables(aggregated_dir=aggregated, tables_dir=tables)

    rows = {
        row["condition_id"]: row
        for row in csv.DictReader((tables / "comfort_diagnostics.csv").open(encoding="utf-8"))
    }
    assert set(rows) == {"cond_a", "cond_b"}
    assert float(rows["cond_a"]["comfort_rate_mean"]) == pytest.approx(0.7)
    assert float(rows["cond_a"]["comfort_rate_sd"]) == pytest.approx(0.1414213562, rel=1e-6)
    assert rows["cond_a"]["comfort_rate_seed_values"] == "0.8;0.6"
    assert rows["cond_a"]["n_seeds"] == "2"
    assert float(rows["cond_b"]["comfort_rate_mean"]) == pytest.approx(0.2)

    markdown = (tables / "comfort_diagnostics.md").read_text(encoding="utf-8")
    assert DIAGNOSTIC_LABEL in markdown
    assert "cond_a" in markdown


def test_build_comfort_tables_optional_ci(tmp_path: Path) -> None:
    aggregated = tmp_path / "aggregated"
    tables = tmp_path / "tables"
    _write_final_eval(
        aggregated / "final_eval_all_runs.csv",
        [_final_eval_row("cond_a", 0, 0.8), _final_eval_row("cond_a", 1, 0.6)],
    )

    build_comfort_tables(aggregated_dir=aggregated, tables_dir=tables, include_ci=True)

    header = next(csv.reader((tables / "comfort_diagnostics.csv").open(encoding="utf-8")))
    assert "comfort_rate_ci95" in header
    for metric in COMFORT_METRICS:
        assert f"{metric}_ci95" in header


def test_build_comfort_tables_tolerates_legacy_runs(tmp_path: Path) -> None:
    """Runs recorded before this feature carry no comfort column at all."""

    aggregated = tmp_path / "aggregated"
    tables = tmp_path / "tables"
    _write_final_eval(
        aggregated / "final_eval_all_runs.csv",
        [
            {
                "condition_id": "cond_legacy",
                "algorithm": "ppo",
                "reward_type": "rulebook",
                "reward_behavior": "scalarized",
                "curriculum_name": "none",
                "rulebook_config": "v5.1",
                "seed": 0,
                "success_rate": 0.5,
            }
        ],
    )

    build_comfort_tables(aggregated_dir=aggregated, tables_dir=tables)

    rows = list(csv.DictReader((tables / "comfort_diagnostics.csv").open(encoding="utf-8")))
    assert rows == []
    assert DIAGNOSTIC_LABEL in (tables / "comfort_diagnostics.md").read_text(encoding="utf-8")


def test_build_comfort_tables_without_any_aggregated_file(tmp_path: Path) -> None:
    tables = tmp_path / "tables"

    build_comfort_tables(aggregated_dir=tmp_path / "missing", tables_dir=tables)

    assert (tables / "comfort_diagnostics.csv").exists()
    assert (tables / "comfort_diagnostics.md").exists()
