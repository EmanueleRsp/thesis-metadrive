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
    COMFORT_AGGREGATE_COLUMNS,
    COMFORT_EPISODE_COLUMNS,
    COMFORT_STATISTICS,
    NUPLAN_COMFORT_BOUNDS,
    ComfortEpisodeAccumulator,
    aggregate_comfort_episodes,
    comfort_aggregate_fields,
    comfort_episode_fields,
    extract_comfort_step,
)
from thesis_rl.runtime.io.csv_recorder import CSVRecorder

DT = 0.1


def _step(
    *,
    a_lon: float = 0.0,
    a_lat: float = 0.0,
    yaw: float = 0.0,
    dt: float = DT,
    a_x: float | None = None,
    a_y: float | None = None,
) -> dict[str, Any]:
    """One step-info carrying the ego kinematics `RuleRewardWrapper` exports."""
    return {
        "ego_state": {
            "acceleration": {
                "longitudinal": a_lon,
                "lateral": a_lat,
                "x": a_lon if a_x is None else a_x,
                "y": a_lat if a_y is None else a_y,
            },
            "yaw": yaw,
            "dt": dt,
        }
    }


def _summary(steps: list[Any]) -> dict[str, Any]:
    accumulator = ComfortEpisodeAccumulator()
    for step in steps:
        accumulator.observe(step)
    return accumulator.finalize()


# --- TEST-CMF-01: closed-form kinematics ---


def test_constant_jerk_trajectory_exact_statistics() -> None:
    """A constant longitudinal jerk of 2 m/s^3 at dt=0.1 s, heading fixed."""

    steps = [_step(a_lon=value) for value in (0.0, 0.2, 0.4)]
    summary = _summary(steps)

    assert summary["valid_step_count"] == 3
    assert summary["max_lon_accel"] == pytest.approx(0.4)
    assert summary["min_lon_accel"] == pytest.approx(0.0)
    assert summary["max_abs_lat_accel"] == pytest.approx(0.0)
    assert summary["max_abs_lon_jerk"] == pytest.approx(2.0)
    assert summary["max_abs_mag_jerk"] == pytest.approx(2.0)
    assert summary["max_abs_yaw_rate"] == pytest.approx(0.0)
    assert summary["max_abs_yaw_accel"] == pytest.approx(0.0)
    assert summary["is_comfortable"] is True


def test_lateral_acceleration_uses_the_ego_frame_channel() -> None:
    """`max_abs_lat_accel` tracks magnitude, so a left turn and a right turn
    of equal severity score the same."""

    left = _summary([_step(a_lat=3.0), _step(a_lat=3.0), _step(a_lat=3.0)])
    right = _summary([_step(a_lat=-3.0), _step(a_lat=-3.0), _step(a_lat=-3.0)])

    assert left["max_abs_lat_accel"] == pytest.approx(3.0)
    assert right["max_abs_lat_accel"] == pytest.approx(3.0)


# --- TEST-CMF-02: every bound flips the verdict on its own ---


def _compliant_steps() -> list[dict[str, Any]]:
    return [_step(a_lon=0.1), _step(a_lon=0.1), _step(a_lon=0.1)]


def test_compliant_trajectory_is_comfortable() -> None:
    assert _summary(_compliant_steps())["is_comfortable"] is True


@pytest.mark.parametrize(
    ("channel", "steps"),
    [
        # Above nuPlan's 2.40 m/s^2 upper longitudinal bound, nothing else.
        ("max_lon_accel", [_step(a_lon=3.0)] * 3),
        # Below the -4.05 m/s^2 lower longitudinal bound (hard braking).
        ("min_lon_accel", [_step(a_lon=-5.0)] * 3),
        # Above the 4.89 m/s^2 lateral bound.
        ("max_abs_lat_accel", [_step(a_lat=5.0)] * 3),
        # 0.5 m/s^2 step over 0.1 s = 5 m/s^3 > the 4.13 longitudinal-jerk
        # bound, while the magnitude jerk stays inside its wider 8.37 bound.
        ("max_abs_lon_jerk", [_step(a_lon=0.0), _step(a_lon=0.5), _step(a_lon=0.0)]),
        # Jerk placed on the lateral axis only: 10 m/s^3 magnitude, zero
        # longitudinal jerk, so only the magnitude bound is breached.
        (
            "max_abs_mag_jerk",
            [
                _step(a_lon=0.0, a_lat=0.0),
                _step(a_lon=0.0, a_lat=1.0),
                _step(a_lon=0.0, a_lat=0.0),
            ],
        ),
        # 0.2 rad over 0.1 s = 2 rad/s > the 0.95 rad/s bound.
        ("max_abs_yaw_rate", [_step(yaw=0.0), _step(yaw=0.2), _step(yaw=0.4)]),
        # Yaw rate 0 then 0.9 rad/s (inside its own bound) in one step gives
        # 9 rad/s^2, above the 1.93 rad/s^2 bound.
        ("max_abs_yaw_accel", [_step(yaw=0.0), _step(yaw=0.0), _step(yaw=0.09)]),
    ],
)
def test_each_bound_flips_is_comfortable(channel: str, steps: list[dict[str, Any]]) -> None:
    summary = _summary(steps)

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


# --- TEST-CMF-03: yaw wrapping ---


def test_yaw_rate_wraps_across_pi() -> None:
    """Crossing +-pi must not fabricate a yaw rate of order 2*pi/dt.

    A steady rotation carried across the branch cut: unwrapped, the second
    step's heading difference is -6.20 rad and would score a yaw rate of 62
    rad/s, far outside every bound. Wrapped, it is the same small increment as
    every other step, so the turn reads as the comfortable manoeuvre it is.
    """

    increment = math.remainder(-3.10 - 3.10, 2.0 * math.pi)
    steps = [_step(yaw=3.10), _step(yaw=-3.10), _step(yaw=-3.10 + increment)]
    summary = _summary(steps)

    assert summary["max_abs_yaw_rate"] == pytest.approx(abs(increment) / DT)
    assert summary["max_abs_yaw_rate"] < NUPLAN_COMFORT_BOUNDS.max_abs_yaw_rate
    assert summary["max_abs_yaw_accel"] == pytest.approx(0.0)
    assert summary["is_comfortable"] is True


# --- TEST-CMF-04: gaps break the derivative chain ---


def test_no_derivative_is_taken_across_a_gap() -> None:
    """A step without usable kinematics must not become a differentiation
    interval; otherwise the jump across the gap invents an enormous jerk."""

    steps = [_step(a_lon=0.0), {"ego_state": None}, _step(a_lon=10.0)]
    summary = _summary(steps)

    assert summary["valid_step_count"] == 2
    assert summary["max_abs_lon_jerk"] is None
    assert summary["max_abs_mag_jerk"] is None
    assert summary["max_lon_accel"] == pytest.approx(10.0)
    assert summary["is_comfortable"] is None


def test_gap_also_resets_the_yaw_rate_chain() -> None:
    """The yaw acceleration needs two consecutive yaw rates, which a gap
    invalidates just as it does the jerk."""

    steps = [_step(yaw=0.0), _step(yaw=0.05), {}, _step(yaw=0.5), _step(yaw=1.0)]
    summary = _summary(steps)

    # The surviving chain is the final pair: one yaw rate, no yaw acceleration.
    assert summary["max_abs_yaw_rate"] == pytest.approx(5.0)
    assert summary["max_abs_yaw_accel"] is None
    assert summary["is_comfortable"] is None


# --- TEST-CMF-05: malformed input is absent, not fatal ---


@pytest.mark.parametrize(
    "step_info",
    [
        None,
        {},
        {"ego_state": {}},
        {"ego_state": {"acceleration": None, "yaw": 0.0, "dt": DT}},
        {"ego_state": {"acceleration": {"longitudinal": "x"}, "yaw": 0.0, "dt": DT}},
        {"ego_state": {"acceleration": {"longitudinal": 1.0, "lateral": 1.0, "x": 1.0, "y": 1.0}}},
        _step(dt=0.0),
        _step(dt=-0.1),
        _step(a_lon=float("nan")),
        _step(a_lon=float("inf")),
        _step(yaw=float("nan")),
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

    step_info = _step(a_lon=1.0, yaw=0.2)
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


# --- TEST-CMF-06: short episodes leave channels undefined ---


def test_single_step_episode_defines_only_the_acceleration_channels() -> None:
    summary = _summary([_step(a_lon=1.0, a_lat=2.0)])

    assert summary["valid_step_count"] == 1
    assert summary["max_lon_accel"] == pytest.approx(1.0)
    assert summary["min_lon_accel"] == pytest.approx(1.0)
    assert summary["max_abs_lat_accel"] == pytest.approx(2.0)
    assert summary["max_abs_lon_jerk"] is None
    assert summary["max_abs_mag_jerk"] is None
    assert summary["max_abs_yaw_rate"] is None
    assert summary["max_abs_yaw_accel"] is None
    assert summary["is_comfortable"] is None


def test_two_step_episode_has_no_yaw_acceleration_and_no_verdict() -> None:
    summary = _summary([_step(a_lon=0.0), _step(a_lon=0.1)])

    assert summary["max_abs_lon_jerk"] == pytest.approx(1.0)
    assert summary["max_abs_yaw_rate"] == pytest.approx(0.0)
    assert summary["max_abs_yaw_accel"] is None
    # Undefined is not "comfortable" (`REQ-CMF-07`).
    assert summary["is_comfortable"] is None


# --- TEST-CMF-07: aggregation excludes and counts undefined episodes ---


def test_undefined_episodes_excluded_and_counted() -> None:
    comfortable = _summary(_compliant_steps())
    uncomfortable = _summary([_step(a_lon=3.0)] * 3)
    undefined = _summary([_step(a_lon=0.0)])

    metrics = aggregate_comfort_episodes([comfortable, uncomfortable, undefined])

    assert metrics["comfort_rate"] == pytest.approx(0.5)
    assert metrics["comfort_episode_count"] == 2
    assert metrics["comfort_excluded_episode_count"] == 1
    # The undefined episode still contributes its defined channels.
    assert metrics["mean_comfort_max_lon_accel"] == pytest.approx((0.1 + 3.0 + 0.0) / 3.0)
    # ... and none of its undefined ones.
    assert metrics["mean_comfort_max_abs_yaw_accel"] == pytest.approx(0.0)


def test_aggregate_with_no_defined_episode_reports_none_not_zero() -> None:
    metrics = aggregate_comfort_episodes([_summary([_step(a_lon=0.0)])])

    assert metrics["comfort_rate"] is None
    assert metrics["comfort_episode_count"] == 0
    assert metrics["comfort_excluded_episode_count"] == 1
    assert metrics["mean_comfort_max_abs_lon_jerk"] is None


def test_undefined_channel_wins_over_a_breached_one() -> None:
    """Regression: `ComfortBounds.satisfied_by` short-circuited on the first
    breached bound, so an episode that also had an *undefined* channel was
    reported as uncomfortable instead of unmeasured. Definedness must be
    settled across every channel before any bound is tested."""

    # One step: the longitudinal bound is breached, every derivative channel
    # is undefined. The verdict is "not measured", not "uncomfortable".
    summary = _summary([_step(a_lon=99.0)])

    assert summary["max_lon_accel"] > NUPLAN_COMFORT_BOUNDS.max_lon_accel
    assert summary["max_abs_lon_jerk"] is None
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
    assert float(rows[0]["comfort_max_lon_accel"]) == pytest.approx(0.1)


# --- TEST-CMF-09: row helpers ---


def test_comfort_episode_fields_indexes_per_episode_vector() -> None:
    first = _summary(_compliant_steps())
    second = _summary([_step(a_lon=3.0)] * 3)
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
    """Two fixed 3-step episodes with distinct, deterministic kinematics."""

    EPISODES: tuple[tuple[tuple[float, float], ...], ...] = (
        ((0.0, 0.0), (0.2, 0.05), (0.4, 0.10)),
        ((0.0, 0.0), (3.0, 0.00), (0.0, 0.00)),
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
        a_lon, yaw = pattern[self._step]
        self._step += 1
        done = self._step >= len(pattern)
        info = _step(a_lon=a_lon, yaw=yaw)
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
