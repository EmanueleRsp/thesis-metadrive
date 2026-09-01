"""Ride-comfort and jerk evaluation diagnostics (EP-COMFORT-DIAG).

Diagnostics only. `RULEBOOK-V5.1` §13 excludes comfort and jerk from the
rulebook and from the reward, and permits exactly this: logging them as
diagnostics. Nothing in this module is read by a rule, a margin, a reward, or a
termination decision, and it lives under `runtime/` rather than `rulebook/`
precisely so it cannot be mistaken for a rulebook channel. See
`docs/implementation/comfort_and_jerk_diagnostics_exec_plan.md`.

The instrument is nuPlan's `ego_is_comfortable`: seven kinematic channels, each
with a published bound determined empirically from expert trajectories, and the
boolean their conjunction implies (`DEC-CMF-001`). Both the bounds and the
Savitzky-Golay derivative parameters below are transcribed from the devkit
rather than chosen here (`DEC-CMF-006`); every value carries its source in a
comment.

Two adaptations are unavoidable and are the only places this differs from the
devkit; both are recorded as `DEV-CMF-001`:

1. **Acceleration source.** nuPlan reads the simulator's own
   `dynamic_car_state.center_acceleration_2d` and smooths it (`savgol`,
   `window_length=8`, `poly_order=2`, `deriv=0`). MetaDrive's authoritative
   snapshot publishes velocity, not acceleration, so the acceleration here is
   the Savitzky-Golay **first derivative** of velocity at the same window and
   polynomial order. Same filter, same window, one differentiation earlier.
2. **Segmentation.** nuPlan filters one contiguous trajectory. An evaluation
   episode may contain steps whose kinematics are unusable, so the series is
   split at those gaps and each contiguous segment is filtered independently,
   with the episode statistic taken across segments. No filter window ever
   spans a gap.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

# The seven per-episode channels, in the order they are reported. `min_lon_accel`
# is a lower bound and every other entry is an upper bound; `_BOUND_IS_LOWER`
# below is the single place that distinction lives.
COMFORT_STATISTICS: tuple[str, ...] = (
    "max_lon_accel",
    "min_lon_accel",
    "max_abs_lat_accel",
    "max_abs_mag_jerk",
    "max_abs_lon_jerk",
    "max_abs_yaw_rate",
    "max_abs_yaw_accel",
)

_BOUND_IS_LOWER: frozenset[str] = frozenset({"min_lon_accel"})

COMFORT_EPISODE_COLUMNS: tuple[str, ...] = (
    "comfort_valid_step_count",
    "comfort_is_comfortable",
) + tuple(f"comfort_{name}" for name in COMFORT_STATISTICS)

COMFORT_AGGREGATE_COLUMNS: tuple[str, ...] = (
    "comfort_rate",
    "comfort_episode_count",
    "comfort_excluded_episode_count",
) + tuple(f"mean_comfort_{name}" for name in COMFORT_STATISTICS)


@dataclass(frozen=True, slots=True)
class SavgolSpec:
    """One Savitzky-Golay derivative configuration, as nuPlan parameterises it."""

    window_length: int
    poly_order: int
    deriv_order: int

    def min_samples(self) -> int:
        """Shortest series this spec can be evaluated on.

        `approximate_derivatives` clamps the window to the series length and
        then requires `poly_order < window_length`, so a series of
        `poly_order + 1` samples is the shortest that does not raise.
        """
        return self.poly_order + 1


# Transcribed from nuplan-devkit `nuplan/planning/metrics/utils/state_extractors.py`:
#
# - `extract_ego_acceleration(..., poly_order=2, window_length=8)` smooths the
#   acceleration series (`deriv=0`); here the same window and order take the
#   first derivative of velocity instead (adaptation 1 above).
# - `extract_ego_jerk(..., deriv_order=1, poly_order=2, window_length=15)`.
# - `extract_ego_yaw_rate(..., deriv_order=1, poly_order=2, window_length=15)`
#   and, for yaw acceleration, `deriv_order=2, poly_order=3`.
#
# The yaw window is **5, not 15**, and that is deliberate: `extract_ego_yaw_rate`
# accepts `window_length` but never forwards it to `approximate_derivatives`,
# which therefore applies its own default of 5. Reproducing the devkit's actual
# behaviour is what makes these numbers comparable to published nuPlan figures;
# passing 15 would silently measure something nuPlan never measured.
ACCEL_SPEC = SavgolSpec(window_length=8, poly_order=2, deriv_order=1)
JERK_SPEC = SavgolSpec(window_length=15, poly_order=2, deriv_order=1)
YAW_RATE_SPEC = SavgolSpec(window_length=5, poly_order=2, deriv_order=1)
YAW_ACCEL_SPEC = SavgolSpec(window_length=5, poly_order=3, deriv_order=2)


@dataclass(frozen=True, slots=True)
class ComfortBounds:
    """One threshold per channel, in SI units."""

    max_lon_accel: float
    min_lon_accel: float
    max_abs_lat_accel: float
    max_abs_mag_jerk: float
    max_abs_lon_jerk: float
    max_abs_yaw_rate: float
    max_abs_yaw_accel: float

    def satisfied_by(self, statistics: Mapping[str, float | None]) -> bool | None:
        """Return the comfort verdict, or `None` if any channel is undefined.

        Undefined is not "comfortable": an episode too short to carry a yaw
        acceleration has no verdict at all, and saying so is the point of
        `REQ-CMF-07`.
        """
        # Definedness is settled across *every* channel before any bound is
        # tested. Short-circuiting on the first breach would report `False`
        # for an episode that is missing a later channel, turning "we could
        # not measure this" into "we measured discomfort".
        if any(statistics.get(name) is None for name in COMFORT_STATISTICS):
            return None
        for name in COMFORT_STATISTICS:
            value = float(statistics[name])  # type: ignore[arg-type]
            bound = float(getattr(self, name))
            if name in _BOUND_IS_LOWER:
                if value < bound:
                    return False
            elif value > bound:
                return False
        return True


# Transcribed from the nuplan-devkit metric configs under
# `nuplan/planning/script/config/common/simulation_metric/low_level/`:
# `ego_lon_acceleration_statistics.yaml` (min -4.05, max 2.40),
# `ego_lat_acceleration_statistics.yaml` (4.89), `ego_jerk_statistics.yaml`
# (8.37), `ego_lon_jerk_statistics.yaml` (4.13),
# `ego_yaw_rate_statistics.yaml` (0.95), `ego_yaw_acceleration_statistics.yaml`
# (1.93). Taken as published rather than refitted: `RULEBOOK-V5.0` §13 already
# records these as the expert-derived anchor for comfort, and re-deriving them
# against this repository's own expert panel is deferred work, not a silent
# substitution.
NUPLAN_COMFORT_BOUNDS = ComfortBounds(
    max_lon_accel=2.40,
    min_lon_accel=-4.05,
    max_abs_lat_accel=4.89,
    max_abs_mag_jerk=8.37,
    max_abs_lon_jerk=4.13,
    max_abs_yaw_rate=0.95,
    max_abs_yaw_accel=1.93,
)


@dataclass(frozen=True, slots=True)
class ComfortStep:
    """One step's authoritative ego kinematics, as the rulebook snapshotted it."""

    sim_time_s: float
    velocity_x: float
    velocity_y: float
    heading_rad: float


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _savgol(values: np.ndarray, delta: float, spec: SavgolSpec) -> np.ndarray | None:
    """Apply one nuPlan-parameterised Savitzky-Golay derivative, or `None`.

    Mirrors `approximate_derivatives`: the window is clamped to the series
    length, and a series too short for the polynomial order yields nothing
    rather than an exception.
    """
    window_length = min(spec.window_length, len(values))
    if window_length <= spec.poly_order or delta <= 0.0:
        return None
    filtered: np.ndarray = savgol_filter(
        values,
        window_length=window_length,
        polyorder=spec.poly_order,
        deriv=spec.deriv_order,
        delta=delta,
    )
    return filtered


def extract_comfort_step(step_info: Any) -> ComfortStep | None:
    """Read one step's kinematics, or `None` when they are unusable.

    The source is `info["ego_kinematics"]`, published by
    `RulebookV2MonitorWrapper.step` from the post-transition `EnvSnapshot`:
    the same ego state the rulebook grades, so the comfort diagnostic and the
    rulebook can never disagree about what the vehicle did.

    Tolerance mirrors `extract_subrule_step`: a missing or malformed
    `step_info` is treated as absent rather than as an error, because
    evaluation must not fail on a diagnostic. The legacy v1 `RuleRewardWrapper`
    path publishes no `ego_kinematics`, so runs on that path report empty
    comfort columns -- "not measured", which is what they are.
    """
    if not isinstance(step_info, Mapping):
        return None
    kinematics = step_info.get("ego_kinematics")
    if not isinstance(kinematics, Mapping):
        return None

    velocity = kinematics.get("velocity_xy")
    if not isinstance(velocity, (list, tuple)) or len(velocity) != 2:
        return None

    sim_time_s = _finite_float(kinematics.get("sim_time_s"))
    velocity_x = _finite_float(velocity[0])
    velocity_y = _finite_float(velocity[1])
    heading_rad = _finite_float(kinematics.get("heading_rad"))
    if sim_time_s is None or velocity_x is None or velocity_y is None or heading_rad is None:
        return None
    return ComfortStep(
        sim_time_s=sim_time_s,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        heading_rad=heading_rad,
    )


def segment_statistics(segment: Sequence[ComfortStep]) -> dict[str, float | None]:
    """Reduce one gap-free run of steps to the seven channels.

    Public because it is the whole numerical contract of this module and is
    worth testing directly against closed-form trajectories.
    """
    stats: dict[str, float | None] = dict.fromkeys(COMFORT_STATISTICS)
    if len(segment) < 2:
        return stats

    times = np.asarray([step.sim_time_s for step in segment], dtype=np.float64)
    intervals = np.diff(times)
    if not np.all(intervals > 0.0):
        return stats
    # `approximate_derivatives` collapses the sample spacing to its mean and
    # feeds that single `delta` to `savgol_filter`; a per-interval spacing is
    # not something the filter accepts.
    delta = float(intervals.mean())

    velocity_x = np.asarray([step.velocity_x for step in segment], dtype=np.float64)
    velocity_y = np.asarray([step.velocity_y for step in segment], dtype=np.float64)
    headings = np.asarray([step.heading_rad for step in segment], dtype=np.float64)

    accel_x = _savgol(velocity_x, delta, ACCEL_SPEC)
    accel_y = _savgol(velocity_y, delta, ACCEL_SPEC)
    if accel_x is not None and accel_y is not None:
        cos_h = np.cos(headings)
        sin_h = np.sin(headings)
        accel_lon = accel_x * cos_h + accel_y * sin_h
        accel_lat = -accel_x * sin_h + accel_y * cos_h
        stats["max_lon_accel"] = float(accel_lon.max())
        stats["min_lon_accel"] = float(accel_lon.min())
        stats["max_abs_lat_accel"] = float(np.abs(accel_lat).max())

        # nuPlan's magnitude jerk differentiates the acceleration *magnitude*
        # (`acceleration_coordinate='magnitude'`), not the acceleration vector,
        # so the two jerk channels are derivatives of two scalar series.
        accel_magnitude = np.hypot(accel_x, accel_y)
        magnitude_jerk = _savgol(accel_magnitude, delta, JERK_SPEC)
        if magnitude_jerk is not None:
            stats["max_abs_mag_jerk"] = float(np.abs(magnitude_jerk).max())
        longitudinal_jerk = _savgol(accel_lon, delta, JERK_SPEC)
        if longitudinal_jerk is not None:
            stats["max_abs_lon_jerk"] = float(np.abs(longitudinal_jerk).max())

    # `phase_unwrap` in the devkit; unwrapping before differentiating is what
    # stops a heading crossing +-pi from fabricating a yaw rate of 2*pi/delta.
    unwrapped = np.unwrap(headings)
    yaw_rate = _savgol(unwrapped, delta, YAW_RATE_SPEC)
    if yaw_rate is not None:
        stats["max_abs_yaw_rate"] = float(np.abs(yaw_rate).max())
    yaw_accel = _savgol(unwrapped, delta, YAW_ACCEL_SPEC)
    if yaw_accel is not None:
        stats["max_abs_yaw_accel"] = float(np.abs(yaw_accel).max())
    return stats


class ComfortEpisodeAccumulator:
    """Reduce one episode's steps to the nuPlan comfort statistics.

    One instance per episode. Steps are buffered into gap-free segments,
    because a Savitzky-Golay window must not span a discontinuity; each
    segment is filtered on `finalize()` and the episode statistic is the
    extremum across segments. A step whose kinematics are unusable, or one
    that does not advance simulation time, closes the current segment.
    """

    __slots__ = ("_bounds", "_segment", "_segments", "_valid_steps")

    def __init__(self, bounds: ComfortBounds = NUPLAN_COMFORT_BOUNDS) -> None:
        self._bounds = bounds
        self._valid_steps = 0
        self._segment: list[ComfortStep] = []
        self._segments: list[list[ComfortStep]] = []

    def _close_segment(self) -> None:
        if self._segment:
            self._segments.append(self._segment)
            self._segment = []

    def observe(self, step_info: Any) -> None:
        step = extract_comfort_step(step_info)
        if step is None:
            self._close_segment()
            return

        self._valid_steps += 1
        if self._segment and step.sim_time_s <= self._segment[-1].sim_time_s:
            # Simulation time did not advance: the filter has no spacing to
            # work with, so this starts a fresh segment rather than corrupting
            # the current one.
            self._close_segment()
        self._segment.append(step)

    def finalize(self) -> dict[str, Any]:
        """Return this episode's summary; `None` marks an undefined channel."""
        self._close_segment()

        stats: dict[str, float | None] = dict.fromkeys(COMFORT_STATISTICS)
        for segment in self._segments:
            for name, value in segment_statistics(segment).items():
                if value is None:
                    continue
                current = stats[name]
                if current is None:
                    stats[name] = value
                elif name in _BOUND_IS_LOWER:
                    stats[name] = min(current, value)
                else:
                    stats[name] = max(current, value)

        summary: dict[str, Any] = {
            "valid_step_count": int(self._valid_steps),
            "is_comfortable": self._bounds.satisfied_by(stats),
        }
        summary.update({name: stats[name] for name in COMFORT_STATISTICS})
        return summary


def aggregate_comfort_episodes(episode_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce per-episode summaries to the seed-level diagnostic metrics.

    `REQ-CMF-07`: an episode whose verdict is undefined is excluded from
    `comfort_rate` and counted in `comfort_excluded_episode_count`, never
    folded in as zero or as comfortable. Each channel mean is taken over the
    episodes where that channel is defined, which is the same episode set as
    the verdict except for episodes too short to carry every channel.
    """
    verdicts: list[float] = []
    channel_values: dict[str, list[float]] = {name: [] for name in COMFORT_STATISTICS}

    for summary in episode_summaries:
        if not isinstance(summary, Mapping):
            continue
        verdict = summary.get("is_comfortable")
        if verdict is not None:
            verdicts.append(1.0 if bool(verdict) else 0.0)
        for name in COMFORT_STATISTICS:
            value = _finite_float(summary.get(name))
            if value is not None:
                channel_values[name].append(value)

    total_episodes = sum(1 for summary in episode_summaries if isinstance(summary, Mapping))
    metrics: dict[str, Any] = {
        "comfort_rate": (sum(verdicts) / len(verdicts)) if verdicts else None,
        "comfort_episode_count": len(verdicts),
        "comfort_excluded_episode_count": total_episodes - len(verdicts),
    }
    for name in COMFORT_STATISTICS:
        values = channel_values[name]
        metrics[f"mean_comfort_{name}"] = (sum(values) / len(values)) if values else None
    return metrics


def comfort_episode_fields(per_episode: Any, index: int) -> dict[str, Any]:
    """Return one episode's CSV fields, keyed by `COMFORT_EPISODE_COLUMNS`.

    Out-of-range or absent data yields empty values rather than an error, so a
    caller reducing metrics recorded before this feature keeps working.
    """
    summary: Mapping[str, Any] = {}
    if isinstance(per_episode, Mapping):
        summaries = per_episode.get("comfort")
        if isinstance(summaries, (list, tuple)) and 0 <= index < len(summaries):
            candidate = summaries[index]
            if isinstance(candidate, Mapping):
                summary = candidate

    verdict = summary.get("is_comfortable")
    fields: dict[str, Any] = {
        "comfort_valid_step_count": summary.get("valid_step_count"),
        # Recorded as 1.0/0.0 to match `success`/`collision`, which are floats
        # in this schema, rather than introducing a third boolean encoding.
        "comfort_is_comfortable": None if verdict is None else (1.0 if bool(verdict) else 0.0),
    }
    fields.update({f"comfort_{name}": summary.get(name) for name in COMFORT_STATISTICS})
    return fields


def comfort_aggregate_fields(metrics: Any) -> dict[str, Any]:
    """Return the seed-level CSV fields, keyed by `COMFORT_AGGREGATE_COLUMNS`."""
    if not isinstance(metrics, Mapping):
        return dict.fromkeys(COMFORT_AGGREGATE_COLUMNS)
    return {column: metrics.get(column) for column in COMFORT_AGGREGATE_COLUMNS}
