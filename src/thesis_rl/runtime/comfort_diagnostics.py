"""Ride-comfort and jerk evaluation diagnostics (EP-COMFORT-DIAG).

Diagnostics only. `RULEBOOK-V5.1` §13 excludes comfort and jerk from the
rulebook and from the reward, and permits exactly this: logging them as
diagnostics. Nothing in this module is read by a rule, a margin, a reward, or a
termination decision, and it lives under `runtime/` rather than `rulebook/`
precisely so it cannot be mistaken for a rulebook channel. See
`docs/implementation/comfort_and_jerk_diagnostics_exec_plan.md`.

The statistic set is nuPlan's `ego_is_comfortable`: seven kinematic channels,
each with a published bound determined empirically from expert trajectories,
and the boolean their conjunction implies (`DEC-CMF-001`).

**Deviation from nuPlan (`DEC-CMF-003` / `DEV-CMF-001`).** nuPlan extracts its
channels from Savitzky-Golay-filtered trajectories; this module uses raw
backward finite differences, because `scipy` is not a declared dependency of
this repository and a hand-rolled filter would be an unapproved numerical
convention. Simulator contact impulses and controller chatter therefore inflate
the max-statistics, which makes the comfort verdict **conservative**: a
trajectory nuPlan would call comfortable may be called uncomfortable here, not
the other way round. Comparisons between arms measured by this same instrument
are unaffected; comparisons against published nuPlan figures are not
like-for-like.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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


# nuPlan devkit `ego_is_comfortable` defaults, in m/s^2, m/s^3, rad/s and
# rad/s^2. Taken as published rather than refitted here: the repository's own
# rulebook documents (`RULEBOOK-V5.0` §13) already record these as the
# expert-derived anchor for comfort, and re-deriving them against this
# repository's expert panel is deferred work, not a silent substitution.
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
    """One step's ego kinematics, in the ego frame of `RuleRewardWrapper`."""

    a_lon: float
    a_lat: float
    a_x: float
    a_y: float
    yaw: float
    dt: float


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _wrap_angle(delta: float) -> float:
    """Wrap a heading difference to `(-pi, pi]`.

    Without this a heading crossing `+-pi` fabricates a yaw rate of order
    `2*pi/dt`, which would breach the yaw bounds on a perfectly smooth turn.
    """
    return math.remainder(delta, 2.0 * math.pi)


def extract_comfort_step(step_info: Any) -> ComfortStep | None:
    """Read one step's kinematics, or `None` when they are unusable.

    Mirrors `extract_subrule_step`'s tolerance: a missing or malformed
    `step_info` is treated as absent rather than as an error, because
    evaluation must not fail on a diagnostic. `dt` is read from `ego_state`
    (`DEC-CMF-005`) so the derivative below is taken over exactly the timestep
    `RuleRewardWrapper._extract_physical_acceleration` differentiated the
    velocity over; no default timestep is assumed.
    """
    if not isinstance(step_info, Mapping):
        return None
    ego_state = step_info.get("ego_state")
    if not isinstance(ego_state, Mapping):
        return None
    acceleration = ego_state.get("acceleration")
    if not isinstance(acceleration, Mapping):
        return None

    a_lon = _finite_float(acceleration.get("longitudinal"))
    a_lat = _finite_float(acceleration.get("lateral"))
    a_x = _finite_float(acceleration.get("x"))
    a_y = _finite_float(acceleration.get("y"))
    yaw = _finite_float(ego_state.get("yaw"))
    dt = _finite_float(ego_state.get("dt"))
    if (
        a_lon is None
        or a_lat is None
        or a_x is None
        or a_y is None
        or yaw is None
        or dt is None
        or dt <= 0.0
    ):
        return None
    return ComfortStep(a_lon=a_lon, a_lat=a_lat, a_x=a_x, a_y=a_y, yaw=yaw, dt=dt)


class ComfortEpisodeAccumulator:
    """Reduce one episode's steps to the nuPlan comfort statistics.

    One instance per episode. Derivatives are backward differences over
    *consecutive* valid steps: a step whose kinematics are unusable resets the
    carried state, so no derivative is ever taken across a gap in the episode.
    """

    __slots__ = ("_bounds", "_prev", "_prev_yaw_rate", "_stats", "_valid_steps")

    def __init__(self, bounds: ComfortBounds = NUPLAN_COMFORT_BOUNDS) -> None:
        self._bounds = bounds
        self._valid_steps = 0
        self._prev: ComfortStep | None = None
        self._prev_yaw_rate: float | None = None
        self._stats: dict[str, float | None] = dict.fromkeys(COMFORT_STATISTICS)

    def _keep_max(self, name: str, value: float) -> None:
        current = self._stats[name]
        self._stats[name] = value if current is None else max(current, value)

    def _keep_min(self, name: str, value: float) -> None:
        current = self._stats[name]
        self._stats[name] = value if current is None else min(current, value)

    def observe(self, step_info: Any) -> None:
        step = extract_comfort_step(step_info)
        if step is None:
            self._prev = None
            self._prev_yaw_rate = None
            return

        self._valid_steps += 1
        self._keep_max("max_lon_accel", step.a_lon)
        self._keep_min("min_lon_accel", step.a_lon)
        self._keep_max("max_abs_lat_accel", abs(step.a_lat))

        previous = self._prev
        if previous is None:
            # First step of a run of consecutive valid steps: no difference to
            # take, and the previous yaw rate belongs to a broken chain.
            self._prev_yaw_rate = None
            self._prev = step
            return

        dt = step.dt
        self._keep_max("max_abs_lon_jerk", abs(step.a_lon - previous.a_lon) / dt)
        self._keep_max(
            "max_abs_mag_jerk",
            math.hypot(step.a_x - previous.a_x, step.a_y - previous.a_y) / dt,
        )
        yaw_rate = _wrap_angle(step.yaw - previous.yaw) / dt
        self._keep_max("max_abs_yaw_rate", abs(yaw_rate))
        if self._prev_yaw_rate is not None:
            self._keep_max("max_abs_yaw_accel", abs(yaw_rate - self._prev_yaw_rate) / dt)
        self._prev_yaw_rate = yaw_rate
        self._prev = step

    def finalize(self) -> dict[str, Any]:
        """Return this episode's summary; `None` marks an undefined channel."""
        summary: dict[str, Any] = {
            "valid_step_count": int(self._valid_steps),
            "is_comfortable": self._bounds.satisfied_by(self._stats),
        }
        summary.update({name: self._stats[name] for name in COMFORT_STATISTICS})
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
