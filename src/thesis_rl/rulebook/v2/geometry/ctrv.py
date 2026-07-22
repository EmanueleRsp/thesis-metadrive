"""Causal filtered CTRV prediction for vehicle conflict-zone occupancy."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from math import ceil, cos, isfinite, pi, sin

from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.prepared import prep

from thesis_rl.rulebook.v2.geometry.continuous_sat import (
    OccupancyInterval,
    predict_occupancy_interval,
)
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorMotionHistory,
    ActorMotionSample,
    ActorSnapshot,
)


def wrap_heading_delta(delta_rad: float) -> float:
    """Return the principal heading difference in (-pi, pi]."""

    wrapped = (delta_rad + pi) % (2.0 * pi) - pi
    return pi if wrapped == -pi else wrapped


def unwrap_headings(headings_rad: tuple[float, ...]) -> tuple[float, ...]:
    """Unwrap an ordered causal heading sequence."""

    if not headings_rad:
        return ()
    values = [float(headings_rad[0])]
    for heading in headings_rad[1:]:
        values.append(values[-1] + wrap_heading_delta(float(heading) - values[-1]))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class YawRateEstimate:
    yaw_rate_rad_s: float
    sample_count: int
    history_span_s: float
    sufficient_history: bool


def estimate_yaw_rate(
    history: ActorMotionHistory,
    *,
    minimum_history_samples: int,
) -> YawRateEstimate:
    """Estimate causal yaw rate with OLS against actual timestamps."""

    if minimum_history_samples < 2:
        raise ValueError("minimum_history_samples must be at least two")
    samples = history.samples
    if any(not isfinite(sample.timestamp_s) for sample in samples):
        raise ValueError("Motion history timestamps must be finite")
    if len(samples) < minimum_history_samples:
        span = 0.0 if len(samples) < 2 else samples[-1].timestamp_s - samples[0].timestamp_s
        return YawRateEstimate(0.0, len(samples), span, False)
    times = tuple(sample.timestamp_s for sample in samples)
    if any(previous >= current for previous, current in zip(times, times[1:])):
        raise ValueError("Motion history timestamps must be strictly increasing")
    headings = unwrap_headings(tuple(sample.heading_rad for sample in samples))
    mean_time = sum(times) / len(times)
    mean_heading = sum(headings) / len(headings)
    denominator = sum((time - mean_time) ** 2 for time in times)
    span = times[-1] - times[0]
    if denominator <= 0.0 or span <= 0.0:
        return YawRateEstimate(0.0, len(samples), span, False)
    numerator = sum(
        (time - mean_time) * (heading - mean_heading) for time, heading in zip(times, headings)
    )
    estimate = numerator / denominator
    if not isfinite(estimate):
        raise ValueError("OLS yaw-rate estimate must be finite")
    return YawRateEstimate(estimate, len(samples), span, True)


def propagate_ctrv_footprint(
    *,
    footprint: Polygon,
    center_xy_m: tuple[float, float],
    heading_rad: float,
    signed_speed_mps: float,
    yaw_rate_rad_s: float,
    offset_s: float,
) -> Polygon:
    """Propagate and rotate a footprint from its current post-state pose."""

    values = (*center_xy_m, heading_rad, signed_speed_mps, yaw_rate_rad_s, offset_s)
    if not all(isfinite(value) for value in values):
        raise ValueError("CTRV pose and kinematics must be finite")
    if abs(yaw_rate_rad_s) <= 1.0e-15:
        position = (
            center_xy_m[0] + signed_speed_mps * cos(heading_rad) * offset_s,
            center_xy_m[1] + signed_speed_mps * sin(heading_rad) * offset_s,
        )
        delta_heading = 0.0
    else:
        position = (
            center_xy_m[0]
            + signed_speed_mps
            / yaw_rate_rad_s
            * (sin(heading_rad + yaw_rate_rad_s * offset_s) - sin(heading_rad)),
            center_xy_m[1]
            + signed_speed_mps
            / yaw_rate_rad_s
            * (-cos(heading_rad + yaw_rate_rad_s * offset_s) + cos(heading_rad)),
        )
        delta_heading = yaw_rate_rad_s * offset_s
    rotated = rotate(
        footprint,
        delta_heading * 180.0 / pi,
        origin=center_xy_m,
        use_radians=False,
    )
    return translate(
        rotated,
        xoff=position[0] - center_xy_m[0],
        yoff=position[1] - center_xy_m[1],
    )


def _refine_entry(
    *,
    lo_s: float,
    hi_s: float,
    overlaps,
    entering: bool,
    tolerance_s: float,
) -> float:
    while hi_s - lo_s > tolerance_s:
        mid_s = (lo_s + hi_s) / 2.0
        if overlaps(mid_s) is entering:
            hi_s = mid_s
        else:
            lo_s = mid_s
    return hi_s if entering else lo_s


def predict_rotating_occupancy_interval(
    *,
    footprint: Polygon,
    center_xy_m: tuple[float, float],
    heading_rad: float,
    signed_speed_mps: float,
    yaw_rate_rad_s: float,
    zone: Polygon,
    horizon_s: float,
    max_step_s: float,
    interval_tolerance_s: float = 1.0e-6,
) -> OccupancyInterval | None:
    """Detect rotating-footprint occupancy with deterministic sweep/bisection."""

    if not isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("horizon_s must be finite and positive")
    if not isfinite(max_step_s) or not 0.0 < max_step_s <= horizon_s:
        raise ValueError("max_step_s must be within the finite horizon")
    if not zone.is_valid or zone.is_empty or not footprint.is_valid or footprint.is_empty:
        raise ValueError("CTRV occupancy requires valid non-empty polygons")
    count = max(1, int(ceil(horizon_s / max_step_s)))
    grid = tuple(horizon_s * index / count for index in range(count + 1))
    prepared_zone = prep(zone)

    def overlaps(offset_s: float) -> bool:
        predicted = propagate_ctrv_footprint(
            footprint=footprint,
            center_xy_m=center_xy_m,
            heading_rad=heading_rad,
            signed_speed_mps=signed_speed_mps,
            yaw_rate_rad_s=yaw_rate_rad_s,
            offset_s=offset_s,
        )
        return bool(prepared_zone.intersects(predicted))

    states = tuple(overlaps(offset_s) for offset_s in grid)
    intervals: list[tuple[float, float | None]] = []
    active_start = 0.0 if states[0] else None
    for index in range(len(grid) - 1):
        if not states[index] and states[index + 1]:
            active_start = _refine_entry(
                lo_s=grid[index],
                hi_s=grid[index + 1],
                overlaps=overlaps,
                entering=True,
                tolerance_s=interval_tolerance_s,
            )
        elif states[index] and not states[index + 1]:
            if active_start is None:
                active_start = grid[index]
            end = _refine_entry(
                lo_s=grid[index],
                hi_s=grid[index + 1],
                overlaps=overlaps,
                entering=False,
                tolerance_s=interval_tolerance_s,
            )
            intervals.append((active_start, end))
            active_start = None
    if states[-1] and active_start is not None:
        intervals.append((active_start, None))
    if not intervals:
        return None
    merged: list[list[float | None]] = []
    for start, end in intervals:
        if not merged:
            merged.append([start, end])
            continue
        previous_end = merged[-1][1]
        if previous_end is None or start <= previous_end + interval_tolerance_s:
            if previous_end is None or end is None:
                merged[-1][1] = None
            else:
                merged[-1][1] = max(previous_end, end)
        else:
            merged.append([start, end])
    selected = (
        next(interval for interval in merged if interval[0] <= 0.0 <= (interval[1] or horizon_s))
        if any(interval[0] <= 0.0 <= (interval[1] or horizon_s) for interval in merged)
        else merged[0]
    )
    return OccupancyInterval(
        float(selected[0]), None if selected[1] is None else float(selected[1])
    )


def predict_vehicle_occupancy_interval(
    *,
    history: ActorMotionHistory,
    footprint: Polygon,
    center_xy_m: tuple[float, float],
    heading_rad: float,
    velocity_xy_mps: tuple[float, float],
    zone: Polygon,
    horizon_s: float,
    minimum_history_samples: int = 3,
    stationary_speed_epsilon_mps: float = 0.1,
    yaw_rate_straight_epsilon_rad_s: float = 1.0e-3,
    rotating_occupancy_max_step_s: float = 0.02,
    interval_tolerance_s: float = 1.0e-6,
) -> tuple[OccupancyInterval | None, dict[str, object]]:
    """Predict one vehicle interval and return finite diagnostics."""

    estimate = estimate_yaw_rate(history, minimum_history_samples=minimum_history_samples)
    signed_speed = velocity_xy_mps[0] * cos(heading_rad) + velocity_xy_mps[1] * sin(heading_rad)
    if not isfinite(signed_speed):
        raise ValueError("Signed vehicle speed must be finite")
    if not estimate.sufficient_history:
        interval = predict_occupancy_interval(
            actor_footprint=footprint,
            actor_velocity_xy=velocity_xy_mps,
            zone=zone,
            horizon_s=horizon_s,
        )
        model = "CV_FALLBACK_INSUFFICIENT_HISTORY"
        used_speed = signed_speed
        used_yaw = 0.0
    elif abs(signed_speed) <= stationary_speed_epsilon_mps:
        interval = predict_occupancy_interval(
            actor_footprint=footprint,
            actor_velocity_xy=(0.0, 0.0),
            zone=zone,
            horizon_s=horizon_s,
        )
        model = "CV_STRAIGHT_LIMIT"
        used_speed = 0.0
        used_yaw = 0.0
    elif abs(estimate.yaw_rate_rad_s) <= yaw_rate_straight_epsilon_rad_s:
        interval = predict_occupancy_interval(
            actor_footprint=footprint,
            actor_velocity_xy=velocity_xy_mps,
            zone=zone,
            horizon_s=horizon_s,
        )
        model = "CV_STRAIGHT_LIMIT"
        used_speed = signed_speed
        used_yaw = 0.0
    else:
        interval = predict_rotating_occupancy_interval(
            footprint=footprint,
            center_xy_m=center_xy_m,
            heading_rad=heading_rad,
            signed_speed_mps=signed_speed,
            yaw_rate_rad_s=estimate.yaw_rate_rad_s,
            zone=zone,
            horizon_s=horizon_s,
            max_step_s=rotating_occupancy_max_step_s,
            interval_tolerance_s=interval_tolerance_s,
        )
        model = "CTRV"
        used_speed = signed_speed
        used_yaw = estimate.yaw_rate_rad_s
    diagnostics = {
        "history_sample_count": estimate.sample_count,
        "history_span_s": estimate.history_span_s,
        "motion_model": model,
        "yaw_rate_raw_ols_rad_s": estimate.yaw_rate_rad_s,
        "yaw_rate_used_rad_s": used_yaw,
        "signed_speed_used_mps": used_speed,
        "prediction_horizon_s": horizon_s,
        "occupancy_sweep_step_s": rotating_occupancy_max_step_s,
        "occupancy_result": "NO_INTERVAL"
        if interval is None
        else ("OPEN_END" if interval.is_open_end else "FINITE"),
    }
    return interval, diagnostics


def predict_actor_occupancy_interval(
    *,
    actor: ActorSnapshot,
    history: ActorMotionHistory | None,
    sim_time_s: float,
    zone: Polygon,
    horizon_s: float,
    minimum_history_samples: int = 3,
    stationary_speed_epsilon_mps: float = 0.1,
    yaw_rate_straight_epsilon_rad_s: float = 1.0e-3,
    rotating_occupancy_max_step_s: float = 0.02,
    interval_tolerance_s: float = 1.0e-6,
) -> tuple[OccupancyInterval | None, dict[str, object]]:
    """Predict one live actor without consulting any future trajectory."""

    if not isfinite(sim_time_s):
        raise ValueError("sim_time_s must be finite")
    if actor.actor_class is ActorClass.VEHICLE:
        resolved_history = history or ActorMotionHistory(
            actor.actor_id,
            (
                ActorMotionSample(
                    sim_time_s,
                    actor.position_xy,
                    actor.heading_rad,
                    actor.velocity_xy,
                ),
            ),
        )
        return predict_vehicle_occupancy_interval(
            history=resolved_history,
            footprint=actor.footprint,
            center_xy_m=actor.position_xy,
            heading_rad=actor.heading_rad,
            velocity_xy_mps=actor.velocity_xy,
            zone=zone,
            horizon_s=horizon_s,
            minimum_history_samples=minimum_history_samples,
            stationary_speed_epsilon_mps=stationary_speed_epsilon_mps,
            yaw_rate_straight_epsilon_rad_s=yaw_rate_straight_epsilon_rad_s,
            rotating_occupancy_max_step_s=rotating_occupancy_max_step_s,
            interval_tolerance_s=interval_tolerance_s,
        )
    velocity = actor.velocity_xy
    if actor.actor_class is ActorClass.STATIC_COLLIDABLE:
        velocity = (0.0, 0.0)
    interval = predict_occupancy_interval(
        actor_footprint=actor.footprint,
        actor_velocity_xy=velocity,
        zone=zone,
        horizon_s=horizon_s,
    )
    return interval, {
        "history_sample_count": 0,
        "history_span_s": 0.0,
        "motion_model": "CV_NON_VEHICLE",
        "yaw_rate_raw_ols_rad_s": 0.0,
        "yaw_rate_used_rad_s": 0.0,
        "signed_speed_used_mps": 0.0,
        "prediction_horizon_s": horizon_s,
        "occupancy_sweep_step_s": rotating_occupancy_max_step_s,
        "occupancy_result": "NO_INTERVAL"
        if interval is None
        else ("OPEN_END" if interval.is_open_end else "FINITE"),
    }


def predict_conflict_zone_occupancy_intervals(
    *,
    ego: ActorSnapshot,
    actors: tuple[ActorSnapshot, ...],
    histories: tuple[ActorMotionHistory, ...],
    sim_time_s: float,
    zone: Polygon,
    horizon_s: float = 3.0,
    minimum_history_samples: int = 3,
    stationary_speed_epsilon_mps: float = 0.1,
    yaw_rate_straight_epsilon_rad_s: float = 1.0e-3,
    rotating_occupancy_max_step_s: float = 0.02,
    interval_tolerance_s: float = 1.0e-6,
) -> tuple[
    OccupancyInterval | None,
    tuple[tuple[str, OccupancyInterval], ...],
    Mapping[str, dict[str, object]],
]:
    """Build the common ego/actor interval view consumed by yield rules."""

    history_by_id = {history.actor_id: history for history in histories}
    if len(history_by_id) != len(histories):
        raise ValueError("Duplicate actor history IDs")
    ego_interval, ego_diagnostics = predict_actor_occupancy_interval(
        actor=ego,
        history=history_by_id.get(ego.actor_id),
        sim_time_s=sim_time_s,
        zone=zone,
        horizon_s=horizon_s,
        minimum_history_samples=minimum_history_samples,
        stationary_speed_epsilon_mps=stationary_speed_epsilon_mps,
        yaw_rate_straight_epsilon_rad_s=yaw_rate_straight_epsilon_rad_s,
        rotating_occupancy_max_step_s=rotating_occupancy_max_step_s,
        interval_tolerance_s=interval_tolerance_s,
    )
    actor_intervals: list[tuple[str, OccupancyInterval]] = []
    diagnostics: dict[str, dict[str, object]] = {ego.actor_id: ego_diagnostics}
    for actor in actors:
        if actor.actor_id == ego.actor_id:
            raise ValueError("Ego actor must not be repeated in actors")
        interval, actor_diagnostics = predict_actor_occupancy_interval(
            actor=actor,
            history=history_by_id.get(actor.actor_id),
            sim_time_s=sim_time_s,
            zone=zone,
            horizon_s=horizon_s,
            minimum_history_samples=minimum_history_samples,
            stationary_speed_epsilon_mps=stationary_speed_epsilon_mps,
            yaw_rate_straight_epsilon_rad_s=yaw_rate_straight_epsilon_rad_s,
            rotating_occupancy_max_step_s=rotating_occupancy_max_step_s,
            interval_tolerance_s=interval_tolerance_s,
        )
        diagnostics[actor.actor_id] = actor_diagnostics
        if interval is not None:
            actor_intervals.append((actor.actor_id, interval))
    return ego_interval, tuple(actor_intervals), diagnostics
