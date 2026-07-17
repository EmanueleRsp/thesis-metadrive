from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.geometry.continuous_sat import predict_occupancy_interval
from thesis_rl.rulebook.v2.geometry.ctrv import (
    estimate_yaw_rate,
    predict_conflict_zone_occupancy_intervals,
    predict_rotating_occupancy_interval,
    predict_vehicle_occupancy_interval,
    unwrap_headings,
    wrap_heading_delta,
)
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box
from thesis_rl.rulebook.v2.memory import (
    build_motion_history_preview,
    initialize_rulebook_memory,
    merge_memory_deltas,
)
from thesis_rl.rulebook.v2.config import RULEBOOK_V2_VERSION, load_rulebook_v2_config
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorMotionHistory,
    ActorMotionSample,
    ActorSnapshot,
    EnvSnapshot,
)
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline


def _history(*headings: float, actor_id: str = "vehicle-1") -> ActorMotionHistory:
    return ActorMotionHistory(
        actor_id,
        tuple(
            ActorMotionSample(index * 0.25, (index * 0.25, 0.0), heading, (1.0, 0.0))
            for index, heading in enumerate(headings)
        ),
    )


def test_heading_unwrap_and_ols_use_causal_actual_timestamps() -> None:
    assert math.isclose(wrap_heading_delta(-math.pi), math.pi)
    unwrapped = unwrap_headings((3.0, -3.0, -2.5))
    assert unwrapped[1] > unwrapped[0]
    estimate = estimate_yaw_rate(_history(0.0, 0.25, 0.5), minimum_history_samples=3)
    assert estimate.sufficient_history
    assert math.isclose(estimate.yaw_rate_rad_s, 1.0, rel_tol=1e-12)


def test_frozen_ctrv_defaults_are_loaded_and_cannot_be_overridden() -> None:
    config = load_rulebook_v2_config(
        {
            "version": RULEBOOK_V2_VERSION,
            "prediction": {
                "conflict_zone_occupancy": {
                    "history_window_s": 0.5,
                    "minimum_history_samples": 3,
                    "stationary_speed_epsilon_mps": 0.1,
                    "yaw_rate_straight_epsilon_rad_s": 1.0e-3,
                    "rotating_occupancy_max_step_s": 0.02,
                }
            },
        }
    )
    assert config.prediction.conflict_zone_occupancy.history_window_s == 0.5
    with pytest.raises(ValueError, match="frozen"):
        load_rulebook_v2_config(
            {"prediction": {"conflict_zone_occupancy": {"history_window_s": 0.25}}}
        )


def test_ctrv_insufficient_history_is_exact_v46_cv_fallback() -> None:
    footprint = oriented_bounding_box(
        center_xy=(0.0, 0.0), heading_rad=0.0, length_m=2.0, width_m=1.0
    )
    zone = Polygon(((1.5, -1.0), (2.5, -1.0), (2.5, 1.0), (1.5, 1.0)))
    interval, diagnostics = predict_vehicle_occupancy_interval(
        history=ActorMotionHistory(
            "vehicle-1", (ActorMotionSample(0.0, (0.0, 0.0), 0.0, (1.0, 0.0)),)
        ),
        footprint=footprint,
        center_xy_m=(0.0, 0.0),
        heading_rad=0.0,
        velocity_xy_mps=(1.0, 0.0),
        zone=zone,
        horizon_s=3.0,
    )
    expected = predict_occupancy_interval(
        actor_footprint=footprint, actor_velocity_xy=(1.0, 0.0), zone=zone, horizon_s=3.0
    )
    assert interval == expected
    assert diagnostics["motion_model"] == "CV_FALLBACK_INSUFFICIENT_HISTORY"


def test_ctrv_rotating_sweep_returns_finite_event_and_open_end() -> None:
    footprint = oriented_bounding_box(
        center_xy=(0.0, 0.0), heading_rad=0.0, length_m=0.5, width_m=0.3
    )
    center_at_one = (math.sin(1.0), 1.0 - math.cos(1.0))
    zone = Polygon(
        (
            (center_at_one[0] - 0.2, center_at_one[1] - 0.2),
            (center_at_one[0] + 0.2, center_at_one[1] - 0.2),
            (center_at_one[0] + 0.2, center_at_one[1] + 0.2),
            (center_at_one[0] - 0.2, center_at_one[1] + 0.2),
        )
    )
    interval = predict_rotating_occupancy_interval(
        footprint=footprint,
        center_xy_m=(0.0, 0.0),
        heading_rad=0.0,
        signed_speed_mps=1.0,
        yaw_rate_rad_s=1.0,
        zone=zone,
        horizon_s=3.0,
        max_step_s=0.02,
    )
    assert interval is not None
    assert 0.3 < interval.start_s < 1.2
    assert interval.end_s is not None
    assert math.isfinite(interval.end_s)
    open_interval = predict_rotating_occupancy_interval(
        footprint=footprint,
        center_xy_m=(0.0, 0.0),
        heading_rad=0.0,
        signed_speed_mps=1.0,
        yaw_rate_rad_s=1.0,
        zone=Polygon(((-3.0, -3.0), (3.0, -3.0), (3.0, 3.0), (-3.0, 3.0))),
        horizon_s=3.0,
        max_step_s=0.02,
    )
    assert open_interval is not None and open_interval.is_open_end


def test_frozen_sweep_matches_finer_synthetic_oracle() -> None:
    footprint = oriented_bounding_box(
        center_xy=(0.0, 0.0), heading_rad=0.0, length_m=0.5, width_m=0.3
    )
    zone = Polygon(((0.65, 0.15), (1.05, 0.15), (1.05, 0.55), (0.65, 0.55)))
    coarse = predict_rotating_occupancy_interval(
        footprint=footprint,
        center_xy_m=(0.0, 0.0),
        heading_rad=0.0,
        signed_speed_mps=1.0,
        yaw_rate_rad_s=1.0,
        zone=zone,
        horizon_s=3.0,
        max_step_s=0.02,
    )
    oracle = predict_rotating_occupancy_interval(
        footprint=footprint,
        center_xy_m=(0.0, 0.0),
        heading_rad=0.0,
        signed_speed_mps=1.0,
        yaw_rate_rad_s=1.0,
        zone=zone,
        horizon_s=3.0,
        max_step_s=0.001,
    )
    assert coarse is not None and oracle is not None
    assert abs(coarse.start_s - oracle.start_s) <= 0.02
    assert coarse.end_s is not None and oracle.end_s is not None
    assert abs(coarse.end_s - oracle.end_s) <= 0.02


def test_common_conflict_zone_query_supplies_vehicle_ctrv_and_nonvehicle_cv() -> None:
    ego_footprint = oriented_bounding_box(
        center_xy=(0.0, 0.0), heading_rad=0.0, length_m=0.5, width_m=0.3
    )
    pedestrian = ActorSnapshot(
        "pedestrian-1",
        ActorClass.PEDESTRIAN,
        (0.8, 0.35),
        0.0,
        0.0,
        (0.0, 0.0),
        oriented_bounding_box(center_xy=(0.8, 0.35), heading_rad=0.0, length_m=0.4, width_m=0.4),
        None,
        None,
    )
    ego = ActorSnapshot(
        "ego",
        ActorClass.VEHICLE,
        (0.0, 0.0),
        0.0,
        0.0,
        (1.0, 0.0),
        ego_footprint,
        "lane",
        20.0,
    )
    ego_interval, actor_intervals, diagnostics = predict_conflict_zone_occupancy_intervals(
        ego=ego,
        actors=(pedestrian,),
        histories=(_history(0.0, 0.25, 0.5, actor_id="ego"),),
        sim_time_s=0.5,
        zone=Polygon(((0.65, 0.15), (1.05, 0.15), (1.05, 0.55), (0.65, 0.55))),
    )
    assert ego_interval is not None
    assert actor_intervals[0][0] == "pedestrian-1"
    assert diagnostics["ego"]["motion_model"] == "CTRV"
    assert diagnostics["pedestrian-1"]["motion_model"] == "CV_NON_VEHICLE"


def test_ctrv_replay_is_deterministic_and_outputs_are_finite_bounded() -> None:
    footprint = oriented_bounding_box(
        center_xy=(0.0, 0.0), heading_rad=0.0, length_m=0.5, width_m=0.3
    )
    history = _history(0.0, 0.25, 0.5, actor_id="ego")
    kwargs = {
        "history": history,
        "footprint": footprint,
        "center_xy_m": (0.0, 0.0),
        "heading_rad": 0.0,
        "velocity_xy_mps": (1.0, 0.0),
        "zone": Polygon(((0.65, 0.15), (1.05, 0.15), (1.05, 0.55), (0.65, 0.55))),
        "horizon_s": 3.0,
    }
    first = predict_vehicle_occupancy_interval(**kwargs)
    second = predict_vehicle_occupancy_interval(**kwargs)
    assert first == second
    interval, diagnostics = first
    assert interval is not None
    assert 0.0 <= interval.start_s <= 3.0
    assert interval.end_s is None or interval.start_s <= interval.end_s <= 3.0
    for value in diagnostics.values():
        if isinstance(value, float):
            assert math.isfinite(value)


def test_history_preview_resets_after_absence_and_rejects_nonmonotonic_time() -> None:
    def actor(actor_id: str, x: float) -> ActorSnapshot:
        return ActorSnapshot(
            actor_id,
            ActorClass.VEHICLE,
            (x, 0.0),
            0.0,
            0.0,
            (1.0, 0.0),
            oriented_bounding_box(center_xy=(x, 0.0), heading_rad=0.0, length_m=2.0, width_m=1.0),
            "lane",
            20.0,
        )

    reset = EnvSnapshot(
        "scenario", 0, 0.0, actor("ego", 0.0), (actor("other", 1.0),), (), frozenset(), {}
    )
    memory = initialize_rulebook_memory(
        reset_snapshot=reset,
        route=RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
        zone_polygons={},
    )
    post = EnvSnapshot("scenario", 1, 0.2, actor("ego", 0.2), (), (), frozenset(), {})
    histories, delta = build_motion_history_preview(
        memory=memory, post_state=post, history_window_s=0.5
    )
    assert tuple(history.actor_id for history in histories) == ("ego",)
    assert delta.writer == "motion_history"
    next_memory = merge_memory_deltas(memory, (delta,))
    reappeared = EnvSnapshot(
        "scenario", 2, 0.4, actor("ego", 0.4), (actor("other", 1.4),), (), frozenset(), {}
    )
    histories, _ = build_motion_history_preview(
        memory=next_memory, post_state=reappeared, history_window_s=0.5
    )
    other_history = next(history for history in histories if history.actor_id == "other")
    assert len(other_history.samples) == 1
    with pytest.raises(ValueError, match="strictly increasing"):
        build_motion_history_preview(memory=memory, post_state=reset, history_window_s=0.5)
