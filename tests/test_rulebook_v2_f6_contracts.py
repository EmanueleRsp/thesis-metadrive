from __future__ import annotations

from math import isfinite

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.components.clearance import evaluate_clearance
from thesis_rl.rulebook.v2.components.collision import evaluate_collision_impact
from thesis_rl.rulebook.v2.components.rss import RSSCalibrationArtifact, RSSCandidate, evaluate_rss
from thesis_rl.rulebook.v2.components.ttc import evaluate_ttc
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, ContactOnsetRecord


def _actor(
    actor_id: str,
    actor_class: ActorClass,
    *,
    x: float = 2.0,
    y: float = 0.0,
    velocity_xy: tuple[float, float] = (0.0, 0.0),
    cap: float | None = 20.0,
) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id,
        actor_class,
        (x, y),
        0.0,
        0.0,
        velocity_xy,
        Polygon(((x, y - 0.5), (x + 1.0, y - 0.5), (x + 1.0, y + 0.5), (x, y - 0.5))),
        None,
        cap,
    )


def _onset(actor_id: str) -> ContactOnsetRecord:
    return ContactOnsetRecord(actor_id, ActorClass.VEHICLE)


def _assert_bounded(result) -> None:
    assert isfinite(result.cost)
    assert 0.0 <= result.cost <= 1.0
    assert result.evaluable is True


def test_collision_handles_tangent_static_vru_and_simultaneous_onsets():
    result, _, _ = evaluate_collision_impact(
        scenario_id="s",
        step_index=1,
        ego_configured_speed_cap_mps=10.0,
        pre_ego=_actor("ego", ActorClass.VEHICLE, x=0.0, y=0.0, velocity_xy=(0.0, 5.0)),
        pre_actors_by_id={
            "static": _actor("static", ActorClass.STATIC_COLLIDABLE, cap=None),
            "ped": _actor("ped", ActorClass.PEDESTRIAN),
        },
        onset_records=(_onset("static"), _onset("ped")),
        previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"static", "ped"}),
    )
    _assert_bounded(result)
    assert {item["actor_id"] for item in result.raw["actors"]} == {"static", "ped"}
    assert result.raw["worst_closing_speed_mps"] == 0.0
    # Rulebook v4.9: a tangent onset still costs strictly more than zero, now
    # structurally rather than through the floor -- the injury-risk curve is
    # positive at zero closing speed.  The pedestrian curve dominates the
    # car-driver curve used for the static obstacle, so it is the worst actor.
    assert result.cost == pytest.approx(0.02366, abs=1e-5)
    assert result.raw["worst_actor_id"] == "ped"


def test_collision_cost_approaches_one_at_extreme_closing_speed():
    """Rulebook v4.9 §7: the cost tends to 1 without ever reaching it.

    This replaces the v4.7 contract that saturated exactly at the configured
    speed cap; that normalizer no longer takes part in the cost (ADR-027).
    """

    result, _, _ = evaluate_collision_impact(
        scenario_id="s",
        step_index=1,
        ego_configured_speed_cap_mps=10.0,
        pre_ego=_actor("ego", ActorClass.VEHICLE, x=0.0, y=0.0, velocity_xy=(5.0, 0.0)),
        pre_actors_by_id={"other": _actor("other", ActorClass.VEHICLE, velocity_xy=(-100.0, 0.0), cap=10.0)},
        onset_records=(_onset("other"),),
        previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"other"}),
    )
    _assert_bounded(result)
    assert result.raw["worst_closing_speed_mps"] == pytest.approx(105.0)
    assert 0.999 < result.cost < 1.0


def test_rss_no_front_vehicle_is_explicitly_not_applicable_and_bounded():
    result, _, _ = evaluate_rss(
        scenario_id="s",
        step_index=1,
        candidates=(),
        calibration=None,
        expected_config_hash="ego",
    )
    _assert_bounded(result)
    assert result.applicable is False
    assert result.status.value == "not_applicable"


def test_ttc_parallel_and_beyond_horizon_are_evaluable_without_cost():
    ego = Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5)))
    parallel, _, _ = evaluate_ttc(
        ego_footprint=ego,
        ego_velocity_xy=(1.0, 0.0),
        actors=(_actor("parallel", ActorClass.VEHICLE, x=2.0, y=3.0, velocity_xy=(1.0, 0.0)),),
        vertically_compatible_actor_ids=frozenset({"parallel"}),
    )
    distant, _, _ = evaluate_ttc(
        ego_footprint=ego,
        ego_velocity_xy=(1.0, 0.0),
        actors=(_actor("distant", ActorClass.STATIC_COLLIDABLE, x=100.0, cap=None),),
        vertically_compatible_actor_ids=frozenset({"distant"}),
    )
    _assert_bounded(parallel)
    _assert_bounded(distant)
    assert parallel.cost == 0.0
    assert distant.raw["worst_ttc_s"] == -1.0


def test_ttc_current_overlap_and_clearance_thresholds_are_bounded():
    ego = Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5)))
    ttc, _, _ = evaluate_ttc(
        ego_footprint=ego,
        ego_velocity_xy=(0.0, 0.0),
        actors=(_actor("overlap", ActorClass.STATIC_COLLIDABLE, x=0.5, cap=None),),
        vertically_compatible_actor_ids=frozenset({"overlap"}),
    )
    clearance, _, _ = evaluate_clearance(
        ego_footprint=ego,
        actors=(
            _actor("vehicle", ActorClass.VEHICLE, x=1.8),
            _actor("cyclist", ActorClass.CYCLIST, x=0.0, y=2.0),
            _actor("static", ActorClass.STATIC_COLLIDABLE, x=2.5, cap=None),
        ),
        vertically_compatible_actor_ids=frozenset({"vehicle", "cyclist", "static"}),
    )
    _assert_bounded(ttc)
    _assert_bounded(clearance)
    assert ttc.cost == pytest.approx(1.0)
    # REQ-R2-01/02: VEHICLE and STATIC_COLLIDABLE no longer contribute to
    # the clearance cost (rulebook v4.8); only the CYCLIST candidate remains.
    assert clearance.raw["worst_actor_id"] == "cyclist"
    assert clearance.cost == pytest.approx(0.0)
    assert clearance.diagnostics["static_polygon_distance_m"] > 0.0


def test_rss_safe_distance_equal_gap_has_zero_margin_and_finite_status():
    calibration = RSSCalibrationArtifact("ego", 3.0)
    gap = 12.0
    result, _, _ = evaluate_rss(
        scenario_id="s",
        step_index=1,
        candidates=(RSSCandidate("front", gap, 5.0, 5.0),),
        calibration=calibration,
        expected_config_hash="ego",
    )
    _assert_bounded(result)
    assert result.applicable is True
    assert result.raw["worst_deficit_m"] == pytest.approx(max(0.0, result.raw["worst_safe_distance_m"] - gap))
