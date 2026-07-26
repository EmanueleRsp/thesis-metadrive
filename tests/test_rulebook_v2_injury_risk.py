"""Rulebook v4.9 R1 injury-risk collision cost.

Requirement IDs refer to `docs/specifications/rulebook_v4.9_specification.md`
and the plan `docs/implementation/r1_injury_risk_collision_cost_v4.9_exec_plan.md`.
"""

from __future__ import annotations

from math import log

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.components.collision import evaluate_collision_impact
from thesis_rl.rulebook.v2.components.injury_risk import (
    REFERENCE_AGE_YEARS,
    SOURCE_MEDIAN_AGE_YEARS,
    closing_speed_at_risk_kmh,
    injury_risk_cost,
    model_for,
)
from thesis_rl.rulebook.v2.errors import RulebookEvaluationError
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, ContactOnsetRecord


MAPPED_CLASSES = (
    ActorClass.PEDESTRIAN,
    ActorClass.CYCLIST,
    ActorClass.VEHICLE,
    ActorClass.STATIC_COLLIDABLE,
)


def _footprint(x: float, y: float) -> Polygon:
    return Polygon(((x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)))


def _actor(
    actor_id: str,
    actor_class: ActorClass,
    *,
    cap: float | None = 20.0,
    x: float = 2.0,
    velocity: tuple[float, float] = (0.0, 0.0),
) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id, actor_class, (x, 0.0), 0.0, 0.0, velocity, _footprint(x, 0.0), None, cap
    )


def _ego(velocity: tuple[float, float], cap: float) -> ActorSnapshot:
    return ActorSnapshot(
        "ego", ActorClass.VEHICLE, (0.0, 0.0), 0.0, 0.0, velocity, _footprint(0.0, 0.0), None, cap
    )


def _collision_cost(*, ego_speed: float, ego_cap: float, actor_cap: float = 20.0) -> float:
    result, _, _ = evaluate_collision_impact(
        scenario_id="scenario",
        step_index=1,
        ego_configured_speed_cap_mps=ego_cap,
        pre_ego=_ego((ego_speed, 0.0), ego_cap),
        pre_actors_by_id={"other": _actor("other", ActorClass.VEHICLE, cap=actor_cap)},
        onset_records=(ContactOnsetRecord("other", ActorClass.VEHICLE),),
        previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"other"}),
    )
    return float(result.cost)


# --- REQ-R1-04: the transcribed coefficients reproduce the published anchors ---


def test_published_ten_percent_anchors_are_reproduced() -> None:
    """AC-R1-05: source §4.4 prints 29 / 44 / 112 km/h at 10% MAIS3+F risk.

    The source publishes coefficients rounded to three decimals, so the anchor
    it prints cannot be reproduced exactly.  Rather than pick an arbitrary
    tolerance, this propagates the +-0.0005 rounding interval of every
    coefficient through the inverted curve and requires the published anchor to
    fall inside the resulting interval.  The car-driver curve is the sensitive
    case: its speed coefficient 0.041 carries only two significant digits, so
    its anchor is determined only to roughly +-2 km/h.
    """

    published_kmh = {
        ActorClass.PEDESTRIAN: 29.0,
        ActorClass.CYCLIST: 44.0,
        ActorClass.VEHICLE: 112.0,
    }
    half_ulp = 0.0005
    for actor_class, published in published_kmh.items():
        model = model_for(actor_class)
        age = SOURCE_MEDIAN_AGE_YEARS[actor_class]
        # z(v) = intercept + per_kmh * v + per_year * age = logit(0.10)
        numerator = log(0.10 / 0.90) - model.intercept - model.per_year * age
        spread = half_ulp * (1.0 + age)
        lowest = (numerator - spread) / (model.per_kmh + half_ulp)
        highest = (numerator + spread) / (model.per_kmh - half_ulp)

        assert lowest <= published <= highest, (
            f"{actor_class.value}: published {published} km/h outside "
            f"[{lowest:.2f}, {highest:.2f}] implied by coefficient rounding"
        )
        # The runtime helper must land inside the same interval.
        calculated = closing_speed_at_risk_kmh(actor_class=actor_class, risk=0.10, age_years=age)
        assert lowest <= calculated <= highest


# --- REQ-R1-01: the cost no longer depends on scenario configuration ---


def test_cost_is_independent_of_configured_speed_cap() -> None:
    """AC-R1-01: same closing speed and class, different caps, identical cost."""

    baseline = _collision_cost(ego_speed=5.0, ego_cap=5.0, actor_cap=5.0)
    for ego_cap, actor_cap in ((10.0, 20.0), (20.0, 40.0), (33.0, 7.0)):
        assert _collision_cost(
            ego_speed=5.0, ego_cap=ego_cap, actor_cap=actor_cap
        ) == pytest.approx(baseline, abs=1e-12)


def test_v47_severity_inversion_regression() -> None:
    """AC-R1-02: regression for the defect that motivated v4.9.

    Under v4.7 the residential impact scored (4/5)^2 = 0.640 and the highway
    impact (15/20)^2 = 0.5625, ranking the low-energy impact as more severe.
    """

    residential = _collision_cost(ego_speed=4.0, ego_cap=5.0, actor_cap=5.0)
    highway = _collision_cost(ego_speed=15.0, ego_cap=20.0, actor_cap=20.0)

    assert highway > residential
    # Guard against a silent return of the superseded formula.
    assert residential != pytest.approx(0.640, abs=1e-3)
    assert highway != pytest.approx(0.5625, abs=1e-3)


# --- REQ-R1-02: the cost is the source's MAIS3+F probability ---


def test_reference_cost_table_and_domain() -> None:
    """AC-R1-04: reproduces the normative table of specification §5."""

    expected = {
        ActorClass.PEDESTRIAN: {0.0: 0.0237, 5.0: 0.0898, 10.0: 0.2866, 20.0: 0.8694, 30.0: 0.9910},
        ActorClass.CYCLIST: {0.0: 0.0120, 5.0: 0.0479, 10.0: 0.1725, 20.0: 0.7818, 30.0: 0.9840},
        ActorClass.VEHICLE: {0.0: 0.0019, 5.0: 0.0039, 10.0: 0.0081, 20.0: 0.0343, 30.0: 0.1346},
    }
    for actor_class, rows in expected.items():
        for speed, value in rows.items():
            cost = injury_risk_cost(actor_class=actor_class, normal_closing_speed_mps=speed)
            assert cost == pytest.approx(value, abs=1e-4)
            assert 0.0 < cost < 1.0


def test_cost_is_strictly_increasing_in_closing_speed() -> None:
    """AC-R1-04: strict monotonicity over the reachable domain."""

    for actor_class in MAPPED_CLASSES:
        costs = [
            injury_risk_cost(actor_class=actor_class, normal_closing_speed_mps=float(speed))
            for speed in range(0, 101)
        ]
        assert all(later > earlier for earlier, later in zip(costs, costs[1:]))
        assert all(0.0 < cost < 1.0 for cost in costs)


def test_cost_is_finite_at_extreme_closing_speed() -> None:
    """Numerical stability: no overflow at speeds far outside the sim range."""

    for actor_class in MAPPED_CLASSES:
        assert injury_risk_cost(
            actor_class=actor_class, normal_closing_speed_mps=1.0e6
        ) == pytest.approx(1.0)


# --- REQ-R1-03: vulnerability ordering follows the source ---


def test_vulnerability_ordering_matches_source() -> None:
    """AC-R1-03: pedestrian > cyclist > car occupant at equal closing speed."""

    for speed in (1.0, 5.0, 10.0, 20.0, 30.0):
        pedestrian = injury_risk_cost(
            actor_class=ActorClass.PEDESTRIAN, normal_closing_speed_mps=speed
        )
        cyclist = injury_risk_cost(actor_class=ActorClass.CYCLIST, normal_closing_speed_mps=speed)
        vehicle = injury_risk_cost(actor_class=ActorClass.VEHICLE, normal_closing_speed_mps=speed)
        assert pedestrian > cyclist > vehicle


# --- REQ-R1-05 / REQ-R1-06: mapping and absence of fallbacks ---


def test_static_collidable_uses_car_driver_curve() -> None:
    """AC-R1-06: the two classes share one curve (specification §4.4)."""

    for speed in (0.0, 7.5, 25.0):
        assert injury_risk_cost(
            actor_class=ActorClass.STATIC_COLLIDABLE, normal_closing_speed_mps=speed
        ) == pytest.approx(
            injury_risk_cost(actor_class=ActorClass.VEHICLE, normal_closing_speed_mps=speed),
            abs=1e-12,
        )
    assert (
        model_for(ActorClass.STATIC_COLLIDABLE).curve_id
        == model_for(ActorClass.VEHICLE).curve_id
        == "car_driver_mais3f"
    )


def test_unmapped_actor_class_is_fatal() -> None:
    """AC-R1-07: no default curve for an unmapped class."""

    with pytest.raises(ValueError, match="no MAIS3\\+F injury-risk curve"):
        injury_risk_cost(
            actor_class=ActorClass.INFRASTRUCTURE_NON_COLLIDABLE,
            normal_closing_speed_mps=5.0,
        )


def test_invalid_closing_speed_is_rejected() -> None:
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="finite and non-negative"):
            injury_risk_cost(actor_class=ActorClass.VEHICLE, normal_closing_speed_mps=bad)


def test_collision_rejects_unmapped_actor_class() -> None:
    """AC-R1-07 through the evaluator: fatal, surfaced as a rulebook error."""

    with pytest.raises(RulebookEvaluationError, match="not collidable"):
        evaluate_collision_impact(
            scenario_id="scenario",
            step_index=1,
            ego_configured_speed_cap_mps=10.0,
            pre_ego=_ego((5.0, 0.0), 10.0),
            pre_actors_by_id={"sign": _actor("sign", ActorClass.INFRASTRUCTURE_NON_COLLIDABLE)},
            onset_records=(ContactOnsetRecord("sign", ActorClass.VEHICLE),),
            previous_contact_ids=frozenset(),
            post_active_contact_ids=frozenset({"sign"}),
        )


# --- Diagnostics contract (specification §6) ---


def test_diagnostics_expose_curve_identity_and_closing_speed() -> None:
    result, _, _ = evaluate_collision_impact(
        scenario_id="scenario",
        step_index=1,
        ego_configured_speed_cap_mps=10.0,
        pre_ego=_ego((6.0, 0.0), 10.0),
        pre_actors_by_id={"walker": _actor("walker", ActorClass.PEDESTRIAN, cap=None)},
        onset_records=(ContactOnsetRecord("walker", ActorClass.VEHICLE),),
        previous_contact_ids=frozenset(),
        post_active_contact_ids=frozenset({"walker"}),
    )

    assert result.raw["worst_closing_speed_mps"] == pytest.approx(6.0)
    assert "worst_raw_closing_speed_squared" not in result.raw
    row = result.raw["actors"][0]
    assert row["actor_class"] == "pedestrian"
    assert row["injury_risk_curve"] == "pedestrian_mais3f"
    assert row["closing_speed_mps"] == pytest.approx(6.0)
    assert "raw_speed_squared" not in row
    assert result.diagnostics["injury_risk_model"] == {
        "severity": "MAIS3+F",
        "age_years": REFERENCE_AGE_YEARS,
        "source": "Lubbe et al. (2022), Traffic Safety Research 2:000006",
    }
    # The pedestrian curve must dominate the vehicle curve at the same speed.
    assert result.cost > injury_risk_cost(
        actor_class=ActorClass.VEHICLE, normal_closing_speed_mps=6.0
    )
