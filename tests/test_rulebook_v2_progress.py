import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.errors import (
    RuntimeScenarioNotEvaluableError,
    RuntimeScenarioNotEvaluableReason,
)
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box
from thesis_rl.rulebook.v2.components.progress import evaluate_progress, route_outside_fraction
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot


def _ego(x, cap=10):
    return ActorSnapshot(
        "ego",
        ActorClass.VEHICLE,
        (x, 0),
        0,
        0,
        (1, 0),
        Polygon(((x - 0.5, -0.5), (x + 0.5, -0.5), (x + 0.5, 0.5), (x - 0.5, 0.5))),
        "lane",
        cap,
    )


def _ego_xy(x, y, cap=10):
    return ActorSnapshot(
        "ego",
        ActorClass.VEHICLE,
        (x, y),
        0,
        0,
        (1, 0),
        Polygon(((x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5))),
        "lane",
        cap,
    )


def test_progress_returns_raw_delta_and_normalized_margin():
    result, delta, _ = evaluate_progress(
        pre_ego=_ego(1),
        post_ego=_ego(3),
        route=RoutePolyline(((0, 0, 0), (10, 0, 0))),
        previous_route_s_m=1,
        delta_t_s=1,
    )
    assert result.raw["route_delta_m"] == pytest.approx(2)
    assert result.cost == pytest.approx(0.2)
    assert dict(delta.writes)["previous_route_s_m"] == pytest.approx(3)


def test_progress_rejects_discontinuous_memory():
    # DEC-EF-XX (user-directed, session of 2026-07-30): a pre-state
    # discontinuity that survives the OPEN-EF-03 jump-bound preference (no
    # plausible candidate near previous_route_s_m) is treated as a runtime
    # scenario data defect eligible for a single-episode data abort, the same
    # bucket as INVALID_SIGNAL_TRANSITION, rather than a fatal ValueError that
    # kills the whole run. See progress.py for the full rationale.
    with pytest.raises(RuntimeScenarioNotEvaluableError) as excinfo:
        evaluate_progress(
            pre_ego=_ego(1),
            post_ego=_ego(2),
            route=RoutePolyline(((0, 0, 0), (10, 0, 0))),
            previous_route_s_m=5,
            delta_t_s=1,
        )
    assert excinfo.value.reason == RuntimeScenarioNotEvaluableReason.ROUTE_PROJECTION_DISCONTINUOUS
    assert excinfo.value.diagnostics["previous_route_s_m"] == 5
    assert excinfo.value.diagnostics["projected_s_m"] == pytest.approx(1.0)


def test_progress_prefers_continuous_branch_over_closer_far_branch_on_self_intersecting_route():
    """Regression for a live crash: a hairpin route running back near itself
    (a Waymo intersection/on-ramp, in practice) has a return leg that can be
    planar-closer to the ego than the outgoing leg it is actually on. Without
    a jump bound on the pre-state projection too, ``route.project`` picks the
    strictly-closer far branch and ``evaluate_progress`` raises a spurious
    "discontinuous with memory" error even though the continuous candidate
    was available. See OPEN-EF-03 and diag-sac-lite-native-noacl-fast /
    diag-sac-lite-scalar-acl-fast crash logs.
    """
    route = RoutePolyline(((0, 0, 0), (10, 0, 0), (10, -0.06, 0), (0, -0.06, 0)))
    pre_ego = _ego_xy(1, -0.05)
    post_ego = _ego_xy(1.5, -0.05)
    result, delta, _ = evaluate_progress(
        pre_ego=pre_ego, post_ego=post_ego, route=route, previous_route_s_m=1.0, delta_t_s=0.1
    )
    assert result.raw["pre_s_m"] == pytest.approx(1.0, abs=1e-6)
    assert dict(delta.writes)["previous_route_s_m"] == pytest.approx(1.5, abs=1e-2)


def test_route_outside_fraction_measures_corridor_adherence_without_changing_cost() -> None:
    """TEST-EF-22 / REQ-EF-15 (M6a, diagnostics only).

    R4 credits projected longitudinal progress with no lateral cut-off, and the
    R3 off-road surface is the union of *every* map lane, so advancing along a
    legal parallel road scores positive progress at zero off-road cost. This
    measures the frequency of that situation so the corridor gate of DEC-EF-06
    can be decided from data. It must not affect the margin.
    """

    corridor = Polygon([(0.0, -1.75), (100.0, -1.75), (100.0, 1.75), (0.0, 1.75)])
    inside = oriented_bounding_box(
        center_xy=(10.0, 0.0), heading_rad=0.0, length_m=4.515, width_m=1.852
    )
    outside = oriented_bounding_box(
        center_xy=(10.0, 20.0), heading_rad=0.0, length_m=4.515, width_m=1.852
    )
    straddling = oriented_bounding_box(
        center_xy=(10.0, 1.75), heading_rad=0.0, length_m=4.515, width_m=1.852
    )
    assert route_outside_fraction(inside, corridor) == pytest.approx(0.0)
    assert route_outside_fraction(outside, corridor) == pytest.approx(1.0)
    partial = route_outside_fraction(straddling, corridor)
    assert 0.0 < partial < 1.0
    assert route_outside_fraction(inside, None) is None
