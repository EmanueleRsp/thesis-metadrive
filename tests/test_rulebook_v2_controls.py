import pytest
from shapely.geometry import LineString, Polygon

from thesis_rl.rulebook.v2.geometry.controls import (
    ControlLineOffRouteError,
    derive_control_line,
)
from thesis_rl.rulebook.v2.geometry.lanes import RouteLaneRecord
from thesis_rl.rulebook.v2.geometry.route import GEOMETRY_EPSILON_M, RoutePolyline
from thesis_rl.rulebook.v2.transition import _selected_control

from thesis_rl.rulebook.v2.components.controls import (
    evaluate_signal_state,
    evaluate_stop,
    select_active_signal_group,
    signal_group_state,
)
from thesis_rl.rulebook.v2.types import (
    ApproachControl,
    ComponentStatus,
    MovementKey,
    TrafficControlRecord,
)


def _control(group, s, ids=("a",)):
    return TrafficControlRecord(
        group,
        ApproachControl.SIGNAL,
        ("lane",),
        MovementKey("a", "n", "e"),
        LineString(((s, -1), (s, 1))),
        s,
        0.0,
        ids,
    )


def test_signal_selection_is_route_ordered_and_skips_resolved():
    active = select_active_signal_group(
        controls=(_control("far", 20), _control("near", 10)),
        ego_front_s_m=5,
        resolved_group_ids=frozenset({"near"}),
    )
    assert active is not None and active.control_group_id == "far"


def test_signal_group_requires_concordant_physical_states():
    control = _control("g", 10, ("a", "b"))
    assert (
        signal_group_state(control=control, signal_states_by_physical_id={"a": "RED", "b": "RED"})
        == "RED"
    )
    with pytest.raises(ValueError):
        signal_group_state(control=control, signal_states_by_physical_id={"a": "RED", "b": "GREEN"})


def test_signal_component_not_applicable_without_group():
    result, delta, _ = evaluate_signal_state(
        control=None,
        ego_front_s_m=0,
        signal_states_by_physical_id={},
        resolved_group_ids=frozenset(),
    )
    assert result.status.value == "not_applicable" and delta.writes == ()


def _stop_control(group_id: str, route_s: float = 10.0):
    return TrafficControlRecord(
        group_id,
        ApproachControl.STOP,
        ("lane",),
        MovementKey("a", "n", "e"),
        LineString(((route_s, -1), (route_s, 1))),
        route_s,
        0.0,
        (),
    )


def test_stop_dwell_state_is_not_inherited_from_a_different_stop_group() -> None:
    """TEST-EF-16 / REQ-EF-09.

    Regression: ``previous_group_id`` was accepted and never used, so a second
    stop sign inherited the standstill performed at the first one and could be
    crossed at 8 m/s for zero cost.
    """

    result, _, _ = evaluate_stop(
        control=_stop_control("stop-b"),
        pre_delta_m=0.1,
        post_delta_m=-0.1,
        speed_mps=8.0,
        previous_continuous_s=1.5,
        previous_best_s=1.5,
        delta_t_s=0.1,
        previous_group_id="stop-a",
        resolved_group_ids=frozenset(),
        crossing=True,
    )
    assert result.cost == pytest.approx(1.0)
    assert result.status is ComponentStatus.VIOLATED
    assert result.raw["best_timer_s"] == pytest.approx(0.0)


def test_stop_dwell_state_is_kept_within_the_same_stop_group() -> None:
    """TEST-EF-16 / REQ-EF-09: the reset is scoped to a group change only."""

    result, _, _ = evaluate_stop(
        control=_stop_control("stop-a"),
        pre_delta_m=0.1,
        post_delta_m=-0.1,
        speed_mps=8.0,
        previous_continuous_s=1.5,
        previous_best_s=1.5,
        delta_t_s=0.1,
        previous_group_id="stop-a",
        resolved_group_ids=frozenset(),
        crossing=True,
    )
    assert result.cost == pytest.approx(0.0)
    assert result.status is ComponentStatus.SATISFIED


def test_control_route_s_is_a_canonical_route_coordinate_not_a_lane_local_one() -> None:
    """TEST-EF-12 / REQ-EF-05 (v4.7 2.9.5).

    Regression: ``derive_control_line`` projected the control point on the
    *controlled lane's* centerline, so a control 5 m into the second of two
    10 m route lanes reported ``route_s_m = 5.0``. That value was then compared
    against an ego front bumper measured on the concatenated route, making every
    control not on the first route lane look already passed.
    """

    lane_a = RouteLaneRecord(
        "A",
        Polygon([(0, -2), (10, -2), (10, 2), (0, 2)]),
        RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))),
        ("B",),
    )
    lane_b = RouteLaneRecord(
        "B",
        Polygon([(10, -2), (20, -2), (20, 2), (10, 2)]),
        RoutePolyline(((10.0, 0.0, 0.0), (20.0, 0.0, 0.0))),
    )
    route = RoutePolyline.from_lane_centerlines(
        (lane_a.centerline.points_xyz, lane_b.centerline.points_xyz)
    )

    line = derive_control_line(
        control_point_xy=(15.0, 0.0),
        control_point_z=0.0,
        controlled_lane=lane_b,
        route=route,
    )
    assert line.route_s_m == pytest.approx(15.0, abs=GEOMETRY_EPSILON_M)

    control = TrafficControlRecord(
        "stop-b",
        ApproachControl.STOP,
        ("B",),
        MovementKey("B", "n", "B"),
        line.geometry,
        line.route_s_m,
        0.0,
        (),
    )
    selected = _selected_control(
        (control,), ApproachControl.STOP, 8.0, frozenset(), route_lane_ids=("A", "B")
    )
    assert selected is control


def test_control_line_off_the_route_is_reported_as_off_route_not_as_invalid() -> None:
    """TEST-EF-13 / REQ-EF-05: a control on another approach is not a defect."""

    lane = RouteLaneRecord(
        "OTHER",
        Polygon([(0, 48), (10, 48), (10, 52), (0, 52)]),
        RoutePolyline(((0.0, 50.0, 0.0), (10.0, 50.0, 0.0))),
    )
    route = RoutePolyline(((0.0, 0.0, 0.0), (20.0, 0.0, 0.0)))
    with pytest.raises(ControlLineOffRouteError):
        derive_control_line(
            control_point_xy=(5.0, 50.0),
            control_point_z=0.0,
            controlled_lane=lane,
            route=route,
        )


def _flat_lane_and_route() -> tuple[RouteLaneRecord, RoutePolyline]:
    lane = RouteLaneRecord(
        "A",
        Polygon([(0, -2), (20, -2), (20, 2), (0, 2)]),
        RoutePolyline(((0.0, 0.0, 0.0), (20.0, 0.0, 0.0))),
    )
    return lane, RoutePolyline.from_lane_centerlines((lane.centerline.points_xyz,))


@pytest.mark.parametrize(
    ("control_point_z", "accepted"),
    [
        (0.0, True),  # traffic-light stop point: measured dz == 0 for all samples
        (2.567, True),  # median measured stop-sign mounting height
        (3.668, True),  # tallest measured stop sign; rejected by the old abs(dz) <= 3 test
        (6.5, True),  # overhead gantry, still this road's control
        (12.0, False),  # far above the carriageway: another level, not a mount
        (-2.5, False),  # below this road: an underpass control the old test accepted
    ],
)
def test_control_vertical_check_bounds_mounting_height_instead_of_absolute_offset(
    control_point_z: float, accepted: bool
) -> None:
    """TEST-EF-24 / REQ-EF-16.

    ``STOP_SIGN.position`` is the *physical sign*, mounted above the road, while
    ``TRAFFIC_LIGHT.stop_point`` lies on the carriageway. Measured over 250 Waymo
    scenarios: signals have ``dz == 0.000`` for all 1319 samples; stop signs have
    median ``dz = +2.567 m`` (p05 +1.693, p95 +2.971, max +3.668), and 3.8% exceed
    the 3.0 m grade-separation tolerance. The old symmetric ``abs(dz) <= 3.0``
    therefore compared a pole height against an overpass clearance, and erred in
    both directions: it dropped tall mounts and admitted controls belonging to a
    road *below* this one.
    """

    lane, route = _flat_lane_and_route()
    if accepted:
        line = derive_control_line(
            control_point_xy=(10.0, 0.0),
            control_point_z=control_point_z,
            controlled_lane=lane,
            route=route,
        )
        assert line.route_s_m == pytest.approx(10.0, abs=GEOMETRY_EPSILON_M)
        # The control line lies on the carriageway, so its elevation is the
        # lane's regardless of how high the sign itself is mounted.
        assert line.elevation_m == pytest.approx(0.0)
    else:
        with pytest.raises(ValueError, match="vertically incompatible"):
            derive_control_line(
                control_point_xy=(10.0, 0.0),
                control_point_z=control_point_z,
                controlled_lane=lane,
                route=route,
            )


def test_route_crossing_level_uses_the_control_line_not_the_mounted_sign() -> None:
    """TEST-EF-25 / REQ-EF-16.

    ``_route_curvilinear_crossing_s_m`` filtered crossings by vertical
    compatibility against the raw control-point elevation. On a route whose
    elevation differs from zero, a legitimately mounted sign then leaked its
    mounting height into that comparison and could discard the only crossing on
    the ego's own level, turning the control into a spurious off-route skip.
    """

    lane = RouteLaneRecord(
        "A",
        Polygon([(0, -2), (20, -2), (20, 2), (0, 2)]),
        RoutePolyline(((0.0, 0.0, 8.0), (20.0, 0.0, 8.0))),
    )
    route = RoutePolyline.from_lane_centerlines((lane.centerline.points_xyz,))
    # Sign mounted 2.6 m above a carriageway that is itself 8 m above datum:
    # dz against the lane is +2.6, but the absolute elevations are 10.6 vs 8.0.
    line = derive_control_line(
        control_point_xy=(10.0, 0.0),
        control_point_z=10.6,
        controlled_lane=lane,
        route=route,
    )
    assert line.route_s_m == pytest.approx(10.0, abs=GEOMETRY_EPSILON_M)
    assert line.elevation_m == pytest.approx(8.0)


def test_selected_control_ignores_controls_outside_the_assigned_route() -> None:
    """TEST-EF-14 / REQ-EF-06 (v4.7 2.9.5 "movimento ego pertinente").

    Without the filter a control governing a foreign approach could be selected,
    mask the correct one, or abort the scenario on an UNKNOWN state.
    """

    def control(group_id: str, approach_lane_id: str, route_s: float):
        return TrafficControlRecord(
            group_id,
            ApproachControl.SIGNAL,
            (approach_lane_id,),
            MovementKey(approach_lane_id, "n", "e"),
            LineString(((route_s, -1), (route_s, 1))),
            route_s,
            0.0,
            ("p",),
        )

    foreign = control("foreign", "OTHER_ROAD", 10.0)
    own = control("own", "B", 20.0)
    selected = _selected_control(
        (foreign, own), ApproachControl.SIGNAL, 0.0, frozenset(), route_lane_ids=("A", "B")
    )
    assert selected is own


def test_selected_control_keeps_a_control_on_a_later_route_lane() -> None:
    """TEST-EF-14 / REQ-EF-06.

    The relevant movement is not the ego's *current* one: a signal at the end
    of the next route lane must stay selectable while the ego is still on the
    current lane.
    """

    later = TrafficControlRecord(
        "later",
        ApproachControl.SIGNAL,
        ("B",),
        MovementKey("B", "n", "C"),
        LineString(((19.0, -1), (19.0, 1))),
        19.0,
        0.0,
        ("p",),
    )
    selected = _selected_control(
        (later,), ApproachControl.SIGNAL, 3.0, frozenset(), route_lane_ids=("A", "B", "C")
    )
    assert selected is later
