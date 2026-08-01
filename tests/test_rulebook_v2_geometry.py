from __future__ import annotations

import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon

from thesis_rl.rulebook.v2.geometry.canonical import (
    CanonicalGeometryError,
    canonical_geometry_wkb,
    stable_geometry_id,
)
from thesis_rl.rulebook.v2.geometry.elevation import PolylineElevation
from thesis_rl.rulebook.v2.geometry.drivable import (
    DrivableLaneRecord,
    _union_selected_surfaces,
    drivable_surface_for_ego,
)
from thesis_rl.rulebook.v2.geometry.controls import derive_control_line
from thesis_rl.rulebook.v2.geometry.continuous_sat import (
    CONSTRAINED_DECOMPOSITION_VERTEX_THRESHOLD,
    deterministic_convex_decomposition,
    predict_occupancy_interval,
)
from thesis_rl.rulebook.v2.geometry.conflict_zones import (
    MovementCorridor,
    attach_route_intervals,
    build_crosswalk_conflict_zone_candidates,
    build_vehicle_conflict_zone_candidates,
    route_interval_for_zone,
    _vertical_overlap_compatible,
    select_first_ahead_or_occupied_zone,
)
from thesis_rl.rulebook.v2.geometry import conflict_zones
from thesis_rl.rulebook.v2.geometry import continuous_sat
from thesis_rl.rulebook.v2.geometry.footprint import (
    front_bumper_segment,
    oriented_bounding_box,
    swept_front_bumper,
)
from thesis_rl.rulebook.v2.geometry.lanes import (
    anchored_frame_extent,
    anchored_lateral_gap,
    tangent_intervals_overlap,
    RouteLaneRecord,
    associate_route_lane,
    bumper_to_bumper_gap,
    derive_lane_movement_key,
    footprint_route_coordinates,
    lateral_edge_to_edge_gap,
    _lane_coverage_polygon,
)
from thesis_rl.rulebook.v2.geometry.route import GEOMETRY_EPSILON_M, RoutePolyline
from thesis_rl.rulebook.v2.geometry.vertical import vertically_compatible_at_xy
from thesis_rl.rulebook.v2.types import MovementKey


def test_lane_movement_key_uses_unique_successor_or_assigned_route() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    unique = RouteLaneRecord(
        "approach",
        Polygon(((-1.0, -2.0), (11.0, -2.0), (11.0, 2.0), (-1.0, 2.0))),
        route,
        ("exit",),
    )
    assert derive_lane_movement_key(unique) is not None
    assert derive_lane_movement_key(unique).exit_lane_id == "exit"

    branching = RouteLaneRecord(
        "approach",
        unique.polygon_xy,
        route,
        ("left", "right"),
    )
    assert derive_lane_movement_key(branching) is None
    resolved = derive_lane_movement_key(
        branching,
        assigned_route_lane_ids=("approach", "right"),
    )
    assert resolved is not None
    assert resolved.exit_lane_id == "right"


def test_lane_association_reuses_the_exact_tolerance_envelope() -> None:
    polygon = Polygon(((0.0, -1.0), (10.0, -1.0), (10.0, 1.0), (0.0, 1.0)))
    _lane_coverage_polygon.cache_clear()
    first = _lane_coverage_polygon(polygon)
    second = _lane_coverage_polygon(polygon)
    assert first.equals(second)
    assert _lane_coverage_polygon.cache_info().hits == 1


def test_lane_association_keeps_points_inside_the_frozen_buffered_bounds() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    lane = RouteLaneRecord(
        "lane-a",
        Polygon(((0.0, -1.0), (10.0, -1.0), (10.0, 1.0), (0.0, 1.0))),
        route,
    )

    association = associate_route_lane(
        position_xy=(10.0 + GEOMETRY_EPSILON_M * 0.5, 0.0),
        position_z=0.0,
        heading_rad=0.0,
        route_lanes=(lane,),
    )

    assert association is not None
    assert association.lane_id == "lane-a"


def test_canonical_wkb_and_synthetic_id_ignore_ring_orientation_and_small_noise() -> None:
    clockwise = Polygon(((0.0004, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0004, 0.0)))
    counterclockwise = Polygon(((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)))
    assert canonical_geometry_wkb(clockwise) == canonical_geometry_wkb(counterclockwise)
    assert stable_geometry_id(
        scenario_id="s", namespace="map", feature_type="lane", geometry=clockwise
    ) == stable_geometry_id(
        scenario_id="s", namespace="map", feature_type="lane", geometry=counterclockwise
    )


def test_canonical_geometry_rejects_collapsed_or_invalid_input() -> None:
    with pytest.raises(CanonicalGeometryError, match="collapsed"):
        canonical_geometry_wkb(LineString(((0.0, 0.0), (0.0004, 0.0))))
    with pytest.raises(CanonicalGeometryError, match="invalid"):
        canonical_geometry_wkb(
            Polygon(((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
        )


def test_vertical_compatibility_does_not_match_overpass_to_underpass() -> None:
    assert vertically_compatible_at_xy(
        first_elevation_at_xy=lambda _x, _y: 10.0,
        second_elevation_at_xy=lambda _x, _y: 12.9,
        x=0.0,
        y=0.0,
    )
    assert not vertically_compatible_at_xy(
        first_elevation_at_xy=lambda _x, _y: 10.0,
        second_elevation_at_xy=lambda _x, _y: 13.1,
        x=0.0,
        y=0.0,
    )


def test_invalid_elevation_is_fail_fast() -> None:
    with pytest.raises(ValueError, match="finite"):
        vertically_compatible_at_xy(
            first_elevation_at_xy=lambda _x, _y: float("nan"),
            second_elevation_at_xy=lambda _x, _y: 0.0,
            x=0.0,
            y=0.0,
        )


def test_oriented_bounding_box_uses_pose_heading_and_strict_dimensions() -> None:
    footprint = oriented_bounding_box(
        center_xy=(0.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0
    )
    assert footprint.bounds == pytest.approx((-2.0, -1.0, 2.0, 1.0))
    with pytest.raises(ValueError, match="strictly positive"):
        oriented_bounding_box(center_xy=(0.0, 0.0), heading_rad=0.0, length_m=0.0, width_m=2.0)


def test_front_bumper_and_swept_region_are_canonical() -> None:
    pre = oriented_bounding_box(center_xy=(0.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    post = oriented_bounding_box(center_xy=(2.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    assert front_bumper_segment(pre, heading_rad=0.0).bounds == pytest.approx((2.0, -1.0, 2.0, 1.0))
    assert swept_front_bumper(
        pre, post, pre_heading_rad=0.0, post_heading_rad=0.0
    ).bounds == pytest.approx((2.0, -1.0, 4.0, 1.0))


def test_elevation_profile_interpolates_and_breaks_nearest_segment_ties_by_index() -> None:
    elevation = PolylineElevation(((0.0, 0.0, 0.0), (10.0, 0.0, 10.0), (10.0, 10.0, 30.0)))
    assert elevation(5.0, 1.0) == pytest.approx(5.0)
    assert elevation(10.0, 0.0) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="duplicate XY"):
        PolylineElevation(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))


def test_route_projection_uses_reset_and_previous_s_tie_breaks_at_self_intersection() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0), (2.0, 0.0, 0.0)))
    at_reset = route.project((1.0, 1.0))
    near_final_crossing = route.project((1.0, 1.0), previous_s_m=route.length_m - 1.0)
    assert at_reset.segment_index == 0
    assert near_final_crossing.segment_index == 2


def test_route_point_at_interpolates_inside_the_containing_segment() -> None:
    """TEST-EF-01 / REQ-EF-01.

    Regression for the nearest-endpoint segment selection, which returned the
    preceding vertex for every arc length in the first half of a segment:
    ``point_at(14.0)`` used to return ``(10.0, 0.0, 0.0)``.
    """

    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 1.0), (20.0, 0.0, 2.0)))
    assert route.point_at(4.0) == pytest.approx((4.0, 0.0, 0.4))
    assert route.point_at(6.0) == pytest.approx((6.0, 0.0, 0.6))
    assert route.point_at(14.0) == pytest.approx((14.0, 0.0, 1.4))
    assert route.point_at(16.0) == pytest.approx((16.0, 0.0, 1.6))


def test_route_point_at_is_exact_at_vertices_and_clamps_outside_the_route() -> None:
    """TEST-EF-02 / REQ-EF-01."""

    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 1.0), (20.0, 0.0, 2.0)))
    assert route.point_at(0.0) == pytest.approx((0.0, 0.0, 0.0))
    assert route.point_at(10.0) == pytest.approx((10.0, 0.0, 1.0))
    assert route.point_at(20.0) == pytest.approx((20.0, 0.0, 2.0))
    assert route.point_at(-5.0) == pytest.approx((0.0, 0.0, 0.0))
    assert route.point_at(25.0) == pytest.approx((20.0, 0.0, 2.0))
    with pytest.raises(ValueError, match="finite"):
        route.point_at(float("nan"))


def test_route_point_at_has_no_sawtooth_on_a_finely_sampled_polyline() -> None:
    """TEST-EF-03 / REQ-EF-01.

    MetaDrive exports lane polylines with ``interval=2`` m, which is what made
    the historical error a sawtooth of up to 1 m in the policy's route
    waypoints rather than a single isolated wrong value.
    """

    route = RoutePolyline(tuple((float(x), 0.0, 0.0) for x in range(0, 101, 2)))
    for step in range(0, 1001):
        target = step / 10.0
        assert route.point_at(target)[0] == pytest.approx(target, abs=1.0e-9)


def test_route_consolidates_noisy_xy_points_with_median_elevation_and_rejects_large_spread() -> (
    None
):
    route = RoutePolyline(((0.0, 0.0, 0.0), (0.0005, 0.0005, 2.0), (10.0, 0.0, 4.0)))
    assert route.points_xyz[0] == pytest.approx((0.00025, 0.00025, 1.0))
    with pytest.raises(ValueError, match="incompatible"):
        RoutePolyline(((0.0, 0.0, 0.0), (0.0005, 0.0, 3.1), (10.0, 0.0, 4.0)))


def test_route_projection_rejects_incompatible_vertical_level() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="vertically compatible"):
        route.project((5.0, 0.0), position_z=3.1)


def test_route_projection_diagnostics_report_nearest_and_vertical_feasibility() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 6.0)))
    diagnostics = route.projection_diagnostics((10.0, 0.0), position_z=2.9)
    assert diagnostics.nearest_planar_segment_index == 0
    assert diagnostics.nearest_planar_s_m == pytest.approx(10.0)
    assert diagnostics.nearest_planar_z_m == pytest.approx(6.0)
    assert diagnostics.nearest_planar_distance_m == pytest.approx(0.0)
    assert diagnostics.minimum_vertical_difference_m == pytest.approx(3.1)
    assert diagnostics.vertically_compatible_segment_count == 0
    assert diagnostics.route_min_z_m == pytest.approx(0.0)
    assert diagnostics.route_max_z_m == pytest.approx(6.0)


def test_lane_association_rejects_vertical_and_geometric_ties() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    lane = RouteLaneRecord(
        "lane-a", Polygon(((0.0, -2.0), (10.0, -2.0), (10.0, 2.0), (0.0, 2.0))), route
    )
    assert (
        associate_route_lane(
            position_xy=(5.0, 0.0), position_z=0.0, heading_rad=0.0, route_lanes=(lane,)
        ).lane_id
        == "lane-a"
    )
    assert (
        associate_route_lane(
            position_xy=(5.0, 0.0), position_z=3.1, heading_rad=0.0, route_lanes=(lane,)
        )
        is None
    )
    tied = RouteLaneRecord("lane-b", lane.polygon_xy, route)
    assert (
        associate_route_lane(
            position_xy=(5.0, 0.0), position_z=0.0, heading_rad=0.0, route_lanes=(lane, tied)
        )
        is None
    )


def test_bumper_gap_handles_separated_tangent_and_overlapping_footprints() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (30.0, 0.0, 0.0)))
    ego = footprint_route_coordinates(
        oriented_bounding_box(center_xy=(5.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        route,
        position_z=0.0,
    )
    other = footprint_route_coordinates(
        oriented_bounding_box(center_xy=(12.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        route,
        position_z=0.0,
    )
    assert bumper_to_bumper_gap(ego, other) == pytest.approx((3.0, True))
    tangent = footprint_route_coordinates(
        oriented_bounding_box(center_xy=(9.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        route,
        position_z=0.0,
    )
    assert bumper_to_bumper_gap(ego, tangent) == pytest.approx((0.0, True))


def test_lateral_edge_to_edge_gap_and_side_on_shared_route_frame() -> None:
    """rulebook v4.8 §7 ``d_i^lat``: edge-to-edge gap on the route normal
    axis, mirroring ``bumper_to_bumper_gap`` on the tangent axis."""
    route = RoutePolyline(((0.0, 0.0, 0.0), (30.0, 0.0, 0.0)))
    ego = footprint_route_coordinates(
        oriented_bounding_box(center_xy=(5.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        route,
        position_z=0.0,
    )
    left = footprint_route_coordinates(
        oriented_bounding_box(center_xy=(5.0, 3.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        route,
        position_z=0.0,
    )
    assert lateral_edge_to_edge_gap(ego, left) == pytest.approx((1.0, 1))
    assert lateral_edge_to_edge_gap(left, ego) == pytest.approx((1.0, -1))
    overlapping = footprint_route_coordinates(
        oriented_bounding_box(center_xy=(5.0, 0.5), heading_rad=0.0, length_m=4.0, width_m=2.0),
        route,
        position_z=0.0,
    )
    gap, _ = lateral_edge_to_edge_gap(ego, overlapping)
    assert gap == pytest.approx(0.0)


def test_anchored_frame_extent_measures_both_footprints_on_one_shared_axis() -> None:
    """TEST-EF-11 / REQ-EF-04 (rulebook v4.8 §3, ADR-035).

    ``footprint_route_coordinates`` lets every vertex select its own nearest
    route segment, so two extents compared as if homogeneous are only valid on
    a straight route. The anchored frame is a scalar product against a single
    ``(tangent, normal)`` pair, which is what the lateral metric requires.
    """

    origin = (5.0, 0.0)
    tangent = (1.0, 0.0)
    ego = anchored_frame_extent(
        oriented_bounding_box(center_xy=(5.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        origin_xy=origin,
        tangent_xy=tangent,
    )
    left = anchored_frame_extent(
        oriented_bounding_box(center_xy=(5.0, 3.0), heading_rad=0.0, length_m=4.0, width_m=2.0),
        origin_xy=origin,
        tangent_xy=tangent,
    )
    assert ego.tangent_min_m == pytest.approx(-2.0)
    assert ego.tangent_max_m == pytest.approx(2.0)
    assert ego.normal_min_m == pytest.approx(-1.0)
    assert ego.normal_max_m == pytest.approx(1.0)
    assert anchored_lateral_gap(ego, left) == pytest.approx((1.0, 1))
    assert anchored_lateral_gap(left, ego) == pytest.approx((1.0, -1))
    assert tangent_intervals_overlap(ego, left) is True


def test_tangent_intervals_overlap_separates_abreast_from_leading_pairs() -> None:
    """TEST-EF-11 / REQ-EF-02: the ADR-035 applicability predicate."""

    origin = (0.0, 0.0)
    tangent = (1.0, 0.0)

    def extent(center_x: float, center_y: float):
        return anchored_frame_extent(
            oriented_bounding_box(
                center_xy=(center_x, center_y), heading_rad=0.0, length_m=4.0, width_m=2.0
            ),
            origin_xy=origin,
            tangent_xy=tangent,
        )

    ego = extent(0.0, 0.0)
    assert tangent_intervals_overlap(ego, extent(0.0, 3.0)) is True
    assert tangent_intervals_overlap(ego, extent(3.0, 3.0)) is True
    assert tangent_intervals_overlap(ego, extent(10.0, 0.0)) is False
    assert tangent_intervals_overlap(ego, extent(-10.0, 0.0)) is False


def test_drivable_surface_uses_current_vertical_layer_and_width_fallback() -> None:
    lower = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    upper = RoutePolyline(((0.0, 0.0, 10.0), (10.0, 0.0, 10.0)))
    ego = oriented_bounding_box(center_xy=(5.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    lanes = (
        DrivableLaneRecord("lower", lower, None, 4.0),
        DrivableLaneRecord("upper", upper, None, 4.0),
    )
    surface = drivable_surface_for_ego(
        ego_footprint=ego,
        ego_position_xy=(5.0, 0.0),
        ego_position_z=0.0,
        lanes=lanes,
    )
    assert surface.bounds == pytest.approx((0.0, -2.0, 10.0, 2.0))
    with pytest.raises(ValueError, match="lane_width"):
        DrivableLaneRecord("broken", lower, None, None)


def test_drivable_surface_remains_available_when_ego_is_off_lane() -> None:
    lane = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    ego = oriented_bounding_box(center_xy=(50.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    surface = drivable_surface_for_ego(
        ego_footprint=ego,
        ego_position_xy=(50.0, 0.0),
        ego_position_z=0.0,
        lanes=(DrivableLaneRecord("lane", lane, None, 4.0),),
    )
    assert not surface.is_empty


def test_drivable_surface_reuses_exact_union_for_same_lane_layer() -> None:
    _union_selected_surfaces.cache_clear()
    lane = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    ego = oriented_bounding_box(center_xy=(5.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0)
    lanes = (DrivableLaneRecord("lane", lane, None, 4.0),)

    first = drivable_surface_for_ego(
        ego_footprint=ego, ego_position_xy=(5.0, 0.0), ego_position_z=0.0, lanes=lanes
    )
    second = drivable_surface_for_ego(
        ego_footprint=ego, ego_position_xy=(5.5, 0.0), ego_position_z=0.0, lanes=lanes
    )

    assert first.equals(second)
    assert _union_selected_surfaces.cache_info().hits == 1


def test_drivable_surface_closes_lane_seams() -> None:
    """TEST-RBCOST-001/002: adjacent lane polygons with a hairline gap at their
    shared edge must union into a surface with no interior holes, and a
    footprint centered on the seam must not be off-road.

    REQ-RBCOST-001.
    """
    from shapely.geometry import Polygon as _Polygon

    left = _Polygon(((0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (0.0, 2.0)))
    # Right lane's shared edge is offset by 1e-3 m, well under the 0.10 m
    # closing radius but enough to leave a sliver in a raw union.
    right = _Polygon(((5.0 + 1e-3, 0.0), (10.0, 0.0), (10.0, 2.0), (5.0 + 1e-3, 2.0)))
    left_route = RoutePolyline(((0.0, 1.0, 0.0), (5.0, 1.0, 0.0)))
    right_route = RoutePolyline(((5.0, 1.0, 0.0), (10.0, 1.0, 0.0)))
    lanes = (
        DrivableLaneRecord("left", left_route, left, None),
        DrivableLaneRecord("right", right_route, right, None),
    )
    ego = oriented_bounding_box(center_xy=(5.0, 1.0), heading_rad=0.0, length_m=1.0, width_m=1.0)
    _union_selected_surfaces.cache_clear()
    surface = drivable_surface_for_ego(
        ego_footprint=ego, ego_position_xy=(5.0, 1.0), ego_position_z=0.0, lanes=lanes
    )
    assert surface.difference(ego).area >= 0.0
    assert ego.difference(surface).area < 1e-6


def test_drivable_surface_closing_is_area_monotone_and_covers_the_raw_union() -> None:
    """TEST-RBCOST-003: closing can only add surface, never remove it."""
    from shapely.geometry import Polygon as _Polygon

    lane_a = _Polygon(((0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (0.0, 2.0)))
    lane_b = _Polygon(
        ((5.0 + 2e-3, 0.0), (10.0, 0.0), (10.0, 2.0 - 3e-3), (5.0 + 2e-3, 2.0 - 3e-3))
    )
    route = RoutePolyline(((0.0, 1.0, 0.0), (10.0, 1.0, 0.0)))
    lanes = (
        DrivableLaneRecord("a", route, lane_a, None),
        DrivableLaneRecord("b", route, lane_b, None),
    )
    ego = oriented_bounding_box(center_xy=(0.0, 0.0), heading_rad=0.0, length_m=0.1, width_m=0.1)
    _union_selected_surfaces.cache_clear()
    closed = drivable_surface_for_ego(
        ego_footprint=ego, ego_position_xy=(0.0, 0.0), ego_position_z=0.0, lanes=lanes
    )
    raw = shapely.union_all([lane_a, lane_b])
    assert closed.area >= raw.area - 1e-9
    # Buffer round-tripping approximates curves with straight segments, so the
    # closed boundary can diverge from the raw one by a negligible polygonal
    # discretization error at corners; it must stay far below the seam sizes
    # (1e-4..5e-4 m^2) this closing exists to remove.
    assert raw.difference(closed).area < 1e-5


def test_drivable_surface_genuine_gap_still_reports_off_road() -> None:
    """TEST-RBCOST-005: a real gap wider than the closing radius must still
    produce off-road area outside the surface -- the closing must not bridge
    genuine map-edge or inter-lane gaps.
    """
    from shapely.geometry import Polygon as _Polygon

    lane = _Polygon(((0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (0.0, 2.0)))
    route = RoutePolyline(((0.0, 1.0, 0.0), (5.0, 1.0, 0.0)))
    lanes = (DrivableLaneRecord("lane", route, lane, None),)
    # Footprint half outside the outermost (only) lane, well past the closing
    # radius from the drivable edge.
    ego = oriented_bounding_box(center_xy=(5.0, 1.0), heading_rad=0.0, length_m=2.0, width_m=2.0)
    _union_selected_surfaces.cache_clear()
    surface = drivable_surface_for_ego(
        ego_footprint=ego, ego_position_xy=(5.0, 1.0), ego_position_z=0.0, lanes=lanes
    )
    outside_ratio = ego.difference(surface).area / ego.area
    assert outside_ratio == pytest.approx(0.5, abs=0.05)


def test_drivable_surface_union_is_order_independent() -> None:
    """TEST-RBCOST-021: the same lane set in a different order yields the same
    closed surface (determinism, no fold-order dependence).
    """
    from shapely.geometry import Polygon as _Polygon

    a = _Polygon(((0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (0.0, 2.0)))
    b = _Polygon(((5.0 + 1e-3, 0.0), (10.0, 0.0), (10.0, 2.0), (5.0 + 1e-3, 2.0)))
    c = _Polygon(((10.0 + 1e-3, 0.0), (15.0, 0.0), (15.0, 2.0), (10.0 + 1e-3, 2.0)))
    route = RoutePolyline(((0.0, 1.0, 0.0), (15.0, 1.0, 0.0)))
    forward = (
        DrivableLaneRecord("a", route, a, None),
        DrivableLaneRecord("b", route, b, None),
        DrivableLaneRecord("c", route, c, None),
    )
    reversed_lanes = tuple(reversed(forward))
    ego = oriented_bounding_box(center_xy=(7.5, 1.0), heading_rad=0.0, length_m=1.0, width_m=1.0)
    _union_selected_surfaces.cache_clear()
    forward_surface = drivable_surface_for_ego(
        ego_footprint=ego, ego_position_xy=(7.5, 1.0), ego_position_z=0.0, lanes=forward
    )
    _union_selected_surfaces.cache_clear()
    reversed_surface = drivable_surface_for_ego(
        ego_footprint=ego, ego_position_xy=(7.5, 1.0), ego_position_z=0.0, lanes=reversed_lanes
    )
    assert forward_surface.symmetric_difference(reversed_surface).area < 1e-9


def test_derived_control_line_is_orthogonal_and_signed_upstream_positive() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    lane = RouteLaneRecord(
        "lane", Polygon(((0.0, -2.0), (10.0, -2.0), (10.0, 2.0), (0.0, 2.0))), route
    )
    control = derive_control_line(
        control_point_xy=(5.0, 0.0),
        control_point_z=0.0,
        controlled_lane=lane,
        route=lane.centerline,
    )
    assert control.geometry.bounds == pytest.approx((5.0, -2.0, 5.0, 2.0))
    assert control.route_s_m == pytest.approx(5.0)
    assert control.signed_distance_m((4.0, 0.0)) == pytest.approx(1.0)
    assert control.signed_distance_m((6.0, 0.0)) == pytest.approx(-1.0)


def test_conflict_zone_candidates_are_wkb_ordered_stable_and_vertical_filtered() -> None:
    ego = MovementCorridor(
        MovementKey("ego-in", "node", "ego-out"),
        Polygon(((0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (0.0, 2.0))),
        lambda _x, _y: 0.0,
    )
    other = MovementCorridor(
        MovementKey("other-in", "node", "other-out"),
        Polygon(((1.0, -1.0), (2.0, -1.0), (2.0, 3.0), (1.0, 3.0))),
        lambda _x, _y: 0.0,
    )
    candidates = build_vehicle_conflict_zone_candidates(
        scenario_id="scenario", ego_corridor=ego, other_corridor=other
    )
    repeated = build_vehicle_conflict_zone_candidates(
        scenario_id="scenario", ego_corridor=ego, other_corridor=other
    )
    assert len(candidates) == 1
    assert candidates[0].zone_id == repeated[0].zone_id
    assert candidates[0].component_index == 0
    elevated_other = MovementCorridor(other.movement_key, other.polygon, lambda _x, _y: 3.1)
    assert not build_vehicle_conflict_zone_candidates(
        scenario_id="scenario", ego_corridor=ego, other_corridor=elevated_other
    )


def test_conflict_zone_filters_a_collapsed_derived_component_but_not_source_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-boolean sliver is no zone; invalid source corridors still fail."""

    ego = MovementCorridor(
        MovementKey("ego-in", "node", "ego-out"),
        Polygon(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))),
        lambda _x, _y: 0.0,
    )
    other = MovementCorridor(
        MovementKey("other-in", "node", "other-out"),
        Polygon(((1.0, -1.0), (2.0, -1.0), (2.0, 3.0), (1.0, 3.0))),
        lambda _x, _y: 0.0,
    )
    original = conflict_zones.canonicalize_geometry
    calls = 0

    def collapse_only_derived(geometry: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise CanonicalGeometryError("Geometry collapsed to empty after precision snapping.")
        return original(geometry)  # type: ignore[arg-type]

    monkeypatch.setattr(conflict_zones, "canonicalize_geometry", collapse_only_derived)
    assert (
        build_vehicle_conflict_zone_candidates(
            scenario_id="scenario", ego_corridor=ego, other_corridor=other
        )
        == ()
    )

    with pytest.raises(CanonicalGeometryError, match="empty"):
        build_vehicle_conflict_zone_candidates(
            scenario_id="scenario",
            ego_corridor=MovementCorridor(
                ego.movement_key,
                Polygon(),
                ego.elevation_at_xy,
            ),
            other_corridor=other,
        )


def test_vertical_overlap_accepts_repaired_multipolygon() -> None:
    geometry = shapely.MultiPolygon(
        (
            Polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
            Polygon(((2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0))),
        )
    )
    assert _vertical_overlap_compatible(geometry, lambda _x, _y: 0.0, lambda _x, _y: 0.0)


def test_conflict_zone_selection_prefers_occupied_then_first_ahead_along_route() -> None:
    route = RoutePolyline(((0.0, 1.0, 0.0), (10.0, 1.0, 0.0)))
    ego_corridor = MovementCorridor(
        MovementKey("ego-in", "node", "ego-out"),
        Polygon(((0.0, 0.0), (10.0, 0.0), (10.0, 2.0), (0.0, 2.0))),
        lambda _x, _y: 0.0,
    )
    other_corridor = MovementCorridor(
        MovementKey("other-in", "node", "other-out"),
        shapely.MultiPolygon(
            (
                Polygon(((1.0, -1.0), (2.0, -1.0), (2.0, 3.0), (1.0, 3.0))),
                Polygon(((5.0, -1.0), (6.0, -1.0), (6.0, 3.0), (5.0, 3.0))),
            )
        ),
        lambda _x, _y: 0.0,
    )
    attached = attach_route_intervals(
        route=route,
        candidates=build_vehicle_conflict_zone_candidates(
            scenario_id="scenario", ego_corridor=ego_corridor, other_corridor=other_corridor
        ),
    )
    ahead = select_first_ahead_or_occupied_zone(
        candidates=attached,
        ego_footprint=oriented_bounding_box(
            center_xy=(0.0, 1.0), heading_rad=0.0, length_m=0.5, width_m=0.5
        ),
        ego_front_s_m=0.25,
    )
    assert ahead is not None
    assert ahead.route_entry_s_m == pytest.approx(0.99)
    occupied = select_first_ahead_or_occupied_zone(
        candidates=attached,
        ego_footprint=oriented_bounding_box(
            center_xy=(5.5, 1.0), heading_rad=0.0, length_m=0.5, width_m=0.5
        ),
        ego_front_s_m=5.75,
    )
    assert occupied is not None
    assert occupied.route_entry_s_m == pytest.approx(4.99)


def test_route_interval_skips_far_segments_without_changing_interval() -> None:
    route = RoutePolyline(
        tuple((float(index), 100.0, 0.0) for index in range(1000))
        + ((1000.0, 1.0, 0.0), (1001.0, 1.0, 0.0))
    )
    polygon = Polygon(((1000.2, 0.0), (1000.8, 0.0), (1000.8, 2.0), (1000.2, 2.0)))
    interval = route_interval_for_zone(route=route, polygon=polygon)
    assert interval is not None
    assert interval[0] == pytest.approx(route._segment_starts_m[-1] + 0.19)
    assert interval[1] == pytest.approx(route._segment_starts_m[-1] + 0.81)


def test_large_polygon_uses_exact_constrained_decomposition() -> None:
    polygon = Point(0.0, 0.0).buffer(10.0, quad_segs=CONSTRAINED_DECOMPOSITION_VERTEX_THRESHOLD)
    components = deterministic_convex_decomposition(polygon)
    covered = shapely.union_all(components)
    assert polygon.covers(covered)
    assert polygon.symmetric_difference(covered).area <= 1.0e-4


def test_vehicle_conflict_bounds_rejects_disjoint_canonical_corridors() -> None:
    ego = MovementCorridor(
        MovementKey("ego-in", "node", "ego-out"),
        Polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        lambda _x, _y: 0.0,
    )
    other = MovementCorridor(
        MovementKey("other-in", "node", "other-out"),
        Polygon(((10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0))),
        lambda _x, _y: 0.0,
    )
    assert (
        build_vehicle_conflict_zone_candidates(
            scenario_id="scenario", ego_corridor=ego, other_corridor=other
        )
        == ()
    )


def test_occupancy_bounds_rejects_unreachable_zone_before_sat(monkeypatch) -> None:
    actor = Polygon(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)))
    zone = Polygon(((20.0, 0.0), (21.0, 0.0), (21.0, 1.0), (20.0, 1.0)))
    calls = 0

    def count_sat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(continuous_sat, "_sat_interval", count_sat)
    assert (
        predict_occupancy_interval(
            actor_footprint=actor,
            actor_velocity_xy=(1.0, 0.0),
            zone=zone,
            horizon_s=3.0,
        )
        is None
    )
    assert calls == 0


def test_crosswalk_zone_uses_its_own_namespace_and_no_other_movement_key() -> None:
    ego = MovementCorridor(
        MovementKey("ego-in", "node", "ego-out"),
        Polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0))),
        lambda _x, _y: 0.0,
    )
    zones = build_crosswalk_conflict_zone_candidates(
        scenario_id="scenario",
        ego_corridor=ego,
        crosswalk_id="crosswalk-1",
        crosswalk_polygon=Polygon(((1.0, -1.0), (3.0, -1.0), (3.0, 3.0), (1.0, 3.0))),
        crosswalk_elevation_at_xy=lambda _x, _y: 0.0,
    )
    assert len(zones) == 1
    assert zones[0].crosswalk_id == "crosswalk-1"
    assert not build_crosswalk_conflict_zone_candidates(
        scenario_id="scenario",
        ego_corridor=ego,
        crosswalk_id="crosswalk-1",
        crosswalk_polygon=zones[0].polygon,
        crosswalk_elevation_at_xy=lambda _x, _y: 3.1,
    )


def test_continuous_sat_returns_no_interval_finite_interval_and_open_end() -> None:
    zone = Polygon(((0.0, -1.0), (2.0, -1.0), (2.0, 1.0), (0.0, 1.0)))
    crossing = Polygon(((-4.0, -0.5), (-2.0, -0.5), (-2.0, 0.5), (-4.0, 0.5)))
    finite = predict_occupancy_interval(
        actor_footprint=crossing, actor_velocity_xy=(4.0, 0.0), zone=zone, horizon_s=2.0
    )
    assert finite is not None
    assert finite.start_s == pytest.approx(0.5)
    assert finite.end_s == pytest.approx(1.5)
    assert (
        predict_occupancy_interval(
            actor_footprint=crossing, actor_velocity_xy=(0.0, 4.0), zone=zone, horizon_s=2.0
        )
        is None
    )
    inside = Polygon(((0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (0.5, 0.5)))
    open_end = predict_occupancy_interval(
        actor_footprint=inside, actor_velocity_xy=(0.0, 0.0), zone=zone, horizon_s=3.0
    )
    assert open_end is not None
    assert open_end.start_s == 0.0
    assert open_end.is_open_end


def test_deterministic_decomposition_handles_concavity_and_hole() -> None:
    concave = Polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 4.0), (0.0, 4.0)))
    holed = Polygon(
        ((0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)),
        holes=(((1.0, 1.0), (1.0, 4.0), (4.0, 4.0), (4.0, 1.0)),),
    )
    for polygon in (concave, holed):
        triangles = deterministic_convex_decomposition(polygon)
        assert all(polygon.covers(triangle) and triangle.area > 0.0 for triangle in triangles)
        assert shapely.union_all(triangles).area == pytest.approx(polygon.area)


def test_deterministic_decomposition_handles_multiple_holes() -> None:
    """A valid multi-hole zone must not exhaust ears after bridge insertion."""

    polygon = Polygon(
        ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
        holes=(
            ((1.0, 1.0), (1.0, 1.6), (1.6, 1.6), (1.6, 1.0)),
            ((1.0, 6.0), (1.0, 6.6), (1.6, 6.6), (1.6, 6.0)),
        ),
    )

    triangles = deterministic_convex_decomposition(polygon)
    repeated = deterministic_convex_decomposition(polygon)

    assert all(polygon.covers(triangle) and triangle.area > 0.0 for triangle in triangles)
    assert shapely.union_all(triangles).area == pytest.approx(polygon.area)
    assert [triangle.wkb for triangle in triangles] == [triangle.wkb for triangle in repeated]


def test_deterministic_decomposition_recovers_from_single_hole_ear_exhaustion() -> None:
    """A valid concave shell with one hole can exhaust the local ear predicate."""

    polygon = Polygon(
        ((0.0, 0.0), (4.0, 0.0), (8.0, 0.0), (8.0, 8.0), (4.0, 8.0), (0.0, 8.0), (0.0, 4.0)),
        holes=(((3.0, 1.0), (3.0, 7.0), (3.2, 7.0), (3.2, 1.0)),),
    )

    triangles = deterministic_convex_decomposition(polygon)
    repeated = deterministic_convex_decomposition(polygon)

    assert all(polygon.covers(triangle) and triangle.area > 0.0 for triangle in triangles)
    assert shapely.union_all(triangles).area == pytest.approx(polygon.area)
    assert [triangle.wkb for triangle in triangles] == [triangle.wkb for triangle in repeated]


def test_deterministic_decomposition_rejects_invalid_geometry_before_recovery() -> None:
    invalid = Polygon(((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)))

    with pytest.raises(ValueError, match="valid, non-empty polygon"):
        deterministic_convex_decomposition(invalid)


def test_merge_corridors_produce_a_single_stable_conflict_component() -> None:
    ego = MovementCorridor(
        MovementKey("merge-ego", "merge-node", "main-out"),
        Polygon(((0, -1), (6, -1), (6, 1), (0, 1))),
        lambda _x, _y: 0.0,
    )
    merging = MovementCorridor(
        MovementKey("merge-other", "merge-node", "main-out"),
        Polygon(((3, -3), (5, -3), (5, 3), (3, 3))),
        lambda _x, _y: 0.0,
    )
    candidates = build_vehicle_conflict_zone_candidates(
        scenario_id="merge",
        ego_corridor=ego,
        other_corridor=merging,
    )
    assert len(candidates) == 1
    assert candidates[0].component_index == 0


def test_roundabout_corridor_intersection_remains_2_5d_filtered() -> None:
    circulating = MovementCorridor(
        MovementKey("roundabout", "rotary", "exit"),
        Polygon(((-4, -1), (4, -1), (4, 1), (-4, 1))),
        lambda _x, _y: 0.0,
    )
    entering = MovementCorridor(
        MovementKey("entry", "rotary", "roundabout"),
        Polygon(((1, -4), (3, -4), (3, 4), (1, 4))),
        lambda _x, _y: 0.0,
    )
    candidates = build_vehicle_conflict_zone_candidates(
        scenario_id="rotary",
        ego_corridor=circulating,
        other_corridor=entering,
    )
    assert len(candidates) == 1
    elevated_entry = MovementCorridor(entering.movement_key, entering.polygon, lambda _x, _y: 3.1)
    assert (
        build_vehicle_conflict_zone_candidates(
            scenario_id="rotary",
            ego_corridor=circulating,
            other_corridor=elevated_entry,
        )
        == ()
    )


def test_route_projection_continuity_bound_rejects_a_far_branch_jump() -> None:
    """OPEN-EF-03 / `F12`.

    On a route that folds back on itself, a point near the crossing is strictly
    closer to the far branch, so `previous_s_m` — which only breaks ties within
    `eps_geom` of the minimum planar distance — cannot prevent the coordinate
    from jumping and reporting spurious progress. The bound makes the choice
    explicit and fails closed when nothing is plausible.
    """

    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 0.4, 0.0), (0.0, 0.4, 0.0)))
    near_the_fold = (9.0, 0.35)

    # Unbounded: continuity is only a tie-break, so the far branch can win.
    unbounded = route.project(near_the_fold, previous_s_m=9.0)

    # Bounded to a plausible one-step advance from s=9.0.
    bounded = route.project(near_the_fold, previous_s_m=9.0, max_s_jump_m=2.0)
    assert abs(bounded.s_m - 9.0) <= 2.0 + GEOMETRY_EPSILON_M
    assert bounded.s_m <= unbounded.s_m

    # The bound is a preference, not a gate: with nothing plausible the
    # unbounded choice is kept, so a rare geometry never becomes an abort.
    degenerate = route.project(near_the_fold, previous_s_m=0.0, max_s_jump_m=0.01)
    assert degenerate.s_m == pytest.approx(unbounded.s_m)

    with pytest.raises(ValueError, match="requires previous_s_m"):
        route.project(near_the_fold, max_s_jump_m=1.0)
