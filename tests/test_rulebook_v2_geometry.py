from __future__ import annotations

import pytest
from shapely.geometry import LineString, Polygon

from thesis_rl.rulebook.v2.geometry.canonical import (
    CanonicalGeometryError,
    canonical_geometry_wkb,
    stable_geometry_id,
)
from thesis_rl.rulebook.v2.geometry.elevation import PolylineElevation
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box
from thesis_rl.rulebook.v2.geometry.lanes import (
    RouteLaneRecord,
    associate_route_lane,
    bumper_to_bumper_gap,
    footprint_route_coordinates,
)
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.geometry.vertical import vertically_compatible_at_xy


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
        canonical_geometry_wkb(Polygon(((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0))))


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


def test_route_consolidates_noisy_xy_points_with_median_elevation_and_rejects_large_spread() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (0.0005, 0.0005, 2.0), (10.0, 0.0, 4.0)))
    assert route.points_xyz[0] == pytest.approx((0.00025, 0.00025, 1.0))
    with pytest.raises(ValueError, match="incompatible"):
        RoutePolyline(((0.0, 0.0, 0.0), (0.0005, 0.0, 3.1), (10.0, 0.0, 4.0)))


def test_route_projection_rejects_incompatible_vertical_level() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="vertically compatible"):
        route.project((5.0, 0.0), position_z=3.1)


def test_lane_association_rejects_vertical_and_geometric_ties() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    lane = RouteLaneRecord("lane-a", Polygon(((0.0, -2.0), (10.0, -2.0), (10.0, 2.0), (0.0, 2.0))), route)
    assert associate_route_lane(
        position_xy=(5.0, 0.0), position_z=0.0, heading_rad=0.0, route_lanes=(lane,)
    ).lane_id == "lane-a"
    assert associate_route_lane(
        position_xy=(5.0, 0.0), position_z=3.1, heading_rad=0.0, route_lanes=(lane,)
    ) is None
    tied = RouteLaneRecord("lane-b", lane.polygon_xy, route)
    assert associate_route_lane(
        position_xy=(5.0, 0.0), position_z=0.0, heading_rad=0.0, route_lanes=(lane, tied)
    ) is None


def test_bumper_gap_handles_separated_tangent_and_overlapping_footprints() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (30.0, 0.0, 0.0)))
    ego = footprint_route_coordinates(
        oriented_bounding_box(center_xy=(5.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0), route, position_z=0.0
    )
    other = footprint_route_coordinates(
        oriented_bounding_box(center_xy=(12.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0), route, position_z=0.0
    )
    assert bumper_to_bumper_gap(ego, other) == pytest.approx((3.0, True))
    tangent = footprint_route_coordinates(
        oriented_bounding_box(center_xy=(9.0, 0.0), heading_rad=0.0, length_m=4.0, width_m=2.0), route, position_z=0.0
    )
    assert bumper_to_bumper_gap(ego, tangent) == pytest.approx((0.0, True))
