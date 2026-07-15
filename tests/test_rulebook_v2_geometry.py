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
