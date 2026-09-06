"""Coverage budgeting for the constrained-triangulation decomposition path.

`_constrained_components_after_ear_exhaustion` discards triangulation slivers
whose individual area is at or below `AREA_EPSILON_M2`, then checks that what
remains still covers the input. Budgeting that *aggregate* shortfall against the
*per-piece* constant is a dimensional error: N discarded slivers, each
individually inside the per-piece bound, sum to as much as N times it. It killed
a 100 000-step training run on an ordinary ScenarioNet conflict zone.

The fixture is the polygon that reproduced the failure: 195 exterior vertices,
261.43 m2, two slivers discarded, shortfall 1.33e-4 m2 against the old 1.0e-4 m2
bound -- 1.3 times a tolerance that is itself 0.13 mm2.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import shapely
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.errors import RuntimeGeometryNotEvaluableError
from thesis_rl.rulebook.v2.geometry import continuous_sat
from thesis_rl.rulebook.v2.geometry.continuous_sat import (
    AREA_EPSILON_M2,
    RELATIVE_COVERAGE_EPSILON,
    deterministic_convex_decomposition,
)


FIXTURE = Path(__file__).parent / "fixtures" / "constrained_decomposition_shortfall_zone.wkt"


@pytest.fixture(scope="module")
def shortfall_zone() -> Polygon:
    polygon = shapely.from_wkt(FIXTURE.read_text())
    assert polygon.is_valid and polygon.geom_type == "Polygon"
    # Guard the fixture itself: it only exercises the defect if it is routed
    # into constrained triangulation rather than ear clipping.
    assert (
        len(polygon.exterior.coords) - 1 > continuous_sat.CONSTRAINED_DECOMPOSITION_VERTEX_THRESHOLD
    )
    return polygon


def test_sliver_shortfall_within_relative_allowance_decomposes(shortfall_zone: Polygon) -> None:
    """The reproduced conflict zone decomposes instead of aborting the run.

    Fails on the pre-fix code with "Constrained decomposition does not cover the
    input polygon".
    """

    triangles = deterministic_convex_decomposition(shortfall_zone)

    assert triangles
    covered = shapely.union_all(triangles)
    # Faithfulness, not merely absence of an exception: the decomposition may
    # not claim area the polygon does not have, and may not lose a
    # geometrically meaningful amount of the area it does have.
    assert shortfall_zone.covers(covered)
    shortfall = shortfall_zone.symmetric_difference(covered).area
    assert shortfall > AREA_EPSILON_M2, (
        "fixture no longer reproduces the defect: its shortfall now sits inside "
        "the old per-piece bound, so this test would pass pre-fix too"
    )
    assert shortfall <= shortfall_zone.area * RELATIVE_COVERAGE_EPSILON


def test_relative_allowance_still_rejects_a_material_shortfall(
    shortfall_zone: Polygon, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must still bite. Relaxing it is not the same as removing it."""

    triangles = [
        triangle
        for triangle in shapely.constrained_delaunay_triangles(shortfall_zone).geoms
        if triangle.geom_type == "Polygon"
    ]
    # Withhold the largest triangle: a shortfall far above any tolerance.
    triangles.sort(key=lambda triangle: triangle.area)
    starved = shapely.geometrycollections(triangles[:-1])

    monkeypatch.setattr(
        continuous_sat.shapely,
        "constrained_delaunay_triangles",
        lambda polygon: starved,
    )

    with pytest.raises(
        RuntimeGeometryNotEvaluableError, match="does not cover the input polygon"
    ) as excinfo:
        deterministic_convex_decomposition(shortfall_zone)
    message = str(excinfo.value)
    # The magnitude is the diagnosis: a fail-fast message that reports no number
    # forces the next reader to guess, which is what made this defect expensive.
    assert "residual" in message and "allowance" in message


def test_a_triangulation_entirely_outside_the_polygon_is_fatal(
    shortfall_zone: Polygon, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A triangulation that shares nothing with its input is not a tolerance case.

    It exercises the no-triangles path rather than the overshoot branch: every
    candidate is rejected by `polygon.covers`, so none is retained. The overshoot
    branch below it is unreachable by construction for exactly that reason -- the
    union of covered triangles cannot escape the polygon -- and is kept as a
    deliberately fatal assertion on a GEOS invariant, not as a recoverable
    condition.
    """

    outside = shapely.affinity.translate(shortfall_zone, xoff=1000.0, yoff=1000.0)
    monkeypatch.setattr(
        continuous_sat.shapely,
        "constrained_delaunay_triangles",
        lambda polygon: shapely.geometrycollections([outside]),
    )

    with pytest.raises(ValueError) as excinfo:
        deterministic_convex_decomposition(shortfall_zone)
    # Fatal, not a geometry abort: nothing here is recoverable.
    assert not isinstance(excinfo.value, RuntimeGeometryNotEvaluableError)
