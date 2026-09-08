"""F8 (REQ-F8-01/02): the vectorised ``RoutePolyline.project`` equals the reference.

The reference is the pre-F8 pure-Python implementation, reproduced here verbatim
in its arithmetic (per-segment fraction, projected point, interpolated z,
vertical filter, ``max_s_jump_m`` preference, ``d_min + eps_geom`` tie set,
lexicographic selection). Equivalence is checked at a *declared tolerance*
(`DEC-F8-001`, approved 2026-09-08): the same ``segment_index`` on every probe,
every float within ``EQUIVALENCE_TOLERANCE_M``, and the same ``ValueError`` when
the reference raises.

Probes: synthetic polylines that exercise each rule (a straight line, a fine
zigzag, a roundabout whose route approaches itself, a vertical ramp, seeded
random walks), plus the routes and lane centerlines of frozen-panel records
when the prepared ScenarioNet validation runtime is available.
"""

from __future__ import annotations

import os
import pickle
from math import cos, hypot, pi, sin
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thesis_rl.rulebook.v2.geometry.route import (
    GEOMETRY_EPSILON_M,
    RoutePolyline,
    RouteProjection,
    build_assigned_route_polyline,
)
from thesis_rl.rulebook.v2.geometry.vertical import VERTICAL_COMPATIBILITY_TOLERANCE_M

# DEC-F8-001. Coordinates are O(1e2) m, so 1e-9 m is ~1e-11 relative: three
# orders above the double-precision rounding of a summed arc length and six
# orders below GEOMETRY_EPSILON_M, the smallest distance the rulebook resolves.
EQUIVALENCE_TOLERANCE_M = 1.0e-9


def _reference_project(
    route: RoutePolyline,
    point_xy: tuple[float, float],
    *,
    position_z: float | None = None,
    previous_s_m: float | None = None,
    max_s_jump_m: float | None = None,
) -> RouteProjection:
    """The pre-F8 implementation of ``RoutePolyline.project`` (main @ 7f4ae9c)."""

    candidates: list[RouteProjection] = []
    for index, (first, second) in enumerate(zip(route.points_xyz, route.points_xyz[1:])):
        length = route._segment_lengths_m[index]
        tangent = ((second[0] - first[0]) / length, (second[1] - first[1]) / length)
        offset_x = point_xy[0] - first[0]
        offset_y = point_xy[1] - first[1]
        fraction = min(1.0, max(0.0, (offset_x * tangent[0] + offset_y * tangent[1]) / length))
        projected_x = first[0] + fraction * length * tangent[0]
        projected_y = first[1] + fraction * length * tangent[1]
        z_m = first[2] + fraction * (second[2] - first[2])
        if position_z is not None and abs(position_z - z_m) > VERTICAL_COMPATIBILITY_TOLERANCE_M:
            continue
        lateral_distance = (point_xy[0] - projected_x) * -tangent[1] + (
            point_xy[1] - projected_y
        ) * tangent[0]
        candidates.append(
            RouteProjection(
                s_m=route._segment_starts_m[index] + fraction * length,
                tangent_xy=tangent,
                z_m=z_m,
                lateral_distance_m=lateral_distance,
                segment_index=index,
            )
        )
    if not candidates:
        raise ValueError("Route projection has no vertically compatible segment")
    if max_s_jump_m is not None:
        assert previous_s_m is not None
        plausible = [
            candidate
            for candidate in candidates
            if abs(candidate.s_m - previous_s_m) <= max_s_jump_m + GEOMETRY_EPSILON_M
        ]
        if plausible:
            candidates = plausible

    def planar_distance(candidate: RouteProjection) -> float:
        first = route.points_xyz[candidate.segment_index]
        along = candidate.s_m - route._segment_starts_m[candidate.segment_index]
        projected_x = first[0] + along * candidate.tangent_xy[0]
        projected_y = first[1] + along * candidate.tangent_xy[1]
        return hypot(point_xy[0] - projected_x, point_xy[1] - projected_y)

    minimum_distance = min(planar_distance(candidate) for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if planar_distance(candidate) <= minimum_distance + GEOMETRY_EPSILON_M
    ]
    if previous_s_m is None:
        return min(tied, key=lambda candidate: (candidate.s_m, candidate.segment_index))
    return min(
        tied,
        key=lambda candidate: (abs(candidate.s_m - previous_s_m), candidate.segment_index),
    )


def _assert_equivalent(
    route: RoutePolyline, point_xy: tuple[float, float], **options: Any
) -> None:
    try:
        expected = _reference_project(route, point_xy, **options)
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error).split(":")[0]):
            route.project(point_xy, **options)
        return
    actual = route.project(point_xy, **options)
    context = f"point={point_xy} options={options}"
    assert actual.segment_index == expected.segment_index, context
    assert abs(actual.s_m - expected.s_m) <= EQUIVALENCE_TOLERANCE_M, context
    assert abs(actual.z_m - expected.z_m) <= EQUIVALENCE_TOLERANCE_M, context
    assert abs(actual.lateral_distance_m - expected.lateral_distance_m) <= EQUIVALENCE_TOLERANCE_M, (
        context
    )
    assert abs(actual.tangent_xy[0] - expected.tangent_xy[0]) <= EQUIVALENCE_TOLERANCE_M, context
    assert abs(actual.tangent_xy[1] - expected.tangent_xy[1]) <= EQUIVALENCE_TOLERANCE_M, context
    assert all(isinstance(value, float) for value in (actual.s_m, actual.z_m, actual.lateral_distance_m))
    assert isinstance(actual.segment_index, int)
    assert isinstance(actual.tangent_xy, tuple) and len(actual.tangent_xy) == 2


def _probe_points(
    route: RoutePolyline, rng: np.random.Generator, *, per_vertex: int = 3
) -> list[tuple[float, float]]:
    """Vertices, joints +-1 mm, mid-segments, lateral offsets and far points."""

    points: list[tuple[float, float]] = []
    vertices = route.points_xyz
    step = max(1, len(vertices) // 40)
    for index in range(0, len(vertices) - 1, step):
        first, second = vertices[index], vertices[index + 1]
        length = route._segment_lengths_m[index]
        tangent = ((second[0] - first[0]) / length, (second[1] - first[1]) / length)
        normal = (-tangent[1], tangent[0])
        for along in (0.0, 1.0e-3, -1.0e-3, 0.5 * length, length - 1.0e-3):
            base = (first[0] + along * tangent[0], first[1] + along * tangent[1])
            for lateral in (0.0, 0.3, -0.3, 2.0, -2.0, 6.0):
                points.append((base[0] + lateral * normal[0], base[1] + lateral * normal[1]))
        for _ in range(per_vertex):
            jitter = rng.normal(scale=3.0, size=2)
            points.append((first[0] + float(jitter[0]), first[1] + float(jitter[1])))
    xs = [p[0] for p in vertices]
    ys = [p[1] for p in vertices]
    for _ in range(20):
        points.append(
            (
                float(rng.uniform(min(xs) - 30.0, max(xs) + 30.0)),
                float(rng.uniform(min(ys) - 30.0, max(ys) + 30.0)),
            )
        )
    return points


def _option_sets(route: RoutePolyline, point_xy: tuple[float, float]) -> list[dict[str, Any]]:
    reset = _reference_project(route, point_xy)
    z_here = reset.z_m
    far_s = (reset.s_m + 0.5 * route.length_m) % route.length_m
    return [
        {},
        {"position_z": z_here},
        {"position_z": z_here + 0.5 * VERTICAL_COMPATIBILITY_TOLERANCE_M},
        {"position_z": z_here + 1.5 * VERTICAL_COMPATIBILITY_TOLERANCE_M},
        {"previous_s_m": reset.s_m},
        {"previous_s_m": far_s},
        {"previous_s_m": reset.s_m, "max_s_jump_m": 5.0},
        {"previous_s_m": far_s, "max_s_jump_m": 5.0},
        {"previous_s_m": far_s, "max_s_jump_m": 0.0},
        {"position_z": z_here, "previous_s_m": reset.s_m, "max_s_jump_m": 2.0},
    ]


def _check_route(route: RoutePolyline, rng: np.random.Generator) -> int:
    checked = 0
    for point in _probe_points(route, rng):
        for options in _option_sets(route, point):
            _assert_equivalent(route, point, **options)
            checked += 1
    return checked


def _roundabout_route() -> RoutePolyline:
    """Approach, a full circle, and exit: the route passes itself twice."""

    points: list[tuple[float, float, float]] = [(-30.0 + 2.0 * i, -12.0, 0.0) for i in range(15)]
    radius = 12.0
    for k in range(1, 73):
        angle = -pi / 2 + 2.0 * pi * k / 72
        points.append((radius * cos(angle), radius * sin(angle), 0.0))
    points.extend((2.0 * i, -12.0, 0.0) for i in range(1, 16))
    return RoutePolyline(tuple(points))


def _ramp_route() -> RoutePolyline:
    return RoutePolyline(tuple((2.0 * i, 0.1 * i * i, 0.25 * i) for i in range(40)))


def _zigzag_route() -> RoutePolyline:
    return RoutePolyline(tuple((0.5 * i, 0.4 * (i % 2), 0.0) for i in range(80)))


def _random_route(rng: np.random.Generator, count: int) -> RoutePolyline:
    heading = 0.0
    x = y = z = 0.0
    points = [(x, y, z)]
    for _ in range(count):
        heading += float(rng.normal(scale=0.35))
        step = float(rng.uniform(0.3, 4.0))
        x += step * cos(heading)
        y += step * sin(heading)
        z += float(rng.normal(scale=0.05))
        points.append((x, y, z))
    return RoutePolyline(tuple(points))


@pytest.mark.parametrize(
    "route_name",
    ["straight", "zigzag", "roundabout", "ramp", "random_0", "random_1", "random_2"],
)
def test_vectorised_projection_matches_reference_on_synthetic_routes(route_name: str) -> None:
    rng = np.random.default_rng(20260908)
    routes = {
        "straight": lambda: RoutePolyline(((0.0, 0.0, 0.0), (50.0, 0.0, 0.0), (100.0, 0.0, 1.0))),
        "zigzag": _zigzag_route,
        "roundabout": _roundabout_route,
        "ramp": _ramp_route,
        "random_0": lambda: _random_route(np.random.default_rng(0), 120),
        "random_1": lambda: _random_route(np.random.default_rng(1), 350),
        "random_2": lambda: _random_route(np.random.default_rng(2), 30),
    }
    assert _check_route(routes[route_name](), rng) > 500


def test_vectorised_projection_preserves_argument_validation() -> None:
    route = RoutePolyline(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="finite"):
        route.project((float("nan"), 0.0))
    with pytest.raises(ValueError, match="position_z"):
        route.project((1.0, 0.0), position_z=float("inf"))
    with pytest.raises(ValueError, match="previous_s_m"):
        route.project((1.0, 0.0), previous_s_m=float("nan"))
    with pytest.raises(ValueError, match="requires previous_s_m"):
        route.project((1.0, 0.0), max_s_jump_m=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        route.project((1.0, 0.0), previous_s_m=0.0, max_s_jump_m=-1.0)
    with pytest.raises(ValueError, match="vertically compatible"):
        route.project((1.0, 0.0), position_z=10.0)


def _runtime_root() -> Path:
    return Path(os.environ.get("SCENARIONET_DATA_ROOT", "data/scenarionet")) / "runtime" / "validation"


def _frozen_records(prefix: str, count: int) -> list[tuple[str, dict[str, Any]]]:
    root = _runtime_root()
    if not (root / "dataset_summary.pkl").is_file():
        pytest.skip("requires the prepared canonical ScenarioNet validation runtime")
    with (root / "dataset_summary.pkl").open("rb") as handle:
        summary = pickle.load(handle)
    with (root / "dataset_mapping.pkl").open("rb") as handle:
        mapping = pickle.load(handle)
    names = [name for name in sorted(summary) if name.startswith(prefix)][:count]
    records = []
    for name in names:
        with (root / mapping[name] / name).open("rb") as handle:
            records.append((name, pickle.load(handle)))
    return records


@pytest.mark.parametrize("source", ["waymo", "pg"])
def test_vectorised_projection_matches_reference_on_frozen_panel_records(source: str) -> None:
    """AC-F8-01 on the frozen validation panel: routes and every lane centerline."""

    if source == "waymo":
        from thesis_rl.rulebook.v2.context.waymo_static_adapter import (
            build_waymo_static_adapter_result as build_static,
        )

        records = _frozen_records("sd_waymo", 3)
    else:
        from thesis_rl.rulebook.v2.context.pg_static_adapter import (
            build_pg_static_adapter_result as build_static,
        )

        records = _frozen_records("sd_pg", 2)
    rng = np.random.default_rng(8)
    checked = 0
    for name, scenario in records:
        result = build_static(scenario, scenario_uid=name)
        lanes = {lane.lane_id: lane for lane in result.route_lanes}
        route = build_assigned_route_polyline(result.task_route.lane_ids, lanes)
        checked += _check_route(route, rng)
        # Lane centerlines are what drivable/carriageway and lane association project on.
        for lane in list(result.route_lanes)[::7]:
            for point in _probe_points(lane.centerline, rng, per_vertex=1)[::5]:
                for options in _option_sets(lane.centerline, point)[:4]:
                    _assert_equivalent(lane.centerline, point, **options)
                    checked += 1
    assert checked > 2000
