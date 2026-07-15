"""R3 road-geometry components (off-road and wrong-way)."""

from __future__ import annotations

from math import atan2, isfinite, pi

from thesis_rl.rulebook.v2.geometry.route import RoutePolyline
from thesis_rl.rulebook.v2.types import ActorSnapshot, CacheDelta, ComponentStatus, MemoryDelta, RuleComponentResult


OFFROAD_AREA_EPSILON_M2 = 1.0e-4
GEOMETRY_EPSILON_M = 1.0e-2
DASHED_T0_S = 1.0
DASHED_TCAP_S = 2.0


def evaluate_offroad(*, ego_footprint, drivable_surface) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate the footprint area outside the current compatible drivable surface."""
    if ego_footprint.is_empty or not ego_footprint.is_valid or ego_footprint.area <= 0.0:
        raise ValueError("Off-road requires a valid, positive-area ego footprint")
    if drivable_surface is None or drivable_surface.is_empty or not drivable_surface.is_valid:
        raise ValueError("Off-road requires a valid drivable surface")
    outside_area = ego_footprint.difference(drivable_surface).area
    if not isfinite(outside_area) or outside_area < 0.0:
        raise ValueError("Off-road difference area must be finite and non-negative")
    if outside_area < OFFROAD_AREA_EPSILON_M2:
        outside_area = 0.0
    ratio = outside_area / ego_footprint.area
    result = RuleComponentResult(
        name="offroad",
        cost=ratio,
        raw={"outside_area_m2": outside_area, "ego_area_m2": ego_footprint.area},
        applicable=True,
        evaluable=True,
        status=ComponentStatus.VIOLATED if ratio > 0.0 else ComponentStatus.SATISFIED,
        diagnostics={"area_epsilon_m2": OFFROAD_AREA_EPSILON_M2},
    )
    return result, MemoryDelta(), CacheDelta()


def _angle_delta(first: float, second: float) -> float:
    return (first - second + pi) % (2.0 * pi) - pi


def evaluate_wrongway(*, ego: ActorSnapshot, route: RoutePolyline, previous_s_m: float | None = None) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Penalize only velocity opposite to the canonical task-route tangent."""
    cap = ego.configured_speed_cap_mps
    if cap is None or not isfinite(cap) or cap <= 0.0:
        raise ValueError("Wrong-way requires a positive configured ego speed cap")
    projection = route.project(ego.position_xy, position_z=ego.position_z, previous_s_m=previous_s_m)
    vx, vy = ego.velocity_xy
    longitudinal_speed = vx * projection.tangent_xy[0] + vy * projection.tangent_xy[1]
    cost = min(max(-longitudinal_speed, 0.0) / cap, 1.0)
    ego_heading = ego.heading_rad
    route_heading = atan2(projection.tangent_xy[1], projection.tangent_xy[0])
    result = RuleComponentResult(
        name="wrongway",
        cost=cost,
        raw={"v_parallel_mps": longitudinal_speed, "route_s_m": projection.s_m},
        applicable=True,
        evaluable=True,
        status=ComponentStatus.VIOLATED if cost > 0.0 else ComponentStatus.SATISFIED,
        diagnostics={
            "ego_heading_rad": ego_heading,
            "route_heading_rad": route_heading,
            "heading_delta_rad": _angle_delta(ego_heading, route_heading),
            "segment_index": projection.segment_index,
            "speed_cap_mps": cap,
        },
    )
    return result, MemoryDelta(), CacheDelta()


def evaluate_solid_line(*, ego_footprint, solid_boundaries: tuple, swept_front_bumper=None) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Detect current occupancy or completed front-bumper crossing of solid lines."""
    if ego_footprint.is_empty or not ego_footprint.is_valid:
        raise ValueError("Solid-line evaluation requires a valid ego footprint")
    occupied: list[str] = []
    crossed: list[str] = []
    for index, boundary in enumerate(solid_boundaries):
        geometry = getattr(boundary, "geometry", boundary)
        boundary_id = getattr(boundary, "logical_boundary_id", None) or getattr(boundary, "feature_id", None) or str(index)
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("Solid boundary geometry must be valid")
        if ego_footprint.intersects(geometry.buffer(GEOMETRY_EPSILON_M)):
            occupied.append(str(boundary_id))
        if swept_front_bumper is not None and swept_front_bumper.intersects(geometry):
            crossed.append(str(boundary_id))
    ids = tuple(sorted(set(occupied + crossed)))
    result = RuleComponentResult(
        name="solid_line",
        cost=1.0 if ids else 0.0,
        raw={"occupied_boundary_ids": tuple(sorted(occupied)), "crossed_boundary_ids": tuple(sorted(crossed))},
        applicable=bool(solid_boundaries),
        evaluable=True,
        status=ComponentStatus.VIOLATED if ids else (ComponentStatus.SATISFIED if solid_boundaries else ComponentStatus.NOT_APPLICABLE),
        diagnostics={"active_boundary_ids": ids, "geometry_epsilon_m": GEOMETRY_EPSILON_M},
    )
    return result, MemoryDelta(), CacheDelta()


def evaluate_dashed_line(*, ego_footprint, dashed_boundaries: tuple, previous_boundary_id: str | None,
                         previous_timer_s: float, delta_t_s: float) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate continuous dashed boundaries and return the timer state delta."""
    if ego_footprint.is_empty or not ego_footprint.is_valid:
        raise ValueError("Dashed-line evaluation requires a valid ego footprint")
    if not isfinite(previous_timer_s) or previous_timer_s < 0.0 or not isfinite(delta_t_s) or delta_t_s <= 0.0:
        raise ValueError("Dashed-line timer inputs must be finite and non-negative")
    candidates: list[tuple[str, object, float]] = []
    center = ego_footprint.centroid
    for index, boundary in enumerate(dashed_boundaries):
        geometry = getattr(boundary, "geometry", boundary)
        boundary_id = str(getattr(boundary, "logical_boundary_id", None) or getattr(boundary, "feature_id", None) or index)
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("Dashed boundary geometry must be valid")
        if ego_footprint.intersects(geometry.buffer(GEOMETRY_EPSILON_M)):
            candidates.append((boundary_id, geometry, center.distance(geometry)))
    selected: tuple[str, object, float] | None = None
    if previous_boundary_id is not None:
        selected = next((item for item in candidates if item[0] == previous_boundary_id), None)
    if selected is None and candidates:
        selected = min(candidates, key=lambda item: (item[2], item[0]))
    active_id = selected[0] if selected is not None else None
    timer = previous_timer_s + delta_t_s if active_id is not None and active_id == previous_boundary_id else (delta_t_s if active_id is not None else 0.0)
    if timer <= DASHED_T0_S:
        cost = 0.0
    elif timer >= DASHED_TCAP_S:
        cost = 1.0
    else:
        cost = ((timer - DASHED_T0_S) / (DASHED_TCAP_S - DASHED_T0_S)) ** 2
    result = RuleComponentResult(
        name="dashed_line", cost=cost,
        raw={"active_boundary_id": active_id, "timer_s": timer},
        applicable=bool(dashed_boundaries), evaluable=True,
        status=ComponentStatus.VIOLATED if cost > 0.0 else (ComponentStatus.SATISFIED if active_id else ComponentStatus.NOT_APPLICABLE),
        diagnostics={"candidate_count": len(candidates), "threshold_s": DASHED_T0_S, "cap_s": DASHED_TCAP_S},
    )
    delta = MemoryDelta(writer="dashed_line", writes=(("active_dashed_boundary_id", active_id), ("dashed_line_timer_s", timer)))
    return result, delta, CacheDelta()
