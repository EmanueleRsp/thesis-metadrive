"""Rulebook v1 rules and their structured output contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from thesis_rl.rulebook.rules.utils import (
    as_geometry,
    effective_radius,
    ego_pos,
    get_mass,
    get_polygon,
    is_vru,
    signed_poly_clearance,
    xy,
)
from thesis_rl.rulebook.types import RuleEvalInput, RuleResult


def _result(
    name: str,
    margin: float = 0.0,
    *,
    available: bool,
    fallback_used: bool = False,
    raw: Mapping[str, Any] | None = None,
) -> RuleResult:
    value = float(margin)
    return RuleResult(
        name=name,
        margin=value,
        violated=value < 0.0,
        severity=max(0.0, -value),
        available=available,
        fallback_used=fallback_used,
        raw=dict(raw or {}),
    )


def _velocity(state: Mapping[str, Any]) -> np.ndarray | None:
    value = state.get("velocity")
    if value is not None:
        vector = xy(value)
        if vector is not None:
            return vector.astype(np.float64)
    speed = state.get("speed_m_s")
    if speed is None:
        return None
    yaw = state.get("yaw", state.get("heading"))
    if yaw is None:
        return np.array([float(speed), 0.0], dtype=np.float64)
    return float(speed) * np.array([np.cos(float(yaw)), np.sin(float(yaw))])


def _overlap_area(ego_polygon: Any, geometry: Any) -> float | None:
    if ego_polygon is None or geometry is None:
        return None
    try:
        return max(0.0, float(ego_polygon.intersection(geometry).area))
    except Exception:
        return None


def _iter_geometries(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [geom for item in value if (geom := as_geometry(item)) is not None]
    geometry = as_geometry(value)
    return [geometry] if geometry is not None else []


def _ego_geometry_or_position(rule_eval_input: RuleEvalInput) -> tuple[Any | None, np.ndarray | None]:
    ego_state = dict(rule_eval_input.ego_state)
    return get_polygon(ego_state), ego_pos(ego_state)


def _objects_contact(
    ego_polygon: Any | None,
    ego_position: np.ndarray | None,
    ego_state: Mapping[str, Any],
    other: Mapping[str, Any],
) -> bool | None:
    other_polygon = get_polygon(dict(other))
    if ego_polygon is not None and other_polygon is not None:
        try:
            return bool(ego_polygon.intersects(other_polygon))
        except Exception:
            return None
    other_position = ego_pos(dict(other))
    if ego_position is not None and other_position is not None:
        distance = float(np.linalg.norm(ego_position - other_position))
        return distance < effective_radius(dict(ego_state)) + effective_radius(dict(other))
    return None


def collision_severity(
    rule_eval_input: RuleEvalInput,
    *,
    include_static_obstacles: bool = True,
    use_reduced_mass: bool = True,
) -> RuleResult:
    """Return the maximum relative kinetic severity among current contacts."""
    name = "collision_severity"
    ego_state = dict(rule_eval_input.ego_state)
    ego_polygon, ego_position = _ego_geometry_or_position(rule_eval_input)
    if ego_polygon is None and ego_position is None:
        return _result(name, available=False)

    ego_velocity = _velocity(ego_state)
    collisions: list[tuple[float, str, bool]] = []
    inspectable_object_found = False
    for neighbor in rule_eval_input.neighbors:
        other = dict(neighbor)
        if is_vru(other):
            continue
        object_type = str(other.get("type", "vehicle")).lower()
        is_static = any(token in object_type for token in ("static", "obstacle", "barrier", "cone"))
        if is_static and not include_static_obstacles:
            continue
        contact = _objects_contact(ego_polygon, ego_position, ego_state, other)
        if contact is None:
            continue
        inspectable_object_found = True
        if not contact:
            continue

        other_velocity = np.zeros(2, dtype=np.float64) if is_static else _velocity(other)
        fallback_used = False
        if ego_velocity is not None and other_velocity is not None:
            relative_speed_sq = float(np.dot(ego_velocity - other_velocity, ego_velocity - other_velocity))
            if use_reduced_mass:
                m_ego, m_other = get_mass(ego_state), get_mass(other)
                severity = (m_ego * m_other / max(m_ego + m_other, 1e-6)) * relative_speed_sq / 2.0
            else:
                severity = relative_speed_sq
        else:
            # Collision is still meaningful when the simulator exposes geometry only.
            severity = 1.0
            fallback_used = True
        collisions.append((max(float(severity), 0.0), "static_obstacle" if is_static else "vehicle", fallback_used))

    if not collisions:
        return _result(
            name,
            available=True,
            raw={"contacts": 0, "objects_inspected": inspectable_object_found},
        )
    severity, object_type, fallback_used = max(collisions, key=lambda item: item[0])
    return _result(
        name,
        -severity,
        available=True,
        fallback_used=fallback_used,
        raw={"contacts": len(collisions), "collision_object_type": object_type},
    )


def allowed_driving_area(
    rule_eval_input: RuleEvalInput,
    *,
    distance_weight: float = 1.0,
) -> RuleResult:
    """Check the footprint against the semantic allowed region when available."""
    name = "allowed_driving_area"
    ego_polygon, ego_position = _ego_geometry_or_position(rule_eval_input)
    if ego_polygon is None and ego_position is None:
        return _result(name, available=False)

    allowed = as_geometry(rule_eval_input.allowed_driving_area)
    fallback_used = False
    if allowed is None:
        drivable = as_geometry(rule_eval_input.drivable_area)
        if drivable is None:
            return _result(name, available=False)
        opposite = as_geometry(rule_eval_input.opposite_carriageway)
        if opposite is not None:
            try:
                allowed = drivable.difference(opposite)
            except Exception:
                allowed = drivable
                fallback_used = True
        else:
            allowed = drivable
            fallback_used = True

    if ego_polygon is not None:
        try:
            outside_area = max(0.0, float(ego_polygon.difference(allowed).area))
            distance = max(0.0, float(allowed.distance(ego_polygon)))
        except Exception:
            return _result(name, available=False, fallback_used=fallback_used)
    else:
        try:
            from shapely.geometry import Point

            point = Point(float(ego_position[0]), float(ego_position[1]))
            inside = bool(allowed.covers(point))
            outside_area, distance = (0.0, 0.0) if inside else (1.0, float(allowed.distance(point)))
        except Exception:
            return _result(name, available=False, fallback_used=fallback_used)

    wrong_way = {"available": False, "overlap_ratio": 0.0, "violated": False}
    opposite = as_geometry(rule_eval_input.opposite_carriageway)
    if ego_polygon is not None and opposite is not None:
        overlap = _overlap_area(ego_polygon, opposite)
        try:
            ego_area = float(ego_polygon.area)
        except Exception:
            ego_area = 0.0
        if overlap is not None and ego_area > 0.0:
            ratio = overlap / ego_area
            wrong_way = {"available": True, "overlap_ratio": ratio, "violated": ratio > 0.0}

    margin = -(outside_area + float(distance_weight) * distance**2)
    return _result(
        name,
        margin,
        available=True,
        fallback_used=fallback_used,
        raw={"outside_area": outside_area, "distance_to_allowed": distance, "wrong_way_diagnostic": wrong_way},
    )


def lane_marking_compliance(
    rule_eval_input: RuleEvalInput,
    *,
    state: dict[str, Any] | None = None,
    alpha_solid: float = 1.0,
    alpha_dashed: float = 0.25,
    dashed_persistence_steps: int = 5,
) -> RuleResult:
    """Penalize solid markings immediately and dashed markings after persistence."""
    name = "lane_marking_compliance"
    ego_polygon, _ = _ego_geometry_or_position(rule_eval_input)
    if ego_polygon is None:
        return _result(name, available=False)
    state = state if state is not None else {}
    solid_overlap = sum(filter(None, (_overlap_area(ego_polygon, geom) for geom in _iter_geometries(rule_eval_input.solid_lane_markings))))
    dashed_overlap = sum(filter(None, (_overlap_area(ego_polygon, geom) for geom in _iter_geometries(rule_eval_input.dashed_lane_markings))))
    markings_available = rule_eval_input.solid_lane_markings is not None or rule_eval_input.dashed_lane_markings is not None
    fallback_used = False
    if not markings_available:
        boundaries = _iter_geometries(rule_eval_input.lane_boundaries)
        if not boundaries:
            return _result(name, available=False)
        solid_overlap = sum(filter(None, (_overlap_area(ego_polygon, geom) for geom in boundaries)))
        dashed_overlap = 0.0
        fallback_used = True

    dashed_steps = int(state.get("dashed_overlap_steps", 0)) + 1 if dashed_overlap > 0.0 else 0
    state["dashed_overlap_steps"] = dashed_steps
    persistent_dashed_overlap = dashed_overlap if dashed_steps >= max(int(dashed_persistence_steps), 1) else 0.0
    margin = -(float(alpha_solid) * solid_overlap + float(alpha_dashed) * persistent_dashed_overlap)
    return _result(
        name,
        margin,
        available=True,
        fallback_used=fallback_used,
        raw={
            "solid_overlap": solid_overlap,
            "dashed_overlap": dashed_overlap,
            "persistent_dashed_overlap": persistent_dashed_overlap,
            "dashed_overlap_steps": dashed_steps,
        },
    )


def local_route_progress(
    rule_eval_input: RuleEvalInput,
    *,
    checkpoint_weights: Sequence[float] = (0.40, 0.25, 0.15, 0.12, 0.08),
) -> RuleResult:
    """Measure transition-based route progress, with checkpoint fallback."""
    name = "local_route_progress"
    if rule_eval_input.route_progress is not None:
        if rule_eval_input.prev_route_progress is None:
            return _result(
                name,
                available=True,
                raw={"implementation": "route_coordinate", "initialization": True},
            )
        delta = float(rule_eval_input.route_progress) - float(rule_eval_input.prev_route_progress)
        return _result(name, delta, available=True, raw={"implementation": "route_coordinate"})

    ego_now = ego_pos(dict(rule_eval_input.ego_state))
    ego_prev = ego_pos(dict(rule_eval_input.prev_ego_state or {}))
    checkpoints = [point for point in (xy(value) for value in (rule_eval_input.route_checkpoints or ())) if point is not None]
    if ego_now is None or ego_prev is None or not checkpoints:
        return _result(name, available=False)
    weights = np.asarray(list(checkpoint_weights)[: len(checkpoints)], dtype=np.float64)
    if weights.size == 0:
        return _result(name, available=False)
    weights = weights / weights.sum()
    points = np.asarray(checkpoints[: len(weights)], dtype=np.float64)
    old_distance = float(np.sum(weights * np.linalg.norm(points - ego_prev, axis=1)))
    new_distance = float(np.sum(weights * np.linalg.norm(points - ego_now, axis=1)))
    return _result(
        name,
        old_distance - new_distance,
        available=True,
        fallback_used=True,
        raw={"implementation": "checkpoint_fallback", "old_distance": old_distance, "new_distance": new_distance},
    )
