from __future__ import annotations

import math
from typing import Any

import numpy as np

from thesis_rl.rulebook.types import RuleEvalInput

_MISSING_DATA_MARGIN = 0.0


def _xy(value: object) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr[:2] if arr.size >= 2 else None


def _ego_pos(state: dict[str, Any]) -> np.ndarray | None:
    if "position" in state:
        return _xy(state["position"])
    if "x" in state and "y" in state:
        return np.array([float(state["x"]), float(state["y"])], dtype=np.float32)
    return None


def _get_polygon(state: dict[str, Any]) -> object | None:
    poly = state.get("polygon")
    if poly is not None and hasattr(poly, "distance") and hasattr(poly, "area"):
        return poly
    return None


def _effective_radius(state: dict[str, Any], default: float = 1.0) -> float:
    if "radius" in state:
        return float(state["radius"])
    if "length" in state and "width" in state:
        return float(math.hypot(float(state["length"]), float(state["width"])) / 2.0)
    return default


def _ego_speed(state: dict[str, Any]) -> float | None:
    # Strict m/s source for physics-based rules (collision energy, dynamics).
    if "speed_m_s" in state:
        return float(state["speed_m_s"])
    if "velocity" in state:
        v = _xy(state["velocity"])
        return float(np.linalg.norm(v)) if v is not None else None
    return None


def _ego_yaw(state: dict[str, Any]) -> float | None:
    if "yaw" in state:
        return float(state["yaw"])
    if "heading" in state:
        return float(state["heading"])
    return None


def _accel_vector(state: dict[str, Any]) -> np.ndarray | None:
    acc = state.get("acceleration")
    if acc is None:
        return None
    if isinstance(acc, dict) and "x" in acc and "y" in acc:
        return np.array([float(acc["x"]), float(acc["y"])], dtype=np.float32)
    return _xy(acc)


def _signed_poly_clearance(poly_ego: object, poly_other: object) -> float:
    dist = float(poly_ego.distance(poly_other))
    if dist > 0.0:
        return dist
    try:
        inter_area = float(poly_ego.intersection(poly_other).area)
    except Exception:
        inter_area = 0.0
    penetration = math.sqrt(inter_area / math.pi) if inter_area > 0.0 else 1e-3
    return -penetration


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    seg = end - start
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    projection = float(np.dot(point - start, seg) / seg_len_sq)
    projection = min(1.0, max(0.0, projection))
    closest = start + projection * seg
    return float(np.linalg.norm(point - closest))


def _distance_point_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float | None:
    if polyline.ndim != 2 or polyline.shape[0] < 2 or polyline.shape[1] < 2:
        return None
    min_dist = float("inf")
    for idx in range(polyline.shape[0] - 1):
        start = polyline[idx, :2]
        end = polyline[idx + 1, :2]
        min_dist = min(min_dist, _point_to_segment_distance(point, start, end))
    return min_dist if min_dist != float("inf") else None


def _is_vru(state: dict[str, Any]) -> bool:
    obj_type = state.get("type", "").lower()
    if obj_type in ("pedestrian", "cyclist", "vru"):
        return True
    if "mass" in state:
        return float(state["mass"]) < 150.0
    if "m" in state:
        return float(state["m"]) < 150.0
    return False


def _get_mass(state: dict[str, Any], vru_default: float = 70.0, vehicle_default: float = 1500.0) -> float:
    if "mass" in state:
        return float(state["mass"])
    if "m" in state:
        return float(state["m"])
    return vru_default if _is_vru(state) else vehicle_default


def _neighbor_id(state: dict[str, Any]) -> str | None:
    entity_id = state.get("entity_id")
    if isinstance(entity_id, str) and entity_id:
        return entity_id
    return None


def check_vru_collision_energy(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    neighbors = [dict(n) for n in rule_eval_input.neighbors]
    prev_ego_state = rule_eval_input.prev_ego_state
    prev_neighbors_by_id = dict(rule_eval_input.prev_neighbors_by_id or {})

    ego_poly = _get_polygon(ego_state)
    ego_pos = _ego_pos(ego_state)
    if ego_poly is None and ego_pos is None:
        return False, _MISSING_DATA_MARGIN

    max_ke_delta = 0.0
    has_vru_contact = False

    for neighbor in neighbors:
        if not _is_vru(neighbor):
            continue

        n_poly = _get_polygon(neighbor)
        if ego_poly is not None and n_poly is not None:
            clearance = _signed_poly_clearance(ego_poly, n_poly)
        else:
            n_pos = _ego_pos(neighbor)
            if ego_pos is None or n_pos is None:
                continue
            center_dist = float(np.linalg.norm(ego_pos - n_pos))
            clearance = center_dist - (_effective_radius(ego_state) + _effective_radius(neighbor))

        if clearance >= 0:
            continue

        has_vru_contact = True
        ego_vel = _ego_speed(ego_state)
        vru_vel = _ego_speed(neighbor)
        if ego_vel is None or vru_vel is None:
            continue

        ego_mass = _get_mass(ego_state)
        vru_mass = _get_mass(neighbor)
        ke_ego_now = 0.5 * ego_mass * (ego_vel**2)
        ke_vru_now = 0.5 * vru_mass * (vru_vel**2)
        total_ke_now = ke_ego_now + 0.5 * ke_vru_now

        neighbor_id = _neighbor_id(neighbor)
        prev_neighbor = prev_neighbors_by_id.get(neighbor_id) if neighbor_id is not None else None
        if prev_ego_state is not None and prev_neighbor is not None:
            prev_ego_vel = _ego_speed(dict(prev_ego_state))
            prev_vru_vel = _ego_speed(dict(prev_neighbor))
            if prev_ego_vel is not None and prev_vru_vel is not None:
                ke_ego_prev = 0.5 * ego_mass * (prev_ego_vel**2)
                ke_vru_prev = 0.5 * vru_mass * (prev_vru_vel**2)
                total_ke_prev = ke_ego_prev + 0.5 * ke_vru_prev
                max_ke_delta = max(max_ke_delta, total_ke_now - total_ke_prev)
                continue

        # Safe fallback when temporal match is unavailable: use current contact energy only.
        max_ke_delta = max(max_ke_delta, total_ke_now)

    if not has_vru_contact:
        return False, _MISSING_DATA_MARGIN
    return True, -max_ke_delta


def check_vehicle_collision_energy(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    neighbors = [dict(n) for n in rule_eval_input.neighbors]
    prev_ego_state = rule_eval_input.prev_ego_state
    prev_neighbors_by_id = dict(rule_eval_input.prev_neighbors_by_id or {})

    ego_poly = _get_polygon(ego_state)
    ego_pos = _ego_pos(ego_state)
    if ego_poly is None and ego_pos is None:
        return False, _MISSING_DATA_MARGIN

    max_ke_delta = 0.0
    has_vehicle_contact = False

    for neighbor in neighbors:
        if _is_vru(neighbor):
            continue

        n_poly = _get_polygon(neighbor)
        if ego_poly is not None and n_poly is not None:
            clearance = _signed_poly_clearance(ego_poly, n_poly)
        else:
            n_pos = _ego_pos(neighbor)
            if ego_pos is None or n_pos is None:
                continue
            center_dist = float(np.linalg.norm(ego_pos - n_pos))
            clearance = center_dist - (_effective_radius(ego_state) + _effective_radius(neighbor))

        if clearance >= 0:
            continue

        has_vehicle_contact = True
        ego_vel = _ego_speed(ego_state)
        other_vel = _ego_speed(neighbor)
        if ego_vel is None or other_vel is None:
            continue

        ego_mass = _get_mass(ego_state)
        other_mass = _get_mass(neighbor)
        ke_now = 0.5 * ego_mass * (ego_vel**2) + 0.5 * other_mass * (other_vel**2)

        neighbor_id = _neighbor_id(neighbor)
        prev_neighbor = prev_neighbors_by_id.get(neighbor_id) if neighbor_id is not None else None
        if prev_ego_state is not None and prev_neighbor is not None:
            prev_ego_vel = _ego_speed(dict(prev_ego_state))
            prev_other_vel = _ego_speed(dict(prev_neighbor))
            if prev_ego_vel is not None and prev_other_vel is not None:
                ke_prev = 0.5 * ego_mass * (prev_ego_vel**2) + 0.5 * other_mass * (prev_other_vel**2)
                max_ke_delta = max(max_ke_delta, ke_now - ke_prev)
                continue

        # Safe fallback when temporal match is unavailable: use current contact energy only.
        max_ke_delta = max(max_ke_delta, ke_now)

    if not has_vehicle_contact:
        return False, _MISSING_DATA_MARGIN
    return True, -max_ke_delta


def check_drivable_area(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    drivable_area = rule_eval_input.drivable_area
    if drivable_area is None:
        return False, _MISSING_DATA_MARGIN

    ego_poly = _get_polygon(ego_state)
    if ego_poly is not None and hasattr(drivable_area, "distance") and hasattr(drivable_area, "area"):
        try:
            outside_area = float(ego_poly.difference(drivable_area).area)
        except Exception:
            outside_area = 0.0
        dist = float(drivable_area.distance(ego_poly))
        violation = outside_area + dist**2
        return violation > 0.0, -violation

    ego_pos = _ego_pos(ego_state)
    if isinstance(drivable_area, dict) and ego_pos is not None:
        xmin = drivable_area.get("xmin")
        xmax = drivable_area.get("xmax")
        ymin = drivable_area.get("ymin")
        ymax = drivable_area.get("ymax")
        if None not in (xmin, xmax, ymin, ymax):
            dx = min(float(ego_pos[0]) - float(xmin), float(xmax) - float(ego_pos[0]))
            dy = min(float(ego_pos[1]) - float(ymin), float(ymax) - float(ego_pos[1]))
            margin = float(min(dx, dy))
            return margin < 0.0, margin

    return False, _MISSING_DATA_MARGIN


def check_wrong_way(rule_eval_input: RuleEvalInput, threshold_ratio: float = 0.0) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    opposite_carriageway = rule_eval_input.opposite_carriageway
    if opposite_carriageway is None:
        return False, _MISSING_DATA_MARGIN

    ego_poly = ego_state.get("polygon")
    if ego_poly is None or not hasattr(ego_poly, "intersection") or not hasattr(ego_poly, "area"):
        return False, _MISSING_DATA_MARGIN
    if not hasattr(opposite_carriageway, "intersection") or not hasattr(opposite_carriageway, "area"):
        return False, _MISSING_DATA_MARGIN

    try:
        inter_area = float(ego_poly.intersection(opposite_carriageway).area)
        ego_area = float(ego_poly.area)
    except Exception:
        return False, _MISSING_DATA_MARGIN
    if ego_area <= 0.0:
        return False, _MISSING_DATA_MARGIN

    invasion_ratio = inter_area / ego_area
    margin = threshold_ratio - invasion_ratio
    return invasion_ratio > threshold_ratio, margin


def check_speed_limit(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    speed_limit = rule_eval_input.speed_limit
    if speed_limit is None:
        return False, _MISSING_DATA_MARGIN

    # Strict km/h source for speed-limit rule.
    if "speed" not in ego_state:
        return False, _MISSING_DATA_MARGIN
    ego_speed = float(ego_state["speed"])

    margin = float(speed_limit) - ego_speed
    return margin < 0.0, margin


def check_longitudinal_accel(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    a_long: float | None = None

    if isinstance(ego_state.get("acceleration"), dict):
        accel_dict = ego_state["acceleration"]
        if "longitudinal" in accel_dict:
            a_long = float(accel_dict["longitudinal"])

    if a_long is None:
        acc_vec = _accel_vector(ego_state)
        yaw = _ego_yaw(ego_state)
        if acc_vec is not None and yaw is not None:
            heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
            a_long = float(np.dot(acc_vec, heading))

    if a_long is None:
        return False, _MISSING_DATA_MARGIN
    return False, -abs(a_long)


def check_lateral_accel(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    a_lat: float | None = None

    if "accel_lat" in ego_state:
        a_lat = float(ego_state["accel_lat"])
    elif isinstance(ego_state.get("acceleration"), dict):
        accel_dict = ego_state["acceleration"]
        if "lateral" in accel_dict:
            a_lat = float(accel_dict["lateral"])

    if a_lat is None:
        steer = ego_state.get("steer")
        length = ego_state.get("length")
        speed = _ego_speed(ego_state)
        if steer is not None and length is not None and speed is not None:
            sin_steer = math.sin(float(steer) * math.pi / 2.0)
            if abs(sin_steer) > 1e-6:
                turning_radius = float(length) / sin_steer
                a_lat = (float(speed) ** 2) / abs(turning_radius)
            else:
                a_lat = 0.0

    if a_lat is None:
        acc_vec = _accel_vector(ego_state)
        yaw = _ego_yaw(ego_state)
        if acc_vec is not None and yaw is not None:
            lateral_axis = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
            a_lat = float(abs(np.dot(acc_vec, lateral_axis)))

    if a_lat is None:
        return False, _MISSING_DATA_MARGIN
    return False, -abs(a_lat)


def check_lane_centering(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    lane_centerline = rule_eval_input.lane_centerline
    if lane_centerline is None:
        return False, _MISSING_DATA_MARGIN

    ego_poly = _get_polygon(ego_state)
    if ego_poly is not None and hasattr(ego_poly, "centroid"):
        try:
            centroid = ego_poly.centroid
            centroid_xy = np.array([float(centroid.x), float(centroid.y)], dtype=np.float32)
        except Exception:
            centroid_xy = None
    else:
        centroid_xy = None

    if centroid_xy is None:
        centroid_xy = _ego_pos(ego_state)
    if centroid_xy is None:
        return False, _MISSING_DATA_MARGIN

    if hasattr(lane_centerline, "local_coordinates"):
        try:
            _, lateral = lane_centerline.local_coordinates(centroid_xy)
            return False, -abs(float(lateral))
        except Exception:
            return False, _MISSING_DATA_MARGIN

    try:
        polyline = np.asarray(lane_centerline, dtype=np.float32)
    except Exception:
        return False, _MISSING_DATA_MARGIN

    distance = _distance_point_to_polyline(centroid_xy, polyline)
    if distance is None:
        return False, _MISSING_DATA_MARGIN
    return False, -distance


def check_goal_progress(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    target_region = rule_eval_input.target_region
    target_point = rule_eval_input.target_point

    if target_region is None and target_point is None:
        return False, _MISSING_DATA_MARGIN

    ego_poly = _get_polygon(ego_state)
    ego_center = None
    if ego_poly is not None and hasattr(ego_poly, "centroid"):
        try:
            centroid = ego_poly.centroid
            ego_center = np.array([float(centroid.x), float(centroid.y)], dtype=np.float32)
        except Exception:
            ego_center = None
    if ego_center is None:
        ego_center = _ego_pos(ego_state)
    if ego_center is None:
        return False, _MISSING_DATA_MARGIN

    outside_target = 0.0
    distance_to_target = None

    if target_region is not None:
        if (
            ego_poly is not None
            and hasattr(target_region, "distance")
            and hasattr(target_region, "area")
            and hasattr(ego_poly, "difference")
        ):
            try:
                outside_target = float(ego_poly.difference(target_region).area)
                distance_to_target = float(target_region.distance(ego_poly))
            except Exception:
                return False, _MISSING_DATA_MARGIN
        elif isinstance(target_region, dict):
            xmin = target_region.get("xmin")
            xmax = target_region.get("xmax")
            ymin = target_region.get("ymin")
            ymax = target_region.get("ymax")
            if None in (xmin, xmax, ymin, ymax):
                return False, _MISSING_DATA_MARGIN
            x = float(ego_center[0])
            y = float(ego_center[1])
            dx = max(float(xmin) - x, 0.0, x - float(xmax))
            dy = max(float(ymin) - y, 0.0, y - float(ymax))
            distance_to_target = float(math.hypot(dx, dy))
            outside_target = 1.0 if distance_to_target > 0.0 else 0.0
        else:
            return False, _MISSING_DATA_MARGIN

    if distance_to_target is None:
        target_xy = _xy(target_point)
        if target_xy is None:
            return False, _MISSING_DATA_MARGIN
        distance_to_target = float(np.linalg.norm(ego_center - target_xy))

    violation = outside_target + distance_to_target**2
    return False, -violation
