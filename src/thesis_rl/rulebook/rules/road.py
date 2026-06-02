from __future__ import annotations

import math

import numpy as np

from thesis_rl.rulebook.rules.utils import (
    MISSING_DATA_MARGIN,
    as_geometry,
    distance_point_to_polyline,
    ego_pos,
    get_polygon,
    xy,
)
from thesis_rl.rulebook.types import RuleEvalInput


def check_drivable_area(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    drivable_area = rule_eval_input.drivable_area
    if drivable_area is None:
        return False, MISSING_DATA_MARGIN

    ego_poly = get_polygon(ego_state)
    drivable_geom = as_geometry(drivable_area)
    if ego_poly is not None and drivable_geom is not None:
        try:
            outside_area = float(ego_poly.difference(drivable_geom).area)
        except Exception:
            outside_area = 0.0
        dist = float(drivable_geom.distance(ego_poly))
        violation = outside_area + dist**2
        return violation > 0.0, -violation

    ego_position = ego_pos(ego_state)
    if isinstance(drivable_area, dict) and ego_position is not None:
        xmin = drivable_area.get("xmin")
        xmax = drivable_area.get("xmax")
        ymin = drivable_area.get("ymin")
        ymax = drivable_area.get("ymax")
        if None not in (xmin, xmax, ymin, ymax):
            dx = min(float(ego_position[0]) - float(xmin), float(xmax) - float(ego_position[0]))
            dy = min(float(ego_position[1]) - float(ymin), float(ymax) - float(ego_position[1]))
            margin = float(min(dx, dy))
            return margin < 0.0, margin

    return False, MISSING_DATA_MARGIN


def check_wrong_way(rule_eval_input: RuleEvalInput, threshold_ratio: float = 0.0) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    opposite_carriageway = as_geometry(rule_eval_input.opposite_carriageway)
    if opposite_carriageway is None:
        return False, MISSING_DATA_MARGIN

    ego_poly = get_polygon(ego_state)
    if ego_poly is None or not hasattr(ego_poly, "intersection") or not hasattr(ego_poly, "area"):
        return False, MISSING_DATA_MARGIN
    if not hasattr(opposite_carriageway, "intersection") or not hasattr(opposite_carriageway, "area"):
        return False, MISSING_DATA_MARGIN

    try:
        inter_area = float(ego_poly.intersection(opposite_carriageway).area)
        ego_area = float(ego_poly.area)
    except Exception:
        return False, MISSING_DATA_MARGIN
    if ego_area <= 0.0:
        return False, MISSING_DATA_MARGIN

    invasion_ratio = inter_area / ego_area
    margin = threshold_ratio - invasion_ratio
    return invasion_ratio > threshold_ratio, margin


def check_speed_limit(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    speed_limit = rule_eval_input.speed_limit
    if speed_limit is None:
        return False, MISSING_DATA_MARGIN

    # Strict km/h source for speed-limit rule.
    if "speed" not in ego_state:
        return False, MISSING_DATA_MARGIN
    ego_speed_kmh = float(ego_state["speed"])

    margin = float(speed_limit) - ego_speed_kmh
    return margin < 0.0, margin


def check_lane_centering(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    lane_centerline = rule_eval_input.lane_centerline
    if lane_centerline is None:
        return False, MISSING_DATA_MARGIN

    ego_poly = get_polygon(ego_state)
    if ego_poly is not None and hasattr(ego_poly, "centroid"):
        try:
            centroid = ego_poly.centroid
            centroid_xy = np.array([float(centroid.x), float(centroid.y)], dtype=np.float32)
        except Exception:
            centroid_xy = None
    else:
        centroid_xy = None

    if centroid_xy is None:
        centroid_xy = ego_pos(ego_state)
    if centroid_xy is None:
        return False, MISSING_DATA_MARGIN

    if hasattr(lane_centerline, "local_coordinates"):
        try:
            _, lateral = lane_centerline.local_coordinates(centroid_xy)
            return False, -abs(float(lateral))
        except Exception:
            return False, MISSING_DATA_MARGIN

    try:
        polyline = np.asarray(lane_centerline, dtype=np.float32)
    except Exception:
        return False, MISSING_DATA_MARGIN

    distance = distance_point_to_polyline(centroid_xy, polyline)
    if distance is None:
        return False, MISSING_DATA_MARGIN
    return False, -distance


def check_goal_progress(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    target_region = as_geometry(rule_eval_input.target_region)
    target_point = rule_eval_input.target_point

    if target_region is None and target_point is None:
        return False, MISSING_DATA_MARGIN

    ego_poly = get_polygon(ego_state)
    ego_center = None
    if ego_poly is not None and hasattr(ego_poly, "centroid"):
        try:
            centroid = ego_poly.centroid
            ego_center = np.array([float(centroid.x), float(centroid.y)], dtype=np.float32)
        except Exception:
            ego_center = None
    if ego_center is None:
        ego_center = ego_pos(ego_state)
    if ego_center is None:
        return False, MISSING_DATA_MARGIN

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
                return False, MISSING_DATA_MARGIN
        elif isinstance(rule_eval_input.target_region, dict):
            target_region_dict = dict(rule_eval_input.target_region)
            xmin = target_region_dict.get("xmin")
            xmax = target_region_dict.get("xmax")
            ymin = target_region_dict.get("ymin")
            ymax = target_region_dict.get("ymax")
            if None in (xmin, xmax, ymin, ymax):
                return False, MISSING_DATA_MARGIN
            x = float(ego_center[0])
            y = float(ego_center[1])
            dx = max(float(xmin) - x, 0.0, x - float(xmax))
            dy = max(float(ymin) - y, 0.0, y - float(ymax))
            distance_to_target = float(math.hypot(dx, dy))
            outside_target = 1.0 if distance_to_target > 0.0 else 0.0
        # If target_region is present but unusable, we fall back to target_point below.

    if distance_to_target is None:
        target_xy = xy(target_point)
        if target_xy is None:
            return False, MISSING_DATA_MARGIN
        distance_to_target = float(np.linalg.norm(ego_center - target_xy))

    violation = outside_target + distance_to_target**2
    return False, -violation
