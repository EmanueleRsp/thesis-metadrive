from __future__ import annotations

import math
from typing import Any

import numpy as np

MISSING_DATA_MARGIN = 0.0


def as_geometry(value: object) -> object | None:
    if value is None:
        return None
    if hasattr(value, "distance") and hasattr(value, "area"):
        return value
    try:
        coords = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] < 2:
        return None
    try:
        from shapely.geometry import Polygon

        poly = Polygon(coords[:, :2])
    except Exception:
        return None
    return poly if hasattr(poly, "distance") and hasattr(poly, "area") else None


def xy(value: object) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr[:2] if arr.size >= 2 else None


def ego_pos(state: dict[str, Any]) -> np.ndarray | None:
    if "position" in state:
        return xy(state["position"])
    if "x" in state and "y" in state:
        return np.array([float(state["x"]), float(state["y"])], dtype=np.float32)
    return None


def get_polygon(state: dict[str, Any]) -> object | None:
    return as_geometry(state.get("polygon"))


def effective_radius(state: dict[str, Any], default: float = 1.0) -> float:
    if "radius" in state:
        return float(state["radius"])
    if "length" in state and "width" in state:
        return float(math.hypot(float(state["length"]), float(state["width"])) / 2.0)
    return default


def ego_speed(state: dict[str, Any]) -> float | None:
    if "speed_m_s" in state:
        return float(state["speed_m_s"])
    if "velocity" in state:
        v = xy(state["velocity"])
        return float(np.linalg.norm(v)) if v is not None else None
    return None


def ego_yaw(state: dict[str, Any]) -> float | None:
    if "yaw" in state:
        return float(state["yaw"])
    if "heading" in state:
        return float(state["heading"])
    return None


def accel_vector(state: dict[str, Any]) -> np.ndarray | None:
    acc = state.get("acceleration")
    if acc is None:
        return None
    if isinstance(acc, dict) and "x" in acc and "y" in acc:
        return np.array([float(acc["x"]), float(acc["y"])], dtype=np.float32)
    return xy(acc)


def signed_poly_clearance(poly_ego: object, poly_other: object) -> float:
    dist = float(poly_ego.distance(poly_other))
    if dist > 0.0:
        return dist
    try:
        inter_area = float(poly_ego.intersection(poly_other).area)
    except Exception:
        inter_area = 0.0
    penetration = math.sqrt(inter_area / math.pi) if inter_area > 0.0 else 1e-3
    return -penetration


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    seg = end - start
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    projection = float(np.dot(point - start, seg) / seg_len_sq)
    projection = min(1.0, max(0.0, projection))
    closest = start + projection * seg
    return float(np.linalg.norm(point - closest))


def distance_point_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float | None:
    if polyline.ndim != 2 or polyline.shape[0] < 2 or polyline.shape[1] < 2:
        return None
    min_dist = float("inf")
    for idx in range(polyline.shape[0] - 1):
        start = polyline[idx, :2]
        end = polyline[idx + 1, :2]
        min_dist = min(min_dist, point_to_segment_distance(point, start, end))
    return min_dist if min_dist != float("inf") else None


def is_vru(state: dict[str, Any]) -> bool:
    obj_type = state.get("type", "").lower()
    if obj_type in ("pedestrian", "cyclist", "vru"):
        return True
    if "mass" in state:
        return float(state["mass"]) < 150.0
    if "m" in state:
        return float(state["m"]) < 150.0
    return False


def get_mass(state: dict[str, Any], vru_default: float = 70.0, vehicle_default: float = 1500.0) -> float:
    if "mass" in state:
        return float(state["mass"])
    if "m" in state:
        return float(state["m"])
    return vru_default if is_vru(state) else vehicle_default


def neighbor_id(state: dict[str, Any]) -> str | None:
    entity_id = state.get("entity_id")
    if isinstance(entity_id, str) and entity_id:
        return entity_id
    return None
