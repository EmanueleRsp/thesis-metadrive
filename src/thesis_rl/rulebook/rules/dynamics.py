from __future__ import annotations

import math

import numpy as np

from thesis_rl.rulebook.rules.utils import MISSING_DATA_MARGIN, accel_vector, ego_speed, ego_yaw
from thesis_rl.rulebook.types import RuleEvalInput


def check_longitudinal_accel(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    a_long: float | None = None

    if isinstance(ego_state.get("acceleration"), dict):
        accel_dict = ego_state["acceleration"]
        if "longitudinal" in accel_dict:
            a_long = float(accel_dict["longitudinal"])

    if a_long is None:
        acc_vec = accel_vector(ego_state)
        yaw = ego_yaw(ego_state)
        if acc_vec is not None and yaw is not None:
            heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
            a_long = float(np.dot(acc_vec, heading))

    if a_long is None:
        return False, MISSING_DATA_MARGIN
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
        speed = ego_speed(ego_state)
        if steer is not None and length is not None and speed is not None:
            sin_steer = math.sin(float(steer) * math.pi / 2.0)
            if abs(sin_steer) > 1e-6:
                turning_radius = float(length) / sin_steer
                a_lat = (float(speed) ** 2) / abs(turning_radius)
            else:
                a_lat = 0.0

    if a_lat is None:
        acc_vec = accel_vector(ego_state)
        yaw = ego_yaw(ego_state)
        if acc_vec is not None and yaw is not None:
            lateral_axis = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
            a_lat = float(abs(np.dot(acc_vec, lateral_axis)))

    if a_lat is None:
        return False, MISSING_DATA_MARGIN
    return False, -abs(a_lat)
