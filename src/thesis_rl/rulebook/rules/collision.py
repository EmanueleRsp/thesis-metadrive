from __future__ import annotations

from thesis_rl.rulebook.rules.utils import (
    MISSING_DATA_MARGIN,
    effective_radius,
    ego_pos,
    ego_speed,
    get_mass,
    get_polygon,
    is_vru,
    neighbor_id,
    signed_poly_clearance,
)
from thesis_rl.rulebook.types import RuleEvalInput


def check_vru_collision_energy(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    neighbors = [dict(n) for n in rule_eval_input.neighbors]
    prev_ego_state = rule_eval_input.prev_ego_state
    prev_neighbors_by_id = dict(rule_eval_input.prev_neighbors_by_id or {})

    ego_poly = get_polygon(ego_state)
    ego_position = ego_pos(ego_state)
    if ego_poly is None and ego_position is None:
        return False, MISSING_DATA_MARGIN

    max_ke_delta = 0.0
    has_vru_contact = False

    for neighbor in neighbors:
        if not is_vru(neighbor):
            continue

        n_poly = get_polygon(neighbor)
        if ego_poly is not None and n_poly is not None:
            clearance = signed_poly_clearance(ego_poly, n_poly)
        else:
            n_pos = ego_pos(neighbor)
            if ego_position is None or n_pos is None:
                continue
            center_dist = float(((ego_position - n_pos) ** 2).sum() ** 0.5)
            clearance = center_dist - (effective_radius(ego_state) + effective_radius(neighbor))

        if clearance >= 0:
            continue

        has_vru_contact = True
        ego_vel = ego_speed(ego_state)
        vru_vel = ego_speed(neighbor)
        if ego_vel is None or vru_vel is None:
            continue

        ego_mass = get_mass(ego_state)
        vru_mass = get_mass(neighbor)
        ke_ego_now = 0.5 * ego_mass * (ego_vel**2)
        ke_vru_now = 0.5 * vru_mass * (vru_vel**2)
        total_ke_now = ke_ego_now + 0.5 * ke_vru_now

        n_id = neighbor_id(neighbor)
        prev_neighbor = prev_neighbors_by_id.get(n_id) if n_id is not None else None
        if prev_ego_state is not None and prev_neighbor is not None:
            prev_ego_vel = ego_speed(dict(prev_ego_state))
            prev_vru_vel = ego_speed(dict(prev_neighbor))
            if prev_ego_vel is not None and prev_vru_vel is not None:
                ke_ego_prev = 0.5 * ego_mass * (prev_ego_vel**2)
                ke_vru_prev = 0.5 * vru_mass * (prev_vru_vel**2)
                total_ke_prev = ke_ego_prev + 0.5 * ke_vru_prev
                max_ke_delta = max(max_ke_delta, total_ke_now - total_ke_prev)
                continue

        max_ke_delta = max(max_ke_delta, total_ke_now)

    if not has_vru_contact:
        return False, MISSING_DATA_MARGIN
    return True, -max_ke_delta


def check_vehicle_collision_energy(rule_eval_input: RuleEvalInput) -> tuple[bool, float]:
    ego_state = dict(rule_eval_input.ego_state)
    neighbors = [dict(n) for n in rule_eval_input.neighbors]
    prev_ego_state = rule_eval_input.prev_ego_state
    prev_neighbors_by_id = dict(rule_eval_input.prev_neighbors_by_id or {})

    ego_poly = get_polygon(ego_state)
    ego_position = ego_pos(ego_state)
    if ego_poly is None and ego_position is None:
        return False, MISSING_DATA_MARGIN

    max_ke_delta = 0.0
    has_vehicle_contact = False

    for neighbor in neighbors:
        if is_vru(neighbor):
            continue

        n_poly = get_polygon(neighbor)
        if ego_poly is not None and n_poly is not None:
            clearance = signed_poly_clearance(ego_poly, n_poly)
        else:
            n_pos = ego_pos(neighbor)
            if ego_position is None or n_pos is None:
                continue
            center_dist = float(((ego_position - n_pos) ** 2).sum() ** 0.5)
            clearance = center_dist - (effective_radius(ego_state) + effective_radius(neighbor))

        if clearance >= 0:
            continue

        has_vehicle_contact = True
        ego_vel = ego_speed(ego_state)
        other_vel = ego_speed(neighbor)
        if ego_vel is None or other_vel is None:
            continue

        ego_mass = get_mass(ego_state)
        other_mass = get_mass(neighbor)
        ke_now = 0.5 * ego_mass * (ego_vel**2) + 0.5 * other_mass * (other_vel**2)

        n_id = neighbor_id(neighbor)
        prev_neighbor = prev_neighbors_by_id.get(n_id) if n_id is not None else None
        if prev_ego_state is not None and prev_neighbor is not None:
            prev_ego_vel = ego_speed(dict(prev_ego_state))
            prev_other_vel = ego_speed(dict(prev_neighbor))
            if prev_ego_vel is not None and prev_other_vel is not None:
                ke_prev = 0.5 * ego_mass * (prev_ego_vel**2) + 0.5 * other_mass * (prev_other_vel**2)
                max_ke_delta = max(max_ke_delta, ke_now - ke_prev)
                continue

        max_ke_delta = max(max_ke_delta, ke_now)

    if not has_vehicle_contact:
        return False, MISSING_DATA_MARGIN
    return True, -max_ke_delta
