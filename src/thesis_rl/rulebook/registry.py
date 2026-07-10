from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from thesis_rl.rulebook.rules import (
    check_drivable_area,
    check_goal_progress,
    check_lane_centering,
    check_lateral_accel,
    check_longitudinal_accel,
    check_speed_limit,
    check_vehicle_collision_energy,
    check_vru_collision_energy,
    check_wrong_way,
    allowed_driving_area,
    collision_severity,
    lane_marking_compliance,
    local_route_progress,
)
from thesis_rl.rulebook.types import RuleSpec

RULE_REGISTRY: dict[str, Callable[..., Any]] = {
    "collision_severity": collision_severity,
    "allowed_driving_area": allowed_driving_area,
    "lane_marking_compliance": lane_marking_compliance,
    "local_route_progress": local_route_progress,
    "vru_collision_energy": check_vru_collision_energy,
    "vehicle_collision_energy": check_vehicle_collision_energy,
    "drivable_area": check_drivable_area,
    "wrong_way": check_wrong_way,
    "speed_limit": check_speed_limit,
    "lane_centering": check_lane_centering,
    "goal_progress": check_goal_progress,
    "longitudinal_accel": check_longitudinal_accel,
    "lateral_accel": check_lateral_accel,
}


def load_rulebook_from_config(config: Mapping[str, Any]) -> list[RuleSpec]:
    """Load ordered RuleSpec list from YAML-compatible mapping.

    Rules are sorted by (priority, yaml order index).
    """
    items = list(config.get("rules", []))
    specs: list[RuleSpec] = []

    for idx, item in enumerate(items):
        rule_name = str(item["name"])
        if rule_name not in RULE_REGISTRY:
            available = ", ".join(sorted(RULE_REGISTRY.keys()))
            raise ValueError(f"Unknown rule '{rule_name}'. Available: {available}")

        specs.append(
            RuleSpec(
                name=rule_name,
                fn=RULE_REGISTRY[rule_name],
                priority=int(item.get("priority", 0)),
                params=dict(item.get("params", {})),
                order=idx,
            )
        )

    return sorted(specs, key=lambda spec: (spec.priority, spec.order))
