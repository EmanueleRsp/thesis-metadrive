from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from thesis_rl.rulebook import rule_functions
from thesis_rl.rulebook.types import RuleSpec

RULE_REGISTRY: dict[str, Callable[..., tuple[bool, float]]] = {
    "vru_collision_energy": rule_functions.check_vru_collision_energy,
    "vehicle_collision_energy": rule_functions.check_vehicle_collision_energy,
    "drivable_area": rule_functions.check_drivable_area,
    "wrong_way": rule_functions.check_wrong_way,
    "speed_limit": rule_functions.check_speed_limit,
    "lane_centering": rule_functions.check_lane_centering,
    "goal_progress": rule_functions.check_goal_progress,
    "longitudinal_accel": rule_functions.check_longitudinal_accel,
    "lateral_accel": rule_functions.check_lateral_accel,
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
