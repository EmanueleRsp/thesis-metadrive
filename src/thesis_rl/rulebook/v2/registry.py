"""Fixed normative component registry for Rulebook v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from thesis_rl.rulebook.v2.types import CacheDelta, MacroRule, MemoryDelta, RuleComponentResult


ComponentEvaluator = Callable[..., tuple[RuleComponentResult, MemoryDelta, CacheDelta]]


@dataclass(frozen=True, slots=True)
class ComponentDefinition:
    name: str
    macro_rule: MacroRule
    evaluator: ComponentEvaluator | None
    owned_memory_fields: frozenset[str] = frozenset()
    normative_output: bool = True


_COMPONENTS: tuple[ComponentDefinition, ...] = (
    ComponentDefinition(
        "collision", MacroRule.COLLISION_IMPACT, None, frozenset({"previous_contact_ids"})
    ),
    ComponentDefinition("rss", MacroRule.DYNAMIC_INTERACTION_SAFETY, None),
    ComponentDefinition("ttc", MacroRule.DYNAMIC_INTERACTION_SAFETY, None),
    ComponentDefinition("clearance", MacroRule.DYNAMIC_INTERACTION_SAFETY, None),
    ComponentDefinition("offroad", MacroRule.ROAD_TRAFFIC_COMPLIANCE, None),
    ComponentDefinition("wrong_way", MacroRule.ROAD_TRAFFIC_COMPLIANCE, None),
    ComponentDefinition("solid_line", MacroRule.ROAD_TRAFFIC_COMPLIANCE, None),
    ComponentDefinition(
        "dashed_line",
        MacroRule.ROAD_TRAFFIC_COMPLIANCE,
        None,
        frozenset({"active_dashed_boundary_id", "dashed_line_timer_s"}),
    ),
    ComponentDefinition(
        "signal",
        MacroRule.ROAD_TRAFFIC_COMPLIANCE,
        None,
        frozenset({
            "active_signal_group_id", "previous_signal_state", "yellow_must_stop",
            "previous_signal_delta_m", "resolved_signal_group_ids",
        }),
    ),
    ComponentDefinition(
        "stop",
        MacroRule.ROAD_TRAFFIC_COMPLIANCE,
        None,
        frozenset({
            "active_stop_group_id", "stop_continuous_timer_s", "stop_best_timer_s",
            "previous_stop_delta_m", "resolved_stop_group_ids",
        }),
    ),
    ComponentDefinition(
        "zone_lifecycle",
        MacroRule.ROAD_TRAFFIC_COMPLIANCE,
        None,
        frozenset({"preexisting_ego_occupancy_zone_ids"}),
        normative_output=False,
    ),
    ComponentDefinition(
        "crosswalk",
        MacroRule.ROAD_TRAFFIC_COMPLIANCE,
        None,
        frozenset({"crosswalk_illegal_entries"}),
    ),
    ComponentDefinition(
        "vehicle_yield",
        MacroRule.ROAD_TRAFFIC_COMPLIANCE,
        None,
        frozenset({"vehicle_yield_illegal_entries", "frozen_actor_movement_keys"}),
    ),
    ComponentDefinition(
        "progress", MacroRule.ROUTE_PROGRESS, None, frozenset({"previous_route_s_m"})
    ),
)


class RulebookV2Registry:
    """Validates the fixed v2 composition before runtime evaluation starts."""

    def __init__(self, components: tuple[ComponentDefinition, ...] = _COMPONENTS) -> None:
        self._components = components
        self.validate()

    @property
    def components(self) -> tuple[ComponentDefinition, ...]:
        return self._components

    def validate(self) -> None:
        names = [component.name for component in self._components]
        if len(names) != len(set(names)):
            raise ValueError("Rulebook v2 registry contains duplicate component names.")
        expected_names = [component.name for component in _COMPONENTS]
        if names != expected_names:
            raise ValueError(
                "Rulebook v2 registry must contain the fixed normative component sequence."
            )
        macro_order = [component.macro_rule for component in self._components]
        if macro_order != sorted(macro_order, key=lambda item: list(MacroRule).index(item)):
            raise ValueError("Rulebook v2 registry component order must follow macro-rule order.")
        owners: dict[str, str] = {}
        for component in self._components:
            for field in component.owned_memory_fields:
                previous = owners.setdefault(field, component.name)
                if previous != component.name:
                    raise ValueError(
                        f"Memory field {field!r} has multiple writers: "
                        f"{previous!r}, {component.name!r}."
                    )
        expected_fields = {
            "previous_contact_ids",
            "active_dashed_boundary_id",
            "dashed_line_timer_s",
            "active_signal_group_id",
            "previous_signal_state",
            "yellow_must_stop",
            "previous_signal_delta_m",
            "resolved_signal_group_ids",
            "active_stop_group_id",
            "stop_continuous_timer_s",
            "stop_best_timer_s",
            "previous_stop_delta_m",
            "resolved_stop_group_ids",
            "crosswalk_illegal_entries",
            "vehicle_yield_illegal_entries",
            "preexisting_ego_occupancy_zone_ids",
            "frozen_actor_movement_keys",
            "previous_route_s_m",
        }
        if set(owners) != expected_fields:
            raise ValueError(
                "Rulebook v2 registry does not assign every RulebookMemory field exactly once."
            )


DEFAULT_RULEBOOK_V2_REGISTRY = RulebookV2Registry()
