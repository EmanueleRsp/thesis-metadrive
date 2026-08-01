"""Fixed normative component registry for Rulebook v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from thesis_rl.rulebook.v2.types import CacheDelta, MacroRule, MemoryDelta, RuleComponentResult
from thesis_rl.rulebook.v2.components.collision import evaluate_collision_impact
from thesis_rl.rulebook.v2.components.clearance import evaluate_clearance
from thesis_rl.rulebook.v2.components.progress import evaluate_progress
from thesis_rl.rulebook.v2.components.rss import evaluate_rss
from thesis_rl.rulebook.v2.components.rss_lateral import evaluate_rss_lateral
from thesis_rl.rulebook.v2.components.road import (
    evaluate_dashed_line,
    evaluate_offroad,
    evaluate_solid_line,
    evaluate_wrong_carriageway,
    evaluate_wrongway,
)
from thesis_rl.rulebook.v2.components.controls import (
    evaluate_crosswalk_yield,
    evaluate_signal_transition,
    evaluate_stop,
    evaluate_vehicle_yield,
)
from thesis_rl.rulebook.v2.components.ttc import evaluate_ttc


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
        "collision",
        MacroRule.COLLISION_IMPACT,
        evaluate_collision_impact,
        frozenset({"previous_contact_ids"}),
    ),
    ComponentDefinition("rss", MacroRule.DYNAMIC_INTERACTION_SAFETY, evaluate_rss),
    ComponentDefinition("rss_lateral", MacroRule.DYNAMIC_INTERACTION_SAFETY, evaluate_rss_lateral),
    ComponentDefinition("ttc", MacroRule.DYNAMIC_INTERACTION_SAFETY, evaluate_ttc),
    ComponentDefinition("clearance", MacroRule.DYNAMIC_INTERACTION_SAFETY, evaluate_clearance),
    ComponentDefinition("offroad", MacroRule.ROAD_TRAFFIC_COMPLIANCE, evaluate_offroad),
    ComponentDefinition("wrong_way", MacroRule.ROAD_TRAFFIC_COMPLIANCE, evaluate_wrongway),
    ComponentDefinition(
        "wrong_carriageway", MacroRule.ROAD_TRAFFIC_COMPLIANCE, evaluate_wrong_carriageway
    ),
    ComponentDefinition("solid_line", MacroRule.ROAD_TRAFFIC_COMPLIANCE, evaluate_solid_line),
    ComponentDefinition(
        "dashed_line",
        MacroRule.ROAD_TRAFFIC_COMPLIANCE,
        evaluate_dashed_line,
        frozenset({"active_dashed_boundary_id", "dashed_line_timer_s"}),
    ),
    ComponentDefinition(
        "signal",
        MacroRule.ROAD_TRAFFIC_COMPLIANCE,
        evaluate_signal_transition,
        frozenset(
            {
                "active_signal_group_id",
                "previous_signal_state",
                "yellow_must_stop",
                "previous_signal_delta_m",
                "resolved_signal_group_ids",
            }
        ),
    ),
    ComponentDefinition(
        "stop",
        MacroRule.ROAD_TRAFFIC_COMPLIANCE,
        evaluate_stop,
        frozenset(
            {
                "active_stop_group_id",
                "stop_continuous_timer_s",
                "stop_best_timer_s",
                "previous_stop_delta_m",
                "resolved_stop_group_ids",
            }
        ),
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
        evaluate_crosswalk_yield,
        frozenset({"crosswalk_illegal_entries"}),
    ),
    ComponentDefinition(
        "vehicle_yield",
        MacroRule.ROAD_TRAFFIC_COMPLIANCE,
        evaluate_vehicle_yield,
        frozenset({"vehicle_yield_illegal_entries", "frozen_actor_movement_keys"}),
    ),
    ComponentDefinition(
        "motion_history",
        MacroRule.ROAD_TRAFFIC_COMPLIANCE,
        None,
        frozenset({"actor_motion_histories", "previous_sim_time_s"}),
        normative_output=False,
    ),
    ComponentDefinition(
        "progress", MacroRule.ROUTE_PROGRESS, evaluate_progress, frozenset({"previous_route_s_m"})
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

    def definition(self, name: str) -> ComponentDefinition:
        """Return one fixed definition by canonical component name."""
        for component in self._components:
            if component.name == name:
                return component
        raise ValueError(f"Unknown Rulebook v2 component: {name!r}")

    def evaluate(self, name: str, **kwargs):
        """Invoke a pure normative evaluator through the fixed registry."""
        definition = self.definition(name)
        if not definition.normative_output or definition.evaluator is None:
            raise ValueError(f"Component {name!r} is infrastructure-only or unbound")
        return definition.evaluator(**kwargs)

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
            "actor_motion_histories",
            "previous_sim_time_s",
        }
        if set(owners) != expected_fields:
            raise ValueError(
                "Rulebook v2 registry does not assign every RulebookMemory field exactly once."
            )


DEFAULT_RULEBOOK_V2_REGISTRY = RulebookV2Registry()
