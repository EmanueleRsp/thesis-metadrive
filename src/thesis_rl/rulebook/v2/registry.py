"""Fixed normative component registry for Rulebook v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from thesis_rl.rulebook.v2.types import CacheDelta, MacroRule, MemoryDelta, RuleComponentResult
from thesis_rl.rulebook.v2.components.collision import evaluate_collision_impact
from thesis_rl.rulebook.v2.components.clearance import evaluate_clearance
from thesis_rl.rulebook.v2.components.progress import evaluate_progress
from thesis_rl.rulebook.v2.components.progress_rate import evaluate_progress_rate
from thesis_rl.rulebook.v2.components.rss import evaluate_rss
from thesis_rl.rulebook.v2.components.rss_lateral import evaluate_rss_lateral
from thesis_rl.rulebook.v2.components.speed_limit import evaluate_speed_limit
from thesis_rl.rulebook.v2.components.road import (
    evaluate_dashed_line,
    evaluate_offroad,
    evaluate_solid_line,
    evaluate_wrong_carriageway,
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
    """One registered sub-rule, and the two independent things it may not do.

    ``normative_output`` is *whether it is evaluated at all*: infrastructure
    entries such as ``motion_history`` own memory fields but produce no cost and
    have no evaluator.

    ``contributes_to_channel`` is *whether its cost reaches a level*. ADR-063
    demoted ``rss`` longitudinal to "a reported diagnostic, never in the
    reward": it is still evaluated and still published, and only aggregation
    ignores it. Overloading the first flag for that would have silently stopped
    reporting it, which is the opposite of what the ADR decided.
    """

    name: str
    macro_rule: MacroRule
    evaluator: ComponentEvaluator | None
    owned_memory_fields: frozenset[str] = frozenset()
    normative_output: bool = True
    contributes_to_channel: bool = True


# RULEBOOK-V5.1 §3. Ordered by level, because `validate` requires it and because
# reading the tuple should read the hierarchy.
_COMPONENTS: tuple[ComponentDefinition, ...] = (
    # L1 -- collision safety
    ComponentDefinition(
        "collision",
        MacroRule.COLLISION_SAFETY,
        evaluate_collision_impact,
        frozenset({"previous_contact_ids"}),
    ),
    # L2 -- interaction risk. `rss` longitudinal stays registered but
    # NON-NORMATIVE (ADR-063): it fires on 18.29 % of applicable expert steps and
    # failed the controlled-invariance test, so it is reported and never priced.
    ComponentDefinition(
        "rss",
        MacroRule.INTERACTION_RISK,
        evaluate_rss,
        contributes_to_channel=False,
    ),
    ComponentDefinition("rss_lateral", MacroRule.INTERACTION_RISK, evaluate_rss_lateral),
    ComponentDefinition("ttc", MacroRule.INTERACTION_RISK, evaluate_ttc),
    ComponentDefinition("clearance", MacroRule.INTERACTION_RISK, evaluate_clearance),
    # L3 -- non-relaxable compliance. `wrongway` is deleted, not demoted
    # (ADR-066): one violated step in 217,189 of expert replay, and
    # `wrong_carriageway` covers the observable subject.
    ComponentDefinition("offroad", MacroRule.NON_RELAXABLE_COMPLIANCE, evaluate_offroad),
    ComponentDefinition(
        "signal",
        MacroRule.NON_RELAXABLE_COMPLIANCE,
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
        MacroRule.NON_RELAXABLE_COMPLIANCE,
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
        "crosswalk",
        MacroRule.NON_RELAXABLE_COMPLIANCE,
        evaluate_crosswalk_yield,
        frozenset({"crosswalk_illegal_entries"}),
    ),
    ComponentDefinition(
        "vehicle_yield",
        MacroRule.NON_RELAXABLE_COMPLIANCE,
        evaluate_vehicle_yield,
        frozenset({"vehicle_yield_illegal_entries", "frozen_actor_movement_keys"}),
    ),
    ComponentDefinition(
        "speed_limit",
        MacroRule.NON_RELAXABLE_COMPLIANCE,
        evaluate_speed_limit,
    ),
    ComponentDefinition(
        "zone_lifecycle",
        MacroRule.NON_RELAXABLE_COMPLIANCE,
        None,
        frozenset({"preexisting_ego_occupancy_zone_ids"}),
        normative_output=False,
    ),
    ComponentDefinition(
        "motion_history",
        MacroRule.NON_RELAXABLE_COMPLIANCE,
        None,
        frozenset({"actor_motion_histories", "previous_sim_time_s"}),
        normative_output=False,
    ),
    # L4 -- mission progress
    ComponentDefinition("progress", MacroRule.MISSION_PROGRESS, evaluate_progress),
    # L5 -- relaxable lane compliance (ADR-072): the rules a competent driver
    # may relax in order to complete a mission, therefore BELOW progress.
    ComponentDefinition("solid_line", MacroRule.RELAXABLE_LANE_COMPLIANCE, evaluate_solid_line),
    ComponentDefinition(
        "wrong_carriageway",
        MacroRule.RELAXABLE_LANE_COMPLIANCE,
        evaluate_wrong_carriageway,
    ),
    ComponentDefinition(
        "dashed_line",
        MacroRule.RELAXABLE_LANE_COMPLIANCE,
        evaluate_dashed_line,
        frozenset({"active_dashed_boundary_id", "dashed_line_timer_s"}),
    ),
    # L6 -- progress rate (ADR-076)
    ComponentDefinition("advance_shortfall", MacroRule.PROGRESS_RATE, evaluate_progress_rate),
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
            "actor_motion_histories",
            "previous_sim_time_s",
        }
        if set(owners) != expected_fields:
            raise ValueError(
                "Rulebook v2 registry does not assign every RulebookMemory field exactly once."
            )


DEFAULT_RULEBOOK_V2_REGISTRY = RulebookV2Registry()
