"""Sub-rule dominance and cost diagnostics for Rulebook v2 R2/R3 (EP-SUBRULE-DIAG).

Additive diagnostic reporting, not a rulebook or reward change: see
`docs/implementation/subrule_dominance_diagnostics_exec_plan.md`. It answers,
representatively over all evaluation episodes, whether the `max` aggregation
of `aggregate_max_component` (`rulebook/v2/aggregation.py`) lets one sub-rule
silently dominate its macro rule (`dominance_share`), how the sub-rules' cost
distributions compare (left to the caller, via raw per-step costs), and how
often more than one sub-rule is violated at once (`multi_violation_share`).

Scope (`DEC-SUB-002`): only `dynamic_interaction_safety` (R2) and
`road_traffic_compliance` (R3). `collision_impact` (R1)'s sub-components are
per-actor contact onsets, not heterogeneous cost metrics, so the
commensurability question this module answers does not apply there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DIAGNOSTIC_MACRO_RULES: tuple[str, ...] = (
    "dynamic_interaction_safety",
    "road_traffic_compliance",
)


@dataclass(frozen=True, slots=True)
class SubruleStepObservation:
    macro_rule: str
    applicable: bool
    cost: float


@dataclass(frozen=True, slots=True)
class MacroStepObservation:
    applicable: bool
    violated: bool
    worst_component: str | None
    violated_subrule_count: int


def extract_subrule_step(
    step_info: Any,
) -> tuple[dict[str, SubruleStepObservation], dict[str, MacroStepObservation]]:
    """Read one step's sub-rule and macro-rule state for the `DIAGNOSTIC_MACRO_RULES`.

    Mirrors `Agent._extract_rule_applicability`: reads the already-computed
    `step_info["rule_components"]` (`RulebookV2MonitorWrapper.step`, every
    component of `aggregate_rulebook_result`) without recomputing the
    rulebook. A missing or malformed entry is treated as absent, not as an
    error, matching the existing extractor's tolerance for incomplete
    `step_info`.
    """
    if not isinstance(step_info, dict):
        return {}, {}
    components = step_info.get("rule_components")
    if not isinstance(components, dict):
        return {}, {}

    macros: dict[str, MacroStepObservation] = {}
    subrules: dict[str, SubruleStepObservation] = {}
    for macro_name in DIAGNOSTIC_MACRO_RULES:
        macro = components.get(macro_name)
        if not isinstance(macro, dict):
            continue
        macro_applicable = bool(macro.get("applicable", False))
        macro_cost = float(macro.get("cost", 0.0))
        macro_violated = macro_applicable and macro_cost > 0.0
        raw = macro.get("raw")
        subcomponents = raw.get("subcomponents") if isinstance(raw, dict) else None
        diagnostics = macro.get("diagnostics")
        worst_component = (
            diagnostics.get("worst_component") if isinstance(diagnostics, dict) else None
        )
        violated_count = 0
        if isinstance(subcomponents, (list, tuple)):
            for sub in subcomponents:
                if not isinstance(sub, dict):
                    continue
                name = sub.get("name")
                if not isinstance(name, str):
                    continue
                sub_applicable = bool(sub.get("applicable", False))
                sub_cost = float(sub.get("cost", 0.0))
                subrules[name] = SubruleStepObservation(
                    macro_rule=macro_name,
                    applicable=sub_applicable,
                    cost=sub_cost,
                )
                if sub_applicable and sub_cost > 0.0:
                    violated_count += 1
        macros[macro_name] = MacroStepObservation(
            applicable=macro_applicable,
            violated=macro_violated,
            worst_component=(worst_component if macro_applicable and macro_violated else None),
            violated_subrule_count=violated_count,
        )
    return subrules, macros


@dataclass
class SubruleEpisodeAccumulator:
    """Per-episode sub-rule accumulator, observed step by step during `Agent.evaluate()`."""

    _applicable_steps: dict[str, int] = field(default_factory=dict)
    _violated_steps: dict[str, int] = field(default_factory=dict)
    _cost_sum: dict[str, float] = field(default_factory=dict)
    _cost_max: dict[str, float] = field(default_factory=dict)
    _worst_count: dict[str, int] = field(default_factory=dict)
    _parent_macro: dict[str, str] = field(default_factory=dict)
    _macro_violated_steps: dict[str, int] = field(default_factory=dict)
    _macro_multi_violation_steps: dict[str, int] = field(default_factory=dict)
    _total_steps: int = 0

    def observe(self, step_info: Any) -> None:
        self._total_steps += 1
        subrules, macros = extract_subrule_step(step_info)
        for macro_name, macro_obs in macros.items():
            if not macro_obs.applicable:
                continue
            if macro_obs.violated:
                self._macro_violated_steps[macro_name] = (
                    self._macro_violated_steps.get(macro_name, 0) + 1
                )
                # REQ-SUB-05: share of macro-violated steps with >=2 violated
                # sub-rules simultaneously (the `max` had a live runner-up).
                if macro_obs.violated_subrule_count >= 2:
                    self._macro_multi_violation_steps[macro_name] = (
                        self._macro_multi_violation_steps.get(macro_name, 0) + 1
                    )
        for name, obs in subrules.items():
            self._parent_macro[name] = obs.macro_rule
            if not obs.applicable:
                continue
            self._applicable_steps[name] = self._applicable_steps.get(name, 0) + 1
            self._cost_sum[name] = self._cost_sum.get(name, 0.0) + obs.cost
            self._cost_max[name] = max(self._cost_max.get(name, 0.0), obs.cost)
            if obs.cost > 0.0:
                self._violated_steps[name] = self._violated_steps.get(name, 0) + 1
            macro_obs = macros.get(obs.macro_rule)
            # REQ-SUB-02: dominance is counted only over steps where the
            # parent macro is actually violated, never over an all-zero max.
            if macro_obs is not None and macro_obs.violated and macro_obs.worst_component == name:
                self._worst_count[name] = self._worst_count.get(name, 0) + 1

    def finalize(self) -> dict[str, dict[str, Any]]:
        """Return per-subrule episode summaries, keyed by subrule name."""
        result: dict[str, dict[str, Any]] = {}
        for name, parent in self._parent_macro.items():
            result[name] = {
                "macro_rule": parent,
                "applicable_step_count": self._applicable_steps.get(name, 0),
                "violated_step_count": self._violated_steps.get(name, 0),
                "cost_sum": self._cost_sum.get(name, 0.0),
                "cost_max": self._cost_max.get(name, 0.0),
                "worst_component_count": self._worst_count.get(name, 0),
                "macro_violated_step_count": self._macro_violated_steps.get(parent, 0),
                "macro_multi_violation_step_count": self._macro_multi_violation_steps.get(
                    parent, 0
                ),
                "total_step_count": self._total_steps,
            }
        return result


def aggregate_subrule_episodes(
    episode_summaries: list[dict[str, dict[str, Any]]],
    episode_sources: list[str | None],
) -> list[dict[str, Any]]:
    """Aggregate per-episode sub-rule summaries into seed-level diagnostic rows.

    One row per `(scenario_source, macro_rule, subrule_name)`, disaggregated
    by scenario source (`REQ-SUB-04`) since Waymo-urban and PG-highway
    distributions differ and a pooled mean would hide both. A missing source
    is grouped under `"unknown"` rather than dropped, so nothing is silently
    excluded.
    """
    if len(episode_summaries) != len(episode_sources):
        raise ValueError("episode_summaries and episode_sources must have equal length")

    buckets: dict[tuple[str, str, str], dict[str, float]] = {}
    included_episodes: dict[tuple[str, str, str], int] = {}
    total_episodes_by_source: dict[str, int] = {}

    for summary, source in zip(episode_summaries, episode_sources):
        source_key = str(source) if source is not None else "unknown"
        total_episodes_by_source[source_key] = total_episodes_by_source.get(source_key, 0) + 1
        for subrule_name, stats in summary.items():
            macro_rule = str(stats["macro_rule"])
            key = (source_key, macro_rule, subrule_name)
            bucket = buckets.setdefault(
                key,
                {
                    "applicable_step_count": 0.0,
                    "violated_step_count": 0.0,
                    "cost_sum": 0.0,
                    "cost_max": 0.0,
                    "worst_component_count": 0.0,
                    "macro_violated_step_count": 0.0,
                    "macro_multi_violation_step_count": 0.0,
                    "total_step_count": 0.0,
                },
            )
            if stats["applicable_step_count"] > 0:
                included_episodes[key] = included_episodes.get(key, 0) + 1
            bucket["applicable_step_count"] += float(stats["applicable_step_count"])
            bucket["violated_step_count"] += float(stats["violated_step_count"])
            bucket["cost_sum"] += float(stats["cost_sum"])
            bucket["cost_max"] = max(bucket["cost_max"], float(stats["cost_max"]))
            bucket["worst_component_count"] += float(stats["worst_component_count"])
            bucket["macro_violated_step_count"] += float(stats["macro_violated_step_count"])
            bucket["macro_multi_violation_step_count"] += float(
                stats["macro_multi_violation_step_count"]
            )
            bucket["total_step_count"] += float(stats["total_step_count"])

    rows: list[dict[str, Any]] = []
    for (source_key, macro_rule, subrule_name), bucket in sorted(buckets.items()):
        applicable = bucket["applicable_step_count"]
        macro_violated = bucket["macro_violated_step_count"]
        total_steps = bucket["total_step_count"]
        included = included_episodes.get((source_key, macro_rule, subrule_name), 0)
        total_for_source = total_episodes_by_source.get(source_key, 0)
        rows.append(
            {
                "scenario_source": source_key,
                "macro_rule": macro_rule,
                "subrule_name": subrule_name,
                "applicability_rate": (applicable / total_steps) if total_steps > 0 else 0.0,
                "violation_rate": (
                    (bucket["violated_step_count"] / applicable) if applicable > 0 else 0.0
                ),
                "mean_cost": (bucket["cost_sum"] / applicable) if applicable > 0 else 0.0,
                "max_cost": bucket["cost_max"],
                "dominance_share": (
                    (bucket["worst_component_count"] / macro_violated)
                    if macro_violated > 0
                    else 0.0
                ),
                "worst_component_count": int(bucket["worst_component_count"]),
                "macro_violated_step_count": int(macro_violated),
                "multi_violation_share": (
                    (bucket["macro_multi_violation_step_count"] / macro_violated)
                    if macro_violated > 0
                    else 0.0
                ),
                "applicable_episode_count": included,
                "excluded_episode_count": int(total_for_source - included),
            }
        )
    return rows
