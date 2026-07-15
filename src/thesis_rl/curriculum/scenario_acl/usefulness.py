"""Scenario usefulness and optional rulebook diagnostics for Scenario ACL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ScenarioUsefulness:
    """Learning-potential usefulness with separately logged rule criticality."""

    rule_criticality: int
    learning_potential: float
    value: float
    dominant_rule: str | None


_CRITICALITY_BY_RULE = {
    # Rulebook v1 names remain supported for existing ACL records.
    "collision_severity": 3,
    "allowed_driving_area": 3,
    "lane_marking_compliance": 2,
    "local_route_progress": 1,
    # Rulebook v2 macro names follow the canonical lexicographic priority.
    "collision_impact": 3,
    "dynamic_interaction_safety": 2,
    "road_traffic_compliance": 2,
    "route_progress": 1,
}


def compute_learning_potential(
    train_summary: Mapping[str, Any],
    *,
    planner_name: str,
) -> float:
    """Return the current algorithm's scenario-level learning signal.

    Semantic ACL selects a generated or replayed scenario before every
    episode. The critic/value loss available when that episode finishes is
    the learning signal used for its MAB feedback: value loss for PPO and
    critic loss for TD3/SAC. Missing or non-finite losses are rejected rather
    than replaced with an evaluation-based proxy.
    """
    backend = str(planner_name).strip().lower()
    if backend not in {"ppo", "ppo_sb3", "td3", "td3_sb3", "sac", "sac_sb3"}:
        raise ValueError(f"Unsupported planner for ACL learning potential: '{planner_name}'.")
    value = train_summary.get("critic_loss_ema")
    if value is None:
        value = train_summary.get("critic_loss")
    try:
        potential = abs(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("ACL learning potential requires a numeric critic/value loss.") from exc
    if not math.isfinite(potential):
        raise ValueError("ACL learning potential requires a finite critic/value loss.")
    if int(train_summary.get("update_calls", 0)) <= 0:
        raise ValueError("ACL learning potential requires at least one planner update per chunk.")
    return potential


def compute_rule_criticality(metrics: Mapping[str, Any]) -> tuple[int, str | None]:
    """Return the criticality of the highest-priority violated Rulebook v1 rule.

    Rulebook v1 constraints are satisfied at margin zero and violated below
    zero.  The progress objective is safety-relevant only when it regresses.
    This deliberately uses no calibrated scalar scale, so its semantics remain
    stable while reward scalarization is tuned separately.
    """
    rows = metrics.get("per_rule", [])
    if not isinstance(rows, Sequence):
        return 0, None

    best_level = 0
    best_name: str | None = None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("rule_name", ""))
        level = _CRITICALITY_BY_RULE.get(name, 0)
        if level <= 0:
            continue
        try:
            violated = float(row.get("min_margin", 0.0)) < 0.0
        except (TypeError, ValueError):
            continue
        if violated and level > best_level:
            best_level = level
            best_name = name
    return best_level, best_name


def compute_scenario_usefulness(
    metrics: Mapping[str, Any],
    *,
    learning_potential: float,
) -> ScenarioUsefulness:
    """Keep replay usefulness equal to learning potential only.

    Rule criticality is retained solely for analysis.  It must not alter MAB
    feedback, buffer replacement, ranks, or replay sampling.
    """
    criticality, dominant_rule = compute_rule_criticality(metrics)
    value = float(learning_potential)
    return ScenarioUsefulness(
        rule_criticality=criticality,
        learning_potential=float(learning_potential),
        value=value,
        dominant_rule=dominant_rule,
    )
