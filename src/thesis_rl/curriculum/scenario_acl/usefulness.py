"""Scenario usefulness and optional rulebook diagnostics for Scenario ACL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


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


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"ACL learning potential requires a non-empty 1-D {name}.")
    if not np.isfinite(array).all():
        raise ValueError(f"ACL learning potential requires finite {name}.")
    return array


def compute_ppo_learning_potential(
    *,
    advantages: Any | None = None,
    rewards: Any | None = None,
    values: Any | None = None,
    next_values: Any | None = None,
    dones: Any | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.9,
) -> float:
    """Compute PPO learning potential from positive GAE-style residuals."""
    if not 0.0 < float(gamma) <= 1.0 or not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError("PPO learning-potential gamma must be in (0,1] and lambda in [0,1].")
    if advantages is not None:
        gae = _finite_vector(advantages, name="advantages")
    else:
        if rewards is None or values is None or next_values is None or dones is None:
            raise ValueError(
                "PPO learning potential requires advantages or rewards, values, "
                "next_values, and dones."
            )
        reward_array = _finite_vector(rewards, name="rewards")
        value_array = _finite_vector(values, name="values")
        next_value_array = _finite_vector(next_values, name="next_values")
        done_array = np.asarray(dones, dtype=bool)
        if done_array.ndim == 0:
            done_array = done_array.reshape(1)
        if not (
            reward_array.shape == value_array.shape == next_value_array.shape == done_array.shape
        ):
            raise ValueError("PPO learning-potential inputs must have equal lengths.")
        deltas = reward_array + float(gamma) * (~done_array) * next_value_array - value_array
        gae = np.zeros_like(deltas)
        running = 0.0
        for index in range(deltas.size - 1, -1, -1):
            running = float(deltas[index]) + float(gamma) * float(gae_lambda) * (
                0.0 if done_array[index] else running
            )
            gae[index] = running
    return float(np.maximum(gae, 0.0).mean())


def compute_td3_learning_potential(td_residuals: Any) -> float:
    """Compute TD3 learning potential as mean positive-part TD residual.

    DEC-006: `max(delta, 0)` instead of `|delta|`, matching the production
    computation at `agent/planners/core/lifecycle.py:acl_learning_potential`
    and restoring the ZPD hopelessness filter PPO already has structurally.
    """
    return float(np.maximum(_finite_vector(td_residuals, name="TD3 residuals"), 0.0).mean())


def compute_sac_learning_potential(td_residuals: Any) -> float:
    """Compute SAC learning potential as mean positive-part entropy-aware residual.

    DEC-006: `max(delta, 0)` instead of `|delta|`; see `compute_td3_learning_potential`.
    """
    return float(np.maximum(_finite_vector(td_residuals, name="SAC residuals"), 0.0).mean())


def compute_td3_td_residuals(
    *,
    rewards: Any,
    dones: Any,
    target_q: Any,
    current_q: Any,
    gamma: float = 0.99,
) -> np.ndarray:
    """Build TD3 residuals from target and current minimum-Q estimates."""
    reward_array = _finite_vector(rewards, name="rewards")
    target_array = _finite_vector(target_q, name="target Q values")
    current_array = _finite_vector(current_q, name="current Q values")
    done_array = np.asarray(dones, dtype=bool).reshape(-1)
    if not (reward_array.shape == target_array.shape == current_array.shape == done_array.shape):
        raise ValueError("TD3 residual inputs must have equal lengths.")
    return reward_array + float(gamma) * (~done_array) * target_array - current_array


def compute_sac_td_residuals(
    *,
    rewards: Any,
    dones: Any,
    target_q: Any,
    current_q: Any,
    log_pi: Any,
    entropy_temperature: float,
    gamma: float = 0.99,
) -> np.ndarray:
    """Build SAC residuals including the target policy entropy term."""
    target_array = _finite_vector(target_q, name="target Q values")
    log_pi_array = _finite_vector(log_pi, name="policy log probabilities")
    if target_array.shape != log_pi_array.shape:
        raise ValueError("SAC target Q and log-probability inputs must have equal lengths.")
    entropy_adjusted_target = target_array - float(entropy_temperature) * log_pi_array
    return compute_td3_td_residuals(
        rewards=rewards,
        dones=dones,
        target_q=entropy_adjusted_target,
        current_q=current_q,
        gamma=gamma,
    )


def compute_learning_potential(
    train_summary: Mapping[str, Any],
    *,
    planner_name: str,
) -> float:
    """Dispatch the ACL §12 algorithm-specific learning-potential formula.

    Backends may provide a precomputed ``learning_potential`` after collecting
    the algorithm's own residuals. Critic/value losses alone are deliberately
    rejected because they are not the ACL §12 definitions.
    """
    backend = str(planner_name).strip().lower()
    if backend not in {"ppo", "ppo_sb3", "td3", "td3_sb3", "sac", "sac_sb3"}:
        raise ValueError(f"Unsupported planner for ACL learning potential: '{planner_name}'.")
    if int(train_summary.get("update_calls", 0)) <= 0:
        raise ValueError("ACL learning potential requires at least one planner update per chunk.")
    if train_summary.get("learning_potential") is not None:
        potential = float(train_summary["learning_potential"])
        if not math.isfinite(potential) or potential < 0.0:
            raise ValueError("ACL learning potential must be finite and non-negative.")
        return potential
    if backend in {"ppo", "ppo_sb3"}:
        return compute_ppo_learning_potential(
            advantages=train_summary.get("ppo_advantages", train_summary.get("advantages")),
            rewards=train_summary.get("rewards"),
            values=train_summary.get("values"),
            next_values=train_summary.get("next_values"),
            dones=train_summary.get("dones"),
            gamma=float(train_summary.get("gamma", 0.99)),
            gae_lambda=float(train_summary.get("gae_lambda", 0.9)),
        )
    if backend in {"td3", "td3_sb3"}:
        residuals = train_summary.get("td3_td_residuals", train_summary.get("td_residuals"))
        if residuals is None:
            required = ("rewards", "dones", "target_q", "current_q")
            if any(train_summary.get(key) is None for key in required):
                raise ValueError("TD3 learning potential requires TD3 residual inputs.")
            residuals = compute_td3_td_residuals(
                rewards=train_summary.get("rewards"),
                dones=train_summary.get("dones"),
                target_q=train_summary.get("target_q"),
                current_q=train_summary.get("current_q"),
                gamma=float(train_summary.get("gamma", 0.99)),
            )
        return compute_td3_learning_potential(residuals)
    residuals = train_summary.get("sac_td_residuals", train_summary.get("td_residuals"))
    if residuals is None:
        required = ("rewards", "dones", "target_q", "current_q", "log_pi")
        if any(train_summary.get(key) is None for key in required):
            raise ValueError("SAC learning potential requires entropy-aware TD residual inputs.")
        residuals = compute_sac_td_residuals(
            rewards=train_summary.get("rewards"),
            dones=train_summary.get("dones"),
            target_q=train_summary.get("target_q"),
            current_q=train_summary.get("current_q"),
            log_pi=train_summary.get("log_pi"),
            entropy_temperature=float(train_summary.get("entropy_temperature", 0.0)),
            gamma=float(train_summary.get("gamma", 0.99)),
        )
    return compute_sac_learning_potential(residuals)


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
