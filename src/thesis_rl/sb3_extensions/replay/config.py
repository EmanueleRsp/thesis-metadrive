"""Validation and normalization for the transition replay contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TransitionReplayConfig:
    """Validated replay configuration consumed by SB3 planner backends."""

    enabled: bool
    n_steps: int
    prioritized: bool
    optimize_memory_usage: bool
    store_reward_vector: bool
    beta_anneal_steps: int | None
    persistence_enabled: bool = False


def require_replay_buffer_for_resume(
    planner: Any,
    *,
    resumed_global_steps: int,
    replay_persistence_enabled: bool,
) -> None:
    """Refuse to resume an off-policy learner whose replay buffer was not persisted.

    A resumed replay learner restores ``num_timesteps`` from its checkpoint, so the
    ``learning_starts`` warm-up is already spent: it starts full-size gradient
    updates on a buffer holding only the first ``n_envs`` transitions, and every
    batch is drawn from those few rows until the buffer refills. Nothing raises, and
    the resumed policy silently overfits and collapses (audit 2026-09-06, A8).
    Failing here makes the loss of the buffer an explicit decision instead.

    On-policy learners keep no replay buffer and are exempt. Both the baseline
    training loop and the scenario-ACL driver call this, because both resume the
    same learners through their own separate resume paths.
    """

    if not hasattr(planner, "load_replay_buffer"):
        return
    if int(resumed_global_steps) <= 0 or replay_persistence_enabled:
        return
    raise RuntimeError(
        "Resuming an off-policy learner requires its persisted replay buffer: "
        f"the checkpoint records {int(resumed_global_steps)} training steps but "
        "`transition_replay.persistence.enabled` is false, so no buffer can be "
        "restored and the learner would update on a near-empty buffer without a "
        "warm-up. Enable replay persistence for the run being resumed, or restart it."
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("transition_replay must be a mapping.")
    return value


def resolve_transition_replay_config(
    raw_config: Any,
    *,
    algorithm_name: str,
    total_timesteps: int | None = None,
    legacy_save_replay_buffer: bool | None = None,
) -> TransitionReplayConfig:
    """Resolve the approved transition replay configuration for one algorithm.

    TD3 and SAC use the approved three-step uniform core by default. PPO may
    carry only an explicitly inactive section and never constructs replay.
    PER validation is reserved for the custom PER milestone.
    """

    algorithm = str(algorithm_name).strip().lower()
    raw = _mapping(raw_config)
    is_ppo = algorithm in {"ppo", "ppo_sb3"}
    if algorithm not in {"td3", "td3_sb3", "sac", "sac_sb3", "ppo", "ppo_sb3"}:
        raise ValueError(
            "transition_replay is not defined for this learner; provide a separate "
            "approved replay specification."
        )
    default_enabled = not is_ppo
    enabled = bool(raw.get("enabled", default_enabled))
    n_steps = int(raw.get("n_steps", 3))
    prioritized = bool(raw.get("prioritized", False))
    optimize_memory_usage = bool(raw.get("optimize_memory_usage", False))
    store_reward_vector = bool(raw.get("store_reward_vector", False))

    if n_steps not in {1, 3}:
        raise ValueError("transition_replay.n_steps must be one of {1, 3}.")
    if optimize_memory_usage:
        raise ValueError("transition_replay.optimize_memory_usage must be false.")

    persistence = _mapping(raw.get("persistence"))
    per = _mapping(raw.get("per"))
    persistence_enabled = bool(persistence.get("enabled", False))
    if legacy_save_replay_buffer:
        raise ValueError(
            "`checkpoint.save_replay_buffer=true` is no longer accepted; migrate to "
            "`transition_replay.persistence.enabled=true` with trigger=final_or_manual."
        )
    trigger = str(persistence.get("trigger", "final_or_manual")).strip().lower()
    if trigger != "final_or_manual":
        raise ValueError("transition_replay.persistence.trigger must be `final_or_manual` in v1.")
    if persistence.get("periodic_frequency_steps") is not None:
        raise ValueError("Periodic replay persistence is not supported in transition replay v1.")
    if int(persistence.get("keep_last", 1)) != 1:
        raise ValueError("transition_replay.persistence.keep_last must equal 1 in v1.")
    custom_replay = raw.get("replay_buffer_class") is not None
    if is_ppo:
        active_non_default = (
            enabled
            or prioritized
            or persistence_enabled
            or custom_replay
            or store_reward_vector
            or n_steps != 3
        )
        if active_non_default:
            raise ValueError(
                "PPO accepts only an explicitly inactive transition_replay "
                "configuration with schema defaults."
            )
        return TransitionReplayConfig(
            enabled=False,
            n_steps=1,
            prioritized=False,
            optimize_memory_usage=False,
            store_reward_vector=False,
            beta_anneal_steps=None,
            persistence_enabled=False,
        )

    if not enabled:
        if prioritized or persistence_enabled or custom_replay:
            raise ValueError(
                "Active transition replay features require transition_replay.enabled=true."
            )
        n_steps = 1

    beta_anneal_steps_raw = per.get("beta_anneal_steps", total_timesteps)
    beta_anneal_steps = None if beta_anneal_steps_raw is None else int(beta_anneal_steps_raw)
    if prioritized and (beta_anneal_steps is None or beta_anneal_steps <= 0):
        raise ValueError("PER requires a positive beta_anneal_steps value.")

    return TransitionReplayConfig(
        enabled=enabled,
        n_steps=n_steps,
        prioritized=prioritized,
        optimize_memory_usage=optimize_memory_usage,
        store_reward_vector=store_reward_vector,
        beta_anneal_steps=beta_anneal_steps,
        persistence_enabled=persistence_enabled,
    )
