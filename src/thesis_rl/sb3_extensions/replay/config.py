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


# The five `per:` keys the planners actually forward to the buffer constructor.
# `td3_sb3.py` and `sac_sb3.py` select exactly these by a hard-coded whitelist
# and drop everything else in the block without a word.
_LIVE_PER_SETTINGS: frozenset[str] = frozenset(
    {"alpha", "beta_initial", "beta_final", "beta_anneal_steps", "epsilon"}
)

# The other five. Each one *accurately describes* what the implementation
# hard-codes, and that is exactly the problem: they read as configuration while
# being documentation, so an override parses, is logged, reaches the checkpoint's
# configuration record, and changes nothing — an ablation driven from them would
# report "no effect" for a knob that was never connected. `C24` (six encoder
# settings) and `C35` (seventeen observation settings) were the same defect, and
# this is their remedy: refuse a divergent value rather than delete the
# declaration, so the block keeps describing the mechanism accurately.
_FROZEN_PER_SETTINGS: tuple[tuple[str, str], ...] = (
    # The SB3 fork's `td3.py`/`sac.py`: 0.5 * the sum of the twin critics' |TD|.
    ("priority_aggregation", "mean_abs_twin_td"),
    # `PrioritizedNStepReplayBuffer._insertion_priority`, per ADR-080 and
    # `TRANSITION-REPLAY-V1.0.1`: the buffer's exact current maximum.
    ("new_transition_priority", "current_max"),
    # `update_priorities` reduces repeated indices with `np.maximum.at`.
    ("duplicate_update_reduction", "max"),
    # `_sample_addresses` draws one address per `np.linspace` stratum.
    ("sampling", "proportional_stratified"),
    # `_SumTree.__init__` allocates its nodes as `np.float64`.
    ("tree_dtype", "float64"),
)

_KNOWN_PER_KEYS: frozenset[str] = _LIVE_PER_SETTINGS | {key for key, _ in _FROZEN_PER_SETTINGS}

_KNOWN_REPLAY_KEYS: frozenset[str] = frozenset(
    {
        "enabled",
        "n_steps",
        "prioritized",
        "optimize_memory_usage",
        "store_reward_vector",
        "persistence",
        "per",
        "replay_buffer_class",
    }
)

_KNOWN_PERSISTENCE_KEYS: frozenset[str] = frozenset(
    {"enabled", "trigger", "periodic_frequency_steps", "keep_last"}
)


def _reject_unknown_keys(section: Mapping[str, Any], *, known: frozenset[str], path: str) -> None:
    """Refuse a key this resolver does not read, so a typo cannot pass silently.

    Every value in these sections is read with a defaulting `.get`, which means a
    misspelled key is indistinguishable from an absent one: `keep_last` mistyped
    silently restores the default the explicit value was there to override, and a
    mistyped `periodic_frequency_steps` silently disables the guard that refuses
    it. Naming the unknown key is the whole fix.
    """

    unknown = sorted(str(key) for key in section if str(key) not in known)
    if unknown:
        raise ValueError(
            f"Unknown {path} key(s) {unknown}: this resolver reads only "
            f"{sorted(known)}. Nothing consumes an unrecognised key, so it would be "
            "logged and recorded in the checkpoint configuration while changing "
            "nothing. Correct the spelling, or extend the contract deliberately."
        )


def _reject_divergent_per_settings(per: Mapping[str, Any]) -> None:
    """Refuse a frozen `per:` setting the configuration tries to change."""

    for key, frozen in _FROZEN_PER_SETTINGS:
        if key not in per:
            continue
        value = per[key]
        if str(value).strip().lower() != frozen:
            raise ValueError(
                f"The prioritized replay implementation hard-codes {key}={frozen!r}; "
                f"configuration requested {value!r}. Nothing reads this key, so the "
                "override cannot take effect: change the implementation, or drop the "
                "override."
            )


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
    _reject_unknown_keys(raw, known=_KNOWN_REPLAY_KEYS, path="transition_replay")
    _reject_unknown_keys(
        persistence, known=_KNOWN_PERSISTENCE_KEYS, path="transition_replay.persistence"
    )
    _reject_unknown_keys(per, known=_KNOWN_PER_KEYS, path="transition_replay.per")
    _reject_divergent_per_settings(per)
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
