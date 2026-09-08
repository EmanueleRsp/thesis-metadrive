"""`C38`: a frozen `transition_replay.per` setting cannot be silently ignored.

`td3_sb3.yaml` and `sac_sb3.yaml` declare ten `per:` keys and the planners forward
exactly five to the buffer constructor, selected by a hard-coded whitelist. The
other five describe what the implementation hard-codes — accurately, which is why
nobody noticed they were inert: they read as configuration while being
documentation, so an override parsed, was logged, reached the checkpoint's
configuration record and changed nothing.

Third instance of one defect: `C24` was six encoder settings, `C35` seventeen
observation settings, and all three take the same remedy — refuse the divergent
value, keep the declaration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from thesis_rl.sb3_extensions.replay.config import (
    _FROZEN_PER_SETTINGS,
    _LIVE_PER_SETTINGS,
    resolve_transition_replay_config,
)
from thesis_rl.sb3_extensions.replay.prioritized import _SumTree

_ALGORITHM_CONFIGS = {
    "td3_sb3": Path(__file__).resolve().parents[1]
    / "conf"
    / "agent"
    / "planner"
    / "algorithm"
    / "td3_sb3.yaml",
    "sac_sb3": Path(__file__).resolve().parents[1]
    / "conf"
    / "agent"
    / "planner"
    / "algorithm"
    / "sac_sb3.yaml",
}


def _shipped_replay_block(algorithm: str) -> dict[str, object]:
    loaded = OmegaConf.load(_ALGORITHM_CONFIGS[algorithm])
    block = OmegaConf.to_container(loaded, resolve=False)["transition_replay"]  # type: ignore[index]
    assert isinstance(block, dict)
    # `beta_anneal_steps` and `persistence.enabled` are interpolations; the
    # resolver takes the former as an argument and the latter only as a bool.
    block["per"]["beta_anneal_steps"] = 200_000
    block["persistence"]["enabled"] = False
    return block


def _resolve(algorithm: str, **per_overrides: object):
    block = _shipped_replay_block(algorithm)
    block["per"].update(per_overrides)
    return resolve_transition_replay_config(
        block, algorithm_name=algorithm, total_timesteps=200_000
    )


@pytest.mark.parametrize("algorithm", sorted(_ALGORITHM_CONFIGS))
def test_the_shipped_block_still_resolves(algorithm: str) -> None:
    """The guard must cost the production path nothing."""

    resolved = _resolve(algorithm)

    assert resolved.enabled is True
    assert resolved.prioritized is True
    assert resolved.n_steps == 3
    assert resolved.beta_anneal_steps == 200_000


@pytest.mark.parametrize(("key", "frozen"), list(_FROZEN_PER_SETTINGS))
def test_overriding_a_hard_coded_setting_fails_instead_of_being_ignored(
    key: str, frozen: str
) -> None:
    with pytest.raises(ValueError, match=key):
        _resolve("td3_sb3", **{key: "something_else"})


def test_an_unknown_key_is_named_rather_than_ignored() -> None:
    """A misspelling is indistinguishable from an absent key under a defaulting read.

    `keep_last` mistyped silently restores the default the explicit value existed
    to override, and a mistyped `periodic_frequency_steps` silently disables the
    guard that refuses it.
    """

    with pytest.raises(ValueError, match="beta_annel_steps"):
        _resolve("td3_sb3", beta_annel_steps=1000)

    block = _shipped_replay_block("td3_sb3")
    block["persistence"]["keep_lastt"] = 1
    with pytest.raises(ValueError, match="keep_lastt"):
        resolve_transition_replay_config(block, algorithm_name="td3_sb3", total_timesteps=200_000)

    block = _shipped_replay_block("td3_sb3")
    block["n_stepss"] = 3
    with pytest.raises(ValueError, match="n_stepss"):
        resolve_transition_replay_config(block, algorithm_name="td3_sb3", total_timesteps=200_000)


@pytest.mark.parametrize("algorithm", sorted(_ALGORITHM_CONFIGS))
def test_every_shipped_per_key_is_either_live_or_guarded(algorithm: str) -> None:
    """The guard list and the shipped block must not drift apart.

    A key added to the YAML without being either forwarded or guarded is exactly
    the defect this test exists to prevent, one knob later.
    """

    guarded = {key for key, _ in _FROZEN_PER_SETTINGS}
    shipped = set(_shipped_replay_block(algorithm)["per"])  # type: ignore[arg-type]

    unaccounted = shipped - guarded - _LIVE_PER_SETTINGS
    assert not unaccounted, f"per: keys neither guarded nor forwarded: {sorted(unaccounted)}"


def test_the_frozen_values_match_what_the_module_implements() -> None:
    """Pin the guard to the implementation, not to the YAML.

    If one of these ever becomes genuinely configurable, the guard must be removed
    rather than left refusing a value that now works.
    """

    frozen = dict(_FROZEN_PER_SETTINGS)

    # `_SumTree` allocates float64 nodes.
    assert frozen["tree_dtype"] == "float64"
    assert _SumTree(8).tree.dtype == np.float64

    # `_insertion_priority` returns the buffer's exact current maximum
    # (ADR-082 / `TRANSITION-REPLAY-V1.1.1`), not a running or floored one.
    assert frozen["new_transition_priority"] == "current_max"

    # `update_priorities` reduces repeated indices with `np.maximum.at`.
    assert frozen["duplicate_update_reduction"] == "max"

    # `_sample_addresses` draws one address per `np.linspace` stratum.
    assert frozen["sampling"] == "proportional_stratified"

    # The SB3 fork aggregates the twin critics as 0.5 * sum of absolute residuals.
    assert frozen["priority_aggregation"] == "mean_abs_twin_td"


def test_ppo_still_accepts_its_inactive_block() -> None:
    """PPO ships `transition_replay: {enabled: false}` and must keep resolving."""

    resolved = resolve_transition_replay_config(
        {"enabled": False}, algorithm_name="ppo_sb3", total_timesteps=200_000
    )

    assert resolved.enabled is False
    assert resolved.prioritized is False
