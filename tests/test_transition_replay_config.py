from __future__ import annotations

import pytest

from thesis_rl.sb3_extensions.replay import resolve_transition_replay_config


def test_td3_defaults_to_enabled_three_step_uniform_replay() -> None:
    config = resolve_transition_replay_config(None, algorithm_name="td3_sb3")

    assert config.enabled is True
    assert config.n_steps == 3
    assert config.prioritized is False
    assert config.optimize_memory_usage is False


def test_sac_accepts_one_step_uniform_replay() -> None:
    config = resolve_transition_replay_config(
        {"enabled": True, "n_steps": 1},
        algorithm_name="sac_sb3",
    )

    assert config.n_steps == 1


def test_ppo_accepts_only_inactive_schema_defaults() -> None:
    config = resolve_transition_replay_config(
        {"enabled": False, "n_steps": 3},
        algorithm_name="ppo_sb3",
    )

    assert config.enabled is False
    assert config.n_steps == 1


@pytest.mark.parametrize(
    "raw_config",
    [
        {"enabled": True},
        {"enabled": False, "prioritized": True},
        {"enabled": False, "n_steps": 1},
        {"enabled": False, "store_reward_vector": True},
        {"enabled": False, "persistence": {"enabled": True}},
    ],
)
def test_ppo_rejects_active_or_non_default_replay_settings(raw_config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="inactive transition_replay"):
        resolve_transition_replay_config(raw_config, algorithm_name="ppo_sb3")


@pytest.mark.parametrize("n_steps", [0, 2, 4])
def test_invalid_n_step_value_fails(n_steps: int) -> None:
    with pytest.raises(ValueError, match=r"one of \{1, 3\}"):
        resolve_transition_replay_config(
            {"enabled": True, "n_steps": n_steps},
            algorithm_name="td3_sb3",
        )


def test_memory_optimization_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be false"):
        resolve_transition_replay_config(
            {"enabled": True, "optimize_memory_usage": True},
            algorithm_name="sac_sb3",
        )


def test_legacy_replay_persistence_requires_migration() -> None:
    with pytest.raises(ValueError, match="migrate"):
        resolve_transition_replay_config(
            {"enabled": False},
            algorithm_name="sac_sb3",
            legacy_save_replay_buffer=True,
        )


def test_periodic_frequency_steps_is_not_a_setting() -> None:
    """`TRANSITION-REPLAY` v1.1 REQ-024: the periodic replay snapshot rides the
    periodic model checkpoint, so a second frequency knob is rejected."""

    with pytest.raises(ValueError, match="periodic_frequency_steps"):
        resolve_transition_replay_config(
            {"persistence": {"enabled": True, "periodic_frequency_steps": 100}},
            algorithm_name="sac_sb3",
        )


def test_periodic_and_final_trigger_is_accepted() -> None:
    """`TEST-RES-006`: v1.1 trigger resolves to periodic replay persistence."""

    config = resolve_transition_replay_config(
        {"persistence": {"enabled": True, "trigger": "periodic_and_final"}},
        algorithm_name="td3_sb3",
    )

    assert config.persistence_enabled is True
    assert config.persistence_trigger == "periodic_and_final"
    assert config.periodic_replay_persistence is True

    legacy = resolve_transition_replay_config(
        {"persistence": {"enabled": True, "trigger": "final_or_manual"}},
        algorithm_name="td3_sb3",
    )
    assert legacy.periodic_replay_persistence is False

    disabled = resolve_transition_replay_config(
        {"persistence": {"enabled": False, "trigger": "periodic_and_final"}},
        algorithm_name="td3_sb3",
    )
    assert disabled.periodic_replay_persistence is False


def test_unknown_persistence_trigger_is_rejected() -> None:
    with pytest.raises(ValueError, match="trigger must be one of"):
        resolve_transition_replay_config(
            {"persistence": {"enabled": True, "trigger": "every_chunk"}},
            algorithm_name="sac_sb3",
        )
