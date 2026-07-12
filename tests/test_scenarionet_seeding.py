from __future__ import annotations

from omegaconf import OmegaConf

from thesis_rl.runtime.execution.seeding import (
    apply_eval_scenario_seed_split,
    eval_base_seed_from_env_overrides,
    train_episode_seed_from_env_overrides,
)


def test_scenarionet_seeding_delegates_selection_to_provider() -> None:
    cfg = OmegaConf.create(
        {
            "env": {"name": "scenarionet", "config": {}},
            "scenario_splits": {"stride": 100, "validation_offset": 10, "test_offset": 20},
        }
    )
    overrides = apply_eval_scenario_seed_split(
        base_run_seed=7,
        eval_env_overrides=None,
        cfg=cfg,
        n_eval_episodes=2,
        split="test",
    )
    assert overrides["split"] == "test"
    assert eval_base_seed_from_env_overrides(overrides, cfg) is None
    assert (
        train_episode_seed_from_env_overrides(
            None,
            cfg,
            run_seed=7,
            chunk_id=0,
            episode_index=0,
        )
        is None
    )
