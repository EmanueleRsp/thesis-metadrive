from __future__ import annotations

import time
from datetime import datetime

import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from thesis_rl.agents.agent import Agent
from thesis_rl.curriculum.config import CurriculumConfig
from thesis_rl.curriculum.manager import CurriculumManager
from thesis_rl.runtime.builders import (
    adapter_space_kwargs,
    build_adapter,
    build_env,
    build_preprocessor,
    load_planner,
)
from thesis_rl.runtime.metadata import save_run_metadata, update_run_metadata


def _apply_eval_scenario_seed_split(
    *,
    base_run_seed: int,
    eval_env_overrides: dict[str, object] | None,
    cfg: DictConfig,
) -> dict[str, object]:
    """Return env overrides with deterministic disjoint scenario seed for evaluation.

    We keep runtime RNG seeded from `cfg.seed`, but shift MetaDrive scenario pool
    (`start_seed`) so evaluation scenarios are disjoint from training scenarios.
    """
    overrides = dict(eval_env_overrides or {})
    base_start_seed = int(overrides.get("start_seed", cfg.env.config.start_seed))
    scenario_offset = 1_000_000 + int(base_run_seed) * 10_000
    overrides["start_seed"] = int(base_start_seed) + scenario_offset
    return overrides


def _resolve_eval_env_overrides(curriculum_cfg: CurriculumConfig) -> tuple[dict[str, object] | None, str | None]:
    """Resolve evaluation env overrides and selected stage name from curriculum config.

    Evaluation is standalone (no persisted curriculum state), so when curriculum mode is
    `auto` we evaluate on the last configured stage by convention.
    """
    if not (curriculum_cfg.enabled and curriculum_cfg.stages):
        return None, None

    mode = str(curriculum_cfg.mode).lower()
    if mode == "auto":
        stage = curriculum_cfg.stages[-1]
        merged = dict(stage.env)
        merged.update(stage.eval_env)
        return merged, stage.name

    curriculum_manager = CurriculumManager(curriculum_cfg)
    return curriculum_manager.get_env_config(evaluation=True), curriculum_manager.get_current_stage().name


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_env_spaces(env, seed: int) -> None:
    action_space = getattr(env, "action_space", None)
    if action_space is not None and hasattr(action_space, "seed"):
        action_space.seed(seed)
    observation_space = getattr(env, "observation_space", None)
    if observation_space is not None and hasattr(observation_space, "seed"):
        observation_space.seed(seed)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== EVALUATION CONFIG ===")
    print(OmegaConf.to_yaml(cfg))

    # Save metadata for this run (config, git info, etc.)
    artifacts_dir = Path(str(cfg.paths.artifacts_dir))
    save_run_metadata(cfg, artifacts_dir)
    start_time = time.time()

    try:

        # Set seeds
        run_seed = int(cfg.seed)
        _set_global_seed(run_seed)

        checkpoint_path = cfg.checkpoint_path
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required. Example: "
                "uv run python -m thesis_rl.evaluate checkpoint_path=checkpoints/baseline_td3.zip"
            )

        ckpt = Path(str(checkpoint_path))
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        curriculum_cfg = CurriculumConfig.from_curriculum_cfg(cfg.curriculum)
        eval_env_overrides, eval_stage_name = _resolve_eval_env_overrides(curriculum_cfg)
        eval_env_overrides = _apply_eval_scenario_seed_split(
            base_run_seed=run_seed,
            eval_env_overrides=eval_env_overrides,
            cfg=cfg,
        )

        env = build_env(cfg, eval_env_overrides)
        _seed_env_spaces(env, run_seed)
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        if eval_stage_name is not None:
            print(f"Evaluation curriculum stage: {eval_stage_name}")
        print(f"Evaluation scenario start_seed: {eval_env_overrides['start_seed']}")

        preprocessor = build_preprocessor(cfg)
        adapter = build_adapter(
            cfg,
            adapter_space_kwargs(env.action_space),
        )
        planner = load_planner(cfg, checkpoint_path=str(ckpt), env=env)
        agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)
        agent.load_adapter(checkpoint_path=ckpt, strict=True)

        metrics = agent.evaluate(
            env=env,
            n_eval_episodes=int(cfg.experiment.eval_episodes),
            deterministic=bool(cfg.experiment.eval_deterministic),
            base_seed=run_seed + 10_000,
        )

        print(f"Evaluation metrics: {metrics}")
        env.close()

        # Update metadata
        update_run_metadata(artifacts_dir, {
            "status": "completed",
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(time.time() - start_time, 2),
        })
    
    except Exception as e:
        update_run_metadata(artifacts_dir, {
            "status": "failed",
            "error": str(e),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(time.time() - start_time, 2),
        })
        raise


if __name__ == "__main__":
    main()
