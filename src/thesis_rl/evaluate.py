from __future__ import annotations

import logging
import time
from datetime import datetime

from pathlib import Path

import hydra
from omegaconf import DictConfig

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
from thesis_rl.runtime.console import print_evaluation_summary, print_run_setup
from thesis_rl.runtime.metadata import save_run_metadata, update_run_metadata
from thesis_rl.runtime.run_logging import log_event, setup_file_logger
from thesis_rl.runtime.seeding import (
    apply_eval_scenario_seed_split,
    eval_base_seed_from_env_overrides,
    seed_env_spaces,
    set_global_seed,
)


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


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Save metadata for this run (config, git info, etc.)
    artifacts_dir = Path(str(cfg.paths.artifacts_dir))
    metadata_path = save_run_metadata(cfg, artifacts_dir)
    hydra_config_path = Path(str(cfg.paths.run_dir)) / "hydra" / "config.yaml"
    print_run_setup(
        title="Evaluation Run",
        cfg=cfg,
        metadata_path=metadata_path,
        hydra_config_path=hydra_config_path,
        checkpoint_path=str(cfg.checkpoint_path),
    )
    start_time = time.time()
    logs_dir = Path(str(cfg.paths.logs_dir))
    eval_log_path = logs_dir / "eval.log"
    errors_log_path = logs_dir / "errors.log"
    events_log_path = logs_dir / "events.jsonl"

    eval_logger = setup_file_logger("thesis_rl.evaluate", "eval", eval_log_path, level=logging.INFO)
    errors_logger = setup_file_logger("thesis_rl.evaluate", "errors", errors_log_path, level=logging.WARNING)

    update_run_metadata(
        artifacts_dir,
        {
            "logs": {
                "eval": str(eval_log_path),
                "errors": str(errors_log_path),
                "events": str(events_log_path),
            }
        },
    )

    try:

        # Set seeds
        run_seed = int(cfg.seed)
        set_global_seed(run_seed)

        checkpoint_path = cfg.checkpoint_path
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required. Example: "
                "uv run python -m thesis_rl.evaluate checkpoint_path=checkpoints/baseline_td3.zip"
            )

        ckpt = Path(str(checkpoint_path))
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        eval_episodes = int(cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes))
        eval_logger.info(
            "Eval run started | checkpoint=%s | seed=%d | episodes=%d | deterministic=%s",
            str(ckpt),
            run_seed,
            eval_episodes,
            bool(cfg.experiment.eval_deterministic),
        )
        log_event(
            events_log_path,
            "eval_run_started",
            checkpoint_path=str(ckpt),
            seed=run_seed,
            eval_episodes=eval_episodes,
            deterministic=bool(cfg.experiment.eval_deterministic),
        )

        curriculum_cfg = CurriculumConfig.from_curriculum_cfg(cfg.curriculum)
        eval_env_overrides, eval_stage_name = _resolve_eval_env_overrides(curriculum_cfg)
        eval_env_overrides = apply_eval_scenario_seed_split(
            base_run_seed=run_seed,
            eval_env_overrides=eval_env_overrides,
            cfg=cfg,
            n_eval_episodes=eval_episodes,
            split="test",
        )
        eval_base_seed = eval_base_seed_from_env_overrides(eval_env_overrides, cfg)

        env = build_env(cfg, eval_env_overrides)
        seed_env_spaces(env, run_seed)
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        if eval_stage_name is not None:
            print(f"Evaluation curriculum stage: {eval_stage_name}")
        print(f"Evaluation scenario start_seed: {eval_base_seed}")
        log_event(
            events_log_path,
            "evaluation_started",
            stage=eval_stage_name or "baseline",
            start_seed=eval_base_seed,
            checkpoint_path=str(ckpt),
        )

        preprocessor = build_preprocessor(cfg)
        adapter = build_adapter(
            cfg,
            adapter_space_kwargs(env.action_space),
        )
        planner = load_planner(cfg, checkpoint_path=str(ckpt), env=env)
        ema_alpha_cfg = float(cfg.planner.get("monitor_ema_alpha", 0.1)) if hasattr(cfg, "planner") else 0.1
        agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter, ema_alpha=ema_alpha_cfg)
        agent.load_adapter(checkpoint_path=ckpt, strict=True)

        metrics = agent.evaluate(
            env=env,
            n_eval_episodes=eval_episodes,
            deterministic=bool(cfg.experiment.eval_deterministic),
            base_seed=eval_base_seed,
            return_episode_metrics=True,
            error_priority_base=float(cfg.reward.get("a", 2.01)),
            show_progress=True,
        )

        print_evaluation_summary(
            title="Evaluation",
            metrics=metrics,
            stage=eval_stage_name or "baseline",
            global_step=0,
            episodes=eval_episodes,
            base_seed=eval_base_seed,
            details_path=eval_log_path,
            checkpoint_path=str(ckpt),
        )
        eval_logger.info(
            "Evaluation finished | stage=%s | metrics=%s",
            eval_stage_name or "baseline",
            metrics,
        )
        log_event(
            events_log_path,
            "evaluation_finished",
            stage=eval_stage_name or "baseline",
            metrics=metrics,
        )
        env.close()
        duration_seconds = round(time.time() - start_time, 2)
        eval_logger.info("Eval run completed | duration_seconds=%.2f", duration_seconds)
        log_event(
            events_log_path,
            "eval_run_completed",
            duration_seconds=duration_seconds,
        )

        # Update metadata
        update_run_metadata(artifacts_dir, {
            "status": "completed",
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration_seconds,
        })
    
    except Exception as e:
        duration_seconds = round(time.time() - start_time, 2)
        errors_logger.exception("Evaluation failed | error=%s", str(e))
        log_event(
            events_log_path,
            "eval_run_failed",
            error=str(e),
            duration_seconds=duration_seconds,
        )
        update_run_metadata(artifacts_dir, {
            "status": "failed",
            "error": str(e),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration_seconds,
        })
        raise


if __name__ == "__main__":
    main()
