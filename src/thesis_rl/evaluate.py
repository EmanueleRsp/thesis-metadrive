from __future__ import annotations

import json
import logging
import time
from datetime import datetime

import random
from pathlib import Path
from typing import Any

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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_default))
        f.write("\n")


def _setup_file_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(f"thesis_rl.evaluate.{name}.{log_file}")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _log_event(events_path: Path, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(fields)
    _append_jsonl(events_path, payload)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== EVALUATION CONFIG ===")
    print(OmegaConf.to_yaml(cfg))

    # Save metadata for this run (config, git info, etc.)
    artifacts_dir = Path(str(cfg.paths.artifacts_dir))
    save_run_metadata(cfg, artifacts_dir)
    start_time = time.time()
    logs_dir = Path(str(cfg.paths.logs_dir))
    eval_log_path = logs_dir / "eval.log"
    errors_log_path = logs_dir / "errors.log"
    events_log_path = logs_dir / "events.jsonl"

    eval_logger = _setup_file_logger("eval", eval_log_path, level=logging.INFO)
    errors_logger = _setup_file_logger("errors", errors_log_path, level=logging.WARNING)

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
        eval_episodes = int(cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes))
        eval_logger.info(
            "Eval run started | checkpoint=%s | seed=%d | episodes=%d | deterministic=%s",
            str(ckpt),
            run_seed,
            eval_episodes,
            bool(cfg.experiment.eval_deterministic),
        )
        _log_event(
            events_log_path,
            "eval_run_started",
            checkpoint_path=str(ckpt),
            seed=run_seed,
            eval_episodes=eval_episodes,
            deterministic=bool(cfg.experiment.eval_deterministic),
        )

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
        _log_event(
            events_log_path,
            "evaluation_started",
            stage=eval_stage_name or "baseline",
            start_seed=int(eval_env_overrides["start_seed"]),
            checkpoint_path=str(ckpt),
        )

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
            n_eval_episodes=eval_episodes,
            deterministic=bool(cfg.experiment.eval_deterministic),
            base_seed=run_seed + 10_000,
            return_episode_metrics=True,
            error_priority_base=float(cfg.reward.get("a", 2.01)),
        )

        print(f"Evaluation metrics: {metrics}")
        eval_logger.info(
            "Evaluation finished | stage=%s | metrics=%s",
            eval_stage_name or "baseline",
            metrics,
        )
        _log_event(
            events_log_path,
            "evaluation_finished",
            stage=eval_stage_name or "baseline",
            metrics=metrics,
        )
        env.close()
        duration_seconds = round(time.time() - start_time, 2)
        eval_logger.info("Eval run completed | duration_seconds=%.2f", duration_seconds)
        _log_event(
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
        _log_event(
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
