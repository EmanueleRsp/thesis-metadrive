from __future__ import annotations

import json
import logging
import time
from datetime import datetime

from dataclasses import asdict
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
    build_planner,
    build_preprocessor,
)
from thesis_rl.runtime.metadata import save_run_metadata, update_run_metadata


def _required_curriculum_metrics(curriculum_cfg: CurriculumConfig) -> list[str]:
    """Derive required metric names from configured curriculum gates.

    Gate keys are expected to follow `<metric>_min` or `<metric>_max`.
    """
    gates_payload = asdict(curriculum_cfg.promotion.gates)
    required: list[str] = []
    seen: set[str] = set()

    for gate_group in gates_payload.values():
        if not isinstance(gate_group, dict):
            continue
        for gate_key in gate_group.keys():
            metric_name = str(gate_key)
            if metric_name.endswith("_min"):
                metric_name = metric_name[: -len("_min")]
            elif metric_name.endswith("_max"):
                metric_name = metric_name[: -len("_max")]
            if metric_name and metric_name not in seen:
                seen.add(metric_name)
                required.append(metric_name)
    return required


def _missing_curriculum_metrics(
    metrics: dict[str, float], curriculum_cfg: CurriculumConfig
) -> list[str]:
    required = _required_curriculum_metrics(curriculum_cfg)
    return [name for name in required if name not in metrics]


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_env_spaces(env: Any, seed: int) -> None:
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
    logger = logging.getLogger(f"thesis_rl.train.{name}.{log_file}")
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
    print("=== TRAIN CONFIG ===")
    print(OmegaConf.to_yaml(cfg))

    # Save metadata for this run (config, git info, etc.)
    artifacts_dir = Path(str(cfg.paths.artifacts_dir))
    save_run_metadata(cfg, artifacts_dir)
    start_time = time.time()
    logs_dir = Path(str(cfg.paths.logs_dir))
    train_log_path = logs_dir / "train.log"
    eval_log_path = logs_dir / "eval.log"
    curriculum_log_path = logs_dir / "curriculum.log"
    errors_log_path = logs_dir / "errors.log"
    events_log_path = logs_dir / "events.jsonl"

    train_logger = _setup_file_logger("train", train_log_path, level=logging.INFO)
    eval_logger = _setup_file_logger("eval", eval_log_path, level=logging.INFO)
    curriculum_logger = _setup_file_logger("curriculum", curriculum_log_path, level=logging.INFO)
    errors_logger = _setup_file_logger("errors", errors_log_path, level=logging.WARNING)

    update_run_metadata(
        artifacts_dir,
        {
            "logs": {
                "train": str(train_log_path),
                "eval": str(eval_log_path),
                "curriculum": str(curriculum_log_path),
                "errors": str(errors_log_path),
                "events": str(events_log_path),
            }
        },
    )

    try:

        run_seed = int(cfg.seed)
        _set_global_seed(run_seed)

        ###################
        ###### SETUP ######
        ###################
        
        # Curriculum manager
        curriculum_cfg = CurriculumConfig.from_curriculum_cfg(cfg.curriculum)
        curriculum_manager: CurriculumManager | None = None
        current_train_overrides: dict[str, Any] | None = None
        if curriculum_cfg.enabled and curriculum_cfg.stages:
            curriculum_manager = CurriculumManager(curriculum_cfg)
            current_train_overrides = curriculum_manager.get_env_config(evaluation=False)

        # Environment
        env = build_env(cfg, current_train_overrides)
        _seed_env_spaces(env, run_seed)
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Agent
        preprocessor = build_preprocessor(cfg)
        adapter = build_adapter(
            cfg,
            adapter_space_kwargs(env.action_space),
        )
        planner = build_planner(cfg, env, seed=run_seed)
        agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter)

        # Training and evaluation params
        total_timesteps = int(cfg.experiment.total_timesteps)
        log_interval = int(cfg.experiment.get("log_interval", 1000))
        eval_interval = int(cfg.experiment.get("eval_interval", total_timesteps))
        if eval_interval <= 0:
            eval_interval = total_timesteps
        stage_name = (
            curriculum_manager.get_current_stage().name
            if curriculum_manager is not None
            else "baseline"
        )
        train_logger.info(
            "Run started | algorithm=%s | seed=%d | device=%s",
            str(cfg.planner.name),
            run_seed,
            str(cfg.device),
        )
        train_logger.info(
            "Training started | total_timesteps=%d | eval_interval=%d | eval_episodes=%d | stage=%s",
            total_timesteps,
            eval_interval,
            int(cfg.experiment.eval_episodes),
            stage_name,
        )
        _log_event(
            events_log_path,
            "run_started",
            algorithm=str(cfg.planner.name),
            seed=run_seed,
            device=str(cfg.device),
            total_timesteps=total_timesteps,
            eval_interval=eval_interval,
            eval_episodes=int(cfg.experiment.eval_episodes),
            stage=stage_name,
        )

        ##################
        ###### LOOP ######
        ##################

        # Training loop with periodic evaluation and optional curriculum progression
        remaining = total_timesteps
        chunk_id = 0
        eval_id = 0
        while remaining > 0:
            chunk_id += 1

            # Stage info and chunk size 
            chunk_steps = min(eval_interval, remaining)
            current_stage_name = "baseline"
            if curriculum_manager is not None:
                current_stage_name = curriculum_manager.get_current_stage().name
            steps_start = total_timesteps - remaining
            steps_end = steps_start + chunk_steps
            print(
                f"Training chunk on stage '{current_stage_name}': "
                f"steps={chunk_steps}, remaining_after={remaining - chunk_steps}"
            )
            train_logger.info(
                "Chunk started | chunk_id=%d | stage=%s | steps=%d->%d",
                chunk_id,
                current_stage_name,
                steps_start,
                steps_end,
            )
            _log_event(
                events_log_path,
                "chunk_started",
                chunk_id=chunk_id,
                stage=current_stage_name,
                steps_start=steps_start,
                steps_end=steps_end,
            )

            ###### TRAINING ######

            # Agent training
            chunk_summary = agent.train(
                env=env,
                chunk_timesteps=chunk_steps,
                global_total_timesteps=total_timesteps,
                global_steps_done=total_timesteps - remaining,
                deterministic=False,
                log_interval=log_interval,
                reset_seed=run_seed + (total_timesteps - remaining),
            )
            remaining -= chunk_steps
            current_global_step = total_timesteps - remaining
            train_logger.info(
                (
                    "Chunk finished | chunk_id=%d | stage=%s | global_step=%d | "
                    "episodes=%d | mean_return=%.3f | mean_len=%.2f | fps=%.1f | n_updates=%d"
                ),
                chunk_id,
                current_stage_name,
                current_global_step,
                int(chunk_summary.get("episodes", 0)),
                float(chunk_summary.get("ep_rew_mean", 0.0)),
                float(chunk_summary.get("ep_len_mean", 0.0)),
                float(chunk_summary.get("fps", 0.0)),
                int(chunk_summary.get("n_updates", 0)),
            )
            _log_event(
                events_log_path,
                "chunk_finished",
                chunk_id=chunk_id,
                stage=current_stage_name,
                global_step=current_global_step,
                summary=chunk_summary,
            )

            # Record training steps in curriculum manager for potential stage progression
            if curriculum_manager is not None:
                curriculum_manager.record_train_steps(chunk_steps)

            # MetaDrive uses a global engine singleton: close training env before creating eval env.
            env.close()
            
            ###### EVALUATION ######

            # Config setup
            eval_env_overrides = None
            if curriculum_manager is not None:
                eval_env_overrides = curriculum_manager.get_env_config(evaluation=True)
            
            # Environment
            eval_env = build_env(cfg, eval_env_overrides)
            _seed_env_spaces(eval_env, run_seed + 100_000 + (total_timesteps - remaining))
            planner.set_env(eval_env)   # Update planner's env reference

            # Evaluation
            eval_id += 1
            eval_logger.info(
                "Evaluation started | eval_id=%d | stage=%s | step=%d | episodes=%d",
                eval_id,
                current_stage_name,
                current_global_step,
                int(cfg.experiment.eval_episodes),
            )
            _log_event(
                events_log_path,
                "evaluation_started",
                eval_id=eval_id,
                stage=current_stage_name,
                global_step=current_global_step,
                episodes=int(cfg.experiment.eval_episodes),
            )
            metrics = agent.evaluate(
                env=eval_env,
                n_eval_episodes=int(cfg.experiment.eval_episodes),
                deterministic=bool(cfg.experiment.eval_deterministic),
                base_seed=run_seed + 200_000 + (total_timesteps - remaining),
            )
            eval_env.close()
            print(f"Evaluation metrics after chunk: {metrics}")
            eval_logger.info(
                "Evaluation finished | eval_id=%d | stage=%s | step=%d | metrics=%s",
                eval_id,
                current_stage_name,
                current_global_step,
                metrics,
            )
            _log_event(
                events_log_path,
                "evaluation_finished",
                eval_id=eval_id,
                stage=current_stage_name,
                global_step=current_global_step,
                metrics=metrics,
            )

            ###### CURRICULUM PROGRESSION ######

            # If no curriculum, just rebuild the env
            if curriculum_manager is None:
                env = build_env(cfg, current_train_overrides)
                _seed_env_spaces(env, run_seed + 300_000 + (total_timesteps - remaining))
                planner.set_env(env)
                continue

            # Metrics check
            if curriculum_cfg.mode.lower() == "auto":
                missing_metrics = _missing_curriculum_metrics(metrics, curriculum_cfg)
                if missing_metrics:
                    curriculum_logger.error(
                        "Gate check failed | stage=%s | eval_id=%d | missing_metrics=%s",
                        current_stage_name,
                        eval_id,
                        missing_metrics,
                    )
                    _log_event(
                        events_log_path,
                        "gate_check",
                        eval_id=eval_id,
                        stage=current_stage_name,
                        global_step=current_global_step,
                        missing_metrics=missing_metrics,
                        all_gates_pass=False,
                    )
                    raise ValueError(
                        "Curriculum auto mode requires metrics: "
                        f"{missing_metrics}. "
                        "Implement metric extraction in Agent.evaluate before enabling auto mode."
                    )

            # Record eval metrics
            curriculum_manager.record_eval_metrics(metrics)

            # Check for promotion and update env config if promoted
            if curriculum_manager.promote():
                previous_stage = current_stage_name
                next_stage = curriculum_manager.get_current_stage().name
                print(f"Curriculum promoted: {previous_stage} -> {next_stage}")
                current_train_overrides = curriculum_manager.get_env_config(evaluation=False)
                curriculum_logger.info(
                    "PROMOTION | from=%s | to=%s | step=%d | eval_id=%d",
                    previous_stage,
                    next_stage,
                    current_global_step,
                    eval_id,
                )
                _log_event(
                    events_log_path,
                    "promotion",
                    from_stage=previous_stage,
                    to_stage=next_stage,
                    global_step=current_global_step,
                    eval_id=eval_id,
                )
            else:
                curriculum_logger.info(
                    "Gate check | stage=%s | eval_id=%d | promoted=false",
                    current_stage_name,
                    eval_id,
                )
                _log_event(
                    events_log_path,
                    "gate_check",
                    eval_id=eval_id,
                    stage=current_stage_name,
                    global_step=current_global_step,
                    promoted=False,
                )

            # Rebuild env with new config (if changed) and update planner's env reference
            env = build_env(cfg, current_train_overrides)
            _seed_env_spaces(env, run_seed + 400_000 + (total_timesteps - remaining))
            planner.set_env(env)    # Update planner's env reference

        ##########################
        ###### FINALIZATION ######
        ##########################

        # Save agent checkpoints
        checkpoints_dir = Path(str(cfg.paths.checkpoints_dir))
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoints_dir / f"{cfg.experiment.name}"
        agent.save(ckpt_path)
        adapter_ckpt_path = agent.adapter_checkpoint_path(ckpt_path)

        ###### FINAL EVAL ######

        # Environment
        env.close()
        final_eval_env_overrides = None
        if curriculum_manager is not None:
            final_eval_env_overrides = curriculum_manager.get_env_config(evaluation=True)
        eval_env = build_env(cfg, final_eval_env_overrides)
        _seed_env_spaces(eval_env, run_seed + 500_000)

        # Evaluation
        metrics = agent.evaluate(
            env=eval_env,
            n_eval_episodes=int(cfg.experiment.eval_episodes),
            deterministic=bool(cfg.experiment.eval_deterministic),
            base_seed=run_seed + 600_000,
        )
        print(f"Evaluation metrics after training: {metrics}")
        print(f"Checkpoint saved at: {ckpt_path}.zip")
        train_logger.info("Checkpoint saved | path=%s.zip", ckpt_path)
        _log_event(
            events_log_path,
            "checkpoint_saved",
            path=f"{ckpt_path}.zip",
            global_step=total_timesteps,
        )
        if bool(getattr(adapter, "requires_training", False)):
            print(f"Adapter checkpoint saved at: {adapter_ckpt_path}")
            train_logger.info("Adapter checkpoint saved | path=%s", adapter_ckpt_path)
            _log_event(
                events_log_path,
                "checkpoint_saved",
                path=str(adapter_ckpt_path),
                global_step=total_timesteps,
                checkpoint_kind="adapter",
            )
        eval_logger.info(
            "Final evaluation finished | step=%d | metrics=%s",
            total_timesteps,
            metrics,
        )
        _log_event(
            events_log_path,
            "evaluation_finished",
            eval_id=eval_id + 1,
            stage=(
                curriculum_manager.get_current_stage().name
                if curriculum_manager is not None
                else "baseline"
            ),
            global_step=total_timesteps,
            metrics=metrics,
            final=True,
        )

        eval_env.close()
        duration_seconds = round(time.time() - start_time, 2)
        train_logger.info(
            "Run completed | total_timesteps=%d | duration_seconds=%.2f",
            total_timesteps,
            duration_seconds,
        )
        _log_event(
            events_log_path,
            "run_completed",
            total_timesteps=total_timesteps,
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
        errors_logger.exception("Training failed | error=%s", str(e))
        _log_event(
            events_log_path,
            "run_failed",
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
