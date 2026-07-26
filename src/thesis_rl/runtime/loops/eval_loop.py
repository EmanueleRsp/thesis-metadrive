from __future__ import annotations

import json
import logging
import time
from datetime import datetime

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from thesis_rl.agent.agent import Agent
from thesis_rl.curriculum.config import CurriculumConfig
from thesis_rl.curriculum.manager import CurriculumManager
from thesis_rl.curriculum.scenario_acl import validate_scenario_acl_runtime_support
from thesis_rl.runtime.io.csv_recorder import CSVRecorder
from thesis_rl.runtime.io.eval_artifacts import maybe_build_live_final_eval_recorder_factory
from thesis_rl.runtime.wiring.builders import (
    adapter_space_kwargs,
    build_eval_env,
    build_adapter,
    build_preprocessor,
    collect_scenario_runtime_stats,
    evaluation_num_workers,
    load_planner,
    merge_env_config_with_overrides,
)
from thesis_rl.runtime.io.console import print_evaluation_summary, print_run_setup
from thesis_rl.runtime.io.metadata import save_run_metadata, update_run_metadata
from thesis_rl.runtime.io.run_logging import (
    configure_logging,
    log_event,
    parse_log_level,
    setup_file_logger,
)
from thesis_rl.runtime.execution.seeding import (
    apply_eval_scenario_seed_split,
    eval_base_seed_from_env_overrides,
    seed_env_spaces,
    set_global_seed,
)


def _append_rule_metrics_rows(
    recorder: CSVRecorder,
    *,
    base_fields: dict[str, object],
    eval_id: int,
    eval_type: str,
    scenario_set: str,
    chunk_id: int,
    stage: str,
    stage_index: int,
    global_step: int,
    metrics: dict[str, object],
) -> None:
    rows = metrics.get("per_rule", [])
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        recorder.append_row(
            "rule_metrics.csv",
            {
                **base_fields,
                "eval_id": eval_id,
                "eval_type": eval_type,
                "scenario_set": scenario_set,
                "chunk_id": chunk_id,
                "stage": stage,
                "stage_index": stage_index,
                "global_step": global_step,
                "rule_name": row.get("rule_name"),
                "rule_priority": row.get("rule_priority"),
                "violated": row.get("violated"),
                "violation_rate": row.get("violation_rate"),
                "violation_count": row.get("violation_count"),
                "mean_margin": row.get("mean_margin"),
                "min_margin": row.get("min_margin"),
                "max_margin": row.get("max_margin"),
                # EVAL-PROTOCOL v1.0 REQ-008: episodes with >=1 applicable
                # step for this rule feeding violation_rate/mean_margin, and
                # the count excluded for having zero applicable steps.
                "applicable_episode_count": row.get("applicable_episode_count"),
                "excluded_episode_count": row.get("excluded_episode_count"),
            },
        )


def _resolve_eval_env_overrides(
    curriculum_cfg: CurriculumConfig,
) -> tuple[dict[str, object] | None, str | None]:
    """Resolve evaluation env overrides and selected stage name from curriculum config.

    Evaluation is standalone (no persisted curriculum state), so when curriculum mode is
    `auto` we evaluate on the last configured stage by convention.
    """
    if not (curriculum_cfg.enabled and curriculum_cfg.is_staged and curriculum_cfg.staged.stages):
        return None, None

    mode = str(curriculum_cfg.staged.mode).lower()
    if mode == "auto":
        stage = curriculum_cfg.staged.stages[-1]
        merged = dict(stage.env)
        merged.update(stage.eval_env)
        return merged, stage.name

    curriculum_manager = CurriculumManager(curriculum_cfg)
    return curriculum_manager.get_env_config(
        evaluation=True
    ), curriculum_manager.get_current_stage().name


def run_evaluation(cfg: DictConfig) -> None:
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
    csv_dir = Path(str(cfg.paths.csv_dir))
    recorder = CSVRecorder(csv_dir)
    run_id = Path(str(cfg.paths.run_dir)).name
    eval_log_path = logs_dir / "eval.log"
    errors_log_path = logs_dir / "errors.log"
    events_log_path = logs_dir / "events.jsonl"
    reward_type = str(cfg.reward.type).strip()
    reward_behavior = str(cfg.reward.behavior).strip()
    rulebook_config = str(cfg.reward.get("rulebook_config", "")).strip()
    curriculum_enabled = bool(cfg.curriculum.get("enabled", False))
    base_csv_fields = {
        "algorithm": str(cfg.agent.planner.algorithm.name),
        "reward_type": reward_type,
        "reward_behavior": reward_behavior,
        "curriculum_name": str(cfg.curriculum.name),
        "rulebook_config": rulebook_config,
        "curriculum_enabled": curriculum_enabled,
        "seed": int(cfg.seed),
        "run_id": run_id,
    }

    logging_cfg = cfg.get("logging", {})
    global_log_level = parse_log_level(logging_cfg.get("level"), default=logging.INFO)
    file_log_level = parse_log_level(logging_cfg.get("file_level"), default=global_log_level)
    console_log_level = parse_log_level(logging_cfg.get("console_level"), default=file_log_level)
    configure_logging(global_log_level, console_level=console_log_level)

    eval_logger = setup_file_logger(
        "thesis_rl.cli.evaluate",
        "eval",
        eval_log_path,
        level=file_log_level,
        console_level=console_log_level,
    )
    errors_logger = setup_file_logger(
        "thesis_rl.cli.evaluate",
        "errors",
        errors_log_path,
        level=parse_log_level("WARNING"),
        console_level=console_log_level,
    )

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
                "uv run python -m thesis_rl.cli.evaluate checkpoint_path=checkpoints/baseline_td3.zip"
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
        validate_scenario_acl_runtime_support(
            cfg,
            curriculum_cfg,
            context="evaluation",
        )
        if curriculum_cfg.is_scenario_acl:
            raise NotImplementedError(
                "Standalone evaluation for curriculum kind 'scenario_acl' is not "
                "implemented yet. Use the training pipeline final evaluation path."
            )
        eval_env_overrides, eval_stage_name = _resolve_eval_env_overrides(curriculum_cfg)
        eval_env_overrides = apply_eval_scenario_seed_split(
            base_run_seed=run_seed,
            eval_env_overrides=eval_env_overrides,
            cfg=cfg,
            n_eval_episodes=eval_episodes,
            split="test",
        )
        eval_base_seed = eval_base_seed_from_env_overrides(eval_env_overrides, cfg)
        resolved_eval_cfg = OmegaConf.to_container(
            merge_env_config_with_overrides(cfg.env, eval_env_overrides or {}),
            resolve=True,
        )
        if not isinstance(resolved_eval_cfg, dict):
            raise TypeError("Resolved eval env config must be a mapping.")
        resolved_eval_env_config = resolved_eval_cfg.get("config", resolved_eval_cfg)
        if not isinstance(resolved_eval_env_config, dict):
            raise TypeError("Resolved eval env config payload must be a mapping.")

        env = build_eval_env(
            cfg,
            eval_env_overrides,
            n_eval_episodes=eval_episodes,
            workers=evaluation_num_workers(cfg, final=True),
        )
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
        planner = load_planner(
            cfg, checkpoint_path=str(ckpt), env=env, validate_rollout_geometry=False
        )
        ema_alpha_cfg = (
            float(cfg.agent.planner.algorithm.get("monitor_ema_alpha", 0.1))
            if hasattr(cfg, "agent")
            else 0.1
        )
        agent = Agent(
            preprocessor=preprocessor, planner=planner, adapter=adapter, ema_alpha=ema_alpha_cfg
        )
        agent.load_adapter(checkpoint_path=ckpt, strict=True)
        eval_id = 1
        stage_name = eval_stage_name or "baseline"
        stage_index = 0
        if (
            curriculum_cfg.enabled
            and curriculum_cfg.is_staged
            and curriculum_cfg.staged.stages
            and eval_stage_name is not None
        ):
            for idx, stage in enumerate(curriculum_cfg.staged.stages):
                if stage.name == eval_stage_name:
                    stage_index = idx
                    break
        artifact_factory = maybe_build_live_final_eval_recorder_factory(
            cfg=cfg,
            run_dir=Path(str(cfg.paths.run_dir)),
            resolved_env_config=resolved_eval_env_config,
            eval_id=eval_id,
            eval_type="final",
            scenario_set="test",
            stage=stage_name,
            stage_index=stage_index,
            checkpoint_path=str(ckpt),
            checkpoint_type="external" if ckpt.name != "final.zip" else "final",
            checkpoint_global_step=0,
        )

        metrics = agent.evaluate(
            env=env,
            n_eval_episodes=eval_episodes,
            deterministic=bool(cfg.experiment.eval_deterministic),
            base_seed=eval_base_seed,
            return_episode_metrics=True,
            error_priority_base=float(cfg.reward.get("a", 2.01)),
            show_progress=True,
            progress_description="Test episodes",
            artifact_recorder_factory=artifact_factory,
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
        recorder.append_row(
            "evals.csv",
            {
                **base_csv_fields,
                "eval_id": eval_id,
                "eval_type": "final",
                "scenario_set": "test",
                "chunk_id": 0,
                "stage": stage_name,
                "stage_index": stage_index,
                "global_step": 0,
                "eval_episodes": eval_episodes,
                "deterministic": bool(cfg.experiment.eval_deterministic),
                "mean_reward": float(metrics.get("mean_reward", 0.0)),
                "std_reward": float(metrics.get("std_reward", 0.0)),
                "mean_env_reward": float(metrics.get("mean_env_reward", 0.0)),
                "std_env_reward": float(metrics.get("std_env_reward", 0.0)),
                "mean_scalar_rule_reward": float(metrics.get("mean_scalar_rule_reward", 0.0))
                if metrics.get("mean_scalar_rule_reward") is not None
                else None,
                "std_scalar_rule_reward": float(metrics.get("std_scalar_rule_reward", 0.0))
                if metrics.get("std_scalar_rule_reward") is not None
                else None,
                "mean_hybrid_reward": float(metrics.get("mean_hybrid_reward", 0.0))
                if metrics.get("mean_hybrid_reward") is not None
                else None,
                "std_hybrid_reward": float(metrics.get("std_hybrid_reward", 0.0))
                if metrics.get("std_hybrid_reward") is not None
                else None,
                "mean_rule_saturation_max": float(metrics.get("mean_rule_saturation_max", 0.0)),
                "collision_rate": float(metrics.get("collision_rate", 0.0)),
                "collision_rate_std": float(metrics.get("collision_rate_std", 0.0)),
                "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
                "success_rate": float(metrics.get("success_rate", 0.0)),
                "success_rate_std": float(metrics.get("success_rate_std", 0.0)),
                "route_completion": float(metrics.get("route_completion", 0.0)),
                "top_rule_violation_rate": float(metrics.get("top_rule_violation_rate", 0.0)),
                "avg_error_value": float(metrics.get("avg_error_value", 0.0)),
                "max_error_value": float(metrics.get("max_error_value", 0.0)),
                "counterexample_rate": float(metrics.get("counterexample_rate", 0.0)),
                "violated_rules_ratio": float(metrics.get("violated_rules_ratio", 0.0)),
                "unique_violation_patterns": int(metrics.get("unique_violation_patterns", 0)),
                "promoted": False,
                "next_stage": stage_name,
            },
        )
        _append_rule_metrics_rows(
            recorder,
            base_fields=base_csv_fields,
            eval_id=eval_id,
            eval_type="final",
            scenario_set="test",
            chunk_id=0,
            stage=stage_name,
            stage_index=stage_index,
            global_step=0,
            metrics=metrics,
        )
        per_episode = metrics.get("per_episode", {})
        episode_returns = list(per_episode.get("returns", []))
        episode_lengths = list(per_episode.get("episode_length", []))
        episode_success = list(per_episode.get("success", []))
        episode_collision = list(per_episode.get("collision", []))
        episode_out_of_road = list(per_episode.get("out_of_road", []))
        episode_timeout = list(per_episode.get("timeout", []))
        episode_route_completion = list(per_episode.get("route_completion", []))
        episode_top_rule_violation_rate = list(per_episode.get("top_rule_violation_rate", []))
        episode_error_value = list(per_episode.get("error_value", []))
        episode_violated_rules = list(per_episode.get("violated_rules", []))
        episode_violation_pattern = list(per_episode.get("violation_pattern", []))
        episode_env_returns = list(per_episode.get("env_returns", []))
        episode_scalar_rule_returns = list(per_episode.get("scalar_rule_returns", []))
        episode_hybrid_returns = list(per_episode.get("hybrid_returns", []))
        episode_rule_rewards_by_rule = list(per_episode.get("rule_rewards_by_rule", []))
        episode_video_paths = list(per_episode.get("video_path", []))
        episode_video_authoritative_paths = list(per_episode.get("video_authoritative_path", []))
        episode_video_manifest_paths = list(per_episode.get("video_manifest_path", []))
        episode_trajectory_log_paths = list(per_episode.get("trajectory_log_path", []))
        episode_video_recorded_live = list(per_episode.get("video_recorded_live", []))
        episode_replay_warnings = list(per_episode.get("replay_warning", []))
        episode_scenario_metadata = list(per_episode.get("scenario_metadata", []))
        for episode_idx in range(len(episode_returns)):
            scenario_metadata = (
                episode_scenario_metadata[episode_idx]
                if episode_idx < len(episode_scenario_metadata)
                and isinstance(episode_scenario_metadata[episode_idx], dict)
                else {}
            )
            scenario_seed = (
                int(eval_base_seed + episode_idx) if eval_base_seed is not None else None
            )
            recorder.append_row(
                "eval_episodes.csv",
                {
                    **base_csv_fields,
                    "eval_id": eval_id,
                    "eval_type": "final",
                    "scenario_set": "test",
                    "episode_id": episode_idx + 1,
                    "stage": stage_name,
                    "stage_index": stage_index,
                    "global_step": 0,
                    "scenario_seed": scenario_seed,
                    "scenario_uid": scenario_metadata.get("scenario_uid"),
                    "scenario_id": scenario_metadata.get("scenario_id", f"seed_{scenario_seed}"),
                    "source": scenario_metadata.get("source"),
                    "split": scenario_metadata.get("split"),
                    "primary_arm": scenario_metadata.get("arm"),
                    "worker_id": scenario_metadata.get("worker_id"),
                    "termination_reason": scenario_metadata.get("termination_reason"),
                    "terminated": scenario_metadata.get("terminated"),
                    "truncated": scenario_metadata.get("truncated"),
                    "sampling_mode": scenario_metadata.get("sampling_mode"),
                    "requested_arm": scenario_metadata.get("requested_arm"),
                    "source_cell_fallback": scenario_metadata.get("source_cell_fallback"),
                    "deterministic": bool(cfg.experiment.eval_deterministic),
                    "reward": float(episode_returns[episode_idx]),
                    "env_reward": float(episode_env_returns[episode_idx])
                    if episode_idx < len(episode_env_returns)
                    else None,
                    "scalar_rule_reward": float(episode_scalar_rule_returns[episode_idx])
                    if episode_idx < len(episode_scalar_rule_returns)
                    and episode_scalar_rule_returns[episode_idx] is not None
                    else None,
                    "hybrid_reward": float(episode_hybrid_returns[episode_idx])
                    if episode_idx < len(episode_hybrid_returns)
                    and episode_hybrid_returns[episode_idx] is not None
                    else None,
                    "rule_rewards_by_rule": json.dumps(
                        episode_rule_rewards_by_rule[episode_idx], ensure_ascii=True
                    )
                    if episode_idx < len(episode_rule_rewards_by_rule)
                    else None,
                    "episode_length": int(episode_lengths[episode_idx])
                    if episode_idx < len(episode_lengths)
                    else None,
                    "success": float(episode_success[episode_idx])
                    if episode_idx < len(episode_success)
                    else None,
                    "collision": float(episode_collision[episode_idx])
                    if episode_idx < len(episode_collision)
                    else None,
                    "out_of_road": float(episode_out_of_road[episode_idx])
                    if episode_idx < len(episode_out_of_road)
                    else None,
                    "timeout": float(episode_timeout[episode_idx])
                    if episode_idx < len(episode_timeout)
                    else None,
                    "route_completion": float(episode_route_completion[episode_idx])
                    if episode_idx < len(episode_route_completion)
                    else None,
                    "top_rule_violation_rate": float(episode_top_rule_violation_rate[episode_idx])
                    if episode_idx < len(episode_top_rule_violation_rate)
                    else None,
                    "error_value": float(episode_error_value[episode_idx])
                    if episode_idx < len(episode_error_value)
                    else None,
                    "violated_rules": str(episode_violated_rules[episode_idx])
                    if episode_idx < len(episode_violated_rules)
                    else None,
                    "violation_pattern": str(episode_violation_pattern[episode_idx])
                    if episode_idx < len(episode_violation_pattern)
                    else None,
                    "video_path": episode_video_paths[episode_idx]
                    if episode_idx < len(episode_video_paths)
                    else None,
                    "video_authoritative_path": episode_video_authoritative_paths[episode_idx]
                    if episode_idx < len(episode_video_authoritative_paths)
                    else None,
                    "video_manifest_path": episode_video_manifest_paths[episode_idx]
                    if episode_idx < len(episode_video_manifest_paths)
                    else None,
                    "trajectory_log_path": episode_trajectory_log_paths[episode_idx]
                    if episode_idx < len(episode_trajectory_log_paths)
                    else None,
                    "video_recorded_live": bool(episode_video_recorded_live[episode_idx])
                    if episode_idx < len(episode_video_recorded_live)
                    else False,
                    "replay_warning": episode_replay_warnings[episode_idx]
                    if episode_idx < len(episode_replay_warnings)
                    else None,
                },
            )
        recorder.append_row(
            "final_eval.csv",
            {
                **base_csv_fields,
                "eval_type": "final",
                "scenario_set": "test",
                "total_timesteps": 0,
                "final_stage": stage_name,
                "final_stage_index": stage_index,
                "final_stage_reached": True,
                "steps_to_final_stage": 0,
                "final_eval_episodes": eval_episodes,
                "deterministic": bool(cfg.experiment.eval_deterministic),
                "mean_reward": float(metrics.get("mean_reward", 0.0)),
                "std_reward": float(metrics.get("std_reward", 0.0)),
                "mean_env_reward": float(metrics.get("mean_env_reward", 0.0)),
                "std_env_reward": float(metrics.get("std_env_reward", 0.0)),
                "mean_scalar_rule_reward": float(metrics.get("mean_scalar_rule_reward", 0.0))
                if metrics.get("mean_scalar_rule_reward") is not None
                else None,
                "std_scalar_rule_reward": float(metrics.get("std_scalar_rule_reward", 0.0))
                if metrics.get("std_scalar_rule_reward") is not None
                else None,
                "mean_hybrid_reward": float(metrics.get("mean_hybrid_reward", 0.0))
                if metrics.get("mean_hybrid_reward") is not None
                else None,
                "std_hybrid_reward": float(metrics.get("std_hybrid_reward", 0.0))
                if metrics.get("std_hybrid_reward") is not None
                else None,
                "mean_rule_saturation_max": float(metrics.get("mean_rule_saturation_max", 0.0)),
                "collision_rate": float(metrics.get("collision_rate", 0.0)),
                "collision_rate_std": float(metrics.get("collision_rate_std", 0.0)),
                "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
                "success_rate": float(metrics.get("success_rate", 0.0)),
                "success_rate_std": float(metrics.get("success_rate_std", 0.0)),
                "route_completion": float(metrics.get("route_completion", 0.0)),
                "top_rule_violation_rate": float(metrics.get("top_rule_violation_rate", 0.0)),
                "avg_error_value": float(metrics.get("avg_error_value", 0.0)),
                "max_error_value": float(metrics.get("max_error_value", 0.0)),
                "counterexample_rate": float(metrics.get("counterexample_rate", 0.0)),
                "violated_rules_ratio": float(metrics.get("violated_rules_ratio", 0.0)),
                "unique_violation_patterns": int(metrics.get("unique_violation_patterns", 0)),
                "checkpoint_path": str(ckpt),
                "checkpoint_type": "external" if ckpt.name != "final.zip" else "final",
                "checkpoint_global_step": 0,
            },
        )
        scenario_runtime_stats = collect_scenario_runtime_stats(env)
        env.close()
        duration_seconds = round(time.time() - start_time, 2)
        eval_logger.info("Eval run completed | duration_seconds=%.2f", duration_seconds)
        log_event(
            events_log_path,
            "eval_run_completed",
            duration_seconds=duration_seconds,
        )

        # Update metadata
        metadata_updates: dict[str, object] = {
            "status": "completed",
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration_seconds,
        }
        if scenario_runtime_stats is not None:
            metadata_updates["scenarionet_runtime_stats"] = scenario_runtime_stats
        update_run_metadata(artifacts_dir, metadata_updates)

    except Exception as e:
        duration_seconds = round(time.time() - start_time, 2)
        errors_logger.exception("Evaluation failed | error=%s", str(e))
        log_event(
            events_log_path,
            "eval_run_failed",
            error=str(e),
            duration_seconds=duration_seconds,
        )
        update_run_metadata(
            artifacts_dir,
            {
                "status": "failed",
                "error": str(e),
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": duration_seconds,
            },
        )
        raise
