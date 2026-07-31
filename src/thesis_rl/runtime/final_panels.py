"""Serial, frozen ScenarioNet final-panel execution and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from omegaconf import DictConfig, OmegaConf

from thesis_rl.runtime.evaluation_plan import EvaluationPanel, resolve_scenarionet_evaluation_panels
from thesis_rl.runtime.io.eval_artifacts import maybe_build_live_final_eval_recorder_factory
from thesis_rl.runtime.wiring.builders import (
    build_eval_env,
    evaluation_num_workers,
    merge_env_config_with_overrides,
)


def _metric_fields(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "mean_reward", "std_reward", "mean_env_reward", "std_env_reward",
        "mean_scalar_rule_reward", "std_scalar_rule_reward", "mean_hybrid_reward",
        "std_hybrid_reward", "mean_rule_saturation_max", "collision_rate",
        "collision_rate_std", "out_of_road_rate", "success_rate", "success_rate_std",
        "route_completion", "top_rule_violation_rate", "avg_error_value", "max_error_value",
        "counterexample_rate", "violated_rules_ratio", "unique_violation_patterns",
    )
    return {key: metrics.get(key) for key in keys}


def _panel_fields(panel: EvaluationPanel, *, batch_id: str, checkpoint_identity: str) -> dict[str, Any]:
    return {
        "evaluation_batch_id": batch_id,
        "panel_name": panel.name,
        "evaluation_scope": panel.scope,
        "panel_sha256": panel.panel_hash,
        "parent_panel_sha256": panel.parent_panel_hash,
        "frozen_selection_hash": panel.selection_hash,
        "checkpoint_identity": checkpoint_identity,
    }


def _data_abort_fields(metrics: Mapping[str, Any]) -> dict[str, Any]:
    coverage = metrics.get("data_abort_coverage")
    if not isinstance(coverage, Mapping):
        return {
            "data_abort_attempted": None,
            "data_abort_valid": None,
            "data_abort_invalid": None,
            "data_abort_coverage": None,
        }
    attempted = coverage.get("attempted")
    valid = coverage.get("valid")
    return {
        "data_abort_attempted": attempted,
        "data_abort_valid": valid,
        "data_abort_invalid": coverage.get("invalid"),
        "data_abort_coverage": (float(valid) / float(attempted)) if attempted else None,
    }


def _append_rule_rows(
    recorder: Any, base_fields: Mapping[str, Any], common: Mapping[str, Any], metrics: Mapping[str, Any]
) -> None:
    for row in metrics.get("per_rule", []):
        if isinstance(row, Mapping):
            recorder.append_row("rule_metrics.csv", {**base_fields, **common, **row})
    for row in metrics.get("per_subrule", []):
        if isinstance(row, Mapping):
            recorder.append_row("subrule_metrics.csv", {**base_fields, **common, **row})


def _append_episode_rows(
    recorder: Any,
    *,
    base_fields: Mapping[str, Any],
    common: Mapping[str, Any],
    metrics: Mapping[str, Any],
    deterministic: bool,
) -> None:
    episode = metrics.get("per_episode", {})
    if not isinstance(episode, Mapping):
        return
    returns = list(episode.get("returns", []))
    vector_keys = (
        "env_returns", "scalar_rule_returns", "hybrid_returns", "rule_rewards_by_rule",
        "episode_length", "success", "collision", "out_of_road", "timeout",
        "route_completion", "top_rule_violation_rate", "error_value", "violated_rules",
        "violation_pattern", "video_path", "video_authoritative_path", "video_manifest_path",
        "trajectory_log_path", "video_recorded_live", "replay_warning", "scenario_metadata",
    )
    vectors = {key: list(episode.get(key, [])) for key in vector_keys}
    for index, reward in enumerate(returns):
        metadata = vectors["scenario_metadata"][index] if index < len(vectors["scenario_metadata"]) else {}
        metadata = metadata if isinstance(metadata, Mapping) else {}
        def value(key: str) -> Any:
            return vectors[key][index] if index < len(vectors[key]) else None
        recorder.append_row(
            "eval_episodes.csv",
            {
                **base_fields, **common, "episode_id": index + 1,
                "scenario_seed": None, "scenario_uid": metadata.get("scenario_uid"),
                "scenario_id": metadata.get("scenario_id"), "source": metadata.get("source"),
                "split": metadata.get("split"), "primary_arm": metadata.get("arm"),
                "worker_id": metadata.get("worker_id"), "termination_reason": metadata.get("termination_reason"),
                "terminated": metadata.get("terminated"), "truncated": metadata.get("truncated"),
                "sampling_mode": metadata.get("sampling_mode"), "requested_arm": metadata.get("requested_arm"),
                "source_cell_fallback": metadata.get("source_cell_fallback"), "deterministic": deterministic,
                "reward": reward, "env_reward": value("env_returns"),
                "scalar_rule_reward": value("scalar_rule_returns"), "hybrid_reward": value("hybrid_returns"),
                "rule_rewards_by_rule": json.dumps(value("rule_rewards_by_rule") or {}, ensure_ascii=True),
                "episode_length": value("episode_length"), "success": value("success"),
                "collision": value("collision"), "out_of_road": value("out_of_road"),
                "timeout": value("timeout"), "route_completion": value("route_completion"),
                "top_rule_violation_rate": value("top_rule_violation_rate"), "error_value": value("error_value"),
                "violated_rules": value("violated_rules"), "violation_pattern": value("violation_pattern"),
                "video_path": value("video_path"), "video_authoritative_path": value("video_authoritative_path"),
                "video_manifest_path": value("video_manifest_path"), "trajectory_log_path": value("trajectory_log_path"),
                "video_recorded_live": value("video_recorded_live"), "replay_warning": value("replay_warning"),
            },
        )


def run_scenarionet_final_panels(
    *,
    cfg: DictConfig,
    recorder: Any,
    base_csv_fields: Mapping[str, Any],
    run_dir: Path,
    checkpoint_stem: Path,
    global_step: int,
    stage: str,
    stage_index: int,
    build_agent: Callable[[Any], Any],
    seed_env: Callable[[Any, int], None],
    event: Callable[..., None],
    eval_id_start: int,
) -> tuple[int, dict[str, dict[str, Any]]]:
    """Execute the three required panels serially from one `final.zip` policy."""
    panels = resolve_scenarionet_evaluation_panels(cfg, final=True)
    checkpoint_path = checkpoint_stem.with_suffix(".zip")
    if checkpoint_path.name != "final.zip":
        raise ValueError(
            "official ScenarioNet final panels must be evaluated from checkpoints/final.zip"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ScenarioNet final evaluation requires {checkpoint_path}")
    results: dict[str, dict[str, Any]] = {}
    batch_id = f"final_{int(global_step):012d}"
    deterministic = bool(cfg.experiment.eval_deterministic)
    for offset, panel in enumerate(panels, start=1):
        eval_id = eval_id_start + offset
        overrides = panel.env_overrides()
        resolved = OmegaConf.to_container(
            merge_env_config_with_overrides(cfg.env, overrides), resolve=True
        )
        if not isinstance(resolved, dict):
            raise TypeError("resolved final panel environment must be a mapping")
        resolved_env = resolved.get("config", resolved)
        if not isinstance(resolved_env, dict):
            raise TypeError("resolved final panel config payload must be a mapping")
        env = build_eval_env(
            cfg, overrides, n_eval_episodes=panel.episode_count, workers=evaluation_num_workers(cfg, final=True)
        )
        try:
            seed_env(env, int(cfg.seed) + 600_000 + offset)
            agent = build_agent(env)
            artifact_factory = maybe_build_live_final_eval_recorder_factory(
                cfg=cfg, run_dir=run_dir, resolved_env_config=resolved_env, eval_id=eval_id,
                eval_type="final", scenario_set=panel.name, stage=stage, stage_index=stage_index,
                checkpoint_path=str(checkpoint_path.relative_to(run_dir)), checkpoint_type="final",
                checkpoint_global_step=int(global_step),
            )
            event("evaluation_started", eval_id=eval_id, batch_id=batch_id, panel=panel.name, final=True)
            metrics = agent.evaluate(
                env=env, n_eval_episodes=panel.episode_count, deterministic=deterministic,
                base_seed=None, return_episode_metrics=True,
                error_priority_base=float(cfg.reward.get("a", 2.01)), show_progress=True,
                progress_description=f"Test {panel.name}", artifact_recorder_factory=artifact_factory,
            )
        finally:
            env.close()
        if not isinstance(metrics, dict) or not metrics.get("per_episode"):
            raise RuntimeError(f"final panel {panel.name} produced no complete per-episode metrics")
        common = {
            "eval_id": eval_id, "eval_type": "final", "scenario_set": panel.scenario_set,
            "chunk_id": 0, "stage": stage, "stage_index": stage_index, "global_step": global_step,
            "eval_episodes": panel.episode_count, "deterministic": deterministic,
            **_panel_fields(panel, batch_id=batch_id, checkpoint_identity=str(checkpoint_path.relative_to(run_dir))),
            **_data_abort_fields(metrics),
        }
        recorder.append_row("evals.csv", {**base_csv_fields, **common, **_metric_fields(metrics), "promoted": False, "next_stage": stage})
        _append_rule_rows(recorder, base_csv_fields, common, metrics)
        _append_episode_rows(recorder, base_fields=base_csv_fields, common=common, metrics=metrics, deterministic=deterministic)
        recorder.append_row(
            "final_eval.csv",
            {**base_csv_fields, **common, "total_timesteps": global_step, "final_stage": stage,
             "final_stage_index": stage_index, "final_stage_reached": True, "steps_to_final_stage": 0,
             "final_eval_episodes": panel.episode_count, **_metric_fields(metrics),
             "checkpoint_path": str(checkpoint_path.relative_to(run_dir)), "checkpoint_type": "final",
             "checkpoint_global_step": global_step},
        )
        event("evaluation_finished", eval_id=eval_id, batch_id=batch_id, panel=panel.name, final=True, metrics=metrics)
        results[panel.name] = metrics
    return eval_id_start + len(panels), results
