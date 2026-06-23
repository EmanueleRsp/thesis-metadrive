from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from thesis_rl.agent.agent import Agent
from thesis_rl.agent.planners.core.utils import count_envs
from thesis_rl.curriculum.config import CurriculumConfig
from thesis_rl.curriculum.scenario_acl.arms import GeneratorArm, build_default_generator_arms
from thesis_rl.curriculum.scenario_acl.buffer import ScenarioBuffer
from thesis_rl.curriculum.scenario_acl.mab import GeneratorArmBandit
from thesis_rl.curriculum.scenario_acl.record import ScenarioRecord
from thesis_rl.curriculum.scenario_acl.scenario_env import build_scenario_replay_env
from thesis_rl.runtime.execution.seeding import (
    apply_eval_scenario_seed_split,
    eval_base_seed_from_env_overrides,
    seed_env_spaces,
    train_episode_seed_from_env_overrides,
)
from thesis_rl.runtime.io.console import print_evaluation_summary, print_run_setup
from thesis_rl.runtime.io.csv_recorder import CSVRecorder
from thesis_rl.runtime.io.eval_artifacts import maybe_build_live_final_eval_recorder_factory
from thesis_rl.runtime.io.metadata import update_run_metadata
from thesis_rl.runtime.io.run_logging import log_event
from thesis_rl.runtime.wiring.builders import (
    adapter_space_kwargs,
    build_adapter,
    build_env,
    build_planner,
    build_preprocessor,
    load_planner,
    merge_env_config_with_overrides,
    set_planner_env_if_compatible,
)


@dataclass(frozen=True)
class ScenarioAclDriverPaths:
    artifacts_dir: Path
    run_dir: Path
    checkpoints_dir: Path
    final_checkpoint_stem: Path
    latest_checkpoint_stem: Path
    events_log_path: Path


@dataclass(frozen=True)
class IterationSpec:
    mode: str
    arm_index: int
    arm_name: str
    arm_probabilities: list[float]
    scenario_seed: int
    train_env_overrides: dict[str, object] | None
    replay_record: ScenarioRecord | None
    replay_probabilities: list[float] | None


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True))
        handle.write("\n")


def _scenario_acl_artifact_paths(artifacts_dir: Path) -> dict[str, Path]:
    root = artifacts_dir / "curriculum"
    return {
        "root": root,
        "state": root / "scenario_acl_state.json",
        "history": root / "mab_history.jsonl",
        "iterations": root / "iterations.jsonl",
        "buffer": root / "scenario_buffer.json",
        "buffer_events": root / "scenario_buffer_events.jsonl",
        "scenarios": root / "scenarios",
    }


def _compute_proxy_usefulness(metrics: dict[str, Any]) -> float:
    collision_rate = float(metrics.get("collision_rate", 0.0))
    out_of_road_rate = float(metrics.get("out_of_road_rate", 0.0))
    top_rule_violation_rate = float(metrics.get("top_rule_violation_rate", 0.0))
    success_gap = max(0.0, 1.0 - float(metrics.get("success_rate", 0.0)))
    route_gap = max(0.0, 1.0 - float(metrics.get("route_completion", 0.0)))
    reward_difficulty = max(0.0, -float(metrics.get("mean_reward", 0.0)))
    return (
        (2.0 * collision_rate)
        + (1.5 * out_of_road_rate)
        + (1.0 * top_rule_violation_rate)
        + (1.0 * success_gap)
        + (0.5 * route_gap)
        + (0.05 * reward_difficulty)
    )


def _normalize_proxy_usefulness(
    current_value: float,
    recent_values: list[float],
) -> float:
    rank_set = [float(v) for v in recent_values] + [float(current_value)]
    if len(rank_set) <= 1:
        return 1.0
    sorted_values = sorted(rank_set, reverse=True)
    rank = sorted_values.index(float(current_value)) + 1
    return 1.0 - ((rank - 1) / float(len(rank_set) - 1))


def _sample_iteration_env(
    *,
    arm: GeneratorArm,
    rng: np.random.Generator,
    chunk_id: int,
    cfg: DictConfig,
) -> tuple[int, dict[str, object]]:
    base_env_cfg = OmegaConf.to_container(cfg.env.config, resolve=True)
    if not isinstance(base_env_cfg, dict):
        raise TypeError("cfg.env.config must resolve to a mapping.")
    base_start_seed = int(base_env_cfg.get("start_seed", 0))
    scenario_seed = int(base_start_seed + chunk_id - 1)
    overrides = arm.sample_env_overrides(
        rng=rng,
        scenario_seed=scenario_seed,
        base_env_config=base_env_cfg,
    )
    return scenario_seed, overrides


def _export_scenario_dataset(
    *,
    cfg: DictConfig,
    env_overrides: dict[str, object],
    agent: Agent,
    chunk_id: int,
    scenario_seed: int,
    dataset_root: Path,
) -> str | None:
    try:
        from metadrive.scenario.utils import save_dataset
    except Exception:
        return None

    export_env = build_env(cfg, env_overrides)
    seed_env_spaces(export_env, int(cfg.seed) + 700_000 + chunk_id)
    dataset_dir = dataset_root / f"chunk_{chunk_id:04d}"
    export_path: str | None = None
    try:
        export_target = getattr(export_env, "unwrapped", export_env)

        def _policy_action(observation: Any):
            obs_value = observation
            if (
                isinstance(observation, tuple)
                and len(observation) == 2
                and isinstance(observation[1], dict)
            ):
                obs_value = observation[0]
            return agent.predict(obs_value, deterministic=True)[0]

        scenarios = export_target.export_scenarios(
            _policy_action,
            scenario_index=int(scenario_seed),
            max_episode_length=int(cfg.env.config.horizon),
            return_done_info=False,
            to_dict=True,
        )
        scenario = scenarios.get(int(scenario_seed))
        if scenario is None:
            return None
        save_dataset(
            [scenario],
            dataset_name="metadrive",
            dataset_version=f"scenario_acl_chunk_{chunk_id:04d}",
            dataset_dir=str(dataset_dir),
        )
        scenario_files = sorted(
            [
                path
                for path in dataset_dir.glob("*.pkl")
                if path.name not in {"dataset_summary.pkl", "dataset_mapping.pkl"}
            ]
        )
        if scenario_files:
            export_path = str(scenario_files[0])
    finally:
        export_env.close()
    return export_path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics_summary(metrics: dict[str, Any], *, generator_config_id: str) -> dict[str, Any]:
    return {
        "success_rate": float(metrics.get("success_rate", 0.0)),
        "route_completion": float(metrics.get("route_completion", 0.0)),
        "episode_return": float(metrics.get("mean_reward", 0.0)),
        "collision_rate": float(metrics.get("collision_rate", 0.0)),
        "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
        "top_rule_violation_rate": float(metrics.get("top_rule_violation_rate", 0.0)),
        "termination_reason": str(metrics.get("termination_reason", "unknown")),
        "generator_config_id": generator_config_id,
    }


def _build_record_from_export(
    *,
    export_path: str,
    env_overrides: dict[str, object],
    chunk_id: int,
    scenario_seed: int,
    arm_name: str,
    proxy_usefulness: float,
    normalized_usefulness: float,
    metrics: dict[str, Any],
    source: str = "generate",
    parent_id: str | None = None,
) -> ScenarioRecord:
    export_file = Path(export_path)
    return ScenarioRecord(
        scenario_id=f"{source}_chunk_{chunk_id:04d}",
        source=source,
        parent_id=parent_id,
        scenario_description_path=str(export_file),
        scenario_description_hash=_hash_file(export_file),
        dataset_directory=str(export_file.parent),
        scenario_index=0,
        env_config=dict(env_overrides),
        reset_seed=int(scenario_seed),
        generator_arm=arm_name,
        mutation_type=None,
        mutation_params=None,
        validation_status="valid",
        rule_criticality=0.0,
        learning_potential=float(proxy_usefulness),
        usefulness=float(proxy_usefulness),
        usefulness_norm=float(normalized_usefulness),
        rank=0,
        num_seen=1,
        last_seen_step=int(chunk_id),
        num_children=0,
        metrics_summary=_metrics_summary(metrics, generator_config_id=arm_name),
    )


def _update_replay_record(
    record: ScenarioRecord,
    *,
    chunk_id: int,
    proxy_usefulness: float,
    normalized_usefulness: float,
    metrics: dict[str, Any],
) -> ScenarioRecord:
    record.learning_potential = float(proxy_usefulness)
    record.usefulness = float(proxy_usefulness)
    record.usefulness_norm = float(normalized_usefulness)
    record.num_seen += 1
    record.last_seen_step = int(chunk_id)
    record.metrics_summary = _metrics_summary(
        metrics,
        generator_config_id=str(record.generator_arm or record.source),
    )
    return record


def _persist_buffer_state(
    *,
    path: Path,
    buffer: ScenarioBuffer,
) -> None:
    path.write_text(
        json.dumps(buffer.state_dict(), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _choose_iteration_spec(
    *,
    cfg: DictConfig,
    curriculum_cfg: CurriculumConfig,
    arms: list[GeneratorArm],
    bandit: GeneratorArmBandit,
    buffer: ScenarioBuffer,
    rng: np.random.Generator,
    chunk_id: int,
) -> IterationSpec:
    scenario_cfg = curriculum_cfg.scenario_acl
    can_exploit = (
        scenario_cfg.use_replay
        and len(buffer) >= int(scenario_cfg.warmup_buffer_size)
    )

    if can_exploit and rng.random() < float(scenario_cfg.exploit_probability):
        selection = buffer.sample_replay(
            rng=rng,
            current_step=chunk_id,
            cfg=scenario_cfg.replay_sampling,
            use_staleness=bool(scenario_cfg.use_staleness),
        )
        return IterationSpec(
            mode="exploit_replay",
            arm_index=-1,
            arm_name=str(selection.record.generator_arm or selection.record.source),
            arm_probabilities=[],
            scenario_seed=int(selection.record.reset_seed),
            train_env_overrides=None,
            replay_record=selection.record,
            replay_probabilities=selection.probabilities,
        )

    if bool(scenario_cfg.use_mab):
        arm_index, arm_probs = bandit.sample_arm(rng)
    else:
        arm_index = int(rng.integers(0, len(arms)))
        arm_probs = np.full(len(arms), 1.0 / len(arms), dtype=np.float64)
    arm = arms[arm_index]
    scenario_seed, train_env_overrides = _sample_iteration_env(
        arm=arm,
        rng=rng,
        chunk_id=chunk_id,
        cfg=cfg,
    )
    return IterationSpec(
        mode="generate",
        arm_index=arm_index,
        arm_name=arm.name,
        arm_probabilities=[float(x) for x in arm_probs.tolist()],
        scenario_seed=int(scenario_seed),
        train_env_overrides=train_env_overrides,
        replay_record=None,
        replay_probabilities=None,
    )


def run_scenario_acl_training(
    *,
    cfg: DictConfig,
    curriculum_cfg: CurriculumConfig,
    recorder: CSVRecorder,
    base_csv_fields: dict[str, Any],
    metadata_path: Path,
    hydra_config_path: Path,
    start_time: float,
    paths: ScenarioAclDriverPaths,
    train_logger: logging.Logger,
    curriculum_logger: logging.Logger,
) -> None:
    if not curriculum_cfg.is_scenario_acl:
        raise ValueError("Scenario ACL driver requires curriculum kind 'scenario_acl'.")
    if bool(cfg.checkpoint.get("resume", {}).get("enabled", False)):
        raise ValueError("Scenario ACL resume is not implemented yet.")
    if bool(curriculum_cfg.scenario_acl.use_mutation):
        raise NotImplementedError(
            "scenario_acl mutation support is not implemented yet. "
            "Use mode='mab_plus_replay' or disable use_mutation."
        )

    total_timesteps = int(cfg.experiment.total_timesteps)
    eval_interval = int(cfg.experiment.get("eval_interval", total_timesteps))
    if eval_interval <= 0:
        eval_interval = total_timesteps
    log_interval = int(cfg.experiment.get("log_interval", 1000))
    run_seed = int(cfg.seed)
    scenario_cfg = curriculum_cfg.scenario_acl
    artifact_paths = _scenario_acl_artifact_paths(paths.artifacts_dir)
    rng = np.random.default_rng(run_seed)

    arms = build_default_generator_arms()
    expected_arms = int(scenario_cfg.mab.num_arms)
    if len(arms) != expected_arms:
        raise ValueError(
            "Configured scenario_acl.mab.num_arms does not match the built-in arm set: "
            f"expected {expected_arms}, got {len(arms)}."
        )

    bandit = GeneratorArmBandit(scenario_cfg.mab)
    buffer = ScenarioBuffer(capacity=int(scenario_cfg.buffer_capacity))
    recent_usefulness: list[float] = []
    current_global_step = 0
    current_eval_id = 0
    current_chunk_id = 0
    generate_count = 0
    replay_count = 0

    first_spec = _choose_iteration_spec(
        cfg=cfg,
        curriculum_cfg=curriculum_cfg,
        arms=arms,
        bandit=bandit,
        buffer=buffer,
        rng=rng,
        chunk_id=1,
    )
    if first_spec.mode == "generate":
        env = build_env(cfg, first_spec.train_env_overrides)
    else:
        if first_spec.replay_record is None:
            raise RuntimeError("Replay iteration requires a replay record.")
        env = build_scenario_replay_env(
            cfg,
            record=first_spec.replay_record,
            scenario_env_cfg=scenario_cfg.scenario_env,
        )
    seed_env_spaces(env, run_seed)
    train_env_count = count_envs(env)

    preprocessor = build_preprocessor(cfg)
    adapter = build_adapter(cfg, adapter_space_kwargs(env.action_space))
    planner = build_planner(cfg, env, seed=run_seed)
    planner_device = str(getattr(planner, "device", cfg.device))
    ema_alpha_cfg = (
        float(cfg.agent.planner.algorithm.get("monitor_ema_alpha", 0.1))
        if hasattr(cfg, "agent")
        else 0.1
    )
    agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter, ema_alpha=ema_alpha_cfg)

    print_run_setup(
        title="Training Run",
        cfg=cfg,
        metadata_path=metadata_path,
        hydra_config_path=hydra_config_path,
        extra_rows=[
            ("Observation space", str(env.observation_space)),
            ("Action space", str(env.action_space)),
            ("Vectorized training envs", str(train_env_count)),
            ("Planner device", planner_device),
            ("Curriculum kind", "scenario_acl"),
            ("Scenario ACL mode", scenario_cfg.mode),
        ],
    )

    try:
        remaining = total_timesteps
        next_spec = first_spec
        while remaining > 0:
            current_chunk_id += 1
            current_spec = next_spec
            if current_chunk_id > 1:
                if current_spec.mode == "generate":
                    env = build_env(cfg, current_spec.train_env_overrides)
                    seed_env_spaces(env, run_seed + 400_000 + current_chunk_id)
                else:
                    if current_spec.replay_record is None:
                        raise RuntimeError("Replay iteration requires a replay record.")
                    env = build_scenario_replay_env(
                        cfg,
                        record=current_spec.replay_record,
                        scenario_env_cfg=scenario_cfg.scenario_env,
                    )
                    seed_env_spaces(env, run_seed + 450_000 + current_chunk_id)
                set_planner_env_if_compatible(planner, env)

            chunk_steps = min(eval_interval, remaining)
            steps_start = current_global_step
            if current_spec.mode == "generate":
                generate_count += 1
            else:
                replay_count += 1

            curriculum_logger.info(
                "Scenario ACL iteration started | chunk_id=%d | mode=%s | arm=%s | scenario_seed=%d",
                current_chunk_id,
                current_spec.mode,
                current_spec.arm_name,
                current_spec.scenario_seed,
            )
            log_event(
                paths.events_log_path,
                "scenario_acl_iteration_started",
                chunk_id=current_chunk_id,
                mode=current_spec.mode,
                arm_name=current_spec.arm_name,
                arm_index=current_spec.arm_index,
                scenario_seed=current_spec.scenario_seed,
                arm_probabilities=current_spec.arm_probabilities,
                replay_probabilities=current_spec.replay_probabilities,
                replay_scenario_id=(
                    current_spec.replay_record.scenario_id
                    if current_spec.replay_record is not None
                    else None
                ),
                env_overrides=current_spec.train_env_overrides,
            )

            def train_reset_seed_for_episode(episode_index: int) -> int | None:
                if current_spec.train_env_overrides is not None:
                    return train_episode_seed_from_env_overrides(
                        current_spec.train_env_overrides,
                        cfg,
                        run_seed=run_seed,
                        chunk_id=current_chunk_id,
                        episode_index=episode_index,
                        stage_index=max(current_spec.arm_index, 0),
                    )
                # ScenarioEnv interprets reset(seed=...) as a scenario index selector.
                # For replay we must not pass arbitrary seeds or we will go out of range.
                return None

            chunk_summary = agent.train(
                env=env,
                chunk_timesteps=chunk_steps,
                global_total_timesteps=total_timesteps,
                global_steps_done=current_global_step,
                stage_name=(
                    current_spec.arm_name
                    if current_spec.mode == "generate"
                    else f"replay:{current_spec.arm_name}"
                ),
                deterministic=False,
                log_interval=log_interval,
                reset_seed_fn=train_reset_seed_for_episode,
            )
            actual_chunk_steps = int(chunk_summary.get("chunk_steps_actual", chunk_steps))
            current_global_step = min(total_timesteps, current_global_step + actual_chunk_steps)
            remaining = max(0, total_timesteps - current_global_step)
            recorder.append_row(
                "train_chunks.csv",
                {
                    **base_csv_fields,
                    "chunk_id": current_chunk_id,
                    "stage": (
                        current_spec.arm_name
                        if current_spec.mode == "generate"
                        else f"replay:{current_spec.arm_name}"
                    ),
                    "stage_index": current_spec.arm_index,
                    "steps_start": steps_start,
                    "steps_end": current_global_step,
                    "global_step": current_global_step,
                    "chunk_steps": actual_chunk_steps,
                    "episodes": int(chunk_summary.get("episodes", 0)),
                    "ep_rew_mean": float(chunk_summary.get("ep_rew_mean", 0.0)),
                    "ep_rew_std": float(chunk_summary.get("ep_rew_std", 0.0)),
                    "ep_rew_ci_95": float(chunk_summary.get("ep_rew_ci_95", 0.0)),
                    "ep_env_rew_mean": chunk_summary.get("ep_env_rew_mean"),
                    "ep_scalar_rule_rew_mean": chunk_summary.get("ep_scalar_rule_rew_mean"),
                    "ep_hybrid_rew_mean": chunk_summary.get("ep_hybrid_rew_mean"),
                    "ep_len_mean": float(chunk_summary.get("ep_len_mean", 0.0)),
                    "ep_len_std": float(chunk_summary.get("ep_len_std", 0.0)),
                    "ep_len_ci_95": float(chunk_summary.get("ep_len_ci_95", 0.0)),
                    "ep_success_rate": chunk_summary.get("ep_success_rate"),
                    "ep_collision_rate": chunk_summary.get("ep_collision_rate"),
                    "ep_out_of_road_rate": chunk_summary.get("ep_out_of_road_rate"),
                    "ep_route_completion_mean": chunk_summary.get("ep_route_completion_mean"),
                    "actor_loss": float(chunk_summary.get("actor_loss", 0.0)),
                    "critic_loss": float(chunk_summary.get("critic_loss", 0.0)),
                    "actor_loss_ema": float(chunk_summary.get("actor_loss_ema") or 0.0),
                    "critic_loss_ema": float(chunk_summary.get("critic_loss_ema") or 0.0),
                    "learning_rate": float(chunk_summary.get("learning_rate", 0.0)),
                    "update_calls": int(chunk_summary.get("update_calls", 0)),
                    "n_updates": int(chunk_summary.get("n_updates", 0)),
                    "fps": float(chunk_summary.get("fps", 0.0)),
                    "elapsed_seconds": float(chunk_summary.get("elapsed_seconds", 0.0)),
                    "train_reset_seed_first": chunk_summary.get("train_reset_seed_first"),
                    "train_reset_seed_last": chunk_summary.get("train_reset_seed_last"),
                    "train_reset_seed_unique_count": chunk_summary.get("train_reset_seed_unique_count"),
                },
            )

            env.close()

            eval_metrics: dict[str, Any]
            eval_agent: Agent
            if current_spec.mode == "generate":
                eval_env_overrides = apply_eval_scenario_seed_split(
                    base_run_seed=run_seed,
                    eval_env_overrides=current_spec.train_env_overrides,
                    cfg=cfg,
                    n_eval_episodes=int(cfg.experiment.eval_episodes),
                    split="validation",
                )
                eval_base_seed = eval_base_seed_from_env_overrides(eval_env_overrides, cfg)
                eval_env = build_env(cfg, eval_env_overrides)
                seed_env_spaces(eval_env, run_seed + 500_000 + current_chunk_id)
            else:
                if current_spec.replay_record is None:
                    raise RuntimeError("Replay iteration requires a replay record.")
                eval_base_seed = None
                eval_env = build_scenario_replay_env(
                    cfg,
                    record=current_spec.replay_record,
                    scenario_env_cfg=scenario_cfg.scenario_env,
                )
                seed_env_spaces(eval_env, run_seed + 550_000 + current_chunk_id)

            eval_snapshot_stem = paths.checkpoints_dir / "eval_snapshot"
            agent.save(eval_snapshot_stem)
            eval_agent = Agent(
                preprocessor=preprocessor,
                planner=load_planner(cfg, checkpoint_path=f"{eval_snapshot_stem}.zip", env=eval_env),
                adapter=adapter,
                ema_alpha=ema_alpha_cfg,
            )
            current_eval_id += 1
            eval_metrics = eval_agent.evaluate(
                env=eval_env,
                n_eval_episodes=int(cfg.experiment.eval_episodes),
                deterministic=bool(cfg.experiment.eval_deterministic),
                base_seed=eval_base_seed,
                return_episode_metrics=False,
                error_priority_base=float(cfg.reward.get("a", 2.01)),
                show_progress=True,
            )
            eval_env.close()
            Path(f"{eval_snapshot_stem}.zip").unlink(missing_ok=True)
            Agent.adapter_checkpoint_path(eval_snapshot_stem).unlink(missing_ok=True)

            proxy_usefulness = _compute_proxy_usefulness(eval_metrics)
            normalized_usefulness = _normalize_proxy_usefulness(
                proxy_usefulness,
                recent_usefulness,
            )
            recent_usefulness.append(proxy_usefulness)
            max_recent = int(scenario_cfg.recent_window_size)
            if len(recent_usefulness) > max_recent:
                recent_usefulness = recent_usefulness[-max_recent:]

            export_path: str | None = None
            buffer_action = "none"
            scenario_record_id: str | None = None
            if current_spec.mode == "generate":
                if current_spec.train_env_overrides is None:
                    raise RuntimeError("Generate iteration requires training env overrides.")
                if bool(scenario_cfg.use_mab):
                    bandit.update(
                        arm_index=current_spec.arm_index,
                        normalized_usefulness=normalized_usefulness,
                        selection_probability=float(
                            current_spec.arm_probabilities[current_spec.arm_index]
                        ),
                    )
                export_path = _export_scenario_dataset(
                    cfg=cfg,
                    env_overrides=current_spec.train_env_overrides,
                    agent=agent,
                    chunk_id=current_chunk_id,
                    scenario_seed=current_spec.scenario_seed,
                    dataset_root=artifact_paths["scenarios"],
                )
                if export_path is not None:
                    record = _build_record_from_export(
                        export_path=export_path,
                        env_overrides=current_spec.train_env_overrides,
                        chunk_id=current_chunk_id,
                        scenario_seed=current_spec.scenario_seed,
                        arm_name=current_spec.arm_name,
                        proxy_usefulness=proxy_usefulness,
                        normalized_usefulness=normalized_usefulness,
                        metrics=eval_metrics,
                    )
                    inserted = buffer.insert(record)
                    buffer_action = "inserted" if inserted else "rejected"
                    scenario_record_id = record.scenario_id
                    _append_jsonl(
                        artifact_paths["buffer_events"],
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "chunk_id": current_chunk_id,
                            "action": buffer_action,
                            "scenario_id": record.scenario_id,
                            "scenario_hash": record.scenario_description_hash,
                            "usefulness": float(proxy_usefulness),
                        },
                    )
            else:
                if current_spec.replay_record is None:
                    raise RuntimeError("Replay iteration requires a replay record.")
                updated_record = _update_replay_record(
                    current_spec.replay_record,
                    chunk_id=current_chunk_id,
                    proxy_usefulness=proxy_usefulness,
                    normalized_usefulness=normalized_usefulness,
                    metrics=eval_metrics,
                )
                buffer.update(updated_record)
                buffer_action = "updated"
                scenario_record_id = updated_record.scenario_id
                _append_jsonl(
                    artifact_paths["buffer_events"],
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "chunk_id": current_chunk_id,
                        "action": buffer_action,
                        "scenario_id": updated_record.scenario_id,
                        "usefulness": float(proxy_usefulness),
                        "num_seen": int(updated_record.num_seen),
                    },
                )

            _persist_buffer_state(path=artifact_paths["buffer"], buffer=buffer)

            recorder.append_row(
                "evals.csv",
                {
                    **base_csv_fields,
                    "eval_id": current_eval_id,
                    "eval_type": "intermediate",
                    "scenario_set": (
                        "scenario_acl_generate_eval"
                        if current_spec.mode == "generate"
                        else "scenario_acl_replay_eval"
                    ),
                    "chunk_id": current_chunk_id,
                    "stage": (
                        current_spec.arm_name
                        if current_spec.mode == "generate"
                        else f"replay:{current_spec.arm_name}"
                    ),
                    "stage_index": current_spec.arm_index,
                    "global_step": current_global_step,
                    "eval_episodes": int(cfg.experiment.eval_episodes),
                    "deterministic": bool(cfg.experiment.eval_deterministic),
                    "mean_reward": float(eval_metrics.get("mean_reward", 0.0)),
                    "std_reward": float(eval_metrics.get("std_reward", 0.0)),
                    "mean_env_reward": float(eval_metrics.get("mean_env_reward", 0.0)),
                    "std_env_reward": float(eval_metrics.get("std_env_reward", 0.0)),
                    "mean_scalar_rule_reward": eval_metrics.get("mean_scalar_rule_reward"),
                    "std_scalar_rule_reward": eval_metrics.get("std_scalar_rule_reward"),
                    "mean_hybrid_reward": eval_metrics.get("mean_hybrid_reward"),
                    "std_hybrid_reward": eval_metrics.get("std_hybrid_reward"),
                    "mean_rule_saturation_max": float(eval_metrics.get("mean_rule_saturation_max", 0.0)),
                    "collision_rate": float(eval_metrics.get("collision_rate", 0.0)),
                    "collision_rate_std": float(eval_metrics.get("collision_rate_std", 0.0)),
                    "out_of_road_rate": float(eval_metrics.get("out_of_road_rate", 0.0)),
                    "success_rate": float(eval_metrics.get("success_rate", 0.0)),
                    "success_rate_std": float(eval_metrics.get("success_rate_std", 0.0)),
                    "route_completion": float(eval_metrics.get("route_completion", 0.0)),
                    "top_rule_violation_rate": float(
                        eval_metrics.get("top_rule_violation_rate", 0.0)
                    ),
                    "avg_error_value": float(eval_metrics.get("avg_error_value", 0.0)),
                    "max_error_value": float(eval_metrics.get("max_error_value", 0.0)),
                    "counterexample_rate": float(eval_metrics.get("counterexample_rate", 0.0)),
                    "violated_rules_ratio": float(eval_metrics.get("violated_rules_ratio", 0.0)),
                    "unique_violation_patterns": int(eval_metrics.get("unique_violation_patterns", 0)),
                    "promoted": False,
                    "next_stage": current_spec.arm_name,
                },
            )

            history_payload = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "chunk_id": current_chunk_id,
                "global_step": current_global_step,
                "mode": current_spec.mode,
                "arm_index": current_spec.arm_index,
                "arm_name": current_spec.arm_name,
                "scenario_seed": current_spec.scenario_seed,
                "selection_probability": (
                    float(current_spec.arm_probabilities[current_spec.arm_index])
                    if current_spec.mode == "generate"
                    else None
                ),
                "arm_probabilities": current_spec.arm_probabilities,
                "replay_probabilities": current_spec.replay_probabilities,
                "replay_scenario_id": (
                    current_spec.replay_record.scenario_id
                    if current_spec.replay_record is not None
                    else None
                ),
                "proxy_usefulness": float(proxy_usefulness),
                "normalized_usefulness": float(normalized_usefulness),
                "env_overrides": current_spec.train_env_overrides,
                "export_path": export_path,
                "scenario_record_id": scenario_record_id,
                "buffer_action": buffer_action,
                "buffer_size": len(buffer),
                "metrics": {
                    "mean_reward": float(eval_metrics.get("mean_reward", 0.0)),
                    "collision_rate": float(eval_metrics.get("collision_rate", 0.0)),
                    "out_of_road_rate": float(eval_metrics.get("out_of_road_rate", 0.0)),
                    "success_rate": float(eval_metrics.get("success_rate", 0.0)),
                    "route_completion": float(eval_metrics.get("route_completion", 0.0)),
                    "top_rule_violation_rate": float(
                        eval_metrics.get("top_rule_violation_rate", 0.0)
                    ),
                },
                "mab": bandit.state_dict(),
            }
            _append_jsonl(artifact_paths["history"], history_payload)
            _append_jsonl(artifact_paths["iterations"], history_payload)
            artifact_paths["state"].write_text(
                json.dumps(
                    {
                        "global_step": int(current_global_step),
                        "chunk_id": int(current_chunk_id),
                        "eval_id": int(current_eval_id),
                        "recent_usefulness": [float(x) for x in recent_usefulness],
                        "last_mode": current_spec.mode,
                        "last_arm_name": current_spec.arm_name,
                        "last_arm_index": int(current_spec.arm_index),
                        "last_scenario_seed": int(current_spec.scenario_seed),
                        "last_replay_scenario_id": (
                            current_spec.replay_record.scenario_id
                            if current_spec.replay_record is not None
                            else None
                        ),
                        "buffer_size": len(buffer),
                        "generate_count": int(generate_count),
                        "replay_count": int(replay_count),
                        "buffer_top": [
                            {
                                "scenario_id": record.scenario_id,
                                "rank": int(record.rank),
                                "usefulness": float(record.usefulness),
                                "num_seen": int(record.num_seen),
                                "source": record.source,
                            }
                            for record in buffer.top_k(5)
                        ],
                        "mab": bandit.state_dict(),
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if bool(cfg.checkpoint.get("save_latest_each_chunk", True)):
                agent.save(paths.latest_checkpoint_stem)

            train_logger.info(
                "Scenario ACL iteration finished | chunk_id=%d | mode=%s | arm=%s | "
                "scenario_seed=%d | global_step=%d | usefulness=%.4f | usefulness_norm=%.4f | "
                "buffer_size=%d",
                current_chunk_id,
                current_spec.mode,
                current_spec.arm_name,
                current_spec.scenario_seed,
                current_global_step,
                proxy_usefulness,
                normalized_usefulness,
                len(buffer),
            )

            if remaining > 0:
                next_spec = _choose_iteration_spec(
                    cfg=cfg,
                    curriculum_cfg=curriculum_cfg,
                    arms=arms,
                    bandit=bandit,
                    buffer=buffer,
                    rng=rng,
                    chunk_id=current_chunk_id + 1,
                )

        if not bool(cfg.checkpoint.get("save_final", True)):
            raise ValueError("checkpoint.save_final must be true for scenario_acl training.")
        agent.save(paths.final_checkpoint_stem)

        final_eval_env_overrides = apply_eval_scenario_seed_split(
            base_run_seed=run_seed,
            eval_env_overrides=None,
            cfg=cfg,
            n_eval_episodes=int(
                cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)
            ),
            split="test",
        )
        final_eval_base_seed = eval_base_seed_from_env_overrides(
            final_eval_env_overrides,
            cfg,
        )
        final_eval_env = build_env(cfg, final_eval_env_overrides)
        seed_env_spaces(final_eval_env, run_seed + 600_000)
        final_eval_agent = Agent(
            preprocessor=preprocessor,
            planner=load_planner(
                cfg,
                checkpoint_path=f"{paths.final_checkpoint_stem}.zip",
                env=final_eval_env,
            ),
            adapter=adapter,
            ema_alpha=ema_alpha_cfg,
        )
        resolved_final_eval_cfg = OmegaConf.to_container(
            merge_env_config_with_overrides(cfg.env, final_eval_env_overrides or {}),
            resolve=True,
        )
        if not isinstance(resolved_final_eval_cfg, dict):
            raise TypeError("Resolved final eval env config must be a mapping.")
        resolved_final_eval_env_config = resolved_final_eval_cfg.get(
            "config",
            resolved_final_eval_cfg,
        )
        if not isinstance(resolved_final_eval_env_config, dict):
            raise TypeError("Resolved final eval env config payload must be a mapping.")
        final_eval_id = current_eval_id + 1
        artifact_factory = maybe_build_live_final_eval_recorder_factory(
            cfg=cfg,
            run_dir=paths.run_dir,
            resolved_env_config=resolved_final_eval_env_config,
            eval_id=final_eval_id,
            eval_type="final",
            scenario_set="test",
            stage="scenario_acl",
            stage_index=0,
            checkpoint_path=str(
                paths.final_checkpoint_stem.with_suffix(".zip").relative_to(paths.run_dir)
            ),
            checkpoint_type="final",
            checkpoint_global_step=int(current_global_step),
        )
        final_metrics = final_eval_agent.evaluate(
            env=final_eval_env,
            n_eval_episodes=int(
                cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)
            ),
            deterministic=bool(cfg.experiment.eval_deterministic),
            base_seed=final_eval_base_seed,
            return_episode_metrics=True,
            error_priority_base=float(cfg.reward.get("a", 2.01)),
            show_progress=True,
            artifact_recorder_factory=artifact_factory,
        )
        final_eval_env.close()
        print_evaluation_summary(
            title="Final Evaluation",
            metrics=final_metrics,
            stage="scenario_acl",
            global_step=current_global_step,
            episodes=int(cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)),
            base_seed=final_eval_base_seed,
            details_path=paths.events_log_path,
            checkpoint_path=f"{paths.final_checkpoint_stem}.zip",
        )
        recorder.append_row(
            "final_eval.csv",
            {
                **base_csv_fields,
                "eval_type": "final",
                "scenario_set": "test",
                "total_timesteps": current_global_step,
                "final_stage": "scenario_acl",
                "final_stage_index": 0,
                "final_stage_reached": True,
                "steps_to_final_stage": 0,
                "final_eval_episodes": int(
                    cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)
                ),
                "deterministic": bool(cfg.experiment.eval_deterministic),
                "mean_reward": float(final_metrics.get("mean_reward", 0.0)),
                "std_reward": float(final_metrics.get("std_reward", 0.0)),
                "mean_env_reward": float(final_metrics.get("mean_env_reward", 0.0)),
                "std_env_reward": float(final_metrics.get("std_env_reward", 0.0)),
                "mean_scalar_rule_reward": final_metrics.get("mean_scalar_rule_reward"),
                "std_scalar_rule_reward": final_metrics.get("std_scalar_rule_reward"),
                "mean_hybrid_reward": final_metrics.get("mean_hybrid_reward"),
                "std_hybrid_reward": final_metrics.get("std_hybrid_reward"),
                "mean_rule_saturation_max": float(
                    final_metrics.get("mean_rule_saturation_max", 0.0)
                ),
                "collision_rate": float(final_metrics.get("collision_rate", 0.0)),
                "collision_rate_std": float(final_metrics.get("collision_rate_std", 0.0)),
                "out_of_road_rate": float(final_metrics.get("out_of_road_rate", 0.0)),
                "success_rate": float(final_metrics.get("success_rate", 0.0)),
                "success_rate_std": float(final_metrics.get("success_rate_std", 0.0)),
                "route_completion": float(final_metrics.get("route_completion", 0.0)),
                "top_rule_violation_rate": float(
                    final_metrics.get("top_rule_violation_rate", 0.0)
                ),
                "avg_error_value": float(final_metrics.get("avg_error_value", 0.0)),
                "max_error_value": float(final_metrics.get("max_error_value", 0.0)),
                "counterexample_rate": float(final_metrics.get("counterexample_rate", 0.0)),
                "violated_rules_ratio": float(
                    final_metrics.get("violated_rules_ratio", 0.0)
                ),
                "unique_violation_patterns": int(
                    final_metrics.get("unique_violation_patterns", 0)
                ),
                "checkpoint_path": str(
                    paths.final_checkpoint_stem.with_suffix(".zip").relative_to(paths.run_dir)
                ),
                "checkpoint_type": "final",
                "checkpoint_global_step": int(current_global_step),
            },
        )

        duration_seconds = round(time.time() - start_time, 2)
        update_run_metadata(
            paths.artifacts_dir,
            {
                "status": "completed",
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": duration_seconds,
                "global_step": int(current_global_step),
                "chunk_id": int(current_chunk_id),
                "eval_id": int(final_eval_id),
                "stage": "scenario_acl",
                "stage_index": 0,
            },
        )
    except KeyboardInterrupt:
        agent.save(paths.latest_checkpoint_stem)
        duration_seconds = round(time.time() - start_time, 2)
        update_run_metadata(
            paths.artifacts_dir,
            {
                "status": "interrupted",
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": duration_seconds,
                "global_step": int(current_global_step),
                "chunk_id": int(current_chunk_id),
                "eval_id": int(current_eval_id),
                "stage": "scenario_acl",
                "stage_index": 0,
            },
        )
        raise SystemExit(130)
