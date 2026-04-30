from __future__ import annotations

import json
import logging
import time
from datetime import datetime
import csv
import pickle

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
    load_planner,
)
from thesis_rl.runtime.console import print_evaluation_summary, print_run_setup
from thesis_rl.runtime.csv_recorder import CSVRecorder
from thesis_rl.runtime.metadata import save_run_metadata, update_run_metadata
from thesis_rl.runtime.run_logging import log_event, setup_file_logger
from thesis_rl.runtime.seeding import (
    apply_eval_scenario_seed_split,
    eval_base_seed_from_env_overrides,
    seed_env_spaces,
    set_global_seed,
    train_episode_seed_from_env_overrides,
)


CHECKPOINT_INDEX_FIELDS = [
    "checkpoint_path",
    "type",
    "global_step",
    "chunk_id",
    "eval_id",
    "stage",
    "stage_index",
    "success_rate",
    "collision_rate",
    "out_of_road_rate",
    "top_rule_violation_rate",
    "route_completion",
    "mean_reward",
    "avg_error_value",
    "max_error_value",
    "reason",
    "timestamp",
]

def _append_checkpoint_index_row(index_path: Path, row: dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = index_path.exists()
    with index_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHECKPOINT_INDEX_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in CHECKPOINT_INDEX_FIELDS})


def _checkpoint_rel(run_dir: Path, checkpoint_stem_path: Path) -> str:
    zip_path = checkpoint_stem_path.with_suffix(".zip")
    return str(zip_path.relative_to(run_dir)).replace("\\", "/")


def _lexicographic_eval_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    return (
        float(metrics.get("collision_rate", float("inf"))),
        float(metrics.get("out_of_road_rate", float("inf"))),
        float(metrics.get("avg_error_value", float("inf"))),
        float(metrics.get("top_rule_violation_rate", float("inf"))),
        -float(metrics.get("success_rate", float("-inf"))),
        -float(metrics.get("route_completion", float("-inf"))),
        -float(metrics.get("mean_reward", float("-inf"))),
    )


def _rulebook_strict_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    rows = metrics.get("per_rule", [])
    if not isinstance(rows, list) or not rows:
        return (float("inf"),)
    ordered = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda row: (int(row.get("rule_priority", 0)), str(row.get("rule_name", ""))),
    )
    # Strict: compare raw margin even when both are >= 0 (higher margin is better).
    return tuple(-float(row.get("min_margin", 0.0)) for row in ordered)


def _rulebook_thresholded_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    rows = metrics.get("per_rule", [])
    if not isinstance(rows, list) or not rows:
        return (float("inf"),)
    ordered = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda row: (int(row.get("rule_priority", 0)), str(row.get("rule_name", ""))),
    )
    key: list[float] = []
    for row in ordered:
        margin = float(row.get("min_margin", 0.0))
        # Thresholded: any margin >= 0 is considered equally satisfied.
        if margin >= 0.0:
            key.extend([0.0, 0.0])
        else:
            # First value separates satisfied vs violated, second compares violation severity.
            key.extend([1.0, -margin])
    return tuple(key)


def _write_best_checkpoints_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(payload), f=str(path))


def _prune_old_periodic_checkpoints(periodic_dir: Path, keep_last: int) -> None:
    keep_n = max(int(keep_last), 0)
    zip_files = sorted(
        [path for path in periodic_dir.glob("step_*.zip") if path.is_file()],
        key=lambda path: path.name,
    )
    if keep_n <= 0:
        to_remove = zip_files
    else:
        to_remove = zip_files[:-keep_n]
    for zip_path in to_remove:
        stem = zip_path.with_suffix("")
        adapter_path = stem.parent / f"{stem.name}.adapter.pt"
        if zip_path.exists():
            zip_path.unlink()
        if adapter_path.exists():
            adapter_path.unlink()


def _save_training_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(payload), f=str(path))


def _load_training_state(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    return dict(OmegaConf.to_container(loaded, resolve=True))


def _planner_model(planner: Any) -> Any | None:
    model = getattr(planner, "model", None)
    if model is not None:
        return model
    return getattr(planner, "sb3_model", None)


def _save_replay_buffer_if_available(planner: Any, path: Path) -> bool:
    model = _planner_model(planner)
    if model is None or not hasattr(model, "save_replay_buffer"):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_replay_buffer(str(path))
    return True


def _load_replay_buffer_if_available(planner: Any, path: Path) -> bool:
    if not path.exists():
        return False
    model = _planner_model(planner)
    if model is None or not hasattr(model, "load_replay_buffer"):
        return False
    model.load_replay_buffer(str(path))
    return True


def _save_rng_state(path: Path) -> None:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu().numpy().tolist(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        payload["cuda"] = [state.cpu().numpy().tolist() for state in torch.cuda.get_rng_state_all()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _load_rng_state(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return False

    py_state = payload.get("python")
    np_state = payload.get("numpy")
    torch_state = payload.get("torch")
    cuda_state = payload.get("cuda")

    if py_state is not None:
        random.setstate(py_state)
    if np_state is not None:
        np.random.set_state(np_state)
    if torch_state is not None:
        torch.set_rng_state(torch.tensor(torch_state, dtype=torch.uint8))
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.tensor(state, dtype=torch.uint8, device="cpu") for state in cuda_state])
    return True


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


def _min_stage_steps(curriculum_cfg: CurriculumConfig, stage_name: str) -> int | None:
    stage_cfg = curriculum_cfg.promotion.per_stage_min_steps.get(stage_name)
    if stage_cfg is not None:
        return int(stage_cfg)
    if curriculum_cfg.promotion.default_min_stage_steps > 0:
        return int(curriculum_cfg.promotion.default_min_stage_steps)
    return None


def _append_rule_metrics_rows(
    recorder: CSVRecorder,
    *,
    base_fields: dict[str, Any],
    eval_id: int,
    chunk_id: int,
    stage: str,
    stage_index: int,
    global_step: int,
    metrics: dict[str, Any],
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
            },
        )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Save metadata for this run (config, git info, etc.)
    artifacts_dir = Path(str(cfg.paths.artifacts_dir))
    metadata_path = save_run_metadata(cfg, artifacts_dir)
    hydra_config_path = Path(str(cfg.paths.run_dir)) / "hydra" / "config.yaml"
    print_run_setup(
        title="Training Run",
        cfg=cfg,
        metadata_path=metadata_path,
        hydra_config_path=hydra_config_path,
    )
    start_time = time.time()
    logs_dir = Path(str(cfg.paths.logs_dir))
    csv_dir = Path(str(cfg.paths.csv_dir))
    recorder = CSVRecorder(csv_dir)
    run_id = Path(str(cfg.paths.run_dir)).name
    base_csv_fields = {
        "algorithm": str(cfg.planner.name),
        "reward_mode": str(cfg.reward.mode),
        "curriculum_name": str(cfg.curriculum.name),
        "seed": int(cfg.seed),
        "run_id": run_id,
    }
    run_dir = Path(str(cfg.paths.run_dir))
    checkpoints_dir = Path(str(cfg.paths.checkpoints_dir))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_best_dir = checkpoints_dir / "best"
    checkpoints_best_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_periodic_dir = checkpoints_dir / "periodic"
    checkpoints_periodic_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_metadata_dir = checkpoints_dir / "metadata"
    checkpoints_metadata_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_index_path = checkpoints_metadata_dir / "checkpoint_index.csv"
    best_checkpoints_yaml_path = checkpoints_metadata_dir / "best_checkpoints.yaml"
    latest_checkpoint_stem = checkpoints_dir / "latest"
    final_checkpoint_stem = checkpoints_dir / "final"
    best_lex_checkpoint_stem = checkpoints_best_dir / "best_lexicographic"
    best_lex_key: tuple[float, float, float, float, float, float, float] | None = None
    best_lex_payload: dict[str, Any] = {}
    best_rulebook_strict_key: tuple[float, ...] | None = None
    best_rulebook_strict_payload: dict[str, Any] = {}
    best_rulebook_thresholded_key: tuple[float, ...] | None = None
    best_rulebook_thresholded_payload: dict[str, Any] = {}
    best_rulebook_strict_checkpoint_stem = checkpoints_best_dir / "best_lexicographic_rulebook"
    best_rulebook_thresholded_checkpoint_stem = (
        checkpoints_best_dir / "best_thresholded_lexicographic_rulebook"
    )
    latest_replay_buffer_path = checkpoints_dir / "latest_replay_buffer.pkl"
    latest_training_state_path = checkpoints_dir / "latest_training_state.yaml"
    latest_rng_state_path = checkpoints_dir / "latest_rng_state.pkl"
    train_log_path = logs_dir / "train.log"
    eval_log_path = logs_dir / "eval.log"
    curriculum_log_path = logs_dir / "curriculum.log"
    errors_log_path = logs_dir / "errors.log"
    events_log_path = logs_dir / "events.jsonl"

    train_logger = setup_file_logger("thesis_rl.train", "train", train_log_path, level=logging.INFO)
    eval_logger = setup_file_logger("thesis_rl.train", "eval", eval_log_path, level=logging.INFO)
    curriculum_logger = setup_file_logger("thesis_rl.train", "curriculum", curriculum_log_path, level=logging.INFO)
    errors_logger = setup_file_logger("thesis_rl.train", "errors", errors_log_path, level=logging.WARNING)

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
        set_global_seed(run_seed)
        resume_cfg = cfg.checkpoint.get("resume", {})
        resume_enabled = bool(resume_cfg.get("enabled", False))
        resume_run_dir_cfg = resume_cfg.get("run_dir")
        resume_run_dir = (
            Path(str(resume_run_dir_cfg))
            if resume_run_dir_cfg not in (None, "", "null")
            else run_dir
        )
        resume_checkpoint_name = str(resume_cfg.get("checkpoint_name", "latest"))
        resume_checkpoint_stem = resume_run_dir / "checkpoints" / resume_checkpoint_name
        resume_checkpoint_zip = resume_checkpoint_stem.with_suffix(".zip")
        resume_replay_buffer_path = resume_run_dir / "checkpoints" / "latest_replay_buffer.pkl"
        resume_training_state_path = resume_run_dir / "checkpoints" / "latest_training_state.yaml"
        resume_rng_state_path = resume_run_dir / "checkpoints" / "latest_rng_state.pkl"
        resume_state: dict[str, Any] | None = None
        resume_global_steps_done = 0
        resume_chunk_id = 0
        resume_eval_id = 0

        ###################
        ###### SETUP ######
        ###################
        
        # Curriculum manager
        curriculum_cfg = CurriculumConfig.from_curriculum_cfg(cfg.curriculum)
        curriculum_manager: CurriculumManager | None = None
        current_train_overrides: dict[str, Any] | None = None
        if curriculum_cfg.enabled and curriculum_cfg.stages:
            curriculum_manager = CurriculumManager(curriculum_cfg)
            if resume_enabled:
                if not resume_training_state_path.exists():
                    raise FileNotFoundError(
                        f"Resume enabled but training state is missing: {resume_training_state_path}"
                    )
                resume_state = _load_training_state(resume_training_state_path)
                curriculum_state = resume_state.get("curriculum", {})
                if isinstance(curriculum_state, dict):
                    curriculum_manager._stage_idx = int(  # noqa: SLF001
                        curriculum_state.get("stage_index", curriculum_manager.stage_index)
                    )
                    curriculum_manager._stage_steps_done = int(  # noqa: SLF001
                        curriculum_state.get("stage_steps_done", curriculum_manager.stage_steps_done)
                    )
                    curriculum_manager._eval_count_at_stage = int(  # noqa: SLF001
                        curriculum_state.get("eval_count_at_stage", curriculum_manager.eval_count_at_stage)
                    )
                    curriculum_manager._consecutive_passes = int(  # noqa: SLF001
                        curriculum_state.get("consecutive_passes", curriculum_manager.consecutive_passes)
                    )
                    curriculum_manager._last_eval_passed = bool(  # noqa: SLF001
                        curriculum_state.get("last_eval_passed", False)
                    )
            current_train_overrides = curriculum_manager.get_env_config(evaluation=False)
        elif resume_enabled:
            if not resume_training_state_path.exists():
                raise FileNotFoundError(
                    f"Resume enabled but training state is missing: {resume_training_state_path}"
                )
            resume_state = _load_training_state(resume_training_state_path)

        # Environment
        env = build_env(cfg, current_train_overrides)
        seed_env_spaces(env, run_seed)
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Agent
        preprocessor = build_preprocessor(cfg)
        adapter = build_adapter(
            cfg,
            adapter_space_kwargs(env.action_space),
        )
        if resume_enabled:
            if not resume_checkpoint_zip.exists():
                raise FileNotFoundError(
                    f"Resume enabled but checkpoint is missing: {resume_checkpoint_zip}"
                )
            planner = load_planner(cfg, checkpoint_path=str(resume_checkpoint_zip), env=env)
        else:
            planner = build_planner(cfg, env, seed=run_seed)
        # Read EMA alpha from planner config if available
        ema_alpha_cfg = float(cfg.planner.get("monitor_ema_alpha", 0.1)) if hasattr(cfg, "planner") else 0.1
        agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter, ema_alpha=ema_alpha_cfg)
        if resume_enabled:
            agent.load_adapter(checkpoint_path=resume_checkpoint_zip, strict=True)
            _load_replay_buffer_if_available(planner, resume_replay_buffer_path)
            if bool(resume_cfg.get("restore_rng_state", True)):
                _load_rng_state(resume_rng_state_path)
            if resume_state is None and resume_training_state_path.exists():
                resume_state = _load_training_state(resume_training_state_path)
            if resume_state is not None:
                resume_global_steps_done = int(resume_state.get("global_steps_done", 0))
                resume_chunk_id = int(resume_state.get("chunk_id", 0))
                resume_eval_id = int(resume_state.get("eval_id", 0))

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
        log_event(
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
        if resume_enabled:
            train_logger.info(
                "Resume enabled | checkpoint=%s | state=%s | replay_buffer=%s | rng_state=%s | resumed_global_steps=%d | resumed_chunk_id=%d | resumed_eval_id=%d",
                str(resume_checkpoint_zip),
                str(resume_training_state_path),
                str(resume_replay_buffer_path),
                str(resume_rng_state_path),
                int(resume_global_steps_done),
                int(resume_chunk_id),
                int(resume_eval_id),
            )
            log_event(
                events_log_path,
                "run_resumed",
                checkpoint=str(resume_checkpoint_zip),
                state=str(resume_training_state_path),
                resumed_global_steps=int(resume_global_steps_done),
                resumed_chunk_id=int(resume_chunk_id),
                resumed_eval_id=int(resume_eval_id),
            )

        def save_intermediate_checkpoints(
            *,
            current_global_step: int,
            chunk_id: int,
            eval_id: int,
            current_stage_name: str,
            current_stage_index: int,
            metrics: dict[str, Any],
        ) -> None:
            """Save all chunk-level checkpoints shared by curriculum and baseline runs."""
            nonlocal best_lex_key
            nonlocal best_lex_payload
            nonlocal best_rulebook_strict_key
            nonlocal best_rulebook_strict_payload
            nonlocal best_rulebook_thresholded_key
            nonlocal best_rulebook_thresholded_payload

            save_best_lex = bool(cfg.checkpoint.get("save_best_lexicographic", True))
            candidate_key = _lexicographic_eval_key(metrics)
            if save_best_lex and (best_lex_key is None or candidate_key < best_lex_key):
                agent.save(best_lex_checkpoint_stem)
                best_lex_key = candidate_key
                best_lex_payload = {
                    "path": _checkpoint_rel(run_dir, best_lex_checkpoint_stem),
                    "global_step": int(current_global_step),
                    "chunk_id": int(chunk_id),
                    "eval_id": int(eval_id),
                    "stage": str(current_stage_name),
                    "stage_index": int(current_stage_index),
                    "success_rate": float(metrics.get("success_rate", 0.0)),
                    "collision_rate": float(metrics.get("collision_rate", 0.0)),
                    "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
                    "top_rule_violation_rate": float(metrics.get("top_rule_violation_rate", 0.0)),
                    "route_completion": float(metrics.get("route_completion", 0.0)),
                    "mean_reward": float(metrics.get("mean_reward", 0.0)),
                    "avg_error_value": float(metrics.get("avg_error_value", 0.0)),
                    "max_error_value": float(metrics.get("max_error_value", 0.0)),
                }
                _write_best_checkpoints_yaml(
                    best_checkpoints_yaml_path,
                    {"best_lexicographic": best_lex_payload},
                )
                _append_checkpoint_index_row(
                    checkpoint_index_path,
                    {
                        "checkpoint_path": _checkpoint_rel(run_dir, best_lex_checkpoint_stem),
                        "type": "best_lexicographic",
                        "global_step": int(current_global_step),
                        "chunk_id": int(chunk_id),
                        "eval_id": int(eval_id),
                        "stage": str(current_stage_name),
                        "stage_index": int(current_stage_index),
                        "success_rate": float(metrics.get("success_rate", 0.0)),
                        "collision_rate": float(metrics.get("collision_rate", 0.0)),
                        "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
                        "top_rule_violation_rate": float(metrics.get("top_rule_violation_rate", 0.0)),
                        "route_completion": float(metrics.get("route_completion", 0.0)),
                        "mean_reward": float(metrics.get("mean_reward", 0.0)),
                        "avg_error_value": float(metrics.get("avg_error_value", 0.0)),
                        "max_error_value": float(metrics.get("max_error_value", 0.0)),
                        "reason": "improved_lexicographic",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                )

            save_best_rulebook_strict = bool(
                cfg.checkpoint.get("save_best_lexicographic_rulebook", True)
            )
            strict_key = _rulebook_strict_key(metrics)
            if save_best_rulebook_strict and (
                best_rulebook_strict_key is None or strict_key < best_rulebook_strict_key
            ):
                agent.save(best_rulebook_strict_checkpoint_stem)
                best_rulebook_strict_key = strict_key
                best_rulebook_strict_payload = {
                    "path": _checkpoint_rel(run_dir, best_rulebook_strict_checkpoint_stem),
                    "global_step": int(current_global_step),
                    "chunk_id": int(chunk_id),
                    "eval_id": int(eval_id),
                    "stage": str(current_stage_name),
                    "stage_index": int(current_stage_index),
                    "rulebook_mode": "strict",
                }
                _append_checkpoint_index_row(
                    checkpoint_index_path,
                    {
                        "checkpoint_path": _checkpoint_rel(run_dir, best_rulebook_strict_checkpoint_stem),
                        "type": "best_lexicographic_rulebook",
                        "global_step": int(current_global_step),
                        "chunk_id": int(chunk_id),
                        "eval_id": int(eval_id),
                        "stage": str(current_stage_name),
                        "stage_index": int(current_stage_index),
                        "success_rate": float(metrics.get("success_rate", 0.0)),
                        "collision_rate": float(metrics.get("collision_rate", 0.0)),
                        "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
                        "top_rule_violation_rate": float(metrics.get("top_rule_violation_rate", 0.0)),
                        "route_completion": float(metrics.get("route_completion", 0.0)),
                        "mean_reward": float(metrics.get("mean_reward", 0.0)),
                        "avg_error_value": float(metrics.get("avg_error_value", 0.0)),
                        "max_error_value": float(metrics.get("max_error_value", 0.0)),
                        "reason": "improved_rulebook_strict",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                )

            save_best_rulebook_thresholded = bool(
                cfg.checkpoint.get("save_best_thresholded_lexicographic_rulebook", True)
            )
            thresholded_key = _rulebook_thresholded_key(metrics)
            if save_best_rulebook_thresholded and (
                best_rulebook_thresholded_key is None or thresholded_key < best_rulebook_thresholded_key
            ):
                agent.save(best_rulebook_thresholded_checkpoint_stem)
                best_rulebook_thresholded_key = thresholded_key
                best_rulebook_thresholded_payload = {
                    "path": _checkpoint_rel(run_dir, best_rulebook_thresholded_checkpoint_stem),
                    "global_step": int(current_global_step),
                    "chunk_id": int(chunk_id),
                    "eval_id": int(eval_id),
                    "stage": str(current_stage_name),
                    "stage_index": int(current_stage_index),
                    "rulebook_mode": "thresholded",
                }
                _append_checkpoint_index_row(
                    checkpoint_index_path,
                    {
                        "checkpoint_path": _checkpoint_rel(run_dir, best_rulebook_thresholded_checkpoint_stem),
                        "type": "best_thresholded_lexicographic_rulebook",
                        "global_step": int(current_global_step),
                        "chunk_id": int(chunk_id),
                        "eval_id": int(eval_id),
                        "stage": str(current_stage_name),
                        "stage_index": int(current_stage_index),
                        "success_rate": float(metrics.get("success_rate", 0.0)),
                        "collision_rate": float(metrics.get("collision_rate", 0.0)),
                        "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
                        "top_rule_violation_rate": float(metrics.get("top_rule_violation_rate", 0.0)),
                        "route_completion": float(metrics.get("route_completion", 0.0)),
                        "mean_reward": float(metrics.get("mean_reward", 0.0)),
                        "avg_error_value": float(metrics.get("avg_error_value", 0.0)),
                        "max_error_value": float(metrics.get("max_error_value", 0.0)),
                        "reason": "improved_rulebook_thresholded",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                )

            if bool(cfg.checkpoint.get("save_latest_each_chunk", True)):
                agent.save(latest_checkpoint_stem)
                _save_replay_buffer_if_available(planner, latest_replay_buffer_path)
                latest_state_payload = {
                    "global_steps_done": int(current_global_step),
                    "chunk_id": int(chunk_id),
                    "eval_id": int(eval_id),
                    "remaining_steps": int(total_timesteps - current_global_step),
                    "curriculum": {
                        "enabled": bool(curriculum_manager is not None),
                        "stage_index": int(curriculum_manager.stage_index) if curriculum_manager is not None else 0,
                        "stage_steps_done": int(curriculum_manager.stage_steps_done) if curriculum_manager is not None else 0,
                        "eval_count_at_stage": int(curriculum_manager.eval_count_at_stage) if curriculum_manager is not None else 0,
                        "consecutive_passes": int(curriculum_manager.consecutive_passes) if curriculum_manager is not None else 0,
                        "last_eval_passed": bool(curriculum_manager._last_eval_passed) if curriculum_manager is not None else False,  # noqa: SLF001
                    },
                    "seed": int(run_seed),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
                _save_training_state(latest_training_state_path, latest_state_payload)
                if bool(cfg.checkpoint.get("save_rng_state", True)):
                    _save_rng_state(latest_rng_state_path)
                _append_checkpoint_index_row(
                    checkpoint_index_path,
                    {
                        "checkpoint_path": _checkpoint_rel(run_dir, latest_checkpoint_stem),
                        "type": "latest",
                        "global_step": int(current_global_step),
                        "chunk_id": int(chunk_id),
                        "eval_id": int(eval_id),
                        "stage": str(current_stage_name),
                        "stage_index": int(current_stage_index),
                        "success_rate": float(metrics.get("success_rate", 0.0)),
                        "collision_rate": float(metrics.get("collision_rate", 0.0)),
                        "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
                        "top_rule_violation_rate": float(metrics.get("top_rule_violation_rate", 0.0)),
                        "route_completion": float(metrics.get("route_completion", 0.0)),
                        "mean_reward": float(metrics.get("mean_reward", 0.0)),
                        "avg_error_value": float(metrics.get("avg_error_value", 0.0)),
                        "max_error_value": float(metrics.get("max_error_value", 0.0)),
                        "reason": "chunk_end_latest",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                )

            if bool(cfg.checkpoint.get("save_periodic", True)):
                periodic_interval = int(cfg.checkpoint.get("periodic_interval_steps", 0))
                if periodic_interval > 0 and (current_global_step % periodic_interval == 0):
                    periodic_stem = checkpoints_periodic_dir / f"step_{current_global_step:08d}"
                    agent.save(periodic_stem)
                    _append_checkpoint_index_row(
                        checkpoint_index_path,
                        {
                            "checkpoint_path": _checkpoint_rel(run_dir, periodic_stem),
                            "type": "periodic",
                            "global_step": int(current_global_step),
                            "chunk_id": int(chunk_id),
                            "eval_id": int(eval_id),
                            "stage": str(current_stage_name),
                            "stage_index": int(current_stage_index),
                            "success_rate": float(metrics.get("success_rate", 0.0)),
                            "collision_rate": float(metrics.get("collision_rate", 0.0)),
                            "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
                            "top_rule_violation_rate": float(metrics.get("top_rule_violation_rate", 0.0)),
                            "route_completion": float(metrics.get("route_completion", 0.0)),
                            "mean_reward": float(metrics.get("mean_reward", 0.0)),
                            "avg_error_value": float(metrics.get("avg_error_value", 0.0)),
                            "max_error_value": float(metrics.get("max_error_value", 0.0)),
                            "reason": "periodic_interval",
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        },
                    )
                    _prune_old_periodic_checkpoints(
                        checkpoints_periodic_dir,
                        keep_last=int(cfg.checkpoint.get("keep_last_periodic", 4)),
                    )

        ##################
        ###### LOOP ######
        ##################

        # Training loop with periodic evaluation and optional curriculum progression
        remaining = max(0, total_timesteps - resume_global_steps_done)
        chunk_id = int(resume_chunk_id)
        eval_id = int(resume_eval_id)
        while remaining > 0:
            chunk_id += 1

            # Stage info and chunk size 
            chunk_steps = min(eval_interval, remaining)
            current_stage_name = "baseline"
            current_stage_index = 0
            if curriculum_manager is not None:
                current_stage_name = curriculum_manager.get_current_stage().name
                current_stage_index = int(curriculum_manager.stage_index)
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
            log_event(
                events_log_path,
                "chunk_started",
                chunk_id=chunk_id,
                stage=current_stage_name,
                steps_start=steps_start,
                steps_end=steps_end,
            )

            ###### TRAINING ######

            def train_reset_seed_for_episode(episode_index: int) -> int:
                return train_episode_seed_from_env_overrides(
                    current_train_overrides,
                    cfg,
                    run_seed=run_seed,
                    chunk_id=chunk_id,
                    episode_index=episode_index,
                    stage_index=current_stage_index,
                )

            # Agent training
            chunk_summary = agent.train(
                env=env,
                chunk_timesteps=chunk_steps,
                global_total_timesteps=total_timesteps,
                global_steps_done=total_timesteps - remaining,
                deterministic=False,
                log_interval=log_interval,
                reset_seed_fn=train_reset_seed_for_episode,
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
            log_event(
                events_log_path,
                "chunk_finished",
                chunk_id=chunk_id,
                stage=current_stage_name,
                global_step=current_global_step,
                summary=chunk_summary,
            )
            recorder.append_row(
                "train_chunks.csv",
                {
                    **base_csv_fields,
                    "chunk_id": chunk_id,
                    "stage": current_stage_name,
                    "stage_index": current_stage_index,
                    "steps_start": steps_start,
                    "steps_end": steps_end,
                    "global_step": current_global_step,
                    "chunk_steps": chunk_steps,
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
                    "actor_loss_ema": float(chunk_summary.get("actor_loss_ema", 0.0)),
                    "critic_loss_ema": float(chunk_summary.get("critic_loss_ema", 0.0)),
                    "critic_loss": float(chunk_summary.get("critic_loss", 0.0)),
                    "learning_rate": float(chunk_summary.get("learning_rate", 0.0)),
                    "n_updates": int(chunk_summary.get("n_updates", 0)),
                    "fps": float(chunk_summary.get("fps", 0.0)),
                    "elapsed_seconds": float(chunk_summary.get("elapsed_seconds", 0.0)),
                    "train_reset_seed_first": chunk_summary.get("train_reset_seed_first"),
                    "train_reset_seed_last": chunk_summary.get("train_reset_seed_last"),
                    "train_reset_seed_unique_count": chunk_summary.get("train_reset_seed_unique_count"),
                },
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
            eval_episode_count = int(cfg.experiment.eval_episodes)
            eval_env_overrides = apply_eval_scenario_seed_split(
                base_run_seed=run_seed,
                eval_env_overrides=eval_env_overrides,
                cfg=cfg,
                n_eval_episodes=eval_episode_count,
                split="validation",
            )
            eval_base_seed = eval_base_seed_from_env_overrides(eval_env_overrides, cfg)
            
            # Environment
            eval_env = build_env(cfg, eval_env_overrides)
            seed_env_spaces(eval_env, run_seed + 100_000 + (total_timesteps - remaining))
            planner.set_env(eval_env)   # Update planner's env reference

            # Evaluation
            eval_id += 1
            eval_logger.info(
                "Evaluation started | eval_id=%d | stage=%s | step=%d | episodes=%d",
                eval_id,
                current_stage_name,
                current_global_step,
                eval_episode_count,
            )
            log_event(
                events_log_path,
                "evaluation_started",
                eval_id=eval_id,
                stage=current_stage_name,
                global_step=current_global_step,
                episodes=eval_episode_count,
                start_seed=eval_base_seed,
            )
            metrics = agent.evaluate(
                env=eval_env,
                n_eval_episodes=eval_episode_count,
                deterministic=bool(cfg.experiment.eval_deterministic),
                base_seed=eval_base_seed,
                return_episode_metrics=True,
                error_priority_base=float(cfg.reward.get("a", 2.01)),
            )
            eval_env.close()
            print_evaluation_summary(
                title=f"Evaluation {eval_id}",
                metrics=metrics,
                stage=current_stage_name,
                global_step=current_global_step,
                episodes=eval_episode_count,
                base_seed=eval_base_seed,
                details_path=eval_log_path,
            )
            eval_logger.info(
                "Evaluation finished | eval_id=%d | stage=%s | step=%d | metrics=%s",
                eval_id,
                current_stage_name,
                current_global_step,
                metrics,
            )
            log_event(
                events_log_path,
                "evaluation_finished",
                eval_id=eval_id,
                stage=current_stage_name,
                global_step=current_global_step,
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
            episode_count = len(episode_returns)
            for episode_idx in range(episode_count):
                scenario_seed = int(eval_base_seed + episode_idx)
                recorder.append_row(
                    "eval_episodes.csv",
                    {
                        **base_csv_fields,
                        "eval_id": eval_id,
                        "episode_id": episode_idx + 1,
                        "stage": current_stage_name,
                        "stage_index": current_stage_index,
                        "global_step": current_global_step,
                        "scenario_seed": scenario_seed,
                        "scenario_id": f"seed_{scenario_seed}",
                        "deterministic": bool(cfg.experiment.eval_deterministic),
                        "reward": float(episode_returns[episode_idx]),
                        "env_reward": float(episode_env_returns[episode_idx]) if episode_idx < len(episode_env_returns) else None,
                        "scalar_rule_reward": float(episode_scalar_rule_returns[episode_idx]) if episode_idx < len(episode_scalar_rule_returns) and episode_scalar_rule_returns[episode_idx] is not None else None,
                        "hybrid_reward": float(episode_hybrid_returns[episode_idx]) if episode_idx < len(episode_hybrid_returns) and episode_hybrid_returns[episode_idx] is not None else None,
                        "rule_rewards_by_rule": json.dumps(episode_rule_rewards_by_rule[episode_idx], ensure_ascii=True) if episode_idx < len(episode_rule_rewards_by_rule) else None,
                        "episode_length": int(episode_lengths[episode_idx]) if episode_idx < len(episode_lengths) else None,
                        "success": float(episode_success[episode_idx]) if episode_idx < len(episode_success) else None,
                        "collision": float(episode_collision[episode_idx]) if episode_idx < len(episode_collision) else None,
                        "out_of_road": float(episode_out_of_road[episode_idx]) if episode_idx < len(episode_out_of_road) else None,
                        "timeout": float(episode_timeout[episode_idx]) if episode_idx < len(episode_timeout) else None,
                        "route_completion": float(episode_route_completion[episode_idx]) if episode_idx < len(episode_route_completion) else None,
                        "top_rule_violation_rate": float(episode_top_rule_violation_rate[episode_idx]) if episode_idx < len(episode_top_rule_violation_rate) else None,
                        "error_value": float(episode_error_value[episode_idx]) if episode_idx < len(episode_error_value) else None,
                        "violated_rules": str(episode_violated_rules[episode_idx]) if episode_idx < len(episode_violated_rules) else None,
                        "violation_pattern": str(episode_violation_pattern[episode_idx]) if episode_idx < len(episode_violation_pattern) else None,
                        "video_path": None,
                    },
                )

            ###### CURRICULUM PROGRESSION ######

            # If no curriculum, just rebuild the env
            if curriculum_manager is None:
                recorder.append_row(
                    "evals.csv",
                    {
                        **base_csv_fields,
                        "eval_id": eval_id,
                        "chunk_id": chunk_id,
                        "stage": current_stage_name,
                        "stage_index": current_stage_index,
                        "global_step": current_global_step,
                        "eval_episodes": int(cfg.experiment.eval_episodes),
                        "deterministic": bool(cfg.experiment.eval_deterministic),
                        "mean_reward": float(metrics.get("mean_reward", 0.0)),
                        "std_reward": float(metrics.get("std_reward", 0.0)),
                        "mean_env_reward": float(metrics.get("mean_env_reward", 0.0)),
                        "std_env_reward": float(metrics.get("std_env_reward", 0.0)),
                        "mean_scalar_rule_reward": float(metrics.get("mean_scalar_rule_reward", 0.0)) if metrics.get("mean_scalar_rule_reward") is not None else None,
                        "std_scalar_rule_reward": float(metrics.get("std_scalar_rule_reward", 0.0)) if metrics.get("std_scalar_rule_reward") is not None else None,
                        "mean_hybrid_reward": float(metrics.get("mean_hybrid_reward", 0.0)) if metrics.get("mean_hybrid_reward") is not None else None,
                        "std_hybrid_reward": float(metrics.get("std_hybrid_reward", 0.0)) if metrics.get("std_hybrid_reward") is not None else None,
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
                        "next_stage": current_stage_name,
                    },
                )
                _append_rule_metrics_rows(
                    recorder,
                    base_fields=base_csv_fields,
                    eval_id=eval_id,
                    chunk_id=chunk_id,
                    stage=current_stage_name,
                    stage_index=current_stage_index,
                    global_step=current_global_step,
                    metrics=metrics,
                )
                save_intermediate_checkpoints(
                    current_global_step=current_global_step,
                    chunk_id=chunk_id,
                    eval_id=eval_id,
                    current_stage_name=current_stage_name,
                    current_stage_index=current_stage_index,
                    metrics=metrics,
                )
                env = build_env(cfg, current_train_overrides)
                seed_env_spaces(env, run_seed + 300_000 + (total_timesteps - remaining))
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
                    log_event(
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
            passed_eval_gates = curriculum_manager.record_eval_metrics(metrics)
            stage_gates = curriculum_cfg.promotion.gates
            gate_success_pass = float(metrics.get("success_rate", float("-inf"))) >= float(stage_gates.task.success_rate_min)
            gate_collision_pass = float(metrics.get("collision_rate", float("inf"))) <= float(stage_gates.safety.collision_rate_max)
            gate_out_of_road_pass = float(metrics.get("out_of_road_rate", float("inf"))) <= float(stage_gates.safety.out_of_road_rate_max)
            gate_top_rule_pass = float(metrics.get("top_rule_violation_rate", float("inf"))) <= float(stage_gates.safety.top_rule_violation_rate_max)
            gate_route_completion_pass = float(metrics.get("route_completion", float("-inf"))) >= float(stage_gates.task.route_completion_min)

            # Check for promotion and update env config if promoted
            next_stage_name = current_stage_name
            pre_promotion_stage_steps_done = int(curriculum_manager.stage_steps_done)
            pre_promotion_consecutive_passes = int(curriculum_manager.consecutive_passes)
            pre_promotion_min_stage_steps = _min_stage_steps(curriculum_cfg, current_stage_name)

            if curriculum_manager.promote():
                previous_stage = current_stage_name
                next_stage = curriculum_manager.get_current_stage().name
                next_stage_name = next_stage
                print(f"Curriculum promoted: {previous_stage} -> {next_stage}")
                current_train_overrides = curriculum_manager.get_env_config(evaluation=False)
                curriculum_logger.info(
                    "PROMOTION | from=%s | to=%s | step=%d | eval_id=%d",
                    previous_stage,
                    next_stage,
                    current_global_step,
                    eval_id,
                )
                log_event(
                    events_log_path,
                    "promotion",
                    from_stage=previous_stage,
                    to_stage=next_stage,
                    global_step=current_global_step,
                    eval_id=eval_id,
                )
                recorder.append_row(
                    "promotions.csv",
                    {
                        **base_csv_fields,
                        "event_type": "promoted",
                        "from_stage": previous_stage,
                        "to_stage": next_stage,
                        "from_stage_index": current_stage_index,
                        "to_stage_index": int(curriculum_manager.stage_index),
                        "eval_id": eval_id,
                        "chunk_id": chunk_id,
                        "global_step": current_global_step,
                        "stage_steps_done": pre_promotion_stage_steps_done,
                        "stage_steps_min_required": pre_promotion_min_stage_steps,
                        "passed_eval_gates": bool(passed_eval_gates),
                        "consecutive_passes": pre_promotion_consecutive_passes,
                        "success_rate": float(metrics.get("success_rate", 0.0)),
                        "collision_rate": float(metrics.get("collision_rate", 0.0)),
                        "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
                        "top_rule_violation_rate": float(metrics.get("top_rule_violation_rate", 0.0)),
                        "route_completion": float(metrics.get("route_completion", 0.0)),
                        "reason": "promotion_gates_satisfied",
                    },
                )
            else:
                curriculum_logger.info(
                    "Gate check | stage=%s | eval_id=%d | promoted=false",
                    current_stage_name,
                    eval_id,
                )
                log_event(
                    events_log_path,
                    "gate_check",
                    eval_id=eval_id,
                    stage=current_stage_name,
                    global_step=current_global_step,
                    promoted=False,
                )

            recorder.append_row(
                "evals.csv",
                {
                    **base_csv_fields,
                    "eval_id": eval_id,
                    "chunk_id": chunk_id,
                    "stage": current_stage_name,
                    "stage_index": current_stage_index,
                    "global_step": current_global_step,
                    "eval_episodes": int(cfg.experiment.eval_episodes),
                    "deterministic": bool(cfg.experiment.eval_deterministic),
                    "mean_reward": float(metrics.get("mean_reward", 0.0)),
                    "std_reward": float(metrics.get("std_reward", 0.0)),
                    "mean_env_reward": float(metrics.get("mean_env_reward", 0.0)),
                    "std_env_reward": float(metrics.get("std_env_reward", 0.0)),
                    "mean_scalar_rule_reward": float(metrics.get("mean_scalar_rule_reward", 0.0)) if metrics.get("mean_scalar_rule_reward") is not None else None,
                    "std_scalar_rule_reward": float(metrics.get("std_scalar_rule_reward", 0.0)) if metrics.get("std_scalar_rule_reward") is not None else None,
                    "mean_hybrid_reward": float(metrics.get("mean_hybrid_reward", 0.0)) if metrics.get("mean_hybrid_reward") is not None else None,
                    "std_hybrid_reward": float(metrics.get("std_hybrid_reward", 0.0)) if metrics.get("std_hybrid_reward") is not None else None,
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
                    "success_rate_min": float(stage_gates.task.success_rate_min),
                    "collision_rate_max": float(stage_gates.safety.collision_rate_max),
                    "out_of_road_rate_max": float(stage_gates.safety.out_of_road_rate_max),
                    "top_rule_violation_rate_max": float(stage_gates.safety.top_rule_violation_rate_max),
                    "route_completion_min": float(stage_gates.task.route_completion_min),
                    "gate_success_pass": bool(gate_success_pass),
                    "gate_collision_pass": bool(gate_collision_pass),
                    "gate_out_of_road_pass": bool(gate_out_of_road_pass),
                    "gate_top_rule_pass": bool(gate_top_rule_pass),
                    "gate_route_completion_pass": bool(gate_route_completion_pass),
                    "passed_eval_gates": bool(passed_eval_gates),
                    "consecutive_passes": pre_promotion_consecutive_passes,
                    "warmup_evals_required": int(curriculum_cfg.promotion.warmup_evals),
                    "consecutive_evals_required": int(curriculum_cfg.promotion.consecutive_evals),
                    "promoted": bool(next_stage_name != current_stage_name),
                    "next_stage": next_stage_name,
                },
            )
            _append_rule_metrics_rows(
                recorder,
                base_fields=base_csv_fields,
                eval_id=eval_id,
                chunk_id=chunk_id,
                stage=current_stage_name,
                stage_index=current_stage_index,
                global_step=current_global_step,
                metrics=metrics,
            )

            save_intermediate_checkpoints(
                current_global_step=current_global_step,
                chunk_id=chunk_id,
                eval_id=eval_id,
                current_stage_name=current_stage_name,
                current_stage_index=current_stage_index,
                metrics=metrics,
            )

            # Rebuild env with new config (if changed) and update planner's env reference
            env = build_env(cfg, current_train_overrides)
            seed_env_spaces(env, run_seed + 400_000 + (total_timesteps - remaining))
            planner.set_env(env)    # Update planner's env reference

        ##########################
        ###### FINALIZATION ######
        ##########################

        # Save final checkpoint used as official final evaluation checkpoint.
        if not bool(cfg.checkpoint.get("save_final", True)):
            raise ValueError("checkpoint.save_final must be true: final evaluation requires final checkpoint.")
        agent.save(final_checkpoint_stem)
        final_adapter_ckpt_path = agent.adapter_checkpoint_path(final_checkpoint_stem)
        _append_checkpoint_index_row(
            checkpoint_index_path,
            {
                "checkpoint_path": _checkpoint_rel(run_dir, final_checkpoint_stem),
                "type": "final",
                "global_step": int(total_timesteps),
                "chunk_id": int(chunk_id),
                "eval_id": int(eval_id),
                "stage": (
                    curriculum_manager.get_current_stage().name
                    if curriculum_manager is not None
                    else "baseline"
                ),
                "stage_index": int(curriculum_manager.stage_index) if curriculum_manager is not None else 0,
                "reason": "training_completed",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )
        final_entry = {
            "path": _checkpoint_rel(run_dir, final_checkpoint_stem),
            "global_step": int(total_timesteps),
        }
        best_payload_out: dict[str, Any] = {"final": final_entry}
        if best_lex_payload:
            best_payload_out["best_lexicographic"] = best_lex_payload
        if best_rulebook_strict_payload:
            best_payload_out["best_lexicographic_rulebook"] = best_rulebook_strict_payload
        if best_rulebook_thresholded_payload:
            best_payload_out["best_thresholded_lexicographic_rulebook"] = (
                best_rulebook_thresholded_payload
            )
        _write_best_checkpoints_yaml(best_checkpoints_yaml_path, best_payload_out)

        ###### FINAL EVAL ######

        # Environment
        env.close()
        final_eval_env_overrides = None
        if curriculum_manager is not None:
            final_eval_env_overrides = curriculum_manager.get_env_config(evaluation=True)
        final_eval_env_overrides = apply_eval_scenario_seed_split(
            base_run_seed=run_seed,
            eval_env_overrides=final_eval_env_overrides,
            cfg=cfg,
            n_eval_episodes=int(cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)),
            split="test",
        )
        final_eval_base_seed = eval_base_seed_from_env_overrides(final_eval_env_overrides, cfg)
        eval_env = build_env(cfg, final_eval_env_overrides)
        seed_env_spaces(eval_env, run_seed + 500_000)

        # Evaluation
        metrics = agent.evaluate(
            env=eval_env,
            n_eval_episodes=int(cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)),
            deterministic=bool(cfg.experiment.eval_deterministic),
            base_seed=final_eval_base_seed,
            return_episode_metrics=True,
            error_priority_base=float(cfg.reward.get("a", 2.01)),
        )
        final_stage_name = (
            curriculum_manager.get_current_stage().name
            if curriculum_manager is not None
            else "baseline"
        )
        final_stage_index = int(curriculum_manager.stage_index) if curriculum_manager is not None else 0
        final_eval_id = eval_id + 1
        final_checkpoint_zip = f"{final_checkpoint_stem}.zip"
        print_evaluation_summary(
            title="Final Evaluation",
            metrics=metrics,
            stage=final_stage_name,
            global_step=total_timesteps,
            episodes=int(cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)),
            base_seed=final_eval_base_seed,
            details_path=eval_log_path,
            checkpoint_path=final_checkpoint_zip,
        )
        train_logger.info("Checkpoint saved | path=%s.zip", final_checkpoint_stem)
        log_event(
            events_log_path,
            "checkpoint_saved",
            path=f"{final_checkpoint_stem}.zip",
            global_step=total_timesteps,
            checkpoint_type="final",
        )
        if bool(getattr(adapter, "requires_training", False)):
            print(f"Adapter checkpoint saved at: {final_adapter_ckpt_path}")
            train_logger.info("Adapter checkpoint saved | path=%s", final_adapter_ckpt_path)
            log_event(
                events_log_path,
                "checkpoint_saved",
                path=str(final_adapter_ckpt_path),
                global_step=total_timesteps,
                checkpoint_kind="adapter",
            )
        eval_logger.info(
            "Final evaluation finished | step=%d | metrics=%s",
            total_timesteps,
            metrics,
        )
        log_event(
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
        recorder.append_row(
            "evals.csv",
            {
                **base_csv_fields,
                "eval_id": final_eval_id,
                "chunk_id": chunk_id,
                "stage": final_stage_name,
                "stage_index": final_stage_index,
                "global_step": total_timesteps,
                "eval_episodes": int(cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)),
                "deterministic": bool(cfg.experiment.eval_deterministic),
                "mean_reward": float(metrics.get("mean_reward", 0.0)),
                "std_reward": float(metrics.get("std_reward", 0.0)),
                "mean_env_reward": float(metrics.get("mean_env_reward", 0.0)),
                "std_env_reward": float(metrics.get("std_env_reward", 0.0)),
                "mean_scalar_rule_reward": float(metrics.get("mean_scalar_rule_reward", 0.0)) if metrics.get("mean_scalar_rule_reward") is not None else None,
                "std_scalar_rule_reward": float(metrics.get("std_scalar_rule_reward", 0.0)) if metrics.get("std_scalar_rule_reward") is not None else None,
                "mean_hybrid_reward": float(metrics.get("mean_hybrid_reward", 0.0)) if metrics.get("mean_hybrid_reward") is not None else None,
                "std_hybrid_reward": float(metrics.get("std_hybrid_reward", 0.0)) if metrics.get("std_hybrid_reward") is not None else None,
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
                "next_stage": final_stage_name,
            },
        )
        _append_rule_metrics_rows(
            recorder,
            base_fields=base_csv_fields,
            eval_id=final_eval_id,
            chunk_id=chunk_id,
            stage=final_stage_name,
            stage_index=final_stage_index,
            global_step=total_timesteps,
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
        for episode_idx in range(len(episode_returns)):
            scenario_seed = int(final_eval_base_seed + episode_idx)
            recorder.append_row(
                "eval_episodes.csv",
                {
                    **base_csv_fields,
                    "eval_id": final_eval_id,
                    "episode_id": episode_idx + 1,
                    "stage": final_stage_name,
                    "stage_index": final_stage_index,
                    "global_step": total_timesteps,
                    "scenario_seed": scenario_seed,
                    "scenario_id": f"seed_{scenario_seed}",
                    "deterministic": bool(cfg.experiment.eval_deterministic),
                    "reward": float(episode_returns[episode_idx]),
                    "env_reward": float(episode_env_returns[episode_idx]) if episode_idx < len(episode_env_returns) else None,
                    "scalar_rule_reward": float(episode_scalar_rule_returns[episode_idx]) if episode_idx < len(episode_scalar_rule_returns) and episode_scalar_rule_returns[episode_idx] is not None else None,
                    "hybrid_reward": float(episode_hybrid_returns[episode_idx]) if episode_idx < len(episode_hybrid_returns) and episode_hybrid_returns[episode_idx] is not None else None,
                    "rule_rewards_by_rule": json.dumps(episode_rule_rewards_by_rule[episode_idx], ensure_ascii=True) if episode_idx < len(episode_rule_rewards_by_rule) else None,
                    "episode_length": int(episode_lengths[episode_idx]) if episode_idx < len(episode_lengths) else None,
                    "success": float(episode_success[episode_idx]) if episode_idx < len(episode_success) else None,
                    "collision": float(episode_collision[episode_idx]) if episode_idx < len(episode_collision) else None,
                    "out_of_road": float(episode_out_of_road[episode_idx]) if episode_idx < len(episode_out_of_road) else None,
                    "timeout": float(episode_timeout[episode_idx]) if episode_idx < len(episode_timeout) else None,
                    "route_completion": float(episode_route_completion[episode_idx]) if episode_idx < len(episode_route_completion) else None,
                    "top_rule_violation_rate": float(episode_top_rule_violation_rate[episode_idx]) if episode_idx < len(episode_top_rule_violation_rate) else None,
                    "error_value": float(episode_error_value[episode_idx]) if episode_idx < len(episode_error_value) else None,
                    "violated_rules": str(episode_violated_rules[episode_idx]) if episode_idx < len(episode_violated_rules) else None,
                    "violation_pattern": str(episode_violation_pattern[episode_idx]) if episode_idx < len(episode_violation_pattern) else None,
                    "video_path": None,
                },
            )
        recorder.append_row(
            "final_eval.csv",
            {
                **base_csv_fields,
                "total_timesteps": total_timesteps,
                "final_stage": final_stage_name,
                "final_stage_index": final_stage_index,
                "final_stage_reached": bool(curriculum_manager.is_finished()) if curriculum_manager is not None else True,
                "final_eval_episodes": int(cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)),
                "deterministic": bool(cfg.experiment.eval_deterministic),
                "mean_reward": float(metrics.get("mean_reward", 0.0)),
                "std_reward": float(metrics.get("std_reward", 0.0)),
                "mean_env_reward": float(metrics.get("mean_env_reward", 0.0)),
                "std_env_reward": float(metrics.get("std_env_reward", 0.0)),
                "mean_scalar_rule_reward": float(metrics.get("mean_scalar_rule_reward", 0.0)) if metrics.get("mean_scalar_rule_reward") is not None else None,
                "std_scalar_rule_reward": float(metrics.get("std_scalar_rule_reward", 0.0)) if metrics.get("std_scalar_rule_reward") is not None else None,
                "mean_hybrid_reward": float(metrics.get("mean_hybrid_reward", 0.0)) if metrics.get("mean_hybrid_reward") is not None else None,
                "std_hybrid_reward": float(metrics.get("std_hybrid_reward", 0.0)) if metrics.get("std_hybrid_reward") is not None else None,
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
                "checkpoint_path": _checkpoint_rel(run_dir, final_checkpoint_stem),
                "checkpoint_type": "final",
                "checkpoint_global_step": int(total_timesteps),
            },
        )

        eval_env.close()
        duration_seconds = round(time.time() - start_time, 2)
        train_logger.info(
            "Run completed | total_timesteps=%d | duration_seconds=%.2f",
            total_timesteps,
            duration_seconds,
        )
        log_event(
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
        log_event(
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
