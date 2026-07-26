from __future__ import annotations

import json
import logging
import time
from datetime import datetime
import csv
import pickle

from dataclasses import asdict
import random
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

# Ensure CuBLAS reproducibility config is set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from omegaconf import DictConfig, OmegaConf

from thesis_rl.agent.agent import Agent
from thesis_rl.agent.planners.core.utils import call_env_method
from thesis_rl.contracts.reward_semantics import build_reward_semantics_identity
from thesis_rl.runtime.data_abort import RuntimeScenarioQuarantine
from thesis_rl.curriculum.config import CurriculumConfig
from thesis_rl.curriculum.manager import CurriculumManager
from thesis_rl.curriculum.scenario_acl import (
    ScenarioAclDriverPaths,
    run_scenario_acl_training,
    validate_scenario_acl_runtime_support,
)
from thesis_rl.runtime.wiring.builders import (
    adapter_space_kwargs,
    build_adapter,
    build_eval_env,
    build_planner,
    build_preprocessor,
    collect_scenario_runtime_stats,
    merge_scenario_runtime_stats,
    build_train_env,
    evaluation_num_workers,
    is_vectorized_training_enabled,
    load_planner,
    merge_env_config_with_overrides,
    set_planner_env_if_compatible,
)
from thesis_rl.runtime.io.console import print_evaluation_summary, print_run_setup
from thesis_rl.runtime.io.csv_recorder import CSVRecorder
from thesis_rl.runtime.io.eval_artifacts import (
    maybe_build_live_final_eval_recorder_factory,
    maybe_build_periodic_tracked_subset_recorder_factory,
)
from thesis_rl.analysis.videos.select_video_episodes import select_tracked_subset_episodes
from thesis_rl.runtime.io.metadata import save_run_metadata, update_run_metadata
from thesis_rl.runtime.io.run_logging import (
    configure_logging,
    log_event,
    parse_log_level,
    setup_file_logger,
)
from thesis_rl.runtime.async_evaluation import AsyncEvaluationManager, EvaluationJob
from thesis_rl.runtime.execution.seeding import (
    apply_eval_scenario_seed_split,
    configure_parent_torch_threads,
    eval_base_seed_from_env_overrides,
    seed_env_spaces,
    set_global_seed,
    train_episode_seed_from_env_overrides,
)
from thesis_rl.sb3_extensions.replay import resolve_transition_replay_config


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


def _checkpoint_hash(checkpoint_stem_path: Path) -> str | None:
    """EVAL-PROTOCOL REQ-006: content hash for the official checkpoint."""
    from thesis_rl.sb3_extensions.checkpointing import sha256_file

    zip_path = checkpoint_stem_path.with_suffix(".zip")
    if not zip_path.is_file():
        return None
    return sha256_file(zip_path)


def _data_abort_coverage_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    """EVAL-PROTOCOL REQ-012: persist the already-computed data-abort coverage.

    ``metrics["data_abort_coverage"]`` is produced by
    ``Agent._evaluate_parallel`` as ``{"attempted", "valid", "invalid",
    "invalid_episodes"}``; this maps it to additive CSV columns without
    recomputing anything.
    """
    coverage = metrics.get("data_abort_coverage")
    if not isinstance(coverage, dict):
        return {
            "data_abort_attempted": None,
            "data_abort_valid": None,
            "data_abort_invalid": None,
            "data_abort_coverage": None,
        }
    attempted = coverage.get("attempted")
    valid = coverage.get("valid")
    invalid = coverage.get("invalid")
    ratio = (float(valid) / float(attempted)) if attempted else None
    return {
        "data_abort_attempted": attempted,
        "data_abort_valid": valid,
        "data_abort_invalid": invalid,
        "data_abort_coverage": ratio,
    }


def _lexicographic_eval_key(
    metrics: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float]:
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


def _save_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        with temporary_path.open("rb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_checkpoint_pair(
    pair_path: Path,
    *,
    checkpoint_name: str,
    replay_path: Path,
    training_timestep: int | None = None,
) -> dict[str, Any]:
    """Validate the model/replay/manifest identity before replay loading."""
    if not pair_path.is_file():
        raise ValueError(
            "Replay continuation requires a checkpoint pair manifest; "
            f"missing {pair_path}. Use model-only transfer initialization instead."
        )
    try:
        payload = json.loads(pair_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read checkpoint pair manifest: {pair_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint pair manifest must be an object: {pair_path}")
    if not str(payload.get("checkpoint_id", "")):
        raise ValueError(f"Checkpoint pair manifest has no checkpoint_id: {pair_path}")
    expected_model_name = f"{checkpoint_name}.zip"
    if payload.get("model_path") != expected_model_name:
        raise ValueError(
            "Checkpoint pair model identity mismatch: "
            f"manifest={payload.get('model_path')!r}, expected={expected_model_name!r}."
        )
    if payload.get("replay_path") != replay_path.name:
        raise ValueError(
            "Checkpoint pair replay identity mismatch: "
            f"manifest={payload.get('replay_path')!r}, expected={replay_path.name!r}."
        )
    if not replay_path.is_file():
        raise ValueError(f"Checkpoint pair replay artifact is missing: {replay_path}")
    if training_timestep is not None and int(payload.get("training_timestep", -1)) != int(
        training_timestep
    ):
        raise ValueError(
            "Checkpoint pair training timestep mismatch: "
            f"manifest={payload.get('training_timestep')!r}, expected={training_timestep!r}."
        )
    return payload


def _load_training_state(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    return dict(OmegaConf.to_container(loaded, resolve=True))


def _save_replay_buffer_if_available(planner: Any, path: Path) -> bool:
    if not hasattr(planner, "save_replay_buffer"):
        return False
    return bool(planner.save_replay_buffer(str(path)))


def _save_replay_buffer_atomically(planner: Any, path: Path) -> bool:
    """Save a replay artifact through a temporary file and atomic replacement."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        saved = _save_replay_buffer_if_available(planner, temporary_path)
        if not saved or not temporary_path.is_file():
            raise RuntimeError(f"Replay-buffer saver did not create {temporary_path}.")
        os.replace(temporary_path, path)
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_replay_buffer_if_available(planner: Any, path: Path) -> bool:
    if not path.exists():
        return False
    if not hasattr(planner, "load_replay_buffer"):
        return False
    return bool(planner.load_replay_buffer(str(path)))


def _validate_replay_buffer_n_envs(planner: Any) -> None:
    planner_n_envs = int(getattr(planner, "n_envs", 1))
    if not hasattr(planner, "replay_buffer_n_envs"):
        return
    buffer_n_envs = int(planner.replay_buffer_n_envs())
    if planner_n_envs != buffer_n_envs:
        raise ValueError(
            "Loaded replay buffer was created with a different number of envs: "
            f"planner.n_envs={planner_n_envs}, replay_buffer.n_envs={buffer_n_envs}. "
            "Resume with the same env.vectorized.num_envs or start a fresh run."
        )


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
        torch.cuda.set_rng_state_all(
            [torch.tensor(state, dtype=torch.uint8, device="cpu") for state in cuda_state]
        )
    return True


def _save_quarantine_state(env: Any, path: Path) -> bool:
    """Persist the run-local runtime-scenario quarantine alongside the checkpoint."""

    if not hasattr(env, "env_method"):
        return False
    quarantine = RuntimeScenarioQuarantine()
    for worker_uids in call_env_method(env, "get_quarantined_scenario_uids"):
        for uid in worker_uids:
            quarantine.add(str(uid), "unspecified")
    quarantine.save(path)
    return True


def _load_quarantine_state(env: Any, path: Path) -> bool:
    """Restore the run-local runtime-scenario quarantine into every worker."""

    if not path.exists() or not hasattr(env, "env_method"):
        return False
    quarantine = RuntimeScenarioQuarantine.load(path)
    for uid in sorted(quarantine.scenario_uids):
        call_env_method(env, "quarantine_scenario_uid", uid)
    return True


def _required_curriculum_metrics(curriculum_cfg: CurriculumConfig) -> list[str]:
    """Derive required metric names from configured curriculum gates.

    Gate keys are expected to follow `<metric>_min` or `<metric>_max`.
    """
    if not curriculum_cfg.is_staged:
        return []
    gates_payload = asdict(curriculum_cfg.staged.promotion.gates)
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
    if not curriculum_cfg.is_staged:
        return None
    stage_cfg = curriculum_cfg.staged.promotion.per_stage_min_steps.get(stage_name)
    if stage_cfg is not None:
        return int(stage_cfg)
    if curriculum_cfg.staged.promotion.default_min_stage_steps > 0:
        return int(curriculum_cfg.staged.promotion.default_min_stage_steps)
    return None


def _tracked_subset_uids_from_resolved_env_cfg(resolved_cfg: Any) -> tuple[str, ...]:
    """REQ-014/DEC-014 (amended 2026-07-25): read the frozen tracked-subset
    ``scenario_uid``s off a resolved eval env config's
    ``provider.panel_manifest_path``, if one is configured (checked at both
    the top level and under a nested ``config`` key, matching the two
    shapes ``merge_env_config_with_overrides`` output is used with in this
    module for periodic vs. final eval). Returns an empty tuple -- tracked-
    subset rendering simply does not fire, never a fallback draw -- when no
    manifest is configured, consistent with `envs/factory.py`'s
    `_resolve_frozen_panel_uids` precedent.
    """
    if not isinstance(resolved_cfg, dict):
        return ()
    candidates = [resolved_cfg]
    nested = resolved_cfg.get("config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for candidate in candidates:
        provider_cfg = candidate.get("provider")
        if isinstance(provider_cfg, dict):
            manifest_path = provider_cfg.get("panel_manifest_path")
            if manifest_path not in (None, "", "null"):
                from thesis_rl.scenarios.panel_manifest import load_panel_manifest

                return load_panel_manifest(str(manifest_path)).tracked_subset_uids
    return ()


def _make_tracked_subset_render_gate(*, interval: int = 100_000):
    """REQ-014/DEC-014 (amended 2026-07-25): cadence gate for the *periodic*
    validation tracked-subset GIF render (approved 2026-07-25: every
    100,000 timesteps -- with the repo's default ``eval_interval=25,000``
    that is every 4th periodic evaluation). Final-test tracked-subset (and
    full-panel) rendering is always unconditional and never calls this
    gate.

    Returns a stateful ``due(global_steps_done) -> bool`` closure that fires
    exactly once per crossed multiple of ``interval``: the *first* periodic
    eval whose ``global_steps_done`` is at or after each multiple of
    ``interval`` (the nearest eval boundary >= that multiple), rather than
    an exact-modulo check. An exact-modulo check would silently never fire
    again once the cadence and ``eval_interval`` fall out of exact
    alignment (e.g. a non-divisor ``eval_interval`` such as 30,000); this
    closure instead tracks the next still-undue threshold explicitly, so it
    is robust to any ``eval_interval`` value while still firing at most
    once per 100,000-timestep window.
    """
    if interval <= 0:
        raise ValueError("interval must be positive")
    state: dict[str, int] = {"next_due": interval}

    def due(global_steps_done: int) -> bool:
        global_steps_done = int(global_steps_done)
        if global_steps_done >= state["next_due"]:
            state["next_due"] = ((global_steps_done // interval) + 1) * interval
            return True
        return False

    return due


def _append_rule_metrics_rows(
    recorder: CSVRecorder,
    *,
    base_fields: dict[str, Any],
    eval_id: int,
    eval_type: str,
    scenario_set: str,
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


def run_training(cfg: DictConfig) -> None:
    # Save metadata for this run (config, git info, etc.)
    artifacts_dir = Path(str(cfg.paths.artifacts_dir))
    metadata_path = save_run_metadata(cfg, artifacts_dir)
    hydra_config_path = Path(str(cfg.paths.run_dir)) / "hydra" / "config.yaml"
    start_time = time.time()
    logs_dir = Path(str(cfg.paths.logs_dir))
    csv_dir = Path(str(cfg.paths.csv_dir))
    recorder = CSVRecorder(csv_dir)
    run_id = Path(str(cfg.paths.run_dir)).name
    reward_type = str(cfg.reward.type).strip()
    if reward_type == "":
        raise ValueError("reward.type must be set for CSV schema.")
    reward_behavior = str(cfg.reward.behavior).strip()
    if reward_behavior == "":
        raise ValueError("reward.behavior must be set for CSV schema.")
    rulebook_config = str(cfg.reward.get("rulebook_config", "")).strip()
    if rulebook_config == "":
        raise ValueError("reward.rulebook_config must be set for CSV schema.")
    curriculum_enabled = bool(cfg.curriculum.get("enabled", False))
    if reward_type not in {"native", "rulebook"}:
        raise ValueError(
            f"Unsupported reward.type '{reward_type}' for strict analysis schema. "
            "Expected one of: native, rulebook."
        )
    if reward_behavior not in {"off", "monitor_only", "scalar_reward"}:
        raise ValueError(
            f"Unsupported reward.behavior '{reward_behavior}' for strict analysis schema. "
            "Expected one of: off, monitor_only, scalar_reward."
        )
    if reward_behavior == "off" and rulebook_config != "none":
        raise ValueError("reward.behavior=off requires reward.rulebook_config=none.")
    if reward_behavior != "off" and rulebook_config == "none":
        raise ValueError(
            "reward.behavior!=off requires reward.rulebook_config to be a valid rulebook name."
        )
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
    final_replay_buffer_path = checkpoints_dir / "final_replay_buffer.pkl"
    final_checkpoint_pair_path = checkpoints_dir / "final_checkpoint_pair.json"
    latest_training_state_path = checkpoints_dir / "latest_training_state.yaml"
    latest_checkpoint_pair_path = checkpoints_dir / "latest_checkpoint_pair.json"
    latest_rng_state_path = checkpoints_dir / "latest_rng_state.pkl"
    latest_quarantine_state_path = checkpoints_dir / "latest_quarantine_state.json"
    train_log_path = logs_dir / "train.log"
    eval_log_path = logs_dir / "eval.log"
    curriculum_log_path = logs_dir / "curriculum.log"
    errors_log_path = logs_dir / "errors.log"
    events_log_path = logs_dir / "events.jsonl"
    data_abort_log_path = logs_dir / "runtime_scenario_data_abort.jsonl"

    logging_cfg = cfg.get("logging", {})
    global_log_level = parse_log_level(logging_cfg.get("level"), default=logging.INFO)
    file_log_level = parse_log_level(logging_cfg.get("file_level"), default=global_log_level)
    console_log_level = parse_log_level(logging_cfg.get("console_level"), default=file_log_level)
    configure_logging(global_log_level, console_level=console_log_level)

    train_logger = setup_file_logger(
        "thesis_rl.cli.train",
        "train",
        train_log_path,
        level=file_log_level,
        console_level=console_log_level,
    )
    eval_logger = setup_file_logger(
        "thesis_rl.cli.train",
        "eval",
        eval_log_path,
        level=file_log_level,
        console_level=console_log_level,
    )
    curriculum_logger = setup_file_logger(
        "thesis_rl.cli.train",
        "curriculum",
        curriculum_log_path,
        level=file_log_level,
        console_level=console_log_level,
    )
    errors_logger = setup_file_logger(
        "thesis_rl.cli.train",
        "errors",
        errors_log_path,
        level=parse_log_level("WARNING"),
        console_level=console_log_level,
    )

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

    total_timesteps = int(cfg.experiment.total_timesteps)
    planner_algorithm_name = str(cfg.agent.planner.algorithm.name)
    transition_replay_config = resolve_transition_replay_config(
        cfg.agent.planner.algorithm.get("transition_replay"),
        algorithm_name=planner_algorithm_name,
        total_timesteps=total_timesteps,
        legacy_save_replay_buffer=bool(cfg.checkpoint.get("save_replay_buffer", False)),
    )
    current_global_step = 0
    chunk_id = 0
    eval_id = 0
    current_stage_name = "baseline"
    current_stage_index = 0
    beta_progress_env_steps = 0

    try:
        run_seed = int(cfg.seed)
        set_global_seed(run_seed)
        runtime_cfg = cfg.get("runtime", {})
        parent_torch_threads = runtime_cfg.get("parent_torch_num_threads")
        configure_parent_torch_threads(
            None if parent_torch_threads is None else int(parent_torch_threads)
        )
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
        resume_replay_buffer_path = (
            resume_run_dir / "checkpoints" / f"{resume_checkpoint_name}_replay_buffer.pkl"
        )
        resume_checkpoint_pair_path = (
            resume_run_dir / "checkpoints" / f"{resume_checkpoint_name}_checkpoint_pair.json"
        )
        resume_training_state_path = resume_run_dir / "checkpoints" / "latest_training_state.yaml"
        resume_rng_state_path = resume_run_dir / "checkpoints" / "latest_rng_state.pkl"
        resume_quarantine_state_path = (
            resume_run_dir / "checkpoints" / "latest_quarantine_state.json"
        )
        resume_state: dict[str, Any] | None = None
        resume_global_steps_done = 0
        resume_chunk_id = 0
        resume_eval_id = 0

        ###################
        ###### SETUP ######
        ###################

        # Curriculum manager
        curriculum_cfg = CurriculumConfig.from_curriculum_cfg(cfg.curriculum)
        validate_scenario_acl_runtime_support(
            cfg,
            curriculum_cfg,
            context="training",
        )
        if curriculum_cfg.is_scenario_acl:
            run_scenario_acl_training(
                cfg=cfg,
                curriculum_cfg=curriculum_cfg,
                recorder=recorder,
                base_csv_fields=base_csv_fields,
                metadata_path=metadata_path,
                hydra_config_path=hydra_config_path,
                start_time=start_time,
                paths=ScenarioAclDriverPaths(
                    artifacts_dir=artifacts_dir,
                    run_dir=run_dir,
                    checkpoints_dir=checkpoints_dir,
                    final_checkpoint_stem=final_checkpoint_stem,
                    latest_checkpoint_stem=latest_checkpoint_stem,
                    events_log_path=events_log_path,
                ),
                train_logger=train_logger,
                curriculum_logger=curriculum_logger,
            )
            return
        curriculum_manager: CurriculumManager | None = None
        current_train_overrides: dict[str, Any] | None = None
        if curriculum_cfg.enabled and curriculum_cfg.is_staged and curriculum_cfg.staged.stages:
            curriculum_manager = CurriculumManager(curriculum_cfg)
            if resume_enabled:
                if not resume_training_state_path.exists():
                    raise FileNotFoundError(
                        f"Resume enabled but training state is missing: {resume_training_state_path}"
                    )
                resume_state = _load_training_state(resume_training_state_path)
                curriculum_state = resume_state.get("curriculum", {})
                if isinstance(curriculum_state, dict):
                    curriculum_manager.load_state(curriculum_state)
            current_train_overrides = curriculum_manager.get_env_config(evaluation=False)
        elif resume_enabled:
            if not resume_training_state_path.exists():
                raise FileNotFoundError(
                    f"Resume enabled but training state is missing: {resume_training_state_path}"
                )
            resume_state = _load_training_state(resume_training_state_path)

        vectorized_training = is_vectorized_training_enabled(cfg)

        # Environment
        env = build_train_env(cfg, current_train_overrides)
        seed_env_spaces(env, run_seed)
        scenario_runtime_stats_total: dict[str, Any] | None = None

        # Agent
        preprocessor = build_preprocessor(cfg)
        adapter = build_adapter(
            cfg,
            adapter_space_kwargs(env.action_space),
        )
        planner_seed: int | None = run_seed
        if vectorized_training and str(cfg.env.get("name", "")).lower() == "metadrive":
            # Vectorized reset seeding can conflict with MetaDrive per-worker
            # worker seeds as (seed + rank). That conflicts with MetaDrive per-worker
            # scenario partitions (start_seed/num_scenarios), causing out-of-range asserts.
            planner_seed = None
            train_logger.info(
                "Planner seed disabled for vectorized MetaDrive to avoid worker reseeding conflicts."
            )

        if resume_enabled:
            if not resume_checkpoint_zip.exists():
                raise FileNotFoundError(
                    f"Resume enabled but checkpoint is missing: {resume_checkpoint_zip}"
                )
            planner = load_planner(cfg, checkpoint_path=str(resume_checkpoint_zip), env=env)
        else:
            planner = build_planner(cfg, env, seed=planner_seed)
        print_run_setup(
            title="Training Run",
            cfg=cfg,
            metadata_path=metadata_path,
            hydra_config_path=hydra_config_path,
        )
        # Read EMA alpha from planner config if available
        ema_alpha_cfg = (
            float(cfg.agent.planner.algorithm.get("monitor_ema_alpha", 0.1))
            if hasattr(cfg, "agent")
            else 0.1
        )
        agent = Agent(
            preprocessor=preprocessor, planner=planner, adapter=adapter, ema_alpha=ema_alpha_cfg
        )
        agent.set_checkpoint_identity(build_reward_semantics_identity(cfg))

        async_evaluation_manager: AsyncEvaluationManager | None = None

        def _make_eval_agent(checkpoint_stem: Path, eval_env: Any) -> tuple[Agent, str]:
            checkpoint_zip = f"{checkpoint_stem}.zip"
            eval_planner = load_planner(
                cfg, checkpoint_path=checkpoint_zip, env=eval_env, validate_rollout_geometry=False
            )
            return (
                Agent(
                    preprocessor=preprocessor,
                    planner=eval_planner,
                    adapter=adapter,
                    ema_alpha=ema_alpha_cfg,
                ),
                checkpoint_zip,
            )

        def _cleanup_checkpoint_artifacts(checkpoint_stem: Path) -> None:
            checkpoint_zip = checkpoint_stem.with_suffix(".zip")
            if checkpoint_zip.exists():
                checkpoint_zip.unlink()
            adapter_checkpoint = Agent.adapter_checkpoint_path(checkpoint_stem)
            if adapter_checkpoint.exists():
                adapter_checkpoint.unlink()

        def _record_async_evaluation(job: EvaluationJob, metrics: dict[str, Any]) -> None:
            """Persist a completed ordinary validation without gating training."""

            per_episode = metrics.get("per_episode", {})
            returns = list(per_episode.get("returns", []))
            fields = {
                **base_csv_fields,
                "eval_id": job.eval_id,
                "eval_type": "intermediate",
                "scenario_set": "curriculum_eval",
                "stage": job.stage,
                "stage_index": job.stage_index,
                "global_step": job.global_step,
                "deterministic": bool(cfg.experiment.eval_deterministic),
            }
            vector_fields = {
                "episode_length": list(per_episode.get("episode_length", [])),
                "success": list(per_episode.get("success", [])),
                "collision": list(per_episode.get("collision", [])),
                "out_of_road": list(per_episode.get("out_of_road", [])),
                "timeout": list(per_episode.get("timeout", [])),
                "route_completion": list(per_episode.get("route_completion", [])),
                "top_rule_violation_rate": list(per_episode.get("top_rule_violation_rate", [])),
                "error_value": list(per_episode.get("error_value", [])),
                "violated_rules": list(per_episode.get("violated_rules", [])),
                "violation_pattern": list(per_episode.get("violation_pattern", [])),
                "env_returns": list(per_episode.get("env_returns", [])),
                "scalar_rule_returns": list(per_episode.get("scalar_rule_returns", [])),
                "hybrid_returns": list(per_episode.get("hybrid_returns", [])),
                "rule_rewards_by_rule": list(per_episode.get("rule_rewards_by_rule", [])),
                "scenario_metadata": list(per_episode.get("scenario_metadata", [])),
            }
            for episode_idx, reward in enumerate(returns):
                metadata = (
                    vector_fields["scenario_metadata"][episode_idx]
                    if episode_idx < len(vector_fields["scenario_metadata"])
                    and isinstance(vector_fields["scenario_metadata"][episode_idx], dict)
                    else {}
                )
                scenario_seed = job.base_seed + episode_idx if job.base_seed is not None else None
                recorder.append_row(
                    "eval_episodes.csv",
                    {
                        **fields,
                        "episode_id": episode_idx + 1,
                        "scenario_seed": scenario_seed,
                        "scenario_id": metadata.get("scenario_id", f"seed_{scenario_seed}"),
                        "reward": float(reward),
                        "env_reward": _indexed_value(vector_fields["env_returns"], episode_idx),
                        "scalar_rule_reward": _indexed_value(
                            vector_fields["scalar_rule_returns"], episode_idx
                        ),
                        "hybrid_reward": _indexed_value(
                            vector_fields["hybrid_returns"], episode_idx
                        ),
                        "rule_rewards_by_rule": json.dumps(
                            _indexed_value(vector_fields["rule_rewards_by_rule"], episode_idx)
                            or {},
                            ensure_ascii=True,
                        ),
                        "episode_length": _indexed_value(
                            vector_fields["episode_length"], episode_idx
                        ),
                        "success": _indexed_value(vector_fields["success"], episode_idx),
                        "collision": _indexed_value(vector_fields["collision"], episode_idx),
                        "out_of_road": _indexed_value(vector_fields["out_of_road"], episode_idx),
                        "timeout": _indexed_value(vector_fields["timeout"], episode_idx),
                        "route_completion": _indexed_value(
                            vector_fields["route_completion"], episode_idx
                        ),
                        "top_rule_violation_rate": _indexed_value(
                            vector_fields["top_rule_violation_rate"], episode_idx
                        ),
                        "error_value": _indexed_value(vector_fields["error_value"], episode_idx),
                        "violated_rules": _indexed_value(
                            vector_fields["violated_rules"], episode_idx
                        ),
                        "violation_pattern": _indexed_value(
                            vector_fields["violation_pattern"], episode_idx
                        ),
                    },
                )
            recorder.append_row(
                "evals.csv",
                {
                    **fields,
                    "eval_episodes": len(returns),
                    **{
                        key: metrics.get(key)
                        for key in (
                            "mean_reward",
                            "std_reward",
                            "mean_env_reward",
                            "std_env_reward",
                            "mean_scalar_rule_reward",
                            "std_scalar_rule_reward",
                            "mean_hybrid_reward",
                            "std_hybrid_reward",
                            "mean_rule_saturation_max",
                            "collision_rate",
                            "collision_rate_std",
                            "out_of_road_rate",
                            "success_rate",
                            "success_rate_std",
                            "route_completion",
                            "top_rule_violation_rate",
                            "avg_error_value",
                            "max_error_value",
                            "counterexample_rate",
                            "violated_rules_ratio",
                            "unique_violation_patterns",
                        )
                    },
                    "promoted": False,
                    "next_stage": job.stage,
                    **_data_abort_coverage_fields(metrics),
                },
            )
            _append_rule_metrics_rows(
                recorder,
                base_fields=base_csv_fields,
                eval_id=job.eval_id,
                eval_type="intermediate",
                scenario_set="curriculum_eval",
                chunk_id=int(job.metadata.get("chunk_id", 0)),
                stage=job.stage,
                stage_index=job.stage_index,
                global_step=job.global_step,
                metrics=metrics,
            )
            save_intermediate_checkpoints(
                current_global_step=job.global_step,
                chunk_id=int(job.metadata.get("chunk_id", 0)),
                eval_id=job.eval_id,
                current_stage_name=job.stage,
                current_stage_index=job.stage_index,
                metrics=metrics,
            )
            eval_logger.info(
                "Asynchronous evaluation finished | eval_id=%d | step=%d | metrics=%s",
                job.eval_id,
                job.global_step,
                metrics,
            )
            log_event(
                events_log_path,
                "evaluation_finished",
                eval_id=job.eval_id,
                stage=job.stage,
                global_step=job.global_step,
                asynchronous=True,
                metrics=metrics,
            )

        def _indexed_value(values: list[Any], index: int) -> Any:
            return values[index] if index < len(values) else None

        if resume_enabled:
            agent.load_adapter(checkpoint_path=resume_checkpoint_zip, strict=True)
            if transition_replay_config.persistence_enabled:
                _validate_checkpoint_pair(
                    resume_checkpoint_pair_path,
                    checkpoint_name=resume_checkpoint_name,
                    replay_path=resume_replay_buffer_path,
                    training_timestep=int(resume_state.get("global_steps_done", 0))
                    if resume_state is not None
                    else None,
                )
                _load_replay_buffer_if_available(planner, resume_replay_buffer_path)
                _validate_replay_buffer_n_envs(planner)
            if bool(resume_cfg.get("restore_rng_state", True)):
                _load_rng_state(resume_rng_state_path)
            _load_quarantine_state(env, resume_quarantine_state_path)
            if resume_state is None and resume_training_state_path.exists():
                resume_state = _load_training_state(resume_training_state_path)
            if resume_state is not None:
                resume_global_steps_done = int(resume_state.get("global_steps_done", 0))
                resume_chunk_id = int(resume_state.get("chunk_id", 0))
                resume_eval_id = int(resume_state.get("eval_id", 0))
                if transition_replay_config.persistence_enabled:
                    beta_progress_env_steps = int(resume_state.get("beta_progress_env_steps", 0))

        # Training and evaluation params
        log_interval = int(cfg.experiment.get("log_interval", 1000))
        eval_interval = int(cfg.experiment.get("eval_interval", total_timesteps))
        if eval_interval <= 0:
            eval_interval = total_timesteps
        # REQ-014/DEC-014 (amended 2026-07-25): periodic tracked-subset GIF
        # render cadence, independent of `eval_interval`.
        tracked_subset_render_due = _make_tracked_subset_render_gate(interval=100_000)
        stage_name = (
            curriculum_manager.get_current_stage().name
            if curriculum_manager is not None
            else "baseline"
        )
        train_logger.info(
            "Run started | algorithm=%s | seed=%d | device=%s",
            str(cfg.agent.planner.algorithm.name),
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
            algorithm=str(cfg.agent.planner.algorithm.name),
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
                        "top_rule_violation_rate": float(
                            metrics.get("top_rule_violation_rate", 0.0)
                        ),
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
                        "checkpoint_path": _checkpoint_rel(
                            run_dir, best_rulebook_strict_checkpoint_stem
                        ),
                        "type": "best_lexicographic_rulebook",
                        "global_step": int(current_global_step),
                        "chunk_id": int(chunk_id),
                        "eval_id": int(eval_id),
                        "stage": str(current_stage_name),
                        "stage_index": int(current_stage_index),
                        "success_rate": float(metrics.get("success_rate", 0.0)),
                        "collision_rate": float(metrics.get("collision_rate", 0.0)),
                        "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
                        "top_rule_violation_rate": float(
                            metrics.get("top_rule_violation_rate", 0.0)
                        ),
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
                best_rulebook_thresholded_key is None
                or thresholded_key < best_rulebook_thresholded_key
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
                        "checkpoint_path": _checkpoint_rel(
                            run_dir, best_rulebook_thresholded_checkpoint_stem
                        ),
                        "type": "best_thresholded_lexicographic_rulebook",
                        "global_step": int(current_global_step),
                        "chunk_id": int(chunk_id),
                        "eval_id": int(eval_id),
                        "stage": str(current_stage_name),
                        "stage_index": int(current_stage_index),
                        "success_rate": float(metrics.get("success_rate", 0.0)),
                        "collision_rate": float(metrics.get("collision_rate", 0.0)),
                        "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
                        "top_rule_violation_rate": float(
                            metrics.get("top_rule_violation_rate", 0.0)
                        ),
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
                curriculum_state_payload = (
                    curriculum_manager.state_dict()
                    if curriculum_manager is not None
                    else {
                        "stage_index": 0,
                        "stage_steps_done": 0,
                        "eval_count_at_stage": 0,
                        "consecutive_passes": 0,
                        "last_eval_passed": False,
                    }
                )
                latest_state_payload = {
                    "global_steps_done": int(current_global_step),
                    "chunk_id": int(chunk_id),
                    "eval_id": int(eval_id),
                    "remaining_steps": int(total_timesteps - current_global_step),
                    "curriculum": {
                        "enabled": bool(curriculum_manager is not None),
                        **curriculum_state_payload,
                    },
                    "seed": int(run_seed),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "beta_progress_env_steps": int(beta_progress_env_steps),
                }
                _save_training_state(latest_training_state_path, latest_state_payload)
                if bool(cfg.checkpoint.get("save_rng_state", True)):
                    _save_rng_state(latest_rng_state_path)
                _save_quarantine_state(env, latest_quarantine_state_path)
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
                        "top_rule_violation_rate": float(
                            metrics.get("top_rule_violation_rate", 0.0)
                        ),
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
                            "top_rule_violation_rate": float(
                                metrics.get("top_rule_violation_rate", 0.0)
                            ),
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

        if curriculum_manager is None:
            async_evaluation_manager = AsyncEvaluationManager(
                checkpoints_dir=checkpoints_dir,
                on_complete=_record_async_evaluation,
                start_method=str(cfg.experiment.get("evaluation_start_method", "spawn")),
                numeric_library_num_threads=(
                    None
                    if cfg.env.get("vectorized", {}).get("worker_library_num_threads") is None
                    else int(cfg.env.get("vectorized", {}).get("worker_library_num_threads"))
                ),
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

            def train_reset_seed_for_episode(episode_index: int) -> int | None:
                return train_episode_seed_from_env_overrides(
                    current_train_overrides,
                    cfg,
                    run_seed=run_seed,
                    chunk_id=chunk_id,
                    episode_index=episode_index,
                    stage_index=current_stage_index,
                )

            # Agent training
            train_fn = agent.train_vectorized if vectorized_training else agent.train
            extra_train_kwargs: dict[str, Any] = {}
            if vectorized_training:
                extra_train_kwargs["data_abort_log_path"] = data_abort_log_path
                extra_train_kwargs["run_id"] = run_id
            provider_driven_scenarionet = str(
                cfg.env.get("name", "")
            ).lower() == "scenarionet" and str(
                cfg.env.get("provider", {}).get("kind", "uniform")
            ).lower() in {"uniform", "fixed_sequence"}
            chunk_summary = train_fn(
                env=env,
                chunk_timesteps=chunk_steps,
                global_total_timesteps=total_timesteps,
                global_steps_done=total_timesteps - remaining,
                stage_name=current_stage_name,
                deterministic=False,
                log_interval=log_interval,
                reset_seed_fn=(
                    None if provider_driven_scenarionet else train_reset_seed_for_episode
                ),
                monitor_event_poll_callback=(
                    async_evaluation_manager.drain_event_messages
                    if async_evaluation_manager is not None
                    else None
                ),
                live_extra_renderables_callback=(
                    async_evaluation_manager.renderables
                    if async_evaluation_manager is not None
                    else None
                ),
                **extra_train_kwargs,
            )
            actual_chunk_steps = int(chunk_summary.get("chunk_steps_actual", chunk_steps))
            beta_progress_env_steps += actual_chunk_steps
            remaining = max(0, remaining - actual_chunk_steps)
            current_global_step = min(total_timesteps, total_timesteps - remaining)
            steps_end = current_global_step
            train_logger.info(
                (
                    "Chunk finished | chunk_id=%d | stage=%s | global_step=%d | "
                    "episodes=%d | mean_return=%.3f | mean_len=%.2f | fps=%.1f | "
                    "update_calls=%d | n_updates=%d"
                ),
                chunk_id,
                current_stage_name,
                current_global_step,
                int(chunk_summary.get("episodes", 0)),
                float(chunk_summary.get("ep_rew_mean", 0.0)),
                float(chunk_summary.get("ep_len_mean", 0.0)),
                float(chunk_summary.get("fps", 0.0)),
                int(chunk_summary.get("update_calls", 0)),
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
                    "actor_loss_ema": float(chunk_summary.get("actor_loss_ema") or 0.0),
                    "critic_loss_ema": float(chunk_summary.get("critic_loss_ema") or 0.0),
                    "critic_loss": float(chunk_summary.get("critic_loss", 0.0)),
                    "learning_rate": float(chunk_summary.get("learning_rate", 0.0)),
                    "update_calls": int(chunk_summary.get("update_calls", 0)),
                    "n_updates": int(chunk_summary.get("n_updates", 0)),
                    "fps": float(chunk_summary.get("fps", 0.0)),
                    "elapsed_seconds": float(chunk_summary.get("elapsed_seconds", 0.0)),
                    "train_reset_seed_first": chunk_summary.get("train_reset_seed_first"),
                    "train_reset_seed_last": chunk_summary.get("train_reset_seed_last"),
                    "train_reset_seed_unique_count": chunk_summary.get(
                        "train_reset_seed_unique_count"
                    ),
                },
            )

            # Record training steps in curriculum manager for potential stage progression
            if curriculum_manager is not None:
                curriculum_manager.record_train_steps(actual_chunk_steps)

            # Staged curriculum retains the synchronous MetaDrive lifecycle. Ordinary
            # validation keeps the live learner environment open while the evaluator
            # runs in its own spawned process.
            if curriculum_manager is not None:
                scenario_runtime_stats_total = merge_scenario_runtime_stats(
                    scenario_runtime_stats_total,
                    collect_scenario_runtime_stats(env),
                )
                env.close()

            ###### EVALUATION ######

            # Config setup
            eval_env_overrides = None
            if curriculum_manager is not None:
                eval_env_overrides = curriculum_manager.get_env_config(evaluation=True)
            eval_episode_count = int(cfg.experiment.eval_episodes)
            if curriculum_manager is None:
                if eval_episode_count > 0:
                    eval_env_overrides = apply_eval_scenario_seed_split(
                        base_run_seed=run_seed,
                        eval_env_overrides=None,
                        cfg=cfg,
                        n_eval_episodes=eval_episode_count,
                        split="validation",
                    )
                    eval_base_seed = eval_base_seed_from_env_overrides(
                        eval_env_overrides,
                        cfg,
                    )
                    eval_id += 1
                    log_event(
                        events_log_path,
                        "evaluation_started",
                        eval_id=eval_id,
                        stage=current_stage_name,
                        global_step=current_global_step,
                        episodes=eval_episode_count,
                        start_seed=eval_base_seed,
                        asynchronous=True,
                    )
                    async_evaluation_manager.enqueue(
                        agent=agent,
                        cfg=cfg,
                        eval_id=eval_id,
                        global_step=current_global_step,
                        stage=current_stage_name,
                        stage_index=current_stage_index,
                        episode_count=eval_episode_count,
                        base_seed=eval_base_seed,
                        env_seed=run_seed + 100_000 + (total_timesteps - remaining),
                        env_overrides=eval_env_overrides,
                        workers=evaluation_num_workers(cfg, final=False),
                        metadata={"chunk_id": chunk_id},
                    )
                continue
            if eval_episode_count <= 0:
                # Explicitly support training-only smoke runs. The training
                # environment was closed above, so rebuild it for another
                # chunk when work remains and skip validation artifacts.
                if remaining > 0:
                    env = build_train_env(cfg, current_train_overrides)
                    seed_env_spaces(env, run_seed + 300_000 + (total_timesteps - remaining))
                    set_planner_env_if_compatible(planner, env)
                continue
            eval_env_overrides = apply_eval_scenario_seed_split(
                base_run_seed=run_seed,
                eval_env_overrides=eval_env_overrides,
                cfg=cfg,
                n_eval_episodes=eval_episode_count,
                split="validation",
            )
            eval_base_seed = eval_base_seed_from_env_overrides(eval_env_overrides, cfg)

            # Environment
            eval_env = build_eval_env(
                cfg,
                eval_env_overrides,
                n_eval_episodes=eval_episode_count,
                workers=evaluation_num_workers(cfg, final=False),
            )
            seed_env_spaces(eval_env, run_seed + 100_000 + (total_timesteps - remaining))
            eval_snapshot_stem = checkpoints_dir / "eval_snapshot"
            agent.save(eval_snapshot_stem)
            try:
                eval_agent, _ = _make_eval_agent(eval_snapshot_stem, eval_env)
            finally:
                _cleanup_checkpoint_artifacts(eval_snapshot_stem)

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
            # REQ-014/DEC-014 (amended 2026-07-25): the periodic tracked-
            # subset GIF render must be wired in *before* `evaluate()` runs
            # (the recorder factory is invoked live, per episode, during the
            # rollout), so the cadence gate and tracked scenario_uids are
            # resolved here rather than after the eval completes. The gate
            # closure has side effects (it advances its internal
            # `next_due` threshold), so it must be called at most once per
            # periodic evaluation; the result is reused below when writing
            # the tracked-subset selection JSON.
            resolved_periodic_eval_cfg = OmegaConf.to_container(
                merge_env_config_with_overrides(cfg.env, eval_env_overrides or {}),
                resolve=True,
            )
            tracked_scenario_uids = _tracked_subset_uids_from_resolved_env_cfg(
                resolved_periodic_eval_cfg if isinstance(resolved_periodic_eval_cfg, dict) else None
            )
            periodic_tracked_subset_render_due = bool(tracked_scenario_uids) and tracked_subset_render_due(
                current_global_step
            )
            periodic_tracked_subset_artifact_factory = None
            if periodic_tracked_subset_render_due:
                resolved_periodic_eval_env_config = (
                    resolved_periodic_eval_cfg.get("config", resolved_periodic_eval_cfg)
                    if isinstance(resolved_periodic_eval_cfg, dict)
                    else {}
                )
                periodic_tracked_subset_artifact_factory = maybe_build_periodic_tracked_subset_recorder_factory(
                    cfg=cfg,
                    run_dir=run_dir,
                    resolved_env_config=resolved_periodic_eval_env_config
                    if isinstance(resolved_periodic_eval_env_config, dict)
                    else {},
                    eval_id=eval_id,
                    global_step=current_global_step,
                    stage=current_stage_name,
                    stage_index=current_stage_index,
                    checkpoint_path=_checkpoint_rel(run_dir, eval_snapshot_stem),
                    checkpoint_type="periodic_validation_snapshot",
                    checkpoint_global_step=current_global_step,
                    tracked_scenario_uids=tracked_scenario_uids,
                )
            metrics = eval_agent.evaluate(
                env=eval_env,
                n_eval_episodes=eval_episode_count,
                deterministic=bool(cfg.experiment.eval_deterministic),
                base_seed=eval_base_seed,
                return_episode_metrics=True,
                error_priority_base=float(cfg.reward.get("a", 2.01)),
                show_progress=True,
                artifact_recorder_factory=periodic_tracked_subset_artifact_factory,
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
            episode_video_paths = list(per_episode.get("video_path", []))
            episode_video_authoritative_paths = list(
                per_episode.get("video_authoritative_path", [])
            )
            episode_video_manifest_paths = list(per_episode.get("video_manifest_path", []))
            episode_trajectory_log_paths = list(per_episode.get("trajectory_log_path", []))
            episode_video_recorded_live = list(per_episode.get("video_recorded_live", []))
            episode_replay_warnings = list(per_episode.get("replay_warning", []))
            episode_scenario_metadata = list(per_episode.get("scenario_metadata", []))
            episode_count = len(episode_returns)
            for episode_idx in range(episode_count):
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
                        "eval_type": "intermediate",
                        "scenario_set": "curriculum_eval",
                        "episode_id": episode_idx + 1,
                        "stage": current_stage_name,
                        "stage_index": current_stage_index,
                        "global_step": current_global_step,
                        "scenario_seed": scenario_seed,
                        "scenario_uid": scenario_metadata.get("scenario_uid"),
                        "scenario_id": scenario_metadata.get(
                            "scenario_id", f"seed_{scenario_seed}"
                        ),
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
                        "top_rule_violation_rate": float(
                            episode_top_rule_violation_rate[episode_idx]
                        )
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

            # REQ-014/DEC-014 (amended 2026-07-25): tracked-subset GIF
            # mechanism for periodic validation. Selection (which recorded
            # eval_episodes.csv rows match the frozen tracked scenario_uids)
            # is a cheap CSV scan and is written unconditionally every
            # periodic eval, keeping the manifest complete. The GIF
            # *rendering* itself was already gated and wired into the
            # `evaluate()` call above (`periodic_tracked_subset_render_due`
            # / `periodic_tracked_subset_artifact_factory`, computed once
            # before the call since the cadence gate is a stateful
            # closure); this block only writes the selection JSON and logs
            # the outcome, reusing those already-computed values instead of
            # re-invoking the gate.
            if tracked_scenario_uids:
                select_tracked_subset_episodes(run_dir, tracked_scenario_uids=tracked_scenario_uids)
                if periodic_tracked_subset_render_due:
                    eval_logger.info(
                        "Tracked-subset GIF render fired at global_step=%d "
                        "(cadence=100000); GIFs written under "
                        "videos_dir/periodic_eval/step_%07d/eval_%04d/ for "
                        "scenario_uids matching the tracked subset "
                        "(%d uid(s)).",
                        current_global_step,
                        current_global_step,
                        eval_id,
                        len(tracked_scenario_uids),
                    )

            ###### CURRICULUM PROGRESSION ######

            # If no curriculum, just rebuild the env
            if curriculum_manager is None:
                recorder.append_row(
                    "evals.csv",
                    {
                        **base_csv_fields,
                        "eval_id": eval_id,
                        "eval_type": "intermediate",
                        "scenario_set": "curriculum_eval",
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
                        "mean_scalar_rule_reward": float(
                            metrics.get("mean_scalar_rule_reward", 0.0)
                        )
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
                        "mean_rule_saturation_max": float(
                            metrics.get("mean_rule_saturation_max", 0.0)
                        ),
                        "collision_rate": float(metrics.get("collision_rate", 0.0)),
                        "collision_rate_std": float(metrics.get("collision_rate_std", 0.0)),
                        "out_of_road_rate": float(metrics.get("out_of_road_rate", 0.0)),
                        "success_rate": float(metrics.get("success_rate", 0.0)),
                        "success_rate_std": float(metrics.get("success_rate_std", 0.0)),
                        "route_completion": float(metrics.get("route_completion", 0.0)),
                        "top_rule_violation_rate": float(
                            metrics.get("top_rule_violation_rate", 0.0)
                        ),
                        "avg_error_value": float(metrics.get("avg_error_value", 0.0)),
                        "max_error_value": float(metrics.get("max_error_value", 0.0)),
                        "counterexample_rate": float(metrics.get("counterexample_rate", 0.0)),
                        "violated_rules_ratio": float(metrics.get("violated_rules_ratio", 0.0)),
                        "unique_violation_patterns": int(
                            metrics.get("unique_violation_patterns", 0)
                        ),
                        "promoted": False,
                        "next_stage": current_stage_name,
                        **_data_abort_coverage_fields(metrics),
                    },
                )
                _append_rule_metrics_rows(
                    recorder,
                    base_fields=base_csv_fields,
                    eval_id=eval_id,
                    eval_type="intermediate",
                    scenario_set="curriculum_eval",
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
                env = build_train_env(cfg, current_train_overrides)
                seed_env_spaces(env, run_seed + 300_000 + (total_timesteps - remaining))
                set_planner_env_if_compatible(planner, env)
                continue

            # Metrics check
            if curriculum_cfg.is_staged and curriculum_cfg.staged.mode.lower() == "auto":
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
            stage_gates = curriculum_cfg.staged.promotion.gates
            gate_success_pass = float(metrics.get("success_rate", float("-inf"))) >= float(
                stage_gates.task.success_rate_min
            )
            gate_collision_pass = float(metrics.get("collision_rate", float("inf"))) <= float(
                stage_gates.safety.collision_rate_max
            )
            gate_out_of_road_pass = float(metrics.get("out_of_road_rate", float("inf"))) <= float(
                stage_gates.safety.out_of_road_rate_max
            )
            gate_top_rule_pass = float(
                metrics.get("top_rule_violation_rate", float("inf"))
            ) <= float(stage_gates.safety.top_rule_violation_rate_max)
            gate_route_completion_pass = float(
                metrics.get("route_completion", float("-inf"))
            ) >= float(stage_gates.task.route_completion_min)

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
                        "top_rule_violation_rate": float(
                            metrics.get("top_rule_violation_rate", 0.0)
                        ),
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
                    "eval_type": "intermediate",
                    "scenario_set": "curriculum_eval",
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
                    "success_rate_min": float(stage_gates.task.success_rate_min),
                    "collision_rate_max": float(stage_gates.safety.collision_rate_max),
                    "out_of_road_rate_max": float(stage_gates.safety.out_of_road_rate_max),
                    "top_rule_violation_rate_max": float(
                        stage_gates.safety.top_rule_violation_rate_max
                    ),
                    "route_completion_min": float(stage_gates.task.route_completion_min),
                    "gate_success_pass": bool(gate_success_pass),
                    "gate_collision_pass": bool(gate_collision_pass),
                    "gate_out_of_road_pass": bool(gate_out_of_road_pass),
                    "gate_top_rule_pass": bool(gate_top_rule_pass),
                    "gate_route_completion_pass": bool(gate_route_completion_pass),
                    "passed_eval_gates": bool(passed_eval_gates),
                    "consecutive_passes": pre_promotion_consecutive_passes,
                    "warmup_evals_required": int(curriculum_cfg.staged.promotion.warmup_evals),
                    "consecutive_evals_required": int(
                        curriculum_cfg.staged.promotion.consecutive_evals
                    ),
                    "promoted": bool(next_stage_name != current_stage_name),
                    "next_stage": next_stage_name,
                    **_data_abort_coverage_fields(metrics),
                },
            )
            _append_rule_metrics_rows(
                recorder,
                base_fields=base_csv_fields,
                eval_id=eval_id,
                eval_type="intermediate",
                scenario_set="curriculum_eval",
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
            env = build_train_env(cfg, current_train_overrides)
            seed_env_spaces(env, run_seed + 400_000 + (total_timesteps - remaining))
            set_planner_env_if_compatible(planner, env)  # Update planner's env reference

        # EVAL-PROTOCOL v1.0 REQ-002/DEC-015: the approved PPO atomic
        # collection/update unit is a complete global rollout. If the target
        # budget was reached mid-rollout, complete it (bounded overshoot)
        # rather than discarding the partially collected transitions without
        # training on them. Zero for algorithms without this concept
        # (TD3/SAC update on every transition) or already at a boundary.
        overshoot_steps = 0
        pending_atomic_steps = int(agent.atomic_boundary_remaining())
        if pending_atomic_steps > 0:
            train_logger.info(
                "Completing final atomic collection unit before stopping | pending_steps=%d",
                pending_atomic_steps,
            )
            overshoot_summary = train_fn(
                env=env,
                chunk_timesteps=pending_atomic_steps,
                global_total_timesteps=total_timesteps,
                global_steps_done=total_timesteps,
                stage_name=current_stage_name,
                deterministic=False,
                log_interval=log_interval,
                reset_seed_fn=(
                    None if provider_driven_scenarionet else train_reset_seed_for_episode
                ),
                monitor_event_poll_callback=(
                    async_evaluation_manager.drain_event_messages
                    if async_evaluation_manager is not None
                    else None
                ),
                live_extra_renderables_callback=(
                    async_evaluation_manager.renderables
                    if async_evaluation_manager is not None
                    else None
                ),
                **extra_train_kwargs,
            )
            overshoot_steps = int(
                overshoot_summary.get("chunk_steps_actual", pending_atomic_steps)
            )
            beta_progress_env_steps += overshoot_steps
            train_logger.info(
                "Atomic collection unit completed | overshoot_steps=%d", overshoot_steps
            )
            log_event(
                events_log_path,
                "atomic_boundary_overshoot_completed",
                overshoot_steps=overshoot_steps,
            )

        ##########################
        ###### FINALIZATION ######
        ##########################

        # Save final checkpoint used as official final evaluation checkpoint.
        if not bool(cfg.checkpoint.get("save_final", True)):
            raise ValueError(
                "checkpoint.save_final must be true: final evaluation requires final checkpoint."
            )
        agent.save(final_checkpoint_stem)
        if transition_replay_config.persistence_enabled:
            _save_replay_buffer_atomically(planner, final_replay_buffer_path)
        _save_json_atomically(
            final_checkpoint_pair_path,
            {
                "checkpoint_id": uuid.uuid4().hex,
                "training_timestep": int(total_timesteps),
                "replay_segment_id": 0,
                "beta_progress_env_steps": int(beta_progress_env_steps),
                "model_path": final_checkpoint_stem.with_suffix(".zip").name,
                "replay_path": (
                    final_replay_buffer_path.name
                    if transition_replay_config.persistence_enabled
                    else None
                ),
            },
        )
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
                "stage_index": int(curriculum_manager.stage_index)
                if curriculum_manager is not None
                else 0,
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

        if async_evaluation_manager is not None:
            async_evaluation_manager.drain()
            async_evaluation_manager.close()
            scenario_runtime_stats_total = merge_scenario_runtime_stats(
                scenario_runtime_stats_total,
                collect_scenario_runtime_stats(env),
            )

        # Environment
        env.close()
        final_eval_episode_count = int(
            cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)
        )
        if final_eval_episode_count <= 0:
            duration_seconds = round(time.time() - start_time, 2)
            update_run_metadata(
                artifacts_dir,
                {
                    "status": "completed",
                    "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": duration_seconds,
                    "final_evaluation": "skipped",
                    "scenarionet_runtime_stats": scenario_runtime_stats_total,
                },
            )
            train_logger.info(
                "Run completed without final evaluation | total_timesteps=%d | duration_seconds=%.2f",
                total_timesteps,
                duration_seconds,
            )
            return
        final_eval_env_overrides = None
        final_stage_name = "baseline"
        final_stage_index = 0
        steps_to_final_stage = 0
        if curriculum_manager is not None:
            if not curriculum_cfg.staged.stages:
                raise ValueError("Curriculum is enabled but no stages are configured.")
            final_stage_index = len(curriculum_cfg.staged.stages) - 1
            final_stage = curriculum_cfg.staged.stages[final_stage_index]
            final_stage_name = final_stage.name
            final_eval_env_overrides = dict(final_stage.env)
            if final_stage.eval_env:
                final_eval_env_overrides.update(final_stage.eval_env)
            promotions_path = Path(str(cfg.paths.csv_dir)) / "promotions.csv"
            promoted_to_final_step: int | None = None
            if promotions_path.exists():
                with promotions_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for prow in reader:
                        try:
                            to_stage_index = int(str(prow.get("to_stage_index", "")).strip())
                            step_value = int(float(str(prow.get("global_step", "")).strip()))
                        except ValueError:
                            continue
                        if to_stage_index == final_stage_index:
                            promoted_to_final_step = step_value
                            break
            if promoted_to_final_step is not None:
                steps_to_final_stage = promoted_to_final_step
            elif bool(curriculum_manager.is_finished()):
                steps_to_final_stage = 0
            else:
                steps_to_final_stage = -1
        final_eval_env_overrides = apply_eval_scenario_seed_split(
            base_run_seed=run_seed,
            eval_env_overrides=final_eval_env_overrides,
            cfg=cfg,
            n_eval_episodes=final_eval_episode_count,
            split="test",
        )
        final_eval_base_seed = eval_base_seed_from_env_overrides(final_eval_env_overrides, cfg)
        eval_env = build_eval_env(
            cfg,
            final_eval_env_overrides,
            n_eval_episodes=final_eval_episode_count,
            workers=evaluation_num_workers(cfg, final=True),
        )
        seed_env_spaces(eval_env, run_seed + 500_000)
        eval_agent, _ = _make_eval_agent(final_checkpoint_stem, eval_env)

        # Evaluation
        final_eval_id = eval_id + 1
        resolved_final_eval_cfg = OmegaConf.to_container(
            merge_env_config_with_overrides(cfg.env, final_eval_env_overrides or {}),
            resolve=True,
        )
        if not isinstance(resolved_final_eval_cfg, dict):
            raise TypeError("Resolved final eval env config must be a mapping.")
        resolved_final_eval_env_config = resolved_final_eval_cfg.get(
            "config", resolved_final_eval_cfg
        )
        if not isinstance(resolved_final_eval_env_config, dict):
            raise TypeError("Resolved final eval env config payload must be a mapping.")
        final_eval_artifact_factory = maybe_build_live_final_eval_recorder_factory(
            cfg=cfg,
            run_dir=run_dir,
            resolved_env_config=resolved_final_eval_env_config,
            eval_id=final_eval_id,
            eval_type="final",
            scenario_set="test",
            stage=final_stage_name,
            stage_index=final_stage_index,
            checkpoint_path=_checkpoint_rel(run_dir, final_checkpoint_stem),
            checkpoint_type="final",
            checkpoint_global_step=int(total_timesteps),
        )
        metrics = eval_agent.evaluate(
            env=eval_env,
            n_eval_episodes=final_eval_episode_count,
            deterministic=bool(cfg.experiment.eval_deterministic),
            base_seed=final_eval_base_seed,
            return_episode_metrics=True,
            error_priority_base=float(cfg.reward.get("a", 2.01)),
            show_progress=True,
            progress_description="Test episodes",
            artifact_recorder_factory=final_eval_artifact_factory,
        )
        final_checkpoint_zip = f"{final_checkpoint_stem}.zip"
        print_evaluation_summary(
            title="Final Evaluation",
            metrics=metrics,
            stage=final_stage_name,
            global_step=total_timesteps,
            episodes=final_eval_episode_count,
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
            stage=final_stage_name,
            global_step=total_timesteps,
            metrics=metrics,
            final=True,
        )
        recorder.append_row(
            "evals.csv",
            {
                **base_csv_fields,
                "eval_id": final_eval_id,
                "eval_type": "final",
                "scenario_set": "test",
                "chunk_id": chunk_id,
                "stage": final_stage_name,
                "stage_index": final_stage_index,
                "global_step": total_timesteps,
                "eval_episodes": final_eval_episode_count,
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
                "next_stage": final_stage_name,
                **_data_abort_coverage_fields(metrics),
            },
        )
        _append_rule_metrics_rows(
            recorder,
            base_fields=base_csv_fields,
            eval_id=final_eval_id,
            eval_type="final",
            scenario_set="test",
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
                int(final_eval_base_seed + episode_idx)
                if final_eval_base_seed is not None
                else None
            )
            recorder.append_row(
                "eval_episodes.csv",
                {
                    **base_csv_fields,
                    "eval_id": final_eval_id,
                    "eval_type": "final",
                    "scenario_set": "test",
                    "episode_id": episode_idx + 1,
                    "stage": final_stage_name,
                    "stage_index": final_stage_index,
                    "global_step": total_timesteps,
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

        # REQ-014/DEC-014 (amended 2026-07-25): the final-test tracked-
        # subset selection JSON is written unconditionally (no cadence gate
        # -- final test always covers the complete panel regardless of the
        # periodic 100,000-timestep cadence). Actual GIF rendering for the
        # tracked subset has the same AWAITING_CONFIRMATION status noted at
        # the periodic call site above; the full test-panel live-eval video
        # recording path (`maybe_build_live_final_eval_recorder_factory`,
        # `eval_type="final"`) already renders unconditionally today and is
        # unaffected by this change.
        final_tracked_scenario_uids = _tracked_subset_uids_from_resolved_env_cfg(resolved_final_eval_cfg)
        if final_tracked_scenario_uids:
            select_tracked_subset_episodes(run_dir, tracked_scenario_uids=final_tracked_scenario_uids)

        recorder.append_row(
            "final_eval.csv",
            {
                **base_csv_fields,
                "eval_type": "final",
                "scenario_set": "test",
                "total_timesteps": total_timesteps,
                "final_stage": final_stage_name,
                "final_stage_index": final_stage_index,
                "final_stage_reached": bool(curriculum_manager.is_finished())
                if curriculum_manager is not None
                else True,
                "steps_to_final_stage": int(steps_to_final_stage),
                "final_eval_episodes": final_eval_episode_count,
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
                "checkpoint_path": _checkpoint_rel(run_dir, final_checkpoint_stem),
                "checkpoint_type": "final",
                "checkpoint_global_step": int(total_timesteps),
                "checkpoint_hash": _checkpoint_hash(final_checkpoint_stem),
                "checkpoint_role": "final",
                **_data_abort_coverage_fields(metrics),
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
        update_run_metadata(
            artifacts_dir,
            {
                "status": "completed",
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": duration_seconds,
                "scenarionet_runtime_stats": scenario_runtime_stats_total,
                "checkpoint": {
                    "path": _checkpoint_rel(run_dir, final_checkpoint_stem),
                    "hash": _checkpoint_hash(final_checkpoint_stem),
                    "role": "final",
                },
                "budget": {
                    "target_timesteps": int(total_timesteps),
                    "actual_completed_timesteps": int(total_timesteps + overshoot_steps),
                    "overshoot_steps": int(overshoot_steps),
                },
            },
        )

    except KeyboardInterrupt:
        duration_seconds = round(time.time() - start_time, 2)
        train_logger.warning(
            "Run interrupted by user (Ctrl+C) | step=%d | chunk_id=%d | eval_id=%d | stage=%s | stage_index=%d | duration_seconds=%.2f",
            int(current_global_step),
            int(chunk_id),
            int(eval_id),
            str(current_stage_name),
            int(current_stage_index),
            duration_seconds,
        )
        log_event(
            events_log_path,
            "run_interrupted",
            global_step=int(current_global_step),
            chunk_id=int(chunk_id),
            eval_id=int(eval_id),
            stage=str(current_stage_name),
            stage_index=int(current_stage_index),
            duration_seconds=duration_seconds,
        )
        try:
            agent.save(latest_checkpoint_stem)
            if transition_replay_config.persistence_enabled:
                _save_replay_buffer_atomically(planner, latest_replay_buffer_path)
                _save_json_atomically(
                    latest_checkpoint_pair_path,
                    {
                        "checkpoint_id": uuid.uuid4().hex,
                        "training_timestep": int(current_global_step),
                        "replay_segment_id": 0,
                        "beta_progress_env_steps": int(beta_progress_env_steps),
                        "model_path": latest_checkpoint_stem.with_suffix(".zip").name,
                        "replay_path": latest_replay_buffer_path.name,
                    },
                )
            curriculum_state_payload = (
                curriculum_manager.state_dict()
                if curriculum_manager is not None
                else {
                    "stage_index": 0,
                    "stage_steps_done": 0,
                    "eval_count_at_stage": 0,
                    "consecutive_passes": 0,
                    "last_eval_passed": False,
                }
            )
            _save_training_state(
                latest_training_state_path,
                {
                    "global_steps_done": int(current_global_step),
                    "chunk_id": int(chunk_id),
                    "eval_id": int(eval_id),
                    "remaining_steps": int(max(0, total_timesteps - current_global_step)),
                    "curriculum": {
                        "enabled": bool(curriculum_manager is not None),
                        **curriculum_state_payload,
                    },
                    "seed": int(run_seed),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            if bool(cfg.checkpoint.get("save_rng_state", True)):
                _save_rng_state(latest_rng_state_path)
            _save_quarantine_state(env, latest_quarantine_state_path)
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
                    "reason": "run_interrupted",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            )
        except Exception as checkpoint_error:
            errors_logger.warning(
                "Interrupt checkpoint save failed | error=%s",
                str(checkpoint_error),
            )
        update_run_metadata(
            artifacts_dir,
            {
                "status": "interrupted",
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": duration_seconds,
                "global_step": int(current_global_step),
                "chunk_id": int(chunk_id),
                "eval_id": int(eval_id),
                "stage": str(current_stage_name),
                "stage_index": int(current_stage_index),
            },
        )
        raise SystemExit(130)
    except Exception as e:
        duration_seconds = round(time.time() - start_time, 2)
        errors_logger.exception("Training failed | error=%s", str(e))
        log_event(
            events_log_path,
            "run_failed",
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
