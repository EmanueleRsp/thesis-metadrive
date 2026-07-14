from __future__ import annotations

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
from thesis_rl.curriculum.scenario_acl.arms import (
    ScenarioArm,
    SCENARIO_ARM_NAMES,
    build_default_scenario_arms,
)
from thesis_rl.curriculum.scenario_acl.buffer import ScenarioBuffer
from thesis_rl.curriculum.scenario_acl.mab import ScenarioArmBandit
from thesis_rl.curriculum.scenario_acl.record import ScenarioRecord
from thesis_rl.curriculum.scenario_acl.usefulness import (
    compute_learning_potential,
    compute_scenario_usefulness,
)
from thesis_rl.runtime.execution.seeding import (
    apply_eval_scenario_seed_split,
    seed_env_spaces,
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


@dataclass
class EpisodeAclOutcome:
    """The ACL decision and feedback belonging to one completed episode."""

    spec: IterationSpec
    record: Any | None
    metrics: dict[str, Any]
    learning_potential: float | None = None
    usefulness_norm: float | None = None


def _summarize_episode_acl_outcomes(
    outcomes: list[EpisodeAclOutcome],
) -> dict[str, Any]:
    """Summarize ACL decisions using completed episodes as the unit of work."""

    modes = sorted({outcome.spec.mode for outcome in outcomes})
    arms = sorted(
        {
            str(
                getattr(outcome.record, "primary_arm", None)
                if outcome.record is not None
                else outcome.spec.arm_name
            )
            for outcome in outcomes
        }
    )
    return {
        "generate_count": sum(
            1 for outcome in outcomes if not _is_replay_iteration(outcome.spec)
        ),
        "replay_count": sum(
            1 for outcome in outcomes if _is_replay_iteration(outcome.spec)
        ),
        "modes": modes,
        "arms": arms,
        "mode": (
            modes[0]
            if len(modes) == 1
            else "mixed"
            if modes
            else "none"
        ),
    }


def _is_replay_iteration(spec: IterationSpec) -> bool:
    return spec.mode.startswith("exploit_replay")


def _selection_source_override(spec: IterationSpec) -> str | None:
    """Return a forced source only for one-sided semantic arms."""

    if spec.train_env_overrides is None:
        return None
    provider = spec.train_env_overrides.get("provider")
    if not isinstance(provider, dict):
        return None
    probabilities = provider.get("source_probability")
    if not isinstance(probabilities, dict):
        return None
    selected = [str(source) for source, probability in probabilities.items() if float(probability) == 1.0]
    return selected[0] if len(selected) == 1 else None


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _short_arm_name(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value).split("_", maxsplit=1)[0]


def _format_optional_float(value: object | None) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return None


_WAYMO_EVAL_ARMS = tuple(arm for arm in SCENARIO_ARM_NAMES if arm != "A0_simple_low_traffic")
_WAYMO_STRATIFIED_SET = "waymo_stratified_a1_a5"


def _select_waymo_eval_arm(episode_index: int) -> str:
    """Cycle uniformly over Waymo-supported semantic arms A1-A5."""

    return _WAYMO_EVAL_ARMS[int(episode_index) % len(_WAYMO_EVAL_ARMS)]


def _configure_waymo_eval_episode(env: Any, episode_index: int) -> None:
    """Apply the deterministic arm schedule before a ScenarioNet eval reset."""

    base_env = _base_env(env)
    base_env.scenario_arm = _select_waymo_eval_arm(episode_index)
    base_env.scenario_source = "waymo"


def _semantic_catalog_overrides(cfg: DictConfig) -> dict[str, object]:
    """Resolve the catalog required for semantic-arm filtering and ACL records."""

    configured = cfg.env.get("catalog_path")
    catalog_path = (
        Path(str(configured)).expanduser()
        if configured not in (None, "", "null")
        else Path(str(cfg.paths.scenarionet_data_root)).expanduser()
        / "catalog"
        / "scenario_catalog.parquet"
    )
    if not catalog_path.is_file():
        raise FileNotFoundError(
            "Semantic Scenario ACL requires the classified ScenarioNet catalog at "
            f"{catalog_path}. Set env.catalog_path or prepare the dataset catalog."
        )
    return {"catalog_path": str(catalog_path)}


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


def _normalize_learning_potential(
    current_value: float,
    recent_values: list[float],
) -> float:
    rank_set = [float(v) for v in recent_values] + [float(current_value)]
    if len(rank_set) <= 1:
        return 1.0
    # Average tied ranks so identical learning signals receive identical,
    # non-maximal feedback when stronger observations are present.
    greater = sum(value > float(current_value) for value in rank_set)
    equal = sum(value == float(current_value) for value in rank_set)
    rank = greater + ((equal + 1) / 2.0)
    return 1.0 - ((rank - 1) / float(len(rank_set) - 1))


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


def _episode_record_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Adapt one episode's feedback to the ScenarioRecord metric schema."""

    return {
        "mean_reward": float(metrics.get("reward", 0.0)),
        "success_rate": float(bool(metrics.get("success", False))),
        "route_completion": float(metrics.get("route_completion", 0.0)),
        "collision_rate": float(bool(metrics.get("collision", False))),
        "out_of_road_rate": float(bool(metrics.get("out_of_road", False))),
        "top_rule_violation_rate": float(metrics.get("top_rule_violation_rate", 0.0)),
        "termination_reason": str(metrics.get("termination_reason", "unknown")),
    }


def _build_record_from_catalog_entry(
    *,
    catalog_record: Any,
    cfg: DictConfig,
    episode_id: int,
    learning_potential: float,
    normalized_usefulness: float,
    metrics: dict[str, Any],
) -> ScenarioRecord:
    """Adapt one selected ScenarioNet record to the ACL buffer contract."""

    split = str(catalog_record.split)
    data_root = Path(str(cfg.paths.scenarionet_data_root)).expanduser()
    runtime_directory = data_root / "runtime" / split
    scenario_uid = str(catalog_record.scenario_uid)
    scenario_arm = str(catalog_record.primary_arm)
    usefulness = compute_scenario_usefulness(
        metrics,
        learning_potential=learning_potential,
    )
    return ScenarioRecord(
        scenario_id=scenario_uid,
        source=str(catalog_record.source),
        parent_id=None,
        scenario_description_path=str(data_root / str(catalog_record.relative_path)),
        scenario_description_hash=hashlib.sha256(scenario_uid.encode("utf-8")).hexdigest(),
        dataset_directory=str(runtime_directory),
        scenario_index=int(catalog_record.runtime_index),
        env_config={"provider": {"arm": scenario_arm}, "split": split},
        reset_seed=int(catalog_record.runtime_index),
        generator_arm=None,
        mutation_type=None,
        mutation_params=None,
        validation_status="valid",
        rule_criticality=float(usefulness.rule_criticality),
        learning_potential=float(usefulness.learning_potential),
        usefulness=float(usefulness.value),
        usefulness_norm=float(normalized_usefulness),
        rank=0,
        num_seen=1,
        last_seen_step=int(episode_id),
        num_children=0,
        metrics_summary=_metrics_summary(metrics, generator_config_id=scenario_arm),
        scenario_arm=scenario_arm,
    )


def _update_replay_record(
    record: ScenarioRecord,
    *,
    episode_id: int,
    learning_potential: float,
    normalized_usefulness: float,
    metrics: dict[str, Any],
) -> ScenarioRecord:
    usefulness = compute_scenario_usefulness(
        metrics,
        learning_potential=learning_potential,
    )
    record.rule_criticality = float(usefulness.rule_criticality)
    record.learning_potential = float(usefulness.learning_potential)
    record.usefulness = float(usefulness.value)
    record.usefulness_norm = float(normalized_usefulness)
    record.num_seen += 1
    record.last_seen_step = int(episode_id)
    record.metrics_summary = _metrics_summary(
        metrics,
        generator_config_id=str(
            record.scenario_arm or record.generator_arm or record.source
        ),
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


def _load_scenario_acl_resume_state(
    *,
    cfg: DictConfig,
    artifact_paths: dict[str, Path],
    scenario_cfg: Any,
    rng: np.random.Generator,
) -> tuple[ScenarioBuffer, ScenarioArmBandit, int, int, int, int, list[float]]:
    resume_cfg = cfg.checkpoint.get("resume", {})
    if not bool(resume_cfg.get("enabled", False)):
        return ScenarioBuffer(capacity=int(scenario_cfg.buffer_capacity)), ScenarioArmBandit(scenario_cfg.mab), 0, 0, 0, 0, []

    configured_run_dir = resume_cfg.get("run_dir")
    resume_run_dir = Path(str(configured_run_dir)) if configured_run_dir not in (None, "", "null") else artifact_paths["root"].parents[1]
    resume_root = resume_run_dir / "artifacts" / "curriculum"
    state_path = resume_root / "scenario_acl_state.json"
    buffer_path = resume_root / "scenario_buffer.json"
    if not state_path.exists() or not buffer_path.exists():
        raise FileNotFoundError(
            "Scenario ACL resume requires both state and buffer artifacts: "
            f"state={state_path}, buffer={buffer_path}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    buffer_payload = json.loads(buffer_path.read_text(encoding="utf-8"))
    buffer = ScenarioBuffer.from_state_dict(buffer_payload)
    bandit_payload = state.get("mab", {})
    if not isinstance(bandit_payload, dict):
        raise ValueError("Scenario ACL resume state has invalid MAB payload.")
    bandit = ScenarioArmBandit.from_state_dict(scenario_cfg.mab, bandit_payload)
    rng_state = state.get("rng_state")
    if bool(resume_cfg.get("restore_rng_state", True)):
        if not isinstance(rng_state, dict):
            raise ValueError("Scenario ACL resume state is missing RNG state.")
        rng.bit_generator.state = rng_state
    return (
        buffer,
        bandit,
        int(state.get("global_step", 0)),
        int(state.get("chunk_id", 0)),
        int(state.get("eval_id", 0)),
        int(state.get("episode_id", 0)),
        [float(value) for value in state.get("recent_usefulness", [])],
    )


def _choose_iteration_spec(
    *,
    cfg: DictConfig,
    curriculum_cfg: CurriculumConfig,
    arms: list[ScenarioArm],
    bandit: ScenarioArmBandit,
    buffer: ScenarioBuffer,
    rng: np.random.Generator,
    episode_id: int,
) -> IterationSpec:
    scenario_cfg = curriculum_cfg.scenario_acl
    can_exploit = (
        scenario_cfg.use_replay
        and len(buffer) >= int(scenario_cfg.warmup_buffer_size)
    )

    if can_exploit and rng.random() < float(scenario_cfg.exploit_probability):
        selection = buffer.sample_replay(
            rng=rng,
            current_step=episode_id,
            cfg=scenario_cfg.replay_sampling,
            use_staleness=bool(scenario_cfg.use_staleness),
        )
        return IterationSpec(
            mode="exploit_replay",
            arm_index=-1,
            arm_name=str(
                selection.record.scenario_arm
                or selection.record.generator_arm
                or selection.record.source
            ),
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
    return IterationSpec(
        mode="sample",
        arm_index=arm_index,
        arm_name=arm.name,
        arm_probabilities=[float(x) for x in arm_probs.tolist()],
        scenario_seed=0,
        train_env_overrides=arm.sample_env_overrides(),
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

    total_timesteps = int(cfg.experiment.total_timesteps)
    eval_interval = int(cfg.experiment.get("eval_interval", total_timesteps))
    if eval_interval <= 0:
        eval_interval = total_timesteps
    log_interval = int(cfg.experiment.get("log_interval", 1000))
    run_seed = int(cfg.seed)
    scenario_cfg = curriculum_cfg.scenario_acl
    artifact_paths = _scenario_acl_artifact_paths(paths.artifacts_dir)
    rng = np.random.default_rng(run_seed)

    if str(cfg.env.get("name", "")).strip().lower() != "scenarionet":
        raise ValueError("scenario_acl requires env=scenarionet.")
    arms: list[ScenarioArm] = list(build_default_scenario_arms())
    semantic_env_overrides = _semantic_catalog_overrides(cfg)
    expected_arms = int(scenario_cfg.mab.num_arms)
    if len(arms) != expected_arms:
        raise ValueError(
            "Configured scenario_acl.mab.num_arms does not match the built-in arm set: "
            f"expected {expected_arms}, got {len(arms)}."
        )

    (
        buffer,
        bandit,
        current_global_step,
        current_chunk_id,
        current_eval_id,
        current_episode_id,
        recent_usefulness,
    ) = _load_scenario_acl_resume_state(
        cfg=cfg,
        artifact_paths=artifact_paths,
        scenario_cfg=scenario_cfg,
        rng=rng,
    )
    generate_count = 0
    replay_count = 0

    # Arms are assigned immediately before every reset. Construct an
    # unfiltered provider once per evaluation chunk so no prior choice leaks.
    env = build_env(cfg, semantic_env_overrides)
    seed_env_spaces(env, run_seed)
    train_env_count = count_envs(env)

    preprocessor = build_preprocessor(cfg)
    adapter = build_adapter(cfg, adapter_space_kwargs(env.action_space))
    resume_cfg = cfg.checkpoint.get("resume", {})
    resume_enabled = bool(resume_cfg.get("enabled", False))
    resume_run_dir_cfg = resume_cfg.get("run_dir")
    resume_run_dir = Path(str(resume_run_dir_cfg)) if resume_run_dir_cfg not in (None, "", "null") else paths.run_dir
    resume_checkpoint_stem = resume_run_dir / "checkpoints" / str(resume_cfg.get("checkpoint_name", "latest"))
    resume_checkpoint_zip = resume_checkpoint_stem.with_suffix(".zip")
    if resume_enabled:
        if not resume_checkpoint_zip.exists():
            raise FileNotFoundError(f"Scenario ACL resume checkpoint is missing: {resume_checkpoint_zip}")
        planner = load_planner(cfg, checkpoint_path=str(resume_checkpoint_zip), env=env)
    else:
        planner = build_planner(cfg, env, seed=run_seed)
    planner_device = str(getattr(planner, "device", cfg.device))
    ema_alpha_cfg = (
        float(cfg.agent.planner.algorithm.get("monitor_ema_alpha", 0.1))
        if hasattr(cfg, "agent")
        else 0.1
    )
    agent = Agent(preprocessor=preprocessor, planner=planner, adapter=adapter, ema_alpha=ema_alpha_cfg)
    if resume_enabled:
        agent.load_adapter(checkpoint_path=resume_checkpoint_zip, strict=True)

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
            ("Scenario ACL selection", "per episode"),
        ],
    )

    try:
        remaining = max(0, total_timesteps - current_global_step)
        while remaining > 0:
            current_chunk_id += 1
            if current_chunk_id > 1:
                env = build_env(cfg, semantic_env_overrides)
                seed_env_spaces(env, run_seed + 400_000 + current_chunk_id)
                set_planner_env_if_compatible(planner, env)

            chunk_steps = min(eval_interval, remaining)
            steps_start = current_global_step

            curriculum_logger.info(
                "Scenario ACL chunk started | chunk_id=%d | arm_selection=%s",
                current_chunk_id,
                "per_episode",
            )
            log_event(
                paths.events_log_path,
                "scenario_acl_chunk_started",
                chunk_id=current_chunk_id,
                arm_selection="per_episode",
                arm_probabilities=[float(value) for value in bandit.probabilities().tolist()],
                mab=bandit.state_dict(),
            )

            episode_specs: list[IterationSpec] = []
            episode_outcomes: list[EpisodeAclOutcome] = []
            buffer_action = "none"
            scenario_record_id: str | None = None
            buffer_actions: dict[str, int] = {}

            def choose_acl_episode(training_env: Any, episode_index: int) -> None:
                base_env = _base_env(training_env)
                spec = _choose_iteration_spec(
                    cfg=cfg,
                    curriculum_cfg=curriculum_cfg,
                    arms=arms,
                    bandit=bandit,
                    buffer=buffer,
                    rng=rng,
                    episode_id=current_episode_id + 1,
                )
                if not _is_replay_iteration(spec):
                    excluded_uids = {
                        str(record.scenario_id) for record in buffer.records()
                    }
                    provider = getattr(base_env, "scenario_provider", None)
                    has_candidate = getattr(provider, "has_candidate", None)
                    source = _selection_source_override(spec)
                    if (
                        callable(has_candidate)
                        and not has_candidate(
                            split=str(base_env.split),
                            source=source,
                            arm=spec.arm_name,
                            excluded_scenario_uids=excluded_uids,
                        )
                    ):
                        if not len(buffer):
                            raise RuntimeError(
                                "Selected ACL arm has no fresh scenarios and the replay buffer is empty."
                            )
                        replay = buffer.sample_replay(
                            rng=rng,
                            current_step=current_episode_id + 1,
                            cfg=scenario_cfg.replay_sampling,
                            use_staleness=bool(scenario_cfg.use_staleness),
                        )
                        spec = IterationSpec(
                            mode="exploit_replay_fallback",
                            arm_index=-1,
                            arm_name=str(
                                replay.record.scenario_arm
                                or replay.record.generator_arm
                                or replay.record.source
                            ),
                            arm_probabilities=[],
                            scenario_seed=int(replay.record.reset_seed),
                            train_env_overrides=None,
                            replay_record=replay.record,
                            replay_probabilities=replay.probabilities,
                        )
                episode_specs.append(spec)
                if _is_replay_iteration(spec):
                    if spec.replay_record is None:
                        raise RuntimeError("Replay selection requires a scenario record.")
                    base_env.scenario_arm = spec.replay_record.scenario_arm
                    base_env.scenario_source = spec.replay_record.source
                    base_env.scenario_excluded_uids = set()
                else:
                    base_env.scenario_arm = spec.arm_name
                    base_env.scenario_source = _selection_source_override(spec)
                    base_env.scenario_excluded_uids = {
                        str(record.scenario_id) for record in buffer.records()
                    }
                base_env._acl_episode_selection = {
                    "origin": (
                        "fallback_replay_exhausted_arm"
                        if spec.mode == "exploit_replay_fallback"
                        else "scenario_buffer"
                        if _is_replay_iteration(spec)
                        else "new"
                    ),
                    "arm": spec.arm_name,
                    "arm_index": spec.arm_index,
                    "selection_probability": (
                        float(spec.arm_probabilities[spec.arm_index])
                        if not _is_replay_iteration(spec)
                        else None
                    ),
                }

            def train_reset_seed_for_episode(episode_index: int) -> int | None:
                spec = episode_specs[episode_index]
                if _is_replay_iteration(spec):
                    if spec.replay_record is None:
                        raise RuntimeError("Replay selection requires a scenario record.")
                    return int(spec.replay_record.reset_seed)
                # Let the provider sample a fresh catalog record filtered by
                # the arm selected immediately before this reset.
                return None

            def collect_catalog_episode(
                training_env: Any,
                episode_index: int,
                episode_metrics: dict[str, Any],
            ) -> None:
                nonlocal current_episode_id, buffer_action, scenario_record_id
                base_env = _base_env(training_env)
                record = getattr(base_env, "current_scenario_record", None)
                spec = episode_specs[episode_index - 1]
                selection = getattr(base_env, "_acl_episode_selection", {})
                episode_usefulness: float | None = None
                normalized_episode_usefulness: float | None = None
                try:
                    episode_usefulness = compute_learning_potential(
                        episode_metrics,
                        planner_name=str(cfg.agent.planner.algorithm.name),
                    )
                except ValueError:
                    # Before learning_starts no critic update exists yet; the
                    # episode remains valid but provides no MAB feedback.
                    pass
                if episode_usefulness is not None:
                    normalized_episode_usefulness = _normalize_learning_potential(
                        episode_usefulness, recent_usefulness
                    )
                    recent_usefulness.append(episode_usefulness)
                    max_recent = int(scenario_cfg.recent_window_size)
                    if len(recent_usefulness) > max_recent:
                        del recent_usefulness[:-max_recent]
                    if bool(scenario_cfg.use_mab) and not _is_replay_iteration(spec):
                        bandit.update(
                            arm_index=spec.arm_index,
                            normalized_usefulness=normalized_episode_usefulness,
                            selection_probability=float(
                                spec.arm_probabilities[spec.arm_index]
                            ),
                        )
                    selection["usefulness"] = episode_usefulness
                    selection["usefulness_norm"] = normalized_episode_usefulness
                    episode_metrics["usefulness"] = episode_usefulness
                    episode_metrics["usefulness_norm"] = normalized_episode_usefulness
                current_episode_id += 1
                if bool(scenario_cfg.use_scenario_buffer):
                    learning_value = episode_usefulness or 0.0
                    normalized_value = normalized_episode_usefulness or 0.0
                    if _is_replay_iteration(spec):
                        if spec.replay_record is None:
                            raise RuntimeError("Replay selection requires a scenario record.")
                        updated_record = _update_replay_record(
                            spec.replay_record,
                            episode_id=current_episode_id,
                            learning_potential=learning_value,
                            normalized_usefulness=normalized_value,
                            metrics=_episode_record_metrics(episode_metrics),
                        )
                        buffer.update(updated_record)
                        buffer_action = "updated"
                        scenario_record_id = updated_record.scenario_id
                    elif record is not None:
                        buffer_record = _build_record_from_catalog_entry(
                            catalog_record=record,
                            cfg=cfg,
                            episode_id=current_episode_id,
                            learning_potential=learning_value,
                            normalized_usefulness=normalized_value,
                            metrics=_episode_record_metrics(episode_metrics),
                        )
                        inserted = buffer.insert(buffer_record)
                        buffer_action = "inserted" if inserted else "rejected"
                        scenario_record_id = buffer_record.scenario_id
                    _append_jsonl(
                        artifact_paths["buffer_events"],
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "chunk_id": current_chunk_id,
                            "episode_id": current_episode_id,
                            "action": buffer_action,
                            "scenario_id": scenario_record_id,
                        },
                    )
                    buffer_actions[buffer_action] = buffer_actions.get(buffer_action, 0) + 1
                log_event(
                    paths.events_log_path,
                    "scenario_acl_episode_ended",
                    chunk_id=current_chunk_id,
                    episode_id=current_episode_id,
                    episode_index=episode_index,
                    origin=selection.get("origin"),
                    arm_name=_short_arm_name(
                        getattr(record, "primary_arm", selection.get("arm"))
                    ),
                    source=getattr(record, "source", None),
                    selection_probability=selection.get("selection_probability"),
                    usefulness=episode_usefulness,
                    usefulness_norm=normalized_episode_usefulness,
                    metrics=episode_metrics,
                )
                episode_outcomes.append(
                    EpisodeAclOutcome(
                        spec=spec,
                        record=record,
                        metrics=dict(episode_metrics),
                        learning_potential=episode_usefulness,
                        usefulness_norm=normalized_episode_usefulness,
                    )
                )

            def episode_context(
                training_env: Any, _info: dict[str, Any]
            ) -> dict[str, Any] | None:
                base_env = _base_env(training_env)
                record = getattr(base_env, "current_scenario_record", None)
                selection = getattr(base_env, "_acl_episode_selection", {})
                return {
                    "arm": _short_arm_name(
                        getattr(record, "primary_arm", selection.get("arm"))
                    ),
                    "source": getattr(record, "source", None),
                    "origin": (
                        f"{selection.get('origin')} |"
                        if selection.get("usefulness") is not None
                        else selection.get("origin")
                    ),
                    "U": _format_optional_float(selection.get("usefulness")),
                    "U_norm": _format_optional_float(
                        selection.get("usefulness_norm")
                    ),
                }

            def mab_monitor_rows() -> list[tuple[str, str]]:
                probabilities = bandit.probabilities()
                labels = ", ".join(
                    f"{_short_arm_name(arm.name)}={probability:.3f}"
                    for arm, probability in zip(arms, probabilities, strict=True)
                )
                return [("ACL arm probabilities", labels)]

            chunk_summary = agent.train(
                env=env,
                chunk_timesteps=chunk_steps,
                global_total_timesteps=total_timesteps,
                global_steps_done=current_global_step,
                stage_name="scenario_acl_episode_sampling",
                deterministic=False,
                log_interval=log_interval,
                reset_seed_fn=train_reset_seed_for_episode,
                episode_end_callback=collect_catalog_episode,
                before_episode_reset_callback=choose_acl_episode,
                episode_context_callback=episode_context,
                monitor_extra_rows_callback=mab_monitor_rows,
            )
            actual_chunk_steps = int(chunk_summary.get("chunk_steps_actual", chunk_steps))
            current_global_step = min(total_timesteps, current_global_step + actual_chunk_steps)
            remaining = max(0, total_timesteps - current_global_step)
            episode_stats = _summarize_episode_acl_outcomes(episode_outcomes)
            chunk_generate_count = int(episode_stats["generate_count"])
            chunk_replay_count = int(episode_stats["replay_count"])
            episode_modes = list(episode_stats["modes"])
            episode_arms = list(episode_stats["arms"])
            chunk_mode = str(episode_stats["mode"])
            generate_count += chunk_generate_count
            replay_count += chunk_replay_count
            chunk_stage = "scenario_acl_episode_sampling"
            recorder.append_row(
                "train_chunks.csv",
                {
                    **base_csv_fields,
                    "chunk_id": current_chunk_id,
                    "stage": chunk_stage,
                    "stage_index": -1,
                    "acl_chunk_mode": chunk_mode,
                    "acl_episode_modes": ",".join(episode_modes),
                    "acl_episode_arms": ",".join(episode_arms),
                    "acl_generate_episodes": chunk_generate_count,
                    "acl_replay_episodes": chunk_replay_count,
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
            eval_env_overrides = apply_eval_scenario_seed_split(
                base_run_seed=run_seed,
                eval_env_overrides=None,
                cfg=cfg,
                n_eval_episodes=int(cfg.experiment.eval_episodes),
                split="validation",
            )
            eval_env_overrides.update(semantic_env_overrides)
            eval_base_seed = None
            eval_env = build_env(cfg, eval_env_overrides)
            seed_env_spaces(eval_env, run_seed + 500_000 + current_chunk_id)

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
                before_episode_reset_callback=_configure_waymo_eval_episode,
            )
            eval_env.close()
            Path(f"{eval_snapshot_stem}.zip").unlink(missing_ok=True)
            Agent.adapter_checkpoint_path(eval_snapshot_stem).unlink(missing_ok=True)

            try:
                chunk_learning_potential = compute_learning_potential(
                    chunk_summary,
                    planner_name=str(cfg.agent.planner.algorithm.name),
                )
            except ValueError:
                chunk_learning_potential = None

            episode_feedback = [
                (float(outcome.learning_potential), float(outcome.usefulness_norm))
                for outcome in episode_outcomes
                if outcome.learning_potential is not None
                and outcome.usefulness_norm is not None
            ]
            if episode_feedback:
                learning_potential = float(np.mean([item[0] for item in episode_feedback]))
                normalized_usefulness = float(np.mean([item[1] for item in episode_feedback]))
            elif chunk_learning_potential is not None:
                learning_potential = float(chunk_learning_potential)
                normalized_usefulness = _normalize_learning_potential(
                    learning_potential,
                    recent_usefulness,
                )
            else:
                learning_potential = 0.0
                normalized_usefulness = 0.0

            # The buffer was updated synchronously in each episode callback;
            # only its durable snapshot is batched by evaluation chunk.
            _persist_buffer_state(path=artifact_paths["buffer"], buffer=buffer)

            recorder.append_row(
                "evals.csv",
                {
                    **base_csv_fields,
                    "eval_id": current_eval_id,
                    "eval_type": "intermediate",
                    "scenario_set": f"validation_{_WAYMO_STRATIFIED_SET}",
                    "chunk_id": current_chunk_id,
                    "stage": (
                        chunk_stage
                    ),
                    "stage_index": -1,
                    "acl_chunk_mode": chunk_mode,
                    "acl_episode_modes": ",".join(episode_modes),
                    "acl_episode_arms": ",".join(episode_arms),
                    "acl_generate_episodes": chunk_generate_count,
                    "acl_replay_episodes": chunk_replay_count,
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
                    "next_stage": chunk_stage,
                },
            )

            history_payload = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "chunk_id": current_chunk_id,
                "global_step": current_global_step,
                "mode": chunk_mode,
                "arm_index": -1,
                "arm_name": chunk_stage,
                "scenario_seed": None,
                "selection_probability": None,
                "arm_probabilities": [float(value) for value in bandit.probabilities().tolist()],
                "replay_probabilities": None,
                "replay_scenario_id": None,
                "learning_potential": float(learning_potential),
                "normalized_usefulness": float(normalized_usefulness),
                "env_overrides": None,
                "scenario_record_id": scenario_record_id,
                "buffer_action": buffer_action,
                "buffer_actions": dict(buffer_actions),
                "acl_generate_episodes": chunk_generate_count,
                "acl_replay_episodes": chunk_replay_count,
                "acl_episode_modes": episode_modes,
                "acl_episode_arms": episode_arms,
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
            log_event(
                paths.events_log_path,
                "scenario_acl_chunk_finished",
                chunk_id=current_chunk_id,
                global_step=current_global_step,
                usefulness=float(learning_potential),
                usefulness_norm=float(normalized_usefulness),
                mode=chunk_mode,
                episode_modes=episode_modes,
                episode_arms=episode_arms,
                generate_episodes=int(chunk_generate_count),
                replay_episodes=int(chunk_replay_count),
                buffer_actions=dict(buffer_actions),
                buffer_size=len(buffer),
            )
            artifact_paths["state"].write_text(
                json.dumps(
                    {
                        "global_step": int(current_global_step),
                        "chunk_id": int(current_chunk_id),
                        "eval_id": int(current_eval_id),
                        "recent_usefulness": [float(x) for x in recent_usefulness],
                        "last_mode": chunk_mode,
                        "last_arm_name": chunk_stage,
                        "episode_id": int(current_episode_id),
                        "last_arm_index": -1,
                        "last_scenario_seed": None,
                        "last_replay_scenario_id": None,
                        "buffer_size": len(buffer),
                        "generate_count": int(generate_count),
                        "replay_count": int(replay_count),
                        "chunk_generate_count": int(chunk_generate_count),
                        "chunk_replay_count": int(chunk_replay_count),
                        "buffer_actions": dict(buffer_actions),
                        "acl_episode_modes": episode_modes,
                        "acl_episode_arms": episode_arms,
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
                        "rng_state": rng.bit_generator.state,
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if bool(cfg.checkpoint.get("save_latest_each_chunk", True)):
                agent.save(paths.latest_checkpoint_stem)

            train_logger.info(
                "Scenario ACL chunk finished | chunk_id=%d | mode=%s | arms=%s | "
                "generate_episodes=%d | replay_episodes=%d | global_step=%d | "
                "usefulness=%.4f | usefulness_norm=%.4f | buffer_size=%d",
                current_chunk_id,
                chunk_mode,
                ",".join(episode_arms),
                chunk_generate_count,
                chunk_replay_count,
                current_global_step,
                learning_potential,
                normalized_usefulness,
                len(buffer),
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
        final_eval_env_overrides.update(semantic_env_overrides)
        final_eval_base_seed = None
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
            scenario_set=f"test_{_WAYMO_STRATIFIED_SET}",
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
            before_episode_reset_callback=_configure_waymo_eval_episode,
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
                "scenario_set": f"test_{_WAYMO_STRATIFIED_SET}",
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
