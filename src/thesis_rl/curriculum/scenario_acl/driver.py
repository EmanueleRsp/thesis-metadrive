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
from thesis_rl.curriculum.scenario_acl.arms import (
    GeneratorArm,
    ScenarioArm,
    SCENARIO_ARM_NAMES,
    build_default_generator_arms,
    build_default_scenario_arms,
)
from thesis_rl.curriculum.scenario_acl.buffer import ScenarioBuffer
from thesis_rl.curriculum.scenario_acl.mab import GeneratorArmBandit
from thesis_rl.curriculum.scenario_acl.record import ScenarioRecord
from thesis_rl.curriculum.scenario_acl.scenario_env import build_scenario_replay_env
from thesis_rl.curriculum.scenario_acl.usefulness import (
    compute_learning_potential,
    compute_scenario_usefulness,
)
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


def _is_replay_iteration(spec: IterationSpec) -> bool:
    return spec.mode == "exploit_replay"


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


def _short_scenario_id(value: object | None) -> str | None:
    if value is None:
        return None
    scenario_id = str(value)
    return scenario_id.rsplit("-", maxsplit=1)[-1]


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
    sorted_values = sorted(rank_set, reverse=True)
    rank = sorted_values.index(float(current_value)) + 1
    return 1.0 - ((rank - 1) / float(len(rank_set) - 1))


def _sample_iteration_env(
    *,
    arm: GeneratorArm | ScenarioArm,
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


def _build_record_from_catalog_entry(
    *,
    catalog_record: Any,
    cfg: DictConfig,
    chunk_id: int,
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
        last_seen_step=int(chunk_id),
        num_children=0,
        metrics_summary=_metrics_summary(metrics, generator_config_id=scenario_arm),
        scenario_arm=scenario_arm,
    )


def _build_record_from_export(
    *,
    export_path: str,
    env_overrides: dict[str, object],
    chunk_id: int,
    scenario_seed: int,
    arm_name: str,
    learning_potential: float,
    normalized_usefulness: float,
    metrics: dict[str, Any],
    source: str = "generate",
    parent_id: str | None = None,
) -> ScenarioRecord:
    export_file = Path(export_path)
    usefulness = compute_scenario_usefulness(
        metrics,
        learning_potential=learning_potential,
    )
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
        rule_criticality=float(usefulness.rule_criticality),
        learning_potential=float(usefulness.learning_potential),
        usefulness=float(usefulness.value),
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
    record.last_seen_step = int(chunk_id)
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
) -> tuple[ScenarioBuffer, GeneratorArmBandit, int, int, int, list[float]]:
    resume_cfg = cfg.checkpoint.get("resume", {})
    if not bool(resume_cfg.get("enabled", False)):
        return ScenarioBuffer(capacity=int(scenario_cfg.buffer_capacity)), GeneratorArmBandit(scenario_cfg.mab), 0, 0, 0, []

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
    bandit = GeneratorArmBandit.from_state_dict(scenario_cfg.mab, bandit_payload)
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
        [float(value) for value in state.get("recent_usefulness", [])],
    )


def _choose_iteration_spec(
    *,
    cfg: DictConfig,
    curriculum_cfg: CurriculumConfig,
    arms: list[GeneratorArm | ScenarioArm],
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
    scenario_seed, train_env_overrides = _sample_iteration_env(
        arm=arm,
        rng=rng,
        chunk_id=chunk_id,
        cfg=cfg,
    )
    return IterationSpec(
        mode="sample" if scenario_cfg.arm_space == "scenario" else "generate",
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

    total_timesteps = int(cfg.experiment.total_timesteps)
    eval_interval = int(cfg.experiment.get("eval_interval", total_timesteps))
    if eval_interval <= 0:
        eval_interval = total_timesteps
    log_interval = int(cfg.experiment.get("log_interval", 1000))
    run_seed = int(cfg.seed)
    scenario_cfg = curriculum_cfg.scenario_acl
    artifact_paths = _scenario_acl_artifact_paths(paths.artifacts_dir)
    rng = np.random.default_rng(run_seed)

    if scenario_cfg.arm_space == "scenario":
        if str(cfg.env.get("name", "")).strip().lower() != "scenarionet":
            raise ValueError(
                "scenario_acl.arm_space='scenario' requires env=scenarionet."
            )
        arms: list[GeneratorArm | ScenarioArm] = list(build_default_scenario_arms())
        semantic_env_overrides = _semantic_catalog_overrides(cfg)
    else:
        arms = list(build_default_generator_arms())
        semantic_env_overrides: dict[str, object] = {}
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
        recent_usefulness,
    ) = _load_scenario_acl_resume_state(
        cfg=cfg,
        artifact_paths=artifact_paths,
        scenario_cfg=scenario_cfg,
        rng=rng,
    )
    generate_count = 0
    replay_count = 0

    first_spec = _choose_iteration_spec(
        cfg=cfg,
        curriculum_cfg=curriculum_cfg,
        arms=arms,
        bandit=bandit,
        buffer=buffer,
        rng=rng,
        chunk_id=current_chunk_id + 1,
    )
    if scenario_cfg.arm_space == "scenario":
        # Arms are assigned immediately before every reset below.  Construct
        # an unfiltered provider here so a previous episode cannot constrain
        # the next arm or source.
        env = build_env(cfg, semantic_env_overrides)
    elif not _is_replay_iteration(first_spec):
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
            ("Scenario ACL mode", scenario_cfg.mode),
        ],
    )

    try:
        remaining = max(0, total_timesteps - current_global_step)
        next_spec = first_spec
        while remaining > 0:
            current_chunk_id += 1
            current_spec = next_spec
            current_catalog_records: list[Any] = []
            if current_chunk_id > 1:
                if scenario_cfg.arm_space == "scenario":
                    env = build_env(cfg, semantic_env_overrides)
                    seed_env_spaces(env, run_seed + 400_000 + current_chunk_id)
                elif not _is_replay_iteration(current_spec):
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
            if not _is_replay_iteration(current_spec):
                generate_count += 1
            else:
                replay_count += 1

            curriculum_logger.info(
                "Scenario ACL chunk started | chunk_id=%d | arm_selection=per_episode",
                current_chunk_id,
            )
            log_event(
                paths.events_log_path,
                "scenario_acl_chunk_started",
                chunk_id=current_chunk_id,
                arm_selection="per_episode" if scenario_cfg.arm_space == "scenario" else "per_chunk",
                arm_probabilities=[float(value) for value in bandit.probabilities().tolist()],
                mab=bandit.state_dict(),
            )

            episode_specs: list[IterationSpec] = []

            def choose_acl_episode(training_env: Any, episode_index: int) -> None:
                if scenario_cfg.arm_space != "scenario":
                    return
                spec = _choose_iteration_spec(
                    cfg=cfg,
                    curriculum_cfg=curriculum_cfg,
                    arms=arms,
                    bandit=bandit,
                    buffer=buffer,
                    rng=rng,
                    chunk_id=current_chunk_id,
                )
                episode_specs.append(spec)
                base_env = _base_env(training_env)
                if _is_replay_iteration(spec):
                    if spec.replay_record is None:
                        raise RuntimeError("Replay selection requires a scenario record.")
                    base_env.scenario_arm = spec.replay_record.scenario_arm
                    base_env.scenario_source = spec.replay_record.source
                else:
                    base_env.scenario_arm = spec.arm_name
                    base_env.scenario_source = _selection_source_override(spec)
                base_env._acl_episode_selection = {
                    "origin": "scenario_buffer" if _is_replay_iteration(spec) else "new",
                    "arm": spec.arm_name,
                    "arm_index": spec.arm_index,
                    "selection_probability": (
                        float(spec.arm_probabilities[spec.arm_index])
                        if not _is_replay_iteration(spec)
                        else None
                    ),
                }

            def train_reset_seed_for_episode(episode_index: int) -> int | None:
                if scenario_cfg.arm_space == "scenario":
                    spec = episode_specs[episode_index]
                    if _is_replay_iteration(spec):
                        if spec.replay_record is None:
                            raise RuntimeError("Replay selection requires a scenario record.")
                        return int(spec.replay_record.reset_seed)
                    # Let the provider sample a fresh catalog record filtered
                    # by the arm selected immediately before this reset.
                    return None
                if current_spec.mode == "sample":
                    # The ScenarioNet provider must choose the record by arm;
                    # reset(seed=...) would bypass it and select a raw index.
                    return None
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

            def collect_catalog_episode(
                training_env: Any,
                episode_index: int,
                episode_metrics: dict[str, Any],
            ) -> None:
                if scenario_cfg.arm_space != "scenario":
                    return
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
                log_event(
                    paths.events_log_path,
                    "scenario_acl_episode_ended",
                    chunk_id=current_chunk_id,
                    episode_index=episode_index,
                    origin=selection.get("origin"),
                    arm_name=_short_arm_name(
                        getattr(record, "primary_arm", selection.get("arm"))
                    ),
                    source=getattr(record, "source", None),
                    scenario_id=_short_scenario_id(
                        getattr(
                            record,
                            "scenario_uid",
                            getattr(record, "scenario_id", None),
                        )
                    ),
                    selection_probability=selection.get("selection_probability"),
                    usefulness=episode_usefulness,
                    usefulness_norm=normalized_episode_usefulness,
                    metrics=episode_metrics,
                )
                if record is not None and not _is_replay_iteration(spec):
                    current_catalog_records.append(record)

            def episode_context(
                training_env: Any, _info: dict[str, Any]
            ) -> dict[str, Any] | None:
                if scenario_cfg.arm_space != "scenario":
                    return None
                base_env = _base_env(training_env)
                record = getattr(base_env, "current_scenario_record", None)
                selection = getattr(base_env, "_acl_episode_selection", {})
                return {
                    "arm": _short_arm_name(
                        getattr(record, "primary_arm", selection.get("arm"))
                    ),
                    "source": getattr(record, "source", None),
                    "scenario_id": _short_scenario_id(
                        getattr(
                            record,
                            "scenario_uid",
                            getattr(record, "scenario_id", None),
                        )
                    ),
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
                stage_name="scenario_acl_episode_sampling" if scenario_cfg.arm_space == "scenario" else (
                    current_spec.arm_name
                    if not _is_replay_iteration(current_spec)
                    else f"replay:{current_spec.arm_name}"
                ),
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
            recorder.append_row(
                "train_chunks.csv",
                {
                    **base_csv_fields,
                    "chunk_id": current_chunk_id,
                    "stage": (
                        current_spec.arm_name
                        if not _is_replay_iteration(current_spec)
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
            if scenario_cfg.arm_space == "scenario":
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
            elif not _is_replay_iteration(current_spec):
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
                before_episode_reset_callback=(
                    _configure_waymo_eval_episode
                    if scenario_cfg.arm_space == "scenario"
                    else None
                ),
            )
            eval_env.close()
            Path(f"{eval_snapshot_stem}.zip").unlink(missing_ok=True)
            Agent.adapter_checkpoint_path(eval_snapshot_stem).unlink(missing_ok=True)

            learning_potential = compute_learning_potential(
                chunk_summary,
                planner_name=str(cfg.agent.planner.algorithm.name),
            )
            normalized_usefulness = _normalize_learning_potential(
                learning_potential,
                recent_usefulness,
            )

            export_path: str | None = None
            buffer_action = "none"
            scenario_record_id: str | None = None
            if not _is_replay_iteration(current_spec):
                if current_spec.train_env_overrides is None:
                    raise RuntimeError("Generate iteration requires training env overrides.")
                if scenario_cfg.arm_space == "scenario":
                    if not current_catalog_records:
                        raise RuntimeError(
                            "Semantic Scenario ACL did not expose completed catalog records."
                        )
                    if bool(scenario_cfg.use_scenario_buffer):
                        for catalog_record in current_catalog_records:
                            record = _build_record_from_catalog_entry(
                                catalog_record=catalog_record,
                                cfg=cfg,
                                chunk_id=current_chunk_id,
                                learning_potential=learning_potential,
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
                                    "usefulness": float(learning_potential),
                                },
                            )
                elif bool(scenario_cfg.use_scenario_buffer):
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
                        learning_potential=learning_potential,
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
                            "usefulness": float(learning_potential),
                        },
                    )
            else:
                if current_spec.replay_record is None:
                    raise RuntimeError("Replay iteration requires a replay record.")
                updated_record = _update_replay_record(
                    current_spec.replay_record,
                    chunk_id=current_chunk_id,
                    learning_potential=learning_potential,
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
                        "usefulness": float(learning_potential),
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
                        f"validation_{_WAYMO_STRATIFIED_SET}"
                        if scenario_cfg.arm_space == "scenario"
                        else "scenario_acl_generate_eval"
                        if current_spec.mode == "generate"
                        else "scenario_acl_arm_eval"
                        if current_spec.mode == "sample"
                        else "scenario_acl_replay_eval"
                    ),
                    "chunk_id": current_chunk_id,
                    "stage": (
                        current_spec.arm_name
                        if not _is_replay_iteration(current_spec)
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
                    if not _is_replay_iteration(current_spec)
                    else None
                ),
                "arm_probabilities": current_spec.arm_probabilities,
                "replay_probabilities": current_spec.replay_probabilities,
                "replay_scenario_id": (
                    current_spec.replay_record.scenario_id
                    if current_spec.replay_record is not None
                    else None
                ),
                "learning_potential": float(learning_potential),
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
            log_event(
                paths.events_log_path,
                "scenario_acl_chunk_finished",
                chunk_id=current_chunk_id,
                global_step=current_global_step,
                usefulness=float(learning_potential),
                usefulness_norm=float(normalized_usefulness),
                buffer_size=len(buffer),
            )
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
                "Scenario ACL iteration finished | chunk_id=%d | mode=%s | arm=%s | "
                "scenario_seed=%d | global_step=%d | usefulness=%.4f | usefulness_norm=%.4f | "
                "buffer_size=%d",
                current_chunk_id,
                current_spec.mode,
                current_spec.arm_name,
                current_spec.scenario_seed,
                current_global_step,
                learning_potential,
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
        if scenario_cfg.arm_space == "scenario":
            final_eval_env_overrides.update(semantic_env_overrides)
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
            before_episode_reset_callback=(
                _configure_waymo_eval_episode
                if scenario_cfg.arm_space == "scenario"
                else None
            ),
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
