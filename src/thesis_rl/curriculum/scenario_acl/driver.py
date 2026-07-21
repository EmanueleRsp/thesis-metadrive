from __future__ import annotations

import json
import hashlib
import logging
import os
import pickle
import random
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from thesis_rl.agent.agent import Agent
from thesis_rl.agent.planners.core.utils import count_envs
from thesis_rl.contracts.reward_semantics import build_reward_semantics_identity
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
from thesis_rl.curriculum.scenario_acl.vectorized import (
    AclCompletion,
    AclEpisodeAccumulator,
    AclSlotSelection,
    AclVectorSelectionCoordinator,
    AclVectorState,
    AclVectorTransaction,
    load_acl_vector_state,
    retain_unresolved_acl_completions,
    save_acl_vector_state,
)
from thesis_rl.runtime.wiring.builders import train_num_envs
from thesis_rl.scenarios.catalog import read_scenario_catalog
from thesis_rl.runtime.execution.seeding import (
    apply_eval_scenario_seed_split,
    seed_env_spaces,
)
from thesis_rl.runtime.io.console import print_evaluation_summary, print_run_setup
from thesis_rl.runtime.io.csv_recorder import CSVRecorder
from thesis_rl.runtime.io.eval_artifacts import maybe_build_live_final_eval_recorder_factory
from thesis_rl.runtime.io.metadata import update_run_metadata
from thesis_rl.runtime.io.run_logging import log_event
from thesis_rl.runtime.async_evaluation import AsyncEvaluationManager, EvaluationJob
from thesis_rl.sb3_extensions.replay import resolve_transition_replay_config
from thesis_rl.runtime.wiring.builders import (
    adapter_space_kwargs,
    build_adapter,
    build_eval_env,
    build_env,
    build_planner,
    build_preprocessor,
    load_planner,
    merge_env_config_with_overrides,
    set_planner_env_if_compatible,
    build_train_env,
    evaluation_num_workers,
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


def _save_acl_replay_buffer(planner: Any, path: Path) -> bool:
    """Persist the learner replay buffer for the custom ACL training path."""

    save = getattr(planner, "save_replay_buffer", None)
    if save is None:
        raise TypeError("ACL transition replay persistence requires planner.save_replay_buffer().")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        saved = bool(save(str(temporary_path)))
        if not saved or not temporary_path.is_file():
            raise RuntimeError(f"Planner did not publish replay buffer: {temporary_path}")
        os.replace(temporary_path, path)
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_acl_replay_buffer(planner: Any, path: Path) -> bool:
    """Load the persisted replay buffer before ACL sampling resumes."""

    load = getattr(planner, "load_replay_buffer", None)
    if load is None:
        raise TypeError("ACL transition replay resume requires planner.load_replay_buffer().")
    if not path.is_file():
        raise FileNotFoundError(f"ACL replay buffer is missing: {path}")
    return bool(load(str(path)))


def _save_acl_rng_state(path: Path) -> None:
    """Persist process RNGs used by the learner and environment wrappers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        pickle.dump(payload, handle)
    os.replace(temporary_path, path)


def _load_acl_rng_state(path: Path) -> None:
    """Restore process RNGs saved by the ACL driver."""

    if not path.is_file():
        raise FileNotFoundError(f"ACL RNG state is missing: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if "cuda" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda"])


def _save_acl_checkpoint_pair(
    *,
    path: Path,
    checkpoint_name: str,
    replay_name: str,
    training_timestep: int,
) -> None:
    """Publish the model/replay pair identity used by ACL resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_id": hashlib.sha256(
            f"{checkpoint_name}:{replay_name}:{training_timestep}".encode()
        ).hexdigest(),
        "training_timestep": int(training_timestep),
        "replay_segment_id": 0,
        "model_path": f"{checkpoint_name}.zip",
        "replay_path": replay_name,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
        "generate_count": sum(1 for outcome in outcomes if not _is_replay_iteration(outcome.spec)),
        "replay_count": sum(1 for outcome in outcomes if _is_replay_iteration(outcome.spec)),
        "modes": modes,
        "arms": arms,
        "mode": (modes[0] if len(modes) == 1 else "mixed" if modes else "none"),
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
    selected = [
        str(source) for source, probability in probabilities.items() if float(probability) == 1.0
    ]
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
        "vector_state": root / "scenario_acl_vector_state.json",
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
    episode_id: int | None = None,
    chunk_id: int | None = None,
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
    effective_episode_id = int(episode_id if episode_id is not None else (chunk_id or 0))
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
        last_seen_step=effective_episode_id,
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
        generator_config_id=str(record.scenario_arm or record.generator_arm or record.source),
    )
    return record


def _persist_buffer_state(
    *,
    path: Path,
    buffer: ScenarioBuffer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        return (
            ScenarioBuffer(capacity=int(scenario_cfg.buffer_capacity)),
            ScenarioArmBandit(scenario_cfg.mab),
            0,
            0,
            0,
            0,
            [],
        )

    configured_run_dir = resume_cfg.get("run_dir")
    resume_run_dir = (
        Path(str(configured_run_dir))
        if configured_run_dir not in (None, "", "null")
        else artifact_paths["root"].parents[1]
    )
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


def _run_scenario_acl_vectorized_training(
    *,
    cfg: DictConfig,
    curriculum_cfg: CurriculumConfig,
    env: Any,
    agent: Agent,
    planner: Any,
    preprocessor: Any,
    adapter: Any,
    recorder: CSVRecorder,
    base_csv_fields: dict[str, Any],
    metadata_path: Path,
    hydra_config_path: Path,
    start_time: float,
    paths: ScenarioAclDriverPaths,
    train_logger: logging.Logger,
    curriculum_logger: logging.Logger,
    artifact_paths: dict[str, Path],
    semantic_env_overrides: dict[str, object],
    transition_replay_config: Any,
    run_seed: int,
    total_timesteps: int,
    eval_interval: int,
    log_interval: int,
    current_global_step: int,
    current_chunk_id: int,
    current_eval_id: int,
    current_episode_id: int,
    buffer: ScenarioBuffer,
    bandit: ScenarioArmBandit,
    rng: np.random.Generator,
    recent_usefulness: list[float],
    generate_count: int,
    replay_count: int,
    resume_enabled: bool,
) -> None:
    """Run the parent-controlled ACL vector path for ``num_envs > 1``."""

    from thesis_rl.curriculum.scenario_acl.vectorized import VECTOR_STATE_VERSION

    n_envs = count_envs(env)
    if n_envs <= 1 or not bool(getattr(env, "acl_mode", False)):
        raise ValueError("Scenario ACL vector driver requires an ACL-mode vector environment.")
    scenario_cfg = curriculum_cfg.scenario_acl
    arms = list(build_default_scenario_arms())
    catalog = read_scenario_catalog(str(semantic_env_overrides["catalog_path"]))
    train_records = tuple(
        sorted(
            catalog.valid_records(split=str(cfg.env.get("split", "train"))),
            key=lambda r: r.scenario_uid,
        )
    )
    if not train_records:
        raise ValueError("Scenario ACL vector execution requires valid train catalog records.")

    vector_state_load_path = artifact_paths["vector_state"]
    if resume_enabled:
        configured_resume_dir = cfg.checkpoint.get("resume", {}).get("run_dir")
        if configured_resume_dir not in (None, "", "null"):
            vector_state_load_path = (
                Path(str(configured_resume_dir))
                / "artifacts"
                / "curriculum"
                / "scenario_acl_vector_state.json"
            )
        if not vector_state_load_path.is_file():
            raise FileNotFoundError(
                "Scenario ACL vector resume requires the versioned vector state: "
                f"{vector_state_load_path}"
            )
        vector_state = load_acl_vector_state(vector_state_load_path, expected_n_envs=n_envs)
        if vector_state.version != VECTOR_STATE_VERSION:
            raise ValueError("Scenario ACL vector state version is incompatible.")
        if vector_state.rng_state is not None:
            rng.bit_generator.state = vector_state.rng_state
    else:
        vector_state = AclVectorState(n_envs=n_envs)

    coordinator = AclVectorSelectionCoordinator(vector_state)
    transaction = AclVectorTransaction(vector_state)
    initial_observations: Any | None = None

    def select_batch(slots: list[int]) -> dict[int, AclSlotSelection]:
        def selector(slot: int, excluded: frozenset[str]) -> AclSlotSelection:
            nonlocal generate_count, replay_count
            episode_id = vector_state.next_episode_id
            vector_state.next_episode_id += 1
            generation = vector_state.next_generation
            effective_excluded = set(excluded)
            effective_excluded.update(str(record.scenario_id) for record in buffer.records())
            can_replay = bool(
                scenario_cfg.use_replay and len(buffer) >= int(scenario_cfg.warmup_buffer_size)
            )
            if can_replay and float(rng.random()) < float(scenario_cfg.exploit_probability):
                replay = buffer.sample_replay(
                    rng=rng,
                    current_step=episode_id,
                    cfg=scenario_cfg.replay_sampling,
                    use_staleness=bool(scenario_cfg.use_staleness),
                )
                record = replay.record
                replay_count += 1
                return AclSlotSelection(
                    slot_id=slot,
                    episode_id=episode_id,
                    generation=generation,
                    mode="replay",
                    arm_index=SCENARIO_ARM_NAMES.index(
                        str(record.scenario_arm or record.generator_arm or record.source)
                    ),
                    arm_name=str(record.scenario_arm or record.generator_arm or record.source),
                    reset_seed=int(record.reset_seed),
                    scenario_uid=str(record.scenario_id),
                    runtime_index=int(record.scenario_index),
                    source=str(record.source),
                )

            if bool(scenario_cfg.use_mab):
                arm_index, probabilities = bandit.sample_arm(rng)
            else:
                arm_index = int(rng.integers(0, len(arms)))
                probabilities = np.full(len(arms), 1.0 / len(arms), dtype=np.float64)
            arm_name = arms[arm_index].name
            candidates = [
                record
                for record in train_records
                if record.primary_arm == arm_name and record.scenario_uid not in effective_excluded
            ]
            if not candidates:
                if not len(buffer):
                    raise RuntimeError(
                        f"Selected ACL arm {arm_name!r} has no fresh catalog record and replay is empty."
                    )
                replay = buffer.sample_replay(
                    rng=rng,
                    current_step=episode_id,
                    cfg=scenario_cfg.replay_sampling,
                    use_staleness=bool(scenario_cfg.use_staleness),
                )
                record = replay.record
                replay_count += 1
                return AclSlotSelection(
                    slot_id=slot,
                    episode_id=episode_id,
                    generation=generation,
                    mode="replay",
                    arm_index=SCENARIO_ARM_NAMES.index(
                        str(record.scenario_arm or record.generator_arm or record.source)
                    ),
                    arm_name=str(record.scenario_arm or record.generator_arm or record.source),
                    reset_seed=int(record.reset_seed),
                    scenario_uid=str(record.scenario_id),
                    runtime_index=int(record.scenario_index),
                    source=str(record.source),
                )
            record = candidates[int(rng.integers(0, len(candidates)))]
            generate_count += 1
            return AclSlotSelection(
                slot_id=slot,
                episode_id=episode_id,
                generation=generation,
                mode="generate",
                arm_index=int(arm_index),
                arm_name=arm_name,
                reset_seed=int(record.runtime_index),
                scenario_uid=str(record.scenario_uid),
                runtime_index=int(record.runtime_index),
                source=str(record.source),
                selection_probability=float(probabilities[arm_index]),
            )

        selected = coordinator.select_batch(slots, selector=selector)
        for slot in sorted(selected):
            vector_state.accumulators[slot] = AclEpisodeAccumulator(
                slot_id=slot, episode_id=selected[slot].episode_id
            )
        return selected

    if resume_enabled and vector_state.active_selections:
        active_slots = set(vector_state.active_selections)
        expected_slots = set(range(n_envs))
        if active_slots != expected_slots:
            raise ValueError(
                "Scenario ACL vector resume requires one persisted active selection per slot: "
                f"expected {sorted(expected_slots)}, got {sorted(active_slots)}."
            )
        reset_results = coordinator.restart_active_slots_for_resume(env)
        initial_observations = np.asarray([reset_results[slot][0] for slot in range(n_envs)])
        for slot in sorted(vector_state.active_selections):
            selection = vector_state.active_selections[slot]
            log_event(
                paths.events_log_path,
                "scenario_acl_active_slot_restarted_on_resume",
                worker_id=int(slot),
                episode_id=int(selection.episode_id),
                selection_generation=int(selection.generation),
                scenario_id=selection.scenario_uid,
                reset_seed=int(selection.reset_seed),
                reason="simulator_state_not_serialized",
            )
    else:
        active_slots = list(range(n_envs))
        if not vector_state.active_selections:
            select_batch(active_slots)
        reset_results = coordinator.configure_and_reset(env, vector_state.active_selections)
        initial_observations = np.asarray([reset_results[slot][0] for slot in active_slots])

    def vector_episode_end_callback(
        vector_env: Any, done_indices: list[int], payloads: list[dict[str, Any]]
    ) -> dict[int, Any]:
        completed: dict[tuple[int, int], AclCompletion] = {}
        learning_potentials: dict[tuple[int, int], float] = {}
        ready: dict[tuple[int, int], float] = {}
        payload_by_key = {
            (int(payload["worker_id"]), int(payload["episode_id"])): payload for payload in payloads
        }
        for payload in payloads:
            ready.update(
                {
                    (int(key[0]), int(key[1])): float(value)
                    for key, value in dict(payload.get("ready_learning_potentials", {})).items()
                }
            )
            slot = int(payload["worker_id"])
            episode_id = int(payload["episode_id"])
            selection = vector_state.active_selections.get(slot)
            if selection is None or selection.episode_id != episode_id:
                raise ValueError(
                    f"ACL completion has no matching active selection for slot {slot}."
                )
            completion = AclCompletion(
                collection_tick=vector_state.collection_tick,
                worker_id=slot,
                episode_id=episode_id,
                selection=selection,
                metrics=dict(payload.get("metrics", {})),
            )
            completed[(slot, episode_id)] = completion
            if payload.get("learning_potential") is not None:
                learning_potentials[(slot, episode_id)] = float(payload["learning_potential"])
            elif (slot, episode_id) in ready:
                learning_potentials[(slot, episode_id)] = ready[(slot, episode_id)]

        pending_by_key = {
            (item.worker_id, item.episode_id): item for item in vector_state.pending_completions
        }
        pending_by_key.update(completed)
        for key, value in ready.items():
            if key in pending_by_key:
                learning_potentials[key] = float(value)
        ready_completions = [
            completion for key, completion in pending_by_key.items() if key in learning_potentials
        ]

        def commit_event(event: dict[str, Any]) -> None:
            key = (int(event["worker_id"]), int(event["episode_id"]))
            completion = pending_by_key[key]
            lp = float(event["learning_potential"])
            normalized = _normalize_learning_potential(lp, recent_usefulness)
            live_event_context = {
                "slot": int(completion.worker_id),
                "episode_id": int(completion.episode_id),
                "arm": _short_arm_name(completion.selection.arm_name),
                "source": completion.selection.source,
                "origin": ("replay" if completion.selection.mode == "replay" else "new"),
                "U": f"{lp:.4f}",
                "U_norm": f"{normalized:.4f}",
            }
            if isinstance(completion.metrics, dict):
                completion.metrics["live_event_context"] = live_event_context
            payload = payload_by_key.get(key)
            if payload is not None:
                payload["live_event_context"] = live_event_context
            recent_usefulness.append(lp)
            max_recent = int(scenario_cfg.recent_window_size)
            if len(recent_usefulness) > max_recent:
                del recent_usefulness[:-max_recent]
            if completion.selection.mode == "generate" and bool(scenario_cfg.use_mab):
                if completion.selection.selection_probability is None:
                    raise ValueError("Fresh ACL selection is missing its MAB probability.")
                bandit.update(
                    arm_index=completion.selection.arm_index,
                    normalized_usefulness=normalized,
                    selection_probability=float(completion.selection.selection_probability),
                )
            metrics = dict(completion.metrics)
            scenario_uid = completion.selection.scenario_uid
            if scenario_uid is None:
                raise ValueError("ACL completion is missing scenario identity.")
            if completion.selection.mode == "replay":
                record = next(
                    (item for item in buffer.records() if item.scenario_id == scenario_uid), None
                )
                if record is None:
                    raise ValueError(
                        f"Replay ACL record is missing from the parent buffer: {scenario_uid}"
                    )
                updated = _update_replay_record(
                    record,
                    episode_id=completion.episode_id,
                    learning_potential=lp,
                    normalized_usefulness=normalized,
                    metrics=_episode_record_metrics(metrics),
                )
                buffer.update(updated)
                action = "updated"
            else:
                catalog_record = catalog.get_by_uid(scenario_uid).record
                inserted = buffer.insert(
                    _build_record_from_catalog_entry(
                        catalog_record=catalog_record,
                        cfg=cfg,
                        episode_id=completion.episode_id,
                        learning_potential=lp,
                        normalized_usefulness=normalized,
                        metrics=_episode_record_metrics(metrics),
                    )
                )
                action = "inserted" if inserted else "rejected"
            _append_jsonl(
                artifact_paths["buffer_events"],
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "collection_tick": int(event["collection_tick"]),
                    "worker_id": int(event["worker_id"]),
                    "episode_id": int(event["episode_id"]),
                    "selection_generation": int(event["selection_generation"]),
                    "scenario_id": scenario_uid,
                    "learning_potential": lp,
                    "buffer_action": action,
                },
            )
            log_event(
                paths.events_log_path,
                "scenario_acl_episode_ended",
                collection_tick=int(event["collection_tick"]),
                worker_id=int(event["worker_id"]),
                episode_id=int(event["episode_id"]),
                selection_generation=int(event["selection_generation"]),
                scenario_id=scenario_uid,
                learning_potential=lp,
                buffer_action=action,
                metrics=metrics,
            )

        transaction.commit_tick(
            ready_completions,
            learning_potentials=learning_potentials,
            commit=commit_event,
        )
        committed_keys = set(learning_potentials).intersection(pending_by_key)
        vector_state.pending_completions = retain_unresolved_acl_completions(
            vector_state.pending_completions,
            completed.values(),
            resolved_keys=committed_keys,
        )

        coordinator.clear_completed(done_indices)
        selected = select_batch(done_indices)
        reset_results = coordinator.configure_and_reset(vector_env, selected)
        return {slot: reset_results[slot][0] for slot in done_indices}

    def mab_monitor_rows() -> list[tuple[str, str]]:
        probabilities = bandit.probabilities()
        labels = ", ".join(
            f"{_short_arm_name(arm.name)}={probability:.3f}"
            for arm, probability in zip(arms, probabilities, strict=True)
        )
        return [("ACL arm probabilities", labels)]

    def record_async_acl_evaluation(job: EvaluationJob, metrics: dict[str, Any]) -> None:
        """Persist completed ACL diagnostics; the result never feeds ACL selection."""

        recorder.append_row(
            "evals.csv",
            {
                **base_csv_fields,
                "eval_id": job.eval_id,
                "eval_type": "intermediate",
                "scenario_set": f"validation_{_WAYMO_STRATIFIED_SET}",
                "chunk_id": int(job.metadata.get("chunk_id", 0)),
                "stage": job.stage,
                "stage_index": job.stage_index,
                "global_step": job.global_step,
                "eval_episodes": job.episode_count,
                "deterministic": bool(cfg.experiment.eval_deterministic),
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
            },
        )
        log_event(
            paths.events_log_path,
            "evaluation_finished",
            eval_id=job.eval_id,
            stage=job.stage,
            global_step=job.global_step,
            asynchronous=True,
            metrics=metrics,
        )

    async_evaluation_manager = AsyncEvaluationManager(
        checkpoints_dir=paths.checkpoints_dir,
        on_complete=record_async_acl_evaluation,
        start_method=str(cfg.experiment.get("evaluation_start_method", "spawn")),
    )

    def run_intermediate_evaluation() -> None:
        """Queue validation while keeping the subprocess train env alive."""

        nonlocal current_eval_id
        eval_episode_count = int(cfg.experiment.eval_episodes)
        if eval_episode_count <= 0:
            return
        eval_env_overrides = apply_eval_scenario_seed_split(
            base_run_seed=run_seed,
            eval_env_overrides=None,
            cfg=cfg,
            n_eval_episodes=eval_episode_count,
            split="validation",
        )
        eval_env_overrides.update(semantic_env_overrides)
        current_eval_id += 1
        log_event(
            paths.events_log_path,
            "evaluation_started",
            eval_id=current_eval_id,
            stage="scenario_acl_vectorized",
            global_step=current_global_step,
            episodes=eval_episode_count,
            split="validation",
            asynchronous=True,
        )
        async_evaluation_manager.enqueue(
            agent=agent,
            cfg=cfg,
            eval_id=current_eval_id,
            global_step=current_global_step,
            stage="scenario_acl_vectorized",
            stage_index=0,
            episode_count=eval_episode_count,
            base_seed=None,
            env_seed=run_seed + 500_000 + current_chunk_id,
            env_overrides=eval_env_overrides,
            workers=evaluation_num_workers(cfg, final=False),
            scenario_arm_schedule=tuple(
                _select_waymo_eval_arm(index) for index in range(eval_episode_count)
            ),
            scenario_source_schedule=("waymo",) * eval_episode_count,
            metadata={"chunk_id": current_chunk_id},
        )

    current_observations = initial_observations
    try:
        while current_global_step < total_timesteps:
            current_chunk_id += 1
            chunk_steps = min(eval_interval, total_timesteps - current_global_step)
            summary = agent.train_vectorized(
                env=env,
                chunk_timesteps=chunk_steps,
                global_total_timesteps=total_timesteps,
                global_steps_done=current_global_step,
                stage_name="scenario_acl_vectorized",
                deterministic=False,
                log_interval=log_interval,
                vector_episode_end_callback=vector_episode_end_callback,
                initial_observations=current_observations,
                monitor_extra_rows_callback=mab_monitor_rows,
                monitor_event_poll_callback=async_evaluation_manager.drain_event_messages,
                live_extra_renderables_callback=async_evaluation_manager.renderables,
            )
            current_observations = summary["last_observations"]
            current_global_step = min(
                total_timesteps, current_global_step + int(summary["chunk_steps_actual"])
            )
            vector_state.last_observations = current_observations
            vector_state.rng_state = rng.bit_generator.state
            _persist_buffer_state(path=artifact_paths["buffer"], buffer=buffer)
            save_acl_vector_state(artifact_paths["vector_state"], vector_state)
            recorder.append_row(
                "train_chunks.csv",
                {
                    **base_csv_fields,
                    "chunk_id": current_chunk_id,
                    "stage": "scenario_acl_vectorized",
                    "steps_start": current_global_step - int(summary["chunk_steps_actual"]),
                    "steps_end": current_global_step,
                    "global_step": current_global_step,
                    "chunk_steps": int(summary["chunk_steps_actual"]),
                    "episodes": int(summary.get("episodes", 0)),
                    "ep_rew_mean": float(summary.get("ep_rew_mean", 0.0)),
                    "ep_len_mean": float(summary.get("ep_len_mean", 0.0)),
                    "learning_potential": summary.get("learning_potential"),
                    "vector_envs": n_envs,
                },
            )
            history_payload = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "chunk_id": current_chunk_id,
                "global_step": current_global_step,
                "mode": "vectorized",
                "vector_envs": n_envs,
                "buffer_size": len(buffer),
                "pending_completions": len(vector_state.pending_completions),
                "mab": bandit.state_dict(),
            }
            _append_jsonl(artifact_paths["history"], history_payload)
            _append_jsonl(artifact_paths["iterations"], history_payload)
            artifact_paths["state"].write_text(
                json.dumps(
                    {
                        "global_step": current_global_step,
                        "chunk_id": current_chunk_id,
                        "eval_id": current_eval_id,
                        "episode_id": vector_state.next_episode_id,
                        "buffer_size": len(buffer),
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
                if transition_replay_config.persistence_enabled:
                    _save_acl_replay_buffer(
                        planner, paths.checkpoints_dir / "latest_replay_buffer.pkl"
                    )
                    _save_acl_checkpoint_pair(
                        path=paths.checkpoints_dir / "latest_checkpoint_pair.json",
                        checkpoint_name="latest",
                        replay_name="latest_replay_buffer.pkl",
                        training_timestep=current_global_step,
                    )
                if bool(cfg.checkpoint.get("save_rng_state", True)):
                    _save_acl_rng_state(paths.checkpoints_dir / "latest_rng_state.pkl")

            run_intermediate_evaluation()

            train_logger.info(
                "Scenario ACL vector chunk finished | chunk_id=%d | global_step=%d | "
                "buffer_size=%d | pending=%d",
                current_chunk_id,
                current_global_step,
                len(buffer),
                len(vector_state.pending_completions),
            )
    finally:
        env.close()

    async_evaluation_manager.drain()
    async_evaluation_manager.close()

    if not bool(cfg.checkpoint.get("save_final", True)):
        raise ValueError("checkpoint.save_final must be true for scenario_acl training.")
    agent.save(paths.final_checkpoint_stem)
    if transition_replay_config.persistence_enabled:
        _save_acl_replay_buffer(planner, paths.checkpoints_dir / "final_replay_buffer.pkl")
        _save_acl_checkpoint_pair(
            path=paths.checkpoints_dir / "final_checkpoint_pair.json",
            checkpoint_name="final",
            replay_name="final_replay_buffer.pkl",
            training_timestep=current_global_step,
        )
    if bool(cfg.checkpoint.get("save_rng_state", True)):
        _save_acl_rng_state(paths.checkpoints_dir / "latest_rng_state.pkl")

    final_eval_overrides = apply_eval_scenario_seed_split(
        base_run_seed=run_seed,
        eval_env_overrides=None,
        cfg=cfg,
        n_eval_episodes=int(
            cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)
        ),
        split="test",
    )
    final_eval_overrides.update(semantic_env_overrides)
    final_eval_episode_count = int(
        cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)
    )
    final_eval_env = build_eval_env(
        cfg,
        final_eval_overrides,
        n_eval_episodes=final_eval_episode_count,
        workers=evaluation_num_workers(cfg, final=True),
        scenario_arm_schedule=tuple(
            _select_waymo_eval_arm(index) for index in range(final_eval_episode_count)
        ),
        scenario_source_schedule=("waymo",) * final_eval_episode_count,
    )
    seed_env_spaces(final_eval_env, run_seed + 600_000)
    final_eval_agent = Agent(
        preprocessor=preprocessor,
        planner=load_planner(
            cfg, checkpoint_path=f"{paths.final_checkpoint_stem}.zip", env=final_eval_env
        ),
        adapter=adapter,
        ema_alpha=float(getattr(agent, "ema_alpha", 0.1)),
    )
    resolved_final_eval_cfg = OmegaConf.to_container(
        merge_env_config_with_overrides(cfg.env, final_eval_overrides or {}),
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
        n_eval_episodes=final_eval_episode_count,
        deterministic=bool(cfg.experiment.eval_deterministic),
        base_seed=None,
        return_episode_metrics=True,
        error_priority_base=float(cfg.reward.get("a", 2.01)),
        show_progress=True,
        progress_description="Test episodes",
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
        base_seed=None,
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
            "final_stage_reached": True,
            "final_eval_episodes": int(
                cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)
            ),
            "mean_reward": float(final_metrics.get("mean_reward", 0.0)),
            "collision_rate": float(final_metrics.get("collision_rate", 0.0)),
            "success_rate": float(final_metrics.get("success_rate", 0.0)),
            "route_completion": float(final_metrics.get("route_completion", 0.0)),
            "checkpoint_path": str(
                paths.final_checkpoint_stem.with_suffix(".zip").relative_to(paths.run_dir)
            ),
            "checkpoint_type": "final",
            "checkpoint_global_step": current_global_step,
        },
    )
    update_run_metadata(
        paths.artifacts_dir,
        {
            "status": "completed",
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(time.time() - start_time, 2),
            "global_step": current_global_step,
            "chunk_id": current_chunk_id,
            "eval_id": current_eval_id + 1,
            "stage": "scenario_acl",
            "stage_index": 0,
        },
    )


def _choose_iteration_spec(
    *,
    cfg: DictConfig,
    curriculum_cfg: CurriculumConfig,
    arms: list[ScenarioArm],
    bandit: ScenarioArmBandit,
    buffer: ScenarioBuffer,
    rng: np.random.Generator,
    episode_id: int | None = None,
    chunk_id: int | None = None,
) -> IterationSpec:
    effective_episode_id = int(episode_id if episode_id is not None else (chunk_id or 0))
    scenario_cfg = curriculum_cfg.scenario_acl
    can_exploit = scenario_cfg.use_replay and len(buffer) >= int(scenario_cfg.warmup_buffer_size)

    if can_exploit and rng.random() < float(scenario_cfg.exploit_probability):
        selection = buffer.sample_replay(
            rng=rng,
            current_step=effective_episode_id,
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
    transition_replay_config = resolve_transition_replay_config(
        cfg.agent.planner.algorithm.get("transition_replay"),
        algorithm_name=str(cfg.agent.planner.algorithm.name),
        total_timesteps=total_timesteps,
        legacy_save_replay_buffer=bool(cfg.checkpoint.get("save_replay_buffer", False)),
    )
    latest_replay_buffer_path = paths.checkpoints_dir / "latest_replay_buffer.pkl"
    final_replay_buffer_path = paths.checkpoints_dir / "final_replay_buffer.pkl"
    latest_checkpoint_pair_path = paths.checkpoints_dir / "latest_checkpoint_pair.json"
    final_checkpoint_pair_path = paths.checkpoints_dir / "final_checkpoint_pair.json"
    latest_rng_state_path = paths.checkpoints_dir / "latest_rng_state.pkl"
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
    env = build_train_env(cfg, semantic_env_overrides)
    seed_env_spaces(env, run_seed)
    train_env_count = count_envs(env)

    preprocessor = build_preprocessor(cfg)
    adapter = build_adapter(cfg, adapter_space_kwargs(env.action_space))
    resume_cfg = cfg.checkpoint.get("resume", {})
    resume_enabled = bool(resume_cfg.get("enabled", False))
    resume_run_dir_cfg = resume_cfg.get("run_dir")
    resume_run_dir = (
        Path(str(resume_run_dir_cfg))
        if resume_run_dir_cfg not in (None, "", "null")
        else paths.run_dir
    )
    resume_checkpoint_stem = (
        resume_run_dir / "checkpoints" / str(resume_cfg.get("checkpoint_name", "latest"))
    )
    resume_checkpoint_zip = resume_checkpoint_stem.with_suffix(".zip")
    if resume_enabled:
        if not resume_checkpoint_zip.exists():
            raise FileNotFoundError(
                f"Scenario ACL resume checkpoint is missing: {resume_checkpoint_zip}"
            )
        planner = load_planner(cfg, checkpoint_path=str(resume_checkpoint_zip), env=env)
    else:
        planner = build_planner(cfg, env, seed=run_seed)
    planner_device = str(getattr(planner, "device", cfg.device))
    ema_alpha_cfg = (
        float(cfg.agent.planner.algorithm.get("monitor_ema_alpha", 0.1))
        if hasattr(cfg, "agent")
        else 0.1
    )
    agent = Agent(
        preprocessor=preprocessor, planner=planner, adapter=adapter, ema_alpha=ema_alpha_cfg
    )
    agent.set_checkpoint_identity(build_reward_semantics_identity(cfg))
    if resume_enabled:
        agent.load_adapter(checkpoint_path=resume_checkpoint_zip, strict=True)
        if transition_replay_config.persistence_enabled:
            resume_replay_path = (
                resume_run_dir
                / "checkpoints"
                / f"{resume_cfg.get('checkpoint_name', 'latest')}_replay_buffer.pkl"
            )
            _load_acl_replay_buffer(planner, resume_replay_path)
            resume_pair_path = (
                resume_run_dir
                / "checkpoints"
                / f"{resume_cfg.get('checkpoint_name', 'latest')}_checkpoint_pair.json"
            )
            if not resume_pair_path.is_file():
                raise FileNotFoundError(f"ACL checkpoint pair is missing: {resume_pair_path}")
            if bool(resume_cfg.get("restore_rng_state", True)):
                _load_acl_rng_state(resume_run_dir / "checkpoints" / "latest_rng_state.pkl")

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

    if train_num_envs(cfg) > 1:
        _run_scenario_acl_vectorized_training(
            cfg=cfg,
            curriculum_cfg=curriculum_cfg,
            env=env,
            agent=agent,
            planner=planner,
            preprocessor=preprocessor,
            adapter=adapter,
            recorder=recorder,
            base_csv_fields=base_csv_fields,
            metadata_path=metadata_path,
            hydra_config_path=hydra_config_path,
            start_time=start_time,
            paths=paths,
            train_logger=train_logger,
            curriculum_logger=curriculum_logger,
            artifact_paths=artifact_paths,
            semantic_env_overrides=semantic_env_overrides,
            transition_replay_config=transition_replay_config,
            run_seed=run_seed,
            total_timesteps=total_timesteps,
            eval_interval=eval_interval,
            log_interval=log_interval,
            current_global_step=current_global_step,
            current_chunk_id=current_chunk_id,
            current_eval_id=current_eval_id,
            current_episode_id=current_episode_id,
            buffer=buffer,
            bandit=bandit,
            rng=rng,
            recent_usefulness=recent_usefulness,
            generate_count=generate_count,
            replay_count=replay_count,
            resume_enabled=resume_enabled,
        )
        return

    def record_async_acl_evaluation(job: EvaluationJob, metrics: dict[str, Any]) -> None:
        recorder.append_row(
            "evals.csv",
            {
                **base_csv_fields,
                "eval_id": job.eval_id,
                "eval_type": "intermediate",
                "scenario_set": f"validation_{_WAYMO_STRATIFIED_SET}",
                "chunk_id": int(job.metadata.get("chunk_id", 0)),
                "stage": job.stage,
                "stage_index": job.stage_index,
                "global_step": job.global_step,
                "eval_episodes": job.episode_count,
                "deterministic": bool(cfg.experiment.eval_deterministic),
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
            },
        )
        log_event(
            paths.events_log_path,
            "evaluation_finished",
            eval_id=job.eval_id,
            stage=job.stage,
            global_step=job.global_step,
            asynchronous=True,
            metrics=metrics,
        )

    async_evaluation_manager = AsyncEvaluationManager(
        checkpoints_dir=paths.checkpoints_dir,
        on_complete=record_async_acl_evaluation,
        start_method=str(cfg.experiment.get("evaluation_start_method", "spawn")),
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
                    excluded_uids = {str(record.scenario_id) for record in buffer.records()}
                    provider = getattr(base_env, "scenario_provider", None)
                    has_candidate = getattr(provider, "has_candidate", None)
                    source = _selection_source_override(spec)
                    if callable(has_candidate) and not has_candidate(
                        split=str(base_env.split),
                        source=source,
                        arm=spec.arm_name,
                        excluded_scenario_uids=excluded_uids,
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
                            selection_probability=float(spec.arm_probabilities[spec.arm_index]),
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
                    arm_name=_short_arm_name(getattr(record, "primary_arm", selection.get("arm"))),
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

            def episode_context(training_env: Any, _info: dict[str, Any]) -> dict[str, Any] | None:
                base_env = _base_env(training_env)
                record = getattr(base_env, "current_scenario_record", None)
                selection = getattr(base_env, "_acl_episode_selection", {})
                return {
                    "arm": _short_arm_name(getattr(record, "primary_arm", selection.get("arm"))),
                    "source": getattr(record, "source", None),
                    "origin": (
                        f"{selection.get('origin')} |"
                        if selection.get("usefulness") is not None
                        else selection.get("origin")
                    ),
                    "U": _format_optional_float(selection.get("usefulness")),
                    "U_norm": _format_optional_float(selection.get("usefulness_norm")),
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
                monitor_event_poll_callback=async_evaluation_manager.drain_event_messages,
                live_extra_renderables_callback=async_evaluation_manager.renderables,
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
                    "train_reset_seed_unique_count": chunk_summary.get(
                        "train_reset_seed_unique_count"
                    ),
                },
            )

            env.close()

            eval_metrics: dict[str, Any] = {}
            eval_env_overrides = apply_eval_scenario_seed_split(
                base_run_seed=run_seed,
                eval_env_overrides=None,
                cfg=cfg,
                n_eval_episodes=int(cfg.experiment.eval_episodes),
                split="validation",
            )
            eval_env_overrides.update(semantic_env_overrides)
            eval_base_seed = None
            eval_episode_count = int(cfg.experiment.eval_episodes)
            if eval_episode_count <= 0:
                continue
            current_eval_id += 1
            log_event(
                paths.events_log_path,
                "evaluation_started",
                eval_id=current_eval_id,
                stage="scenario_acl_episode_sampling",
                global_step=current_global_step,
                episodes=eval_episode_count,
                split="validation",
                asynchronous=True,
            )
            async_evaluation_manager.enqueue(
                agent=agent,
                cfg=cfg,
                eval_id=current_eval_id,
                global_step=current_global_step,
                stage="scenario_acl_episode_sampling",
                stage_index=0,
                episode_count=eval_episode_count,
                base_seed=eval_base_seed,
                env_seed=run_seed + 500_000 + current_chunk_id,
                env_overrides=eval_env_overrides,
                workers=evaluation_num_workers(cfg, final=False),
                scenario_arm_schedule=tuple(
                    _select_waymo_eval_arm(index) for index in range(eval_episode_count)
                ),
                scenario_source_schedule=("waymo",) * eval_episode_count,
                metadata={"chunk_id": current_chunk_id},
            )

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
                if outcome.learning_potential is not None and outcome.usefulness_norm is not None
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

            if eval_metrics:
                recorder.append_row(
                    "evals.csv",
                    {
                        **base_csv_fields,
                        "eval_id": current_eval_id,
                        "eval_type": "intermediate",
                        "scenario_set": f"validation_{_WAYMO_STRATIFIED_SET}",
                        "chunk_id": current_chunk_id,
                        "stage": (chunk_stage),
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
                        "mean_rule_saturation_max": float(
                            eval_metrics.get("mean_rule_saturation_max", 0.0)
                        ),
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
                        "violated_rules_ratio": float(
                            eval_metrics.get("violated_rules_ratio", 0.0)
                        ),
                        "unique_violation_patterns": int(
                            eval_metrics.get("unique_violation_patterns", 0)
                        ),
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
                if transition_replay_config.persistence_enabled:
                    _save_acl_replay_buffer(planner, latest_replay_buffer_path)
                    _save_acl_checkpoint_pair(
                        path=latest_checkpoint_pair_path,
                        checkpoint_name="latest",
                        replay_name=latest_replay_buffer_path.name,
                        training_timestep=current_global_step,
                    )
                    if bool(cfg.checkpoint.get("save_rng_state", True)):
                        _save_acl_rng_state(latest_rng_state_path)

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

        async_evaluation_manager.drain()
        async_evaluation_manager.close()

        if not bool(cfg.checkpoint.get("save_final", True)):
            raise ValueError("checkpoint.save_final must be true for scenario_acl training.")
        agent.save(paths.final_checkpoint_stem)
        if transition_replay_config.persistence_enabled:
            _save_acl_replay_buffer(planner, final_replay_buffer_path)
            _save_acl_checkpoint_pair(
                path=final_checkpoint_pair_path,
                checkpoint_name="final",
                replay_name=final_replay_buffer_path.name,
                training_timestep=current_global_step,
            )
            if bool(cfg.checkpoint.get("save_rng_state", True)):
                _save_acl_rng_state(latest_rng_state_path)

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
        final_eval_episode_count = int(
            cfg.experiment.get("final_eval_episodes", cfg.experiment.eval_episodes)
        )
        final_eval_env = build_eval_env(
            cfg,
            final_eval_env_overrides,
            n_eval_episodes=final_eval_episode_count,
            workers=evaluation_num_workers(cfg, final=True),
            scenario_arm_schedule=tuple(
                _select_waymo_eval_arm(index) for index in range(final_eval_episode_count)
            ),
            scenario_source_schedule=("waymo",) * final_eval_episode_count,
        )
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
            n_eval_episodes=final_eval_episode_count,
            deterministic=bool(cfg.experiment.eval_deterministic),
            base_seed=final_eval_base_seed,
            return_episode_metrics=True,
            error_priority_base=float(cfg.reward.get("a", 2.01)),
            show_progress=True,
            progress_description="Test episodes",
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
                "top_rule_violation_rate": float(final_metrics.get("top_rule_violation_rate", 0.0)),
                "avg_error_value": float(final_metrics.get("avg_error_value", 0.0)),
                "max_error_value": float(final_metrics.get("max_error_value", 0.0)),
                "counterexample_rate": float(final_metrics.get("counterexample_rate", 0.0)),
                "violated_rules_ratio": float(final_metrics.get("violated_rules_ratio", 0.0)),
                "unique_violation_patterns": int(final_metrics.get("unique_violation_patterns", 0)),
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
        if transition_replay_config.persistence_enabled:
            _save_acl_replay_buffer(planner, latest_replay_buffer_path)
            _save_acl_checkpoint_pair(
                path=latest_checkpoint_pair_path,
                checkpoint_name="latest",
                replay_name=latest_replay_buffer_path.name,
                training_timestep=current_global_step,
            )
            if bool(cfg.checkpoint.get("save_rng_state", True)):
                _save_acl_rng_state(latest_rng_state_path)
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
