from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
import hashlib
import os
import shutil
import socket
import subprocess

import yaml
import torch
from omegaconf import DictConfig, OmegaConf

# Specification identities that are currently frozen repository-wide rather
# than per-run configurable (EVAL-PROTOCOL REQ-015 / DEC-EP metadata gap).
OBSERVATION_SPECIFICATION_ID = "OBS-V1.2"
ENCODER_SPECIFICATION_ID = "ENC-V1.2"
ACL_SPECIFICATION_ID = "ACL-SN-EMA-001"
ACL_SPECIFICATION_VERSION = "1.1"
TRANSITION_REPLAY_SPECIFICATION_ID = "TRANSITION-REPLAY"
TRANSITION_REPLAY_SPECIFICATION_VERSION = "1.0"
EVALUATION_PROTOCOL_SPECIFICATION_ID = "EVAL-PROTOCOL"
EVALUATION_PROTOCOL_SPECIFICATION_VERSION = "1.0"


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_git_branch() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_git_dirty() -> bool | None:
    """Return whether the tracked working tree has uncommitted changes.

    Git-ignored and untracked-but-ignored files never make the tree dirty
    (``git status --porcelain`` with default flags already excludes files
    matched by ``.gitignore``); untracked files that are *not* ignored do
    count, matching a plain ``git status`` reading of "dirty".
    """

    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
    except Exception:
        return None
    return bool(output.strip())


def get_dependency_versions() -> dict[str, str | None]:
    """Best-effort collection of exact dependency versions for reproducibility."""

    def _package_version(name: str) -> str | None:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return None

    sb3_commit = "unknown"
    try:
        import stable_baselines3

        sb3_root = Path(stable_baselines3.__file__).resolve().parents[1]
        sb3_commit = (
            subprocess.check_output(
                ["git", "-C", str(sb3_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        sb3_commit = "unknown"

    return {
        # torch.__version__ is a TorchVersion (str subclass) that PyYAML's
        # SafeDumper cannot represent directly; coerce to a plain str.
        "torch_version": str(torch.__version__),
        "sb3_version": _package_version("stable-baselines3"),
        "sb3_commit": sb3_commit,
        "metadrive_version": _package_version("metadrive-simulator") or _package_version("metadrive"),
        "scenarionet_version": _package_version("scenarionet"),
    }


def _cfg_get(cfg: DictConfig, key: str, default=None):
    """Safely access nested OmegaConf values via dotted keys."""
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def _snapshot_scenarionet_artifacts(cfg: DictConfig, artifacts_dir: Path) -> dict[str, str]:
    """Copy small dataset-definition artifacts and return their run-relative paths."""

    data_root = Path(os.environ.get("SCENARIONET_DATA_ROOT", "data/scenarionet")).expanduser()
    candidates: dict[str, Path] = {
        "dataset_manifest": data_root / "manifest.yaml",
        "split_manifest": data_root / "splits" / "split_manifest.yaml",
        "arm_thresholds": data_root / "catalog" / "arm_thresholds.json",
    }
    catalog_path = _cfg_get(cfg, "env.catalog_path")
    if catalog_path:
        candidates["scenario_catalog"] = Path(str(catalog_path)).expanduser()

    manifest_path = _cfg_get(cfg, "env.provider.panel_manifest_path")
    if manifest_path:
        resolved = Path(str(manifest_path)).expanduser()
        candidates[f"panel_manifest_{resolved.stem}"] = resolved

    snapshot_dir = artifacts_dir / "scenarionet"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}
    for name, source in candidates.items():
        if not source.is_file():
            continue
        target = snapshot_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        result[name] = str(target.relative_to(artifacts_dir))
        result[f"{name}_sha256"] = digest
    return result


def save_run_metadata(cfg: DictConfig, artifacts_dir: str | Path) -> Path:
    """Create or overwrite artifacts/run_metadata.yaml for the current run."""
    artifacts_dir = Path(artifacts_dir)
    metadata_path = artifacts_dir / "run_metadata.yaml"

    selected_scalarization = OmegaConf.select(cfg, "scalarization")
    scalarization_cfg = (
        OmegaConf.to_container(selected_scalarization, resolve=True)
        if selected_scalarization is not None
        else {}
    )
    if not isinstance(scalarization_cfg, dict):
        scalarization_cfg = {}
    legacy_cfg = scalarization_cfg.get("legacy", {})
    if not isinstance(legacy_cfg, dict):
        legacy_cfg = {}
    transition_replay_cfg = OmegaConf.select(cfg, "agent.planner.algorithm.transition_replay")
    transition_replay = (
        OmegaConf.to_container(transition_replay_cfg, resolve=True)
        if transition_replay_cfg is not None
        else {}
    )
    if not isinstance(transition_replay, dict):
        transition_replay = {}
    algorithm_name = str(_cfg_get(cfg, "agent.planner.algorithm.name", default=""))
    ppo_geometry = None
    if algorithm_name == "ppo_sb3":
        num_envs = int(_cfg_get(cfg, "env.vectorized.num_envs", default=1))
        n_steps = int(_cfg_get(cfg, "agent.planner.algorithm.n_steps", default=2048))
        batch_size = int(_cfg_get(cfg, "agent.planner.algorithm.batch_size", default=64))
        n_epochs = int(_cfg_get(cfg, "agent.planner.algorithm.n_epochs", default=10))
        global_rollout_size = num_envs * n_steps
        if n_steps <= 0 or batch_size <= 1 or n_epochs <= 0:
            raise ValueError("Invalid PPO geometry in resolved configuration.")
        if global_rollout_size % batch_size != 0:
            raise ValueError("PPO global rollout size must be divisible by batch_size.")
        ppo_geometry = {
            "num_envs": num_envs,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "global_rollout_size": global_rollout_size,
            "minibatches_per_epoch": global_rollout_size // batch_size,
            "optimizer_steps_per_update": (global_rollout_size // batch_size) * n_epochs,
        }
    metadata = {
        "name": _cfg_get(cfg, "name"),
        "algorithm": algorithm_name or _cfg_get(cfg, "planner.name", default="unknown"),
        "task_contract": _cfg_get(cfg, "env.name", default="unknown"),
        "run_profile": _cfg_get(cfg, "run_profile.name", default="unknown"),
        "reward_type": _cfg_get(cfg, "reward.type", default="unknown"),
        "reward_behavior": _cfg_get(cfg, "reward.behavior", default="unknown"),
        "rulebook_config": _cfg_get(cfg, "reward.rulebook_config", default="none"),
        "rulebook": {
            "implementation_family": _cfg_get(cfg, "rulebook.implementation_family", default="v1"),
            "specification_id": _cfg_get(
                cfg, "rulebook.specification_id", default="not-applicable"
            ),
            "version": _cfg_get(cfg, "rulebook.version", default="not-applicable"),
        },
        "scalarization": {
            "specification_id": scalarization_cfg.get("specification_id", "not-applicable"),
            "version": scalarization_cfg.get("version", "not-applicable"),
            "mode": scalarization_cfg.get("mode", "not-applicable"),
            "vector_schema_id": scalarization_cfg.get("vector_schema_id"),
            "priority_base": scalarization_cfg.get("priority_base"),
            "sigmoid_sharpness": (
                scalarization_cfg.get("sigmoid_sharpness")
                or (scalarization_cfg.get("sigmoid") or {}).get("sharpness")
            ),
            "numerical_tolerance": scalarization_cfg.get("numerical_tolerance"),
            "native_environment_reward_weight": scalarization_cfg.get(
                "native_environment_reward_weight"
            ),
            "legacy": {
                "vector_schema_id": legacy_cfg.get("vector_schema_id"),
                "rule_scales": legacy_cfg.get("rule_scales"),
                "source_path": legacy_cfg.get("source_path"),
                "source_sha256": legacy_cfg.get("source_sha256"),
                "source_commit": legacy_cfg.get("source_commit"),
            },
            "conditional_extensions": {
                "n_step": _cfg_get(cfg, "scalarization.extensions.n_step", default=None),
                "per": _cfg_get(cfg, "scalarization.extensions.per", default=None),
            },
        },
        "curriculum_name": _cfg_get(cfg, "curriculum.name", default="unknown"),
        "transition_replay": {
            "enabled": transition_replay.get("enabled"),
            "n_steps": transition_replay.get("n_steps"),
            "prioritized": transition_replay.get("prioritized"),
            "persistence": transition_replay.get("persistence", {}),
            "specification_id": TRANSITION_REPLAY_SPECIFICATION_ID,
            "specification_version": TRANSITION_REPLAY_SPECIFICATION_VERSION,
        },
        "observation": {"specification_id": OBSERVATION_SPECIFICATION_ID},
        "encoder": {"specification_id": ENCODER_SPECIFICATION_ID},
        "acl": {
            "specification_id": ACL_SPECIFICATION_ID,
            "version": ACL_SPECIFICATION_VERSION,
        },
        "evaluation_protocol": {
            "specification_id": EVALUATION_PROTOCOL_SPECIFICATION_ID,
            "version": EVALUATION_PROTOCOL_SPECIFICATION_VERSION,
        },
        "experiment_group": _cfg_get(cfg, "analysis.experiment_group"),
        "include_in_comparison": bool(_cfg_get(cfg, "analysis.include_in_comparison", True)),
        "seed": _cfg_get(cfg, "seed"),
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_timesteps": _cfg_get(cfg, "experiment.total_timesteps"),
        "eval_interval": _cfg_get(cfg, "experiment.eval_interval"),
        "eval_episodes": _cfg_get(cfg, "experiment.eval_episodes"),
        "checkpoint_interval": _cfg_get(cfg, "checkpoint.periodic_interval"),
        "ppo_geometry": ppo_geometry,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hostname": socket.gethostname(),
        "git": {
            "branch": get_git_branch(),
            "commit": get_git_commit(),
            "dirty": get_git_dirty(),
        },
        "dependencies": get_dependency_versions(),
        "status": "running",
    }
    if str(_cfg_get(cfg, "env.name", default="")).lower() == "scenarionet":
        metadata["scenarionet"] = {
            "split": _cfg_get(cfg, "env.split", default="train"),
            "catalog_path": _cfg_get(cfg, "env.catalog_path"),
            "global_seed": _cfg_get(cfg, "env.global_seed", default=0),
            "provider": OmegaConf.to_container(OmegaConf.select(cfg, "env.provider"), resolve=True),
        }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if str(_cfg_get(cfg, "env.name", default="")).lower() == "scenarionet":
        metadata["scenarionet"]["artifact_snapshots"] = _snapshot_scenarionet_artifacts(
            cfg, artifacts_dir
        )

    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    return metadata_path


def update_run_metadata(artifacts_dir: str | Path, updates: dict) -> Path:
    """Patch artifacts/run_metadata.yaml with new fields."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = artifacts_dir / "run_metadata.yaml"

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f) or {}
    else:
        metadata = {}

    metadata.update(updates)

    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    return metadata_path
