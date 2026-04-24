from datetime import datetime
from pathlib import Path
import socket
import subprocess

import yaml
import torch
from omegaconf import DictConfig, OmegaConf


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _cfg_get(cfg: DictConfig, key: str, default=None):
    """Safely access nested OmegaConf values via dotted keys."""
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def save_run_metadata(cfg: DictConfig, artifacts_dir: str | Path) -> Path:
    """Create or overwrite artifacts/run_metadata.yaml for the current run."""
    artifacts_dir = Path(artifacts_dir)
    metadata_path = artifacts_dir / "run_metadata.yaml"

    metadata = {
        "name": _cfg_get(cfg, "name"),
        "algorithm": _cfg_get(cfg, "planner.name", default="unknown"),
        "seed": _cfg_get(cfg, "seed"),
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_timesteps": _cfg_get(cfg, "experiment.total_timesteps"),
        "eval_interval": _cfg_get(cfg, "experiment.eval_interval"),
        "eval_episodes": _cfg_get(cfg, "experiment.eval_episodes"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hostname": socket.gethostname(),
        "git": {
            "branch": _get_git_branch(),
            "commit": _get_git_commit(),
        },
        "status": "running",
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)

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
