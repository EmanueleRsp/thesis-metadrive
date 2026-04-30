from __future__ import annotations

from typing import Any

import hashlib
import random

import numpy as np
import torch
from omegaconf import DictConfig


_DEFAULT_SCENARIO_SPLIT_STRIDE = 100_000
_DEFAULT_SCENARIO_SPLIT_OFFSETS = {
    "eval": 1_000_000,
    "validation": 1_000_000,
    "test": 2_000_000,
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_env_spaces(env: Any, seed: int) -> None:
    action_space = getattr(env, "action_space", None)
    if action_space is not None and hasattr(action_space, "seed"):
        action_space.seed(seed)
    observation_space = getattr(env, "observation_space", None)
    if observation_space is not None and hasattr(observation_space, "seed"):
        observation_space.seed(seed)


def scenario_split_settings(cfg: DictConfig) -> tuple[int, dict[str, int]]:
    split_cfg = cfg.get("scenario_splits", {})
    stride = int(split_cfg.get("stride", _DEFAULT_SCENARIO_SPLIT_STRIDE))
    validation_offset = int(
        split_cfg.get("validation_offset", _DEFAULT_SCENARIO_SPLIT_OFFSETS["validation"])
    )
    test_offset = int(split_cfg.get("test_offset", _DEFAULT_SCENARIO_SPLIT_OFFSETS["test"]))
    return stride, {
        "eval": validation_offset,
        "validation": validation_offset,
        "test": test_offset,
    }


def apply_eval_scenario_seed_split(
    *,
    base_run_seed: int,
    eval_env_overrides: dict[str, object] | None,
    cfg: DictConfig,
    n_eval_episodes: int | None = None,
    split: str = "eval",
) -> dict[str, object]:
    """Return env overrides with a deterministic disjoint MetaDrive scenario pool."""
    overrides = dict(eval_env_overrides or {})
    base_start_seed = int(overrides.get("start_seed", cfg.env.config.start_seed))
    split_stride, split_offsets = scenario_split_settings(cfg)
    split_name = str(split).lower()
    if split_name not in split_offsets:
        allowed = ", ".join(sorted(split_offsets))
        raise ValueError(f"Unknown scenario split '{split}'. Expected one of: {allowed}.")

    scenario_offset = split_offsets[split_name] + int(base_run_seed) * split_stride
    overrides["start_seed"] = int(base_start_seed) + scenario_offset
    if n_eval_episodes is not None:
        configured_count = int(overrides.get("num_scenarios", cfg.env.config.num_scenarios))
        overrides["num_scenarios"] = max(configured_count, int(n_eval_episodes))
        if base_start_seed + int(overrides["num_scenarios"]) > split_stride:
            raise ValueError(
                "Evaluation scenario window exceeds reserved split stride: "
                f"split='{split_name}', base_start_seed={base_start_seed}, "
                f"num_scenarios={overrides['num_scenarios']}, "
                f"stride={split_stride}."
            )
    return overrides


def eval_base_seed_from_env_overrides(
    eval_env_overrides: dict[str, object],
    cfg: DictConfig,
) -> int:
    """Return the first valid MetaDrive scenario seed for an evaluation env."""
    return int(eval_env_overrides.get("start_seed", cfg.env.config.start_seed))


def train_reset_seed_from_env_overrides(
    train_env_overrides: dict[str, object] | None,
    cfg: DictConfig,
    reset_offset: int,
) -> int:
    """Return a deterministic training reset seed inside the configured train pool."""
    overrides = dict(train_env_overrides or {})
    start_seed = int(overrides.get("start_seed", cfg.env.config.start_seed))
    num_scenarios = int(overrides.get("num_scenarios", cfg.env.config.num_scenarios))
    if num_scenarios <= 0:
        raise ValueError(f"Training env `num_scenarios` must be > 0, got {num_scenarios}.")
    return start_seed + (int(reset_offset) % num_scenarios)


def train_episode_seed_from_env_overrides(
    train_env_overrides: dict[str, object] | None,
    cfg: DictConfig,
    *,
    run_seed: int,
    chunk_id: int,
    episode_index: int,
    stage_index: int = 0,
) -> int:
    """Return a deterministic pseudo-random training scenario seed inside the train pool."""
    overrides = dict(train_env_overrides or {})
    start_seed = int(overrides.get("start_seed", cfg.env.config.start_seed))
    num_scenarios = int(overrides.get("num_scenarios", cfg.env.config.num_scenarios))
    if num_scenarios <= 0:
        raise ValueError(f"Training env `num_scenarios` must be > 0, got {num_scenarios}.")

    payload = f"{int(run_seed)}:{int(stage_index)}:{int(chunk_id)}:{int(episode_index)}".encode(
        "ascii"
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    scenario_offset = int.from_bytes(digest, byteorder="big", signed=False) % num_scenarios
    return start_seed + scenario_offset
