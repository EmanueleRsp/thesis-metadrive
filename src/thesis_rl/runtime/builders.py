from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf

from thesis_rl.adapters.base import BaseAdapter
from thesis_rl.adapters.identity import IdentityAdapter
from thesis_rl.adapters.neural_adapter import NeuralAdapter
from thesis_rl.adapters.policy_adapter import PolicyAdapter
from thesis_rl.agents.base import BasePlanner
from thesis_rl.agents.planner_agent import Td3PlannerBackend
from thesis_rl.envs.factory import make_env
from thesis_rl.envs.wrappers import RuleRewardWrapper
from thesis_rl.preprocessors.base import BasePreprocessor
from thesis_rl.preprocessors.identity import IdentityPreprocessor
from thesis_rl.reward.reward_manager import HybridRulebookRewardManager


def _load_rulebook_cfg_from_reward(cfg: DictConfig) -> DictConfig:
    rulebook_name = str(cfg.reward.get("rulebook", "")).strip()
    if not rulebook_name:
        raise ValueError(
            "Reward mode 'rulebook' requires `reward.rulebook` to be set "
            "(example: reward.rulebook=selection)."
        )

    repo_root = Path(__file__).resolve().parents[3]
    rulebook_path = repo_root / "conf" / "rulebook" / f"{rulebook_name}.yaml"
    if not rulebook_path.exists():
        raise FileNotFoundError(
            f"Rulebook config file not found: {rulebook_path}. "
            f"Check `reward.rulebook={rulebook_name}`."
        )

    loaded = OmegaConf.load(rulebook_path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(
            "Loaded rulebook config is not a DictConfig mapping: "
            f"path={rulebook_path}"
        )
    return loaded


def adapter_space_kwargs(action_space) -> dict[str, object]:
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(
            f"Identity/policy/neural adapters require a Box action space, got {type(action_space).__name__}"
        )

    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)

    if low.shape == ():
        low_value = float(low)
        high_value = float(high)
    else:
        if not np.allclose(low, low.flat[0]) or not np.allclose(high, high.flat[0]):
            raise ValueError(
                "Adapter currently expects uniform Box bounds across all action dimensions"
            )
        low_value = float(low.flat[0])
        high_value = float(high.flat[0])

    return {
        "low": low_value,
        "high": high_value,
        "expected_shape": tuple(action_space.shape),
    }


def build_preprocessor(cfg: DictConfig) -> BasePreprocessor:
    name = str(cfg.preprocessor.name).lower()
    if name == "identity":
        return IdentityPreprocessor()
    raise ValueError(f"Unsupported preprocessor: {cfg.preprocessor.name}")


def build_adapter(cfg: DictConfig, common_kwargs: dict[str, object]) -> BaseAdapter:
    name = str(cfg.adapter.name).lower()

    if name in {"identity", "direct_action"}:
        return IdentityAdapter(**common_kwargs)

    if name == "neural_adapter":
        return NeuralAdapter(
            **common_kwargs,
            clip=bool(cfg.adapter.get("clip", True)),
            hidden_dim=int(cfg.adapter.get("hidden_dim", 64)),
            learning_rate=float(cfg.adapter.get("learning_rate", 1e-3)),
            batch_size=int(cfg.adapter.get("batch_size", 64)),
            update_interval=int(cfg.adapter.get("update_interval", 1)),
            buffer_capacity=int(cfg.adapter.get("buffer_capacity", 10000)),
            device=str(cfg.device),
        )

    if name == "policy_adapter":
        return PolicyAdapter(
            **common_kwargs,
            clip=bool(cfg.adapter.get("clip", True)),
            policy_name=str(cfg.adapter.get("policy_name", "EnvInputPolicy")),
            action_check=bool(cfg.adapter.get("action_check", True)),
        )

    raise ValueError(f"Unsupported adapter: {name}")


def build_planner(cfg: DictConfig, env: Any, seed: int | None = None) -> BasePlanner:
    name = str(cfg.planner.name).lower()
    if name in {"td3", "sb3_td3"}:
        return Td3PlannerBackend.build(
            env=env,
            cfg_planner=cfg.planner,
            device=str(cfg.device),
            seed=seed,
        )
    raise ValueError(f"Unsupported planner backend: {cfg.planner.name}")


def load_planner(cfg: DictConfig, checkpoint_path: str, env: Any) -> BasePlanner:
    name = str(cfg.planner.name).lower()
    if name in {"td3", "sb3_td3"}:
        return Td3PlannerBackend.load(
            checkpoint_path=checkpoint_path,
            env=env,
            device=str(cfg.device),
        )
    raise ValueError(f"Unsupported planner backend: {cfg.planner.name}")


def maybe_wrap_env_with_reward_manager(env, cfg: DictConfig):
    mode = str(cfg.reward.mode).lower()
    if mode == "scalar_native":
        return env

    supported_modes = {
        "scalar_default",
        "rulebook",
        "scalar_rulebook",
        "hybrid",
        "lexicographic",
    }
    if mode not in supported_modes:
        raise ValueError(
            "Unsupported reward mode. "
            f"Got mode='{cfg.reward.mode}', expected one of: "
            f"{', '.join(sorted(supported_modes))}"
        )

    cfg_rulebook = _load_rulebook_cfg_from_reward(cfg)

    manager = HybridRulebookRewardManager.from_configs(
        cfg_reward=cfg.reward,
        cfg_rulebook=cfg_rulebook,
    )
    return RuleRewardWrapper(
        env=env,
        reward_manager=manager,
        reward_mode=mode,
        attach_info=bool(cfg.reward.get("attach_info", True)),
        rule_margin_log_path=cfg.reward.get("rule_margin_log_path"),
    )


def merge_env_config_with_overrides(cfg_env: DictConfig, env_overrides: dict[str, Any]) -> DictConfig:
    merged_cfg_env = OmegaConf.create(OmegaConf.to_container(cfg_env, resolve=True))
    merged_cfg_env.config = OmegaConf.merge(merged_cfg_env.config, dict(env_overrides))
    return merged_cfg_env


def build_env(cfg: DictConfig, env_overrides: dict[str, Any] | None = None):
    cfg_env = cfg.env
    if env_overrides:
        cfg_env = merge_env_config_with_overrides(cfg.env, env_overrides)

    env = make_env(cfg_env)
    return maybe_wrap_env_with_reward_manager(env, cfg)
