from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

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

_LOGGER = logging.getLogger(__name__)


class _CrashLoggingEnvWrapper(gym.Wrapper):
    """Persist worker-side traceback on reset/step failures."""

    def __init__(self, env: gym.Env, crash_log_path: Path) -> None:
        super().__init__(env)
        self._crash_log_path = crash_log_path

    @property
    def start_index(self) -> Any:
        return getattr(self.env, "start_index")

    @property
    def num_scenarios(self) -> Any:
        return getattr(self.env, "num_scenarios")

    def get_worker_seed_bounds(self) -> tuple[int, int]:
        """Return (start_index, num_scenarios) from the base env without deprecated wrapper fallback."""
        base_env = getattr(self, "unwrapped", self.env)
        return int(getattr(base_env, "start_index")), int(getattr(base_env, "num_scenarios"))

    def _log_and_reraise(self) -> None:
        try:
            self._crash_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._crash_log_path.write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        raise

    def reset(self, **kwargs):
        try:
            return self.env.reset(**kwargs)
        except Exception:
            self._log_and_reraise()

    def step(self, action):
        try:
            return self.env.step(action)
        except Exception:
            self._log_and_reraise()


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
    if train_num_envs(cfg) > 1 and name == "neural_adapter":
        raise ValueError(
            "Vectorized training currently supports only stateless adapters "
            "(identity/direct_action or policy_adapter). Set env.vectorized.num_envs=1 "
            "or use adapter=identity/policy_adapter."
        )

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


def train_num_envs(cfg: DictConfig) -> int:
    vector_cfg = cfg.env.get("vectorized", {})
    if not bool(vector_cfg.get("enabled", False)):
        return 1
    return max(int(vector_cfg.get("num_envs", 1)), 1)


def is_vectorized_training_enabled(cfg: DictConfig) -> bool:
    return train_num_envs(cfg) > 1


def _worker_env_overrides(
    cfg: DictConfig,
    env_overrides: dict[str, Any] | None,
    *,
    rank: int,
    num_envs: int,
) -> dict[str, Any]:
    overrides = dict(env_overrides or {})
    base_start_seed = int(overrides.get("start_seed", cfg.env.config.start_seed))
    total_scenarios = int(overrides.get("num_scenarios", cfg.env.config.num_scenarios))
    if total_scenarios <= 0:
        raise ValueError(f"Training env `num_scenarios` must be > 0, got {total_scenarios}.")

    worker_scenarios = max(total_scenarios // int(num_envs), 1)
    worker_start_seed = base_start_seed + int(rank) * worker_scenarios
    overrides["start_seed"] = int(worker_start_seed)
    overrides["num_scenarios"] = int(worker_scenarios)
    return overrides


def build_train_env(cfg: DictConfig, env_overrides: dict[str, Any] | None = None):
    num_envs = train_num_envs(cfg)
    if num_envs <= 1:
        return build_env(cfg, env_overrides)

    cfg_plain = OmegaConf.to_container(cfg, resolve=True)
    start_method = str(cfg.env.get("vectorized", {}).get("start_method", "forkserver"))
    env_name = str(cfg.env.get("name", "")).lower()
    if env_name == "metadrive" and start_method != "spawn":
        _LOGGER.warning(
            "MetaDrive vectorized training with start_method='%s' may be unstable. "
            "Recommended: env.vectorized.start_method='spawn'.",
            start_method,
        )

    def make_thunk(rank: int):
        worker_overrides = _worker_env_overrides(
            cfg,
            env_overrides,
            rank=rank,
            num_envs=num_envs,
        )

        def _init():
            worker_cfg = OmegaConf.create(cfg_plain)
            try:
                env = build_env(worker_cfg, worker_overrides)
            except Exception:
                # Persist worker traceback so parent can inspect the real cause of EOFError.
                try:
                    logs_dir = Path(str(worker_cfg.paths.logs_dir))
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    crash_log = logs_dir / f"subproc_worker_{rank}_crash.log"
                    crash_log.write_text(traceback.format_exc(), encoding="utf-8")
                except Exception:
                    pass
                raise
            logs_dir = Path(str(worker_cfg.paths.logs_dir))
            crash_log = logs_dir / f"subproc_worker_{rank}_crash.log"
            return _CrashLoggingEnvWrapper(env, crash_log)

        return _init

    return SubprocVecEnv(
        [make_thunk(rank) for rank in range(num_envs)],
        start_method=start_method,
    )


def set_planner_env_if_compatible(planner: BasePlanner, env: Any) -> None:
    model = getattr(planner, "model", getattr(planner, "sb3_model", None))
    current_n_envs = getattr(model, "n_envs", None)
    next_n_envs = int(env.num_envs) if isinstance(env, VecEnv) else 1
    if current_n_envs is not None and int(current_n_envs) != int(next_n_envs):
        return
    planner.set_env(env)
