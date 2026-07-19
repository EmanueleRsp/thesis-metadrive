from __future__ import annotations

import logging
import os
import traceback
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf

from thesis_rl.agent.adapters.interfaces.base import BaseAdapter
from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.planners.core.utils import count_envs
from thesis_rl.envs.factory import make_env
from thesis_rl.envs.wrappers import RuleRewardWrapper
from thesis_rl.rulebook.v2.wrapper import RulebookV2MonitorWrapper
from thesis_rl.agent.preprocessors.interfaces.base import BasePreprocessor
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor
from thesis_rl.contracts.reward_semantics import (
    assert_reward_semantics_compatible,
    build_reward_semantics_identity,
)
from thesis_rl.reward.managers.hybrid_rulebook_manager import HybridRulebookRewardManager
from thesis_rl.reward.scalarization import RulebookScalarizer, ScalarizationConfig
from thesis_rl.runtime.execution.deterministic_subproc_vec_env import DeterministicSubprocVecEnv

if TYPE_CHECKING:
    from thesis_rl.agent.planners.interfaces.planner import BasePlanner

_LOGGER = logging.getLogger(__name__)


def collect_scenario_runtime_stats(env: Any) -> dict[str, Any] | None:
    """Collect and merge ScenarioNet counters from direct or vectorized envs."""

    if hasattr(env, "env_method"):
        raw_stats = env.env_method("get_runtime_stats")
    else:
        base_env = getattr(env, "unwrapped", env)
        getter = getattr(base_env, "get_runtime_stats", None)
        raw_stats = [getter()] if callable(getter) else []
    if not raw_stats:
        return None

    merged: dict[str, Any] = {
        "resets": 0,
        "steps": 0,
        "episodes": 0,
        "resets_by_source": Counter(),
        "steps_by_source": Counter(),
        "episodes_by_arm": Counter(),
        "resets_by_source_arm": {},
        "steps_by_source_arm": {},
        "episodes_by_source_arm": {},
        "termination_reasons": Counter(),
    }
    for stats in raw_stats:
        if not isinstance(stats, dict):
            continue
        for key in ("resets", "steps", "episodes"):
            merged[key] += int(stats.get(key, 0))
        for key in (
            "resets_by_source",
            "steps_by_source",
            "episodes_by_arm",
            "termination_reasons",
        ):
            values = stats.get(key, {})
            if isinstance(values, dict):
                merged[key].update({str(name): int(count) for name, count in values.items()})
        for key in ("resets_by_source_arm", "steps_by_source_arm", "episodes_by_source_arm"):
            values = stats.get(key, {})
            if isinstance(values, dict):
                _merge_nested_runtime_counts(merged[key], values)
    return {
        key: (dict(value) if isinstance(value, Counter) else value) for key, value in merged.items()
    }


def merge_scenario_runtime_stats(
    accumulated: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Add one worker/chunk counter snapshot to an accumulated snapshot."""

    if current is None:
        return accumulated
    if accumulated is None:
        return {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in current.items()
        }
    for key in ("resets", "steps", "episodes"):
        accumulated[key] = int(accumulated.get(key, 0)) + int(current.get(key, 0))
    for key in (
        "resets_by_source",
        "steps_by_source",
        "episodes_by_arm",
        "termination_reasons",
    ):
        target = accumulated.setdefault(key, {})
        if not isinstance(target, dict):
            target = accumulated[key] = {}
        values = current.get(key, {})
        if isinstance(values, dict):
            for name, count in values.items():
                target[str(name)] = int(target.get(str(name), 0)) + int(count)
    for key in ("resets_by_source_arm", "steps_by_source_arm", "episodes_by_source_arm"):
        target = accumulated.setdefault(key, {})
        values = current.get(key, {})
        if isinstance(target, dict) and isinstance(values, dict):
            _merge_nested_runtime_counts(target, values)
    return accumulated


def _merge_nested_runtime_counts(target: dict[str, Any], values: dict[str, Any]) -> None:
    """Merge JSON-safe ``source -> arm -> count`` runtime matrices."""

    for source, arms in values.items():
        if not isinstance(arms, dict):
            continue
        source_target = target.setdefault(str(source), {})
        if not isinstance(source_target, dict):
            source_target = target[str(source)] = {}
        for arm, count in arms.items():
            source_target[str(arm)] = int(source_target.get(str(arm), 0)) + int(count)


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
    rulebook_name = str(cfg.reward.get("rulebook_config", "")).strip()
    if not rulebook_name:
        raise ValueError(
            "Reward behavior requires `reward.rulebook_config` to be set "
            "(example: reward.rulebook_config=selection)."
        )
    if rulebook_name == "none":
        raise ValueError(
            "reward.rulebook_config='none' is invalid when rulebook behavior is active."
        )

    repo_root = Path(__file__).resolve().parents[4]
    rulebook_path = repo_root / "conf" / "rulebook" / f"{rulebook_name}.yaml"
    if not rulebook_path.exists():
        raise FileNotFoundError(
            f"Rulebook config file not found: {rulebook_path}. "
            f"Check `reward.rulebook_config={rulebook_name}`."
        )

    loaded = OmegaConf.load(rulebook_path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"Loaded rulebook config is not a DictConfig mapping: path={rulebook_path}")
    return loaded


def adapter_space_kwargs(action_space) -> dict[str, object]:
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(
            f"Identity adapter requires a Box action space, got {type(action_space).__name__}"
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
    name = str(cfg.agent.preprocessor.name).lower()
    if name == "identity":
        return IdentityPreprocessor()
    raise ValueError(f"Unsupported preprocessor: {cfg.agent.preprocessor.name}")


def build_adapter(cfg: DictConfig, common_kwargs: dict[str, object]) -> BaseAdapter:
    name = str(cfg.agent.adapter.name).lower()
    if name == "identity":
        return IdentityAdapter(**common_kwargs)

    raise ValueError(
        f"Unsupported adapter: {cfg.agent.adapter.name}. "
        "Only agent/adapter=identity is currently supported."
    )


def _resolve_planner_cfg(cfg: DictConfig) -> DictConfig:
    """Merge run-profile planner overrides on top of algorithm defaults."""
    base_cfg = OmegaConf.create(OmegaConf.to_container(cfg.agent.planner.algorithm, resolve=True))
    planner_overrides = cfg.get("planner")
    if planner_overrides is None:
        return base_cfg
    return OmegaConf.merge(base_cfg, planner_overrides)


def build_planner(cfg: DictConfig, env: Any, seed: int | None = None) -> "BasePlanner":
    from thesis_rl.agent.planners.factory import build_planner_backend

    return build_planner_backend(
        planner_name=str(cfg.agent.planner.algorithm.name),
        env=env,
        cfg_planner=_resolve_planner_cfg(cfg),
        cfg_encoder=cfg.agent.planner.encoder,
        cfg_decoder=cfg.agent.planner.decoder,
        cfg_obs=cfg.get("obs"),
        device=str(cfg.device),
        seed=seed,
    )


def load_planner(cfg: DictConfig, checkpoint_path: str, env: Any) -> "BasePlanner":
    from thesis_rl.agent.planners.factory import load_planner_backend

    assert_reward_semantics_compatible(
        checkpoint_path,
        build_reward_semantics_identity(cfg),
    )
    return load_planner_backend(
        planner_name=str(cfg.agent.planner.algorithm.name),
        checkpoint_path=checkpoint_path,
        env=env,
        cfg_planner=_resolve_planner_cfg(cfg),
        cfg_encoder=cfg.agent.planner.encoder,
        cfg_decoder=cfg.agent.planner.decoder,
        cfg_obs=cfg.get("obs"),
        device=str(cfg.device),
    )


def maybe_wrap_env_with_reward_manager(env, cfg: DictConfig):
    # v1 remains the default and keeps its historical reward manager path.
    # v2 adapters are deliberately explicit: the live environment must supply
    # canonical snapshot/cache factories, otherwise silently falling back to
    # v1 would violate the fail-fast contract.
    rulebook_version = str(cfg.get("rulebook", {}).get("version", "v1")).lower()
    if rulebook_version in {
        "v2",
        "v4.6",
        "v4.7",
        "4.6",
        "4.7",
        "4.6-final-implementation-complete",
        "4.7-final-implementation-complete",
    }:
        adapter = getattr(env, "rulebook_v2_adapter", None)
        if adapter is None:
            adapter = getattr(getattr(env, "unwrapped", None), "rulebook_v2_adapter", None)
        adapter_factory = None
        if adapter is None:
            adapter_factory = getattr(env, "make_rulebook_v2_adapter", None)
            if not callable(adapter_factory):
                raise ValueError(
                    "Rulebook v2 requires env.rulebook_v2_adapter with snapshotter, "
                    "transition_evaluator, initial_memory and initial_cache"
                )
            setattr(getattr(env, "unwrapped", env), "_rulebook_v2_requested", True)
        scalarizer = None
        if str(cfg.reward.behavior).lower() == "scalar_reward":
            scalarization_cfg = cfg.get("scalarization")
            if scalarization_cfg is None:
                raise ValueError(
                    "scalar_reward with Rulebook v2 requires a `scalarization` configuration."
                )
            scalarizer = RulebookScalarizer(
                ScalarizationConfig.from_mapping(
                    OmegaConf.to_container(scalarization_cfg, resolve=True)
                )
            )
        if adapter_factory is not None:
            def snapshotter(_env):
                raise RuntimeError("Deferred Rulebook v2 adapter is unavailable before reset")

            def transition_evaluator(**_kwargs):
                raise RuntimeError("Deferred Rulebook v2 adapter is unavailable before reset")

            initial_memory = None
            initial_cache = None
        else:
            snapshotter = adapter.snapshotter
            transition_evaluator = adapter.transition_evaluator
            initial_memory = adapter.initial_memory
            initial_cache = adapter.initial_cache
        return RulebookV2MonitorWrapper(
            env,
            snapshotter=snapshotter,
            transition_evaluator=transition_evaluator,
            initial_memory=initial_memory,
            initial_cache=initial_cache,
            scalarizer=scalarizer,
            adapter_factory=adapter_factory,
        )
    mode = str(cfg.reward.behavior).lower()
    if mode == "off":
        return env

    supported_modes = {
        "monitor_only",
        "scalar_reward",
        "hybrid",
        "lexicographic",
    }
    if mode not in supported_modes:
        raise ValueError(
            "Unsupported reward mode. "
            f"Got behavior='{cfg.reward.behavior}', expected one of: "
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
        runtime_info_debug_enabled=bool(cfg.reward.get("runtime_info_debug_enabled", False)),
        runtime_info_debug_path=cfg.reward.get("runtime_info_debug_path"),
        logger_level=cfg.get("logging", {}).get("console_level", "INFO"),
    )


def merge_env_config_with_overrides(
    cfg_env: DictConfig, env_overrides: dict[str, Any]
) -> DictConfig:
    merged_cfg_env = OmegaConf.create(OmegaConf.to_container(cfg_env, resolve=True))
    config_overrides = dict(env_overrides)
    # ScenarioNet keeps split/provider/episode-control outside the native
    # MetaDrive config. Moving these fields here prevents unknown-key errors
    # while allowing evaluation to select validation/test runtime views.
    for top_level_key in (
        "split",
        "catalog_path",
        "global_seed",
        "provider",
        "episode_control",
    ):
        if top_level_key in config_overrides:
            override = config_overrides.pop(top_level_key)
            if top_level_key == "provider" and isinstance(override, dict):
                current = getattr(merged_cfg_env, top_level_key, {})
                setattr(merged_cfg_env, top_level_key, OmegaConf.merge(current, override))
            else:
                setattr(merged_cfg_env, top_level_key, override)
    merged_cfg_env.config = OmegaConf.merge(merged_cfg_env.config, config_overrides)
    return merged_cfg_env


def _attach_observation_group(cfg: DictConfig, cfg_env: DictConfig) -> DictConfig:
    if "obs" not in cfg:
        return cfg_env
    obs_cfg = cfg.get("obs")
    if obs_cfg is None:
        return cfg_env
    merged_cfg_env = OmegaConf.create(OmegaConf.to_container(cfg_env, resolve=True))
    merged_cfg_env.observation = OmegaConf.create(OmegaConf.to_container(obs_cfg, resolve=True))
    return merged_cfg_env


def build_env(cfg: DictConfig, env_overrides: dict[str, Any] | None = None):
    cfg_env = _attach_observation_group(cfg, cfg.env)
    if env_overrides:
        cfg_env = merge_env_config_with_overrides(cfg_env, env_overrides)

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
    env_name = str(cfg.env.get("name", "")).lower()
    if env_name == "scenarionet":
        base_start_seed = int(
            overrides.get(
                "start_scenario_index",
                cfg.env.config.get("start_scenario_index", 0),
            )
        )
    else:
        base_start_seed = int(overrides.get("start_seed", cfg.env.config.start_seed))
    total_scenarios = int(overrides.get("num_scenarios", cfg.env.config.num_scenarios))
    if env_name == "scenarionet" and total_scenarios <= 0:
        catalog_path = cfg.env.get("catalog_path")
        if not catalog_path:
            dataset_root = cfg.env.get("dataset_root") or os.environ.get("SCENARIONET_DATA_ROOT")
            if dataset_root:
                catalog_path = Path(str(dataset_root)) / "catalog" / "scenario_catalog.parquet"
        if not catalog_path:
            raise ValueError(
                "ScenarioNet vectorization requires env.dataset_root or env.catalog_path "
                "when num_scenarios=-1."
            )
        from thesis_rl.scenarios.catalog import read_scenario_catalog

        catalog = read_scenario_catalog(str(catalog_path))
        total_scenarios = len(catalog.valid_records(split=str(cfg.env.get("split", "train"))))
    provider_kind = str(cfg.env.get("provider", {}).get("kind", "uniform")).lower()
    if env_name == "scenarionet" and provider_kind == "uniform" and int(num_envs) > 1:
        # Keep the native ScenarioDataManager on one complete index range and
        # partition provider records round-robin. This preserves both sources
        # in every worker and lets provider-driven auto-reset remain strict.
        overrides["start_scenario_index"] = int(base_start_seed)
        overrides["num_scenarios"] = int(total_scenarios)
        overrides["worker_index"] = 0
        overrides["num_workers"] = 1
        overrides["provider_worker_index"] = int(rank)
        overrides["provider_worker_count"] = int(num_envs)
        return overrides
    if total_scenarios <= 0:
        raise ValueError(f"Training env `num_scenarios` must be > 0, got {total_scenarios}.")

    num_envs_int = int(num_envs)
    base = max(total_scenarios // num_envs_int, 1)
    remainder = max(total_scenarios - base * num_envs_int, 0)
    worker_scenarios = base + (1 if int(rank) < remainder else 0)
    worker_start_seed = base_start_seed + int(rank) * base + min(int(rank), remainder)
    overrides["start_scenario_index" if env_name == "scenarionet" else "start_seed"] = int(
        worker_start_seed
    )
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
            from thesis_rl.runtime.execution.seeding import set_global_seed

            worker_cfg = OmegaConf.create(cfg_plain)
            # Seed worker RNGs deterministically to improve reproducibility across runs.
            try:
                base_worker_seed = int(worker_cfg.get("seed", 0))
            except Exception:
                base_worker_seed = 0
            try:
                set_global_seed(base_worker_seed + int(rank))
            except Exception:
                pass
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

    return DeterministicSubprocVecEnv(
        [make_thunk(rank) for rank in range(num_envs)],
        start_method=start_method,
    )


def set_planner_env_if_compatible(planner: "BasePlanner", env: Any) -> None:
    next_n_envs = count_envs(env)
    current_n_envs = int(getattr(planner, "n_envs", next_n_envs))
    replay_n_envs = int(getattr(planner, "replay_buffer_n_envs", lambda: current_n_envs)())
    if replay_n_envs != next_n_envs or current_n_envs != next_n_envs:
        raise RuntimeError(
            "Cannot change n_envs on an initialized planner: "
            f"planner.n_envs={current_n_envs}, replay_buffer.n_envs={replay_n_envs}, env.n_envs={next_n_envs}. "
            "Recreate planner or rebuild env with the same number of workers."
        )

    planner.set_env(env)
