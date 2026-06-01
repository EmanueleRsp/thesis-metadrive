from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "adapter_space_kwargs": ("thesis_rl.runtime.builders", "adapter_space_kwargs"),
    "build_adapter": ("thesis_rl.runtime.builders", "build_adapter"),
    "build_env": ("thesis_rl.runtime.builders", "build_env"),
    "build_planner": ("thesis_rl.runtime.builders", "build_planner"),
    "build_preprocessor": ("thesis_rl.runtime.builders", "build_preprocessor"),
    "load_planner": ("thesis_rl.runtime.builders", "load_planner"),
    "maybe_wrap_env_with_reward_manager": ("thesis_rl.runtime.builders", "maybe_wrap_env_with_reward_manager"),
    "merge_env_config_with_overrides": ("thesis_rl.runtime.builders", "merge_env_config_with_overrides"),
    "save_run_metadata": ("thesis_rl.runtime.metadata", "save_run_metadata"),
    "update_run_metadata": ("thesis_rl.runtime.metadata", "update_run_metadata"),
    "log_event": ("thesis_rl.runtime.run_logging", "log_event"),
    "setup_file_logger": ("thesis_rl.runtime.run_logging", "setup_file_logger"),
    "apply_eval_scenario_seed_split": ("thesis_rl.runtime.seeding", "apply_eval_scenario_seed_split"),
    "eval_base_seed_from_env_overrides": ("thesis_rl.runtime.seeding", "eval_base_seed_from_env_overrides"),
    "seed_env_spaces": ("thesis_rl.runtime.seeding", "seed_env_spaces"),
    "set_global_seed": ("thesis_rl.runtime.seeding", "set_global_seed"),
    "train_episode_seed_from_env_overrides": ("thesis_rl.runtime.seeding", "train_episode_seed_from_env_overrides"),
    "train_reset_seed_from_env_overrides": ("thesis_rl.runtime.seeding", "train_reset_seed_from_env_overrides"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value
