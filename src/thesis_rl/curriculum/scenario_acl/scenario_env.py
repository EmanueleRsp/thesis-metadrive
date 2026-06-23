from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from thesis_rl.curriculum.config import ScenarioAclScenarioEnvConfig
from thesis_rl.curriculum.scenario_acl.record import ScenarioRecord
from thesis_rl.envs.factory import _configure_agent_observation, _resolve_agent_policy
from thesis_rl.runtime.wiring.builders import maybe_wrap_env_with_reward_manager


_SUPPORTED_SCENARIO_ENV_KEYS = {
    "horizon",
    "truncate_as_terminate",
    "out_of_route_done",
    "crash_vehicle_done",
    "crash_object_done",
    "crash_human_done",
    "reactive_traffic",
    "agent_policy",
    "log_level",
}


def scenario_env_runtime_config(
    scenario_env_cfg: ScenarioAclScenarioEnvConfig,
) -> dict[str, object]:
    raw_env_config = asdict(scenario_env_cfg)
    return {
        key: value
        for key, value in raw_env_config.items()
        if key in _SUPPORTED_SCENARIO_ENV_KEYS
    }


def build_scenario_replay_env(
    cfg: DictConfig,
    *,
    record: ScenarioRecord,
    scenario_env_cfg: ScenarioAclScenarioEnvConfig,
):
    from metadrive.envs.scenario_env import ScenarioEnv

    data_directory = Path(record.dataset_directory)
    if not data_directory.exists():
        raise FileNotFoundError(
            "Scenario replay dataset directory does not exist: "
            f"{data_directory}"
        )

    env_config = scenario_env_runtime_config(scenario_env_cfg)
    env_config.update(
        {
            "data_directory": str(data_directory),
            "num_scenarios": 1,
            "start_scenario_index": int(record.scenario_index),
            "sequential_seed": False,
            "agent_policy": _resolve_agent_policy(env_config.get("agent_policy", "env_input_policy")),
        }
    )

    if "obs" in cfg and cfg.get("obs") is not None:
        obs_cfg = cfg.get("obs")
        if obs_cfg is not None:
            resolved_obs_cfg = OmegaConf.to_container(obs_cfg, resolve=True)
            if isinstance(resolved_obs_cfg, dict):
                _configure_agent_observation(env_config, resolved_obs_cfg)

    env = ScenarioEnv(env_config)
    return maybe_wrap_env_with_reward_manager(env, cfg)
