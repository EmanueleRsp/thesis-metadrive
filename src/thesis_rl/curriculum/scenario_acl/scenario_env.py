from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

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


def _is_scenarionet_env(cfg: DictConfig) -> bool:
    env_cfg = cfg.get("env", {})
    name = str(env_cfg.get("name", "")).strip().lower()
    env_id = str(env_cfg.get("env_id", "")).strip().lower()
    return name == "scenarionet" or env_id == "thesisscenarioenv"


def _build_scenarionet_replay_config(
    cfg: DictConfig,
    *,
    record: ScenarioRecord,
    scenario_env_cfg: ScenarioAclScenarioEnvConfig,
) -> dict[str, object]:
    """Build the canonical ScenarioNet config for one ACL replay record.

    Scenario ACL has a legacy ``ScenarioEnv`` configuration with a fixed
    horizon.  That configuration must not leak into the ScenarioNet runtime:
    ScenarioNet episodes use the exported scenario length plus the thesis
    tail controlled by ``env.episode_control``.
    """

    env_cfg = cfg.get("env", {})
    raw_base_config = OmegaConf.to_container(env_cfg.get("config", {}), resolve=True)
    if not isinstance(raw_base_config, dict):
        raise TypeError("cfg.env.config must resolve to a mapping for ScenarioNet replay.")

    env_config: dict[str, object] = {
        str(key): value for key, value in raw_base_config.items()
    }
    acl_config = scenario_env_runtime_config(scenario_env_cfg)

    # These settings are still useful ACL overrides, while the episode
    # contract itself remains owned by the ScenarioNet environment config.
    for key in (
        "truncate_as_terminate",
        "out_of_route_done",
        "crash_vehicle_done",
        "crash_object_done",
        "crash_human_done",
        "reactive_traffic",
        "agent_policy",
        "log_level",
    ):
        if key in acl_config:
            env_config[key] = acl_config[key]

    # Native MetaDrive horizon handling is deliberately disabled.  The
    # ThesisScenarioEnv hook adds MAX_STEP at scenario_length + tail instead.
    env_config["horizon"] = None
    env_config["allowed_more_steps"] = None
    env_config["truncate_as_terminate"] = False

    episode_control = env_cfg.get("episode_control", {})
    extra_steps = int(episode_control.get("extra_steps_after_scenario", 50))
    if extra_steps < 0:
        raise ValueError("env.episode_control.extra_steps_after_scenario must be non-negative.")

    env_config.update(
        {
            "data_directory": str(Path(record.dataset_directory)),
            "num_scenarios": 1,
            "start_scenario_index": int(record.scenario_index),
            "sequential_seed": False,
            "worker_index": 0,
            "num_workers": 1,
            "extra_steps_after_scenario": extra_steps,
            "agent_policy": _resolve_agent_policy(
                str(env_config.get("agent_policy", "env_input_policy"))
            ),
        }
    )
    return env_config


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
    data_directory = Path(record.dataset_directory)
    if not data_directory.exists():
        raise FileNotFoundError(
            "Scenario replay dataset directory does not exist: "
            f"{data_directory}"
        )

    if _is_scenarionet_env(cfg):
        from thesis_rl.envs.thesis_scenario_env import ThesisScenarioEnv

        env_config = _build_scenarionet_replay_config(
            cfg,
            record=record,
            scenario_env_cfg=scenario_env_cfg,
        )
        split = str(cfg.env.get("split", record.env_config.get("split", "train")))
    else:
        from metadrive.envs.scenario_env import ScenarioEnv  # type: ignore[import-not-found]

        env_config = scenario_env_runtime_config(scenario_env_cfg)
        env_config.update(
            {
                "data_directory": str(data_directory),
                "num_scenarios": 1,
                "start_scenario_index": int(record.scenario_index),
                "sequential_seed": False,
                "agent_policy": _resolve_agent_policy(
                    str(env_config.get("agent_policy", "env_input_policy"))
                ),
            }
        )

    if "obs" in cfg and cfg.get("obs") is not None:
        obs_cfg = cfg.get("obs")
        if obs_cfg is not None:
            resolved_obs_cfg = OmegaConf.to_container(obs_cfg, resolve=True)
            if isinstance(resolved_obs_cfg, dict):
                observation_config: dict[str, Any] = {
                    str(key): value for key, value in resolved_obs_cfg.items()
                }
                _configure_agent_observation(env_config, observation_config)

    if _is_scenarionet_env(cfg):
        env = ThesisScenarioEnv(
            env_config,
            split=split,
        )
        # Keep the ACL record available for runtime metadata and consistency
        # checks, even though the single-scenario range already fixes resets.
        env.current_scenario_record = record
    else:
        env = ScenarioEnv(env_config)
    return maybe_wrap_env_with_reward_manager(env, cfg)
