from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _to_plain_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _resolve_agent_policy(policy_name: str):
    name = policy_name.lower()
    if name in {"envinputpolicy", "env_input_policy", "env_input"}:
        from metadrive.policy.env_input_policy import EnvInputPolicy

        return EnvInputPolicy

    if name in {"expertpolicy", "expert_policy", "expert"}:
        from metadrive.policy.expert_policy import ExpertPolicy

        return ExpertPolicy

    if name in {"idmpolicy", "idm_policy", "idm"}:
        from metadrive.policy.idm_policy import IDMPolicy

        return IDMPolicy

    raise ValueError(
        f"Unsupported env policy '{policy_name}'. "
        "Supported values: env_input_policy, expert_policy, idm_policy"
    )


def _configure_agent_observation(
    env_cfg: dict[str, Any],
    observation_cfg: dict[str, Any],
) -> None:
    """Mutate env_cfg to install a selected observation class/config."""
    obs_type = str(observation_cfg.get("type", "lidar_state")).strip().lower()

    if obs_type in {"lidar", "lidar_state", "lidarstateobservation"}:
        # MetaDrive default when `agent_observation` is unset and image_observation=False.
        return

    if obs_type in {"semantic", "semantic_state", "semanticstateobservation"}:
        from thesis_rl.envs.observations.semantic_state import SemanticStateObservation

        semantic_cfg = dict(observation_cfg)
        semantic_cfg.pop("type", None)
        semantic_cfg.pop("name", None)

        SemanticStateObservation.set_external_config(semantic_cfg)
        env_cfg["agent_observation"] = SemanticStateObservation

        if bool(semantic_cfg.get("disable_lidar_sensors", True)):
            vehicle_cfg = env_cfg.setdefault("vehicle_config", {})
            lidar_cfg = vehicle_cfg.setdefault("lidar", {})
            lidar_cfg["num_lasers"] = 0
            lidar_cfg["num_others"] = 0
            lidar_cfg["distance"] = 0

            side_cfg = vehicle_cfg.setdefault("side_detector", {})
            side_cfg["num_lasers"] = 0
            side_cfg["distance"] = 0

            lane_line_cfg = vehicle_cfg.setdefault("lane_line_detector", {})
            lane_line_cfg["num_lasers"] = 0
            lane_line_cfg["distance"] = 0
        return

    raise ValueError(
        f"Unsupported observation type '{obs_type}'. "
        "Supported values: lidar_state, semantic_state"
    )


def make_env(
    cfg_env: Any,
    *,
    scenario_provider: Any | None = None,
    catalog: Any | None = None,
):
    """Create a MetaDrive environment from Hydra env config."""

    # try to import metadrive, and raise a clear error if it's not installed
    try:
        from metadrive import MetaDriveEnv
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "metadrive is not installed. Run `uv sync` before training/evaluation."
        ) from exc

    # Convert Hydra config to plain dict for env initialization
    env_cfg = _to_plain_dict(cfg_env.config)

    observation_cfg = _to_plain_dict(getattr(cfg_env, "observation", {}))
    if observation_cfg:
        _configure_agent_observation(env_cfg, observation_cfg)

    # Handle policy mode configuration if enabled
    policy_mode_cfg = _to_plain_dict(getattr(cfg_env, "policy_mode", {}))
    if policy_mode_cfg.get("enabled", False):
        env_cfg["action_check"] = bool(policy_mode_cfg.get("action_check", True))
        policy_name = str(policy_mode_cfg.get("agent_policy", "env_input_policy"))
        agent_policy_cls = _resolve_agent_policy(policy_name)
        env_cfg["agent_policy"] = agent_policy_cls
    elif isinstance(env_cfg.get("agent_policy"), str):
        # Allow direct config override like env_overrides["agent_policy"] = "expert_policy".
        env_cfg["agent_policy"] = _resolve_agent_policy(str(env_cfg["agent_policy"]))

    env_name = str(getattr(cfg_env, "name", "metadrive")).strip().lower()
    env_id = str(getattr(cfg_env, "env_id", "")).strip().lower()
    if env_name == "scenarionet" or env_id == "thesisscenarioenv":
        from thesis_rl.envs.thesis_scenario_env import ThesisScenarioEnv
        from thesis_rl.scenarios.catalog import read_scenario_catalog
        from thesis_rl.scenarios.provider import (
            FixedSequenceScenarioProvider,
            UniformScenarioProvider,
        )

        split = str(getattr(cfg_env, "split", "train"))
        episode_control = _to_plain_dict(getattr(cfg_env, "episode_control", {}))
        if "extra_steps_after_scenario" in episode_control:
            env_cfg["extra_steps_after_scenario"] = int(
                episode_control["extra_steps_after_scenario"]
            )
        if not env_cfg.get("data_directory"):
            data_root = os.environ.get("SCENARIONET_DATA_ROOT")
            if data_root:
                env_cfg["data_directory"] = str(
                    (Path(data_root).expanduser() / "runtime" / split).resolve()
                )
        provider_worker_id = int(
            env_cfg.pop("provider_worker_index", env_cfg.get("worker_index", 0))
        )
        provider_worker_count = int(env_cfg.pop("provider_worker_count", 1))
        worker_id = provider_worker_id
        catalog_path = getattr(cfg_env, "catalog_path", None) or os.environ.get(
            "SCENARIONET_CATALOG_PATH"
        )
        if catalog_path and catalog is None:
            catalog = read_scenario_catalog(str(catalog_path))
        if scenario_provider is None and catalog is not None:
            provider_cfg = _to_plain_dict(getattr(cfg_env, "provider", {}))
            provider_kind = str(provider_cfg.get("kind", "uniform")).lower()
            start_index = int(env_cfg.get("start_scenario_index", 0))
            num_scenarios = int(env_cfg.get("num_scenarios", -1))
            records = tuple(
                record
                for record in catalog.records
                if (
                    provider_worker_count > 1
                    and record.runtime_index is not None
                    and record.runtime_index % provider_worker_count == provider_worker_id
                )
                or (
                    provider_worker_count <= 1
                    and (
                        num_scenarios <= 0
                        or (
                            record.runtime_index is not None
                            and start_index <= record.runtime_index < start_index + num_scenarios
                        )
                    )
                )
            )
            if not records:
                raise ValueError(
                    "ScenarioNet catalog has no records in the configured worker range: "
                    f"start_scenario_index={start_index}, num_scenarios={num_scenarios}"
                )
            if provider_kind == "uniform":
                probabilities = provider_cfg.get("source_probability", {})
                scenario_provider = UniformScenarioProvider(
                    records,
                    global_seed=int(getattr(cfg_env, "global_seed", 0)),
                    source_probabilities={
                        "waymo": float(probabilities.get("waymo", 0.5)),
                        "pg": float(probabilities.get("pg", 0.5)),
                    },
                    strict=bool(provider_cfg.get("strict", True)),
                    allow_fallback=bool(provider_cfg.get("allow_fallback", False)),
                )
            elif provider_kind == "fixed_sequence":
                scenario_provider = FixedSequenceScenarioProvider(
                    tuple(
                        sorted(
                            (record for record in records if record.split == split),
                            key=lambda record: (
                                record.runtime_index is None,
                                record.runtime_index if record.runtime_index is not None else 0,
                            ),
                        )
                    ),
                    repeat=bool(provider_cfg.get("repeat", False)),
                )
            else:
                raise ValueError(f"Unsupported ScenarioNet provider kind: {provider_kind!r}")
        return ThesisScenarioEnv(
            env_cfg,
            scenario_provider=scenario_provider,
            catalog=catalog,
            split=split,
            worker_id=worker_id,
        )

    return MetaDriveEnv(env_cfg)
