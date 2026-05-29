from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def _to_plain_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _resolve_agent_policy(policy_name: str):
    name = policy_name.lower()
    if name in {"env_input_policy", "env_input"}:
        from metadrive.policy.env_input_policy import EnvInputPolicy

        return EnvInputPolicy

    if name in {"thesis_policy_bridge", "thesis_bridge"}:
        from thesis_rl.policies.metadrive_policy_bridge import ThesisPolicyBridge

        return ThesisPolicyBridge

    if name in {"expert_policy", "expert"}:
        from metadrive.policy.expert_policy import ExpertPolicy

        return ExpertPolicy

    if name in {"idm_policy", "idm"}:
        from metadrive.policy.idm_policy import IDMPolicy

        return IDMPolicy

    raise ValueError(
        f"Unsupported env policy '{policy_name}'. "
        "Supported values: env_input_policy, thesis_policy_bridge, expert_policy, idm_policy"
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
        from thesis_rl.envs.semantic_state_observation import SemanticStateObservation

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


def make_env(cfg_env: Any):
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
        if policy_name.lower() == "thesis_policy_bridge":
            agent_policy_cls.POLICY_LOW = float(policy_mode_cfg.get("low", -1.0))
            agent_policy_cls.POLICY_HIGH = float(policy_mode_cfg.get("high", 1.0))

        env_cfg["agent_policy"] = agent_policy_cls
    elif isinstance(env_cfg.get("agent_policy"), str):
        # Allow direct config override like env_overrides["agent_policy"] = "expert_policy".
        env_cfg["agent_policy"] = _resolve_agent_policy(str(env_cfg["agent_policy"]))

    return MetaDriveEnv(env_cfg)
