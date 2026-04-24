from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def _to_plain_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _resolve_agent_policy(policy_name: str):
    name = policy_name.lower()
    if name == "env_input_policy":
        from metadrive.policy.env_input_policy import EnvInputPolicy

        return EnvInputPolicy

    if name == "thesis_policy_bridge":
        from thesis_rl.policies.metadrive_policy_bridge import ThesisPolicyBridge

        return ThesisPolicyBridge

    raise ValueError(
        f"Unsupported env policy '{policy_name}'. "
        "Supported values: env_input_policy, thesis_policy_bridge"
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

    return MetaDriveEnv(env_cfg)
