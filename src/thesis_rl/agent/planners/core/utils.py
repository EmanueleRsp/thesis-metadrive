from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from thesis_rl.agent.planners.encoders.factory import build_encoder
from thesis_rl.contracts.observation_schema import (
    SemanticObservationSchemaV11,
    SemanticObservationSchemaV12,
)


def to_plain_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(cfg)


def normalize_checkpoint_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    if checkpoint.suffix != ".zip":
        checkpoint = checkpoint.with_suffix(".zip")
    return checkpoint


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def count_envs(env: Any) -> int:
    if hasattr(env, "get_wrapper_attr"):
        try:
            return int(env.get_wrapper_attr("num_envs"))
        except Exception:
            pass
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "num_envs"):
        return int(unwrapped.num_envs)
    try:
        return int(object.__getattribute__(env, "num_envs"))
    except Exception:
        return 1


def call_env_method(env: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call ``env_method`` without tripping gymnasium's deprecated attribute forwarding.

    ``gym.Wrapper.__getattr__`` logs a deprecation warning whenever an
    attribute (such as ``env_method``) falls through to the wrapped env
    instead of being defined on the wrapper itself. ``get_wrapper_attr``
    resolves the same attribute by walking the wrapper chain explicitly, with
    no warning, but itself raises ``AttributeError`` as soon as the chain
    reaches an env that isn't a ``gym.Wrapper`` (e.g. the innermost SB3
    ``VecEnv``, which defines ``env_method`` directly but has no
    ``get_wrapper_attr``). The fallback therefore goes through
    ``env.unwrapped`` rather than plain attribute access on ``env``:
    ``unwrapped`` is a real property that walks the chain without going
    through ``__getattr__``, so it reaches ``env_method`` without logging the
    deprecation warning at every wrapper level along the way.
    """
    if hasattr(env, "get_wrapper_attr"):
        try:
            return env.get_wrapper_attr("env_method")(method_name, *args, **kwargs)
        except AttributeError:
            pass
    base = getattr(env, "unwrapped", env)
    return base.env_method(method_name, *args, **kwargs)


def to_batch_obs(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - tau)
            tgt_param.data.add_(tau * src_param.data)


def assert_box_spaces(env: Any) -> tuple[gym.spaces.Box, gym.spaces.Box]:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError(
            f"Only Box observation spaces are supported, got {type(env.observation_space).__name__}"
        )
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError(
            f"Only Box action spaces are supported, got {type(env.action_space).__name__}"
        )
    return env.observation_space, env.action_space


def safe_atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    clipped = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return torch.atanh(clipped)


def build_encoder_for_env(cfg_encoder: Any, cfg_obs: Any, obs_dim: int):
    enc_cfg = to_plain_dict(cfg_encoder)
    enc_type = str(enc_cfg.get("type", "none")).lower()
    obs_cfg = to_plain_dict(cfg_obs)
    observation_type = str(obs_cfg.get("type", "")).lower()
    observation_schema = (
        SemanticObservationSchemaV11()
        if observation_type in {"semantic", "semantic_state", "semantic_v2"}
        else SemanticObservationSchemaV12() if observation_type == "semantic_v3" else None
    )
    if enc_type == "none":
        enc_cfg["output_dim"] = obs_dim
    return build_encoder(
        cfg_encoder=enc_cfg,
        input_dim=obs_dim,
        observation_schema=observation_schema,
    )
