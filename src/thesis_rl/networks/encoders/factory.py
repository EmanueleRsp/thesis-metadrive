from __future__ import annotations

from typing import Any

from thesis_rl.networks.encoders.base import BaseEncoder
from thesis_rl.networks.encoders.lq_encoder import LQEncoder
from thesis_rl.networks.encoders.mlp_encoder import MLPEncoder
from thesis_rl.networks.encoders.none_encoder import NoneEncoder
from thesis_rl.observations.spec import ObservationSpec


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def build_encoder(cfg_encoder: Any, obs_spec: ObservationSpec) -> BaseEncoder:
    enc_type = str(_cfg_get(cfg_encoder, "type", "none")).lower()

    if enc_type == "none":
        return NoneEncoder(input_dim=int(_cfg_get(cfg_encoder, "input_dim", obs_spec.flat_dim)))

    if enc_type == "mlp":
        return MLPEncoder(
            input_dim=int(_cfg_get(cfg_encoder, "input_dim", obs_spec.flat_dim)),
            hidden_layers=list(_cfg_get(cfg_encoder, "hidden_layers", [512, 512, 256])),
            output_dim=int(_cfg_get(cfg_encoder, "output_dim", 256)),
            activation=str(_cfg_get(cfg_encoder, "activation", "relu")),
            layer_norm=bool(_cfg_get(cfg_encoder, "layer_norm", True)),
            dropout=float(_cfg_get(cfg_encoder, "dropout", 0.0)),
        )

    if enc_type == "lq":
        return LQEncoder(
            obs_spec=obs_spec,
            d_model=int(_cfg_get(cfg_encoder, "d_model", 64)),
            num_latents=int(_cfg_get(cfg_encoder, "num_latents", 16)),
            latent_dim=int(_cfg_get(cfg_encoder, "latent_dim", 128)),
            output_dim=int(_cfg_get(cfg_encoder, "output_dim", 256)),
            depth=int(_cfg_get(cfg_encoder, "depth", 4)),
            num_heads=int(_cfg_get(cfg_encoder, "num_heads", 4)),
            ff_dim=int(_cfg_get(cfg_encoder, "ff_dim", 256)),
            activation=str(_cfg_get(cfg_encoder, "activation", "relu")),
            dropout=float(_cfg_get(cfg_encoder, "dropout", 0.0)),
            attention_dropout=float(_cfg_get(cfg_encoder, "attention_dropout", 0.0)),
            residual_gating=bool(_cfg_get(cfg_encoder, "residual_gating", True)),
            pooling=str(_cfg_get(cfg_encoder, "pooling", "mean")),
            type_embedding=bool(_cfg_get(cfg_encoder, "type_embedding", True)),
            time_embedding=bool(_cfg_get(cfg_encoder, "time_embedding", True)),
            slot_embedding=bool(_cfg_get(cfg_encoder, "slot_embedding", True)),
        )

    raise ValueError(f"Unknown encoder type: {enc_type}")
