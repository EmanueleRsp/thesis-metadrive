"""Factory for the frozen encoder v1.0 configurations."""

from __future__ import annotations

from typing import Any

from thesis_rl.agent.planners.encoders.base import BaseEncoder
from thesis_rl.agent.planners.encoders.lq_encoder import LatentQueryEncoderV2, LatentQueryEncoderV3
from thesis_rl.agent.planners.encoders.mlp_encoder import FlatMLPEncoder
from thesis_rl.agent.planners.encoders.none_encoder import NoneEncoder
from thesis_rl.contracts.observation_schema import SemanticObservationSchemaV11, SemanticObservationSchemaV12


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    return getter(key, default) if callable(getter) else getattr(config, key, default)


def build_encoder(
    cfg_encoder: Any,
    *,
    input_dim: int,
    observation_schema: SemanticObservationSchemaV11 | SemanticObservationSchemaV12 | None,
) -> BaseEncoder:
    """Build an encoder from serializable configuration only."""

    encoder_type = str(_get(cfg_encoder, "type", "none")).lower()
    if encoder_type in {"none", "identity", "flat_no_compression"}:
        return NoneEncoder(input_dim=input_dim)
    if encoder_type == "mlp":
        return FlatMLPEncoder(
            input_dim=input_dim,
            hidden_layers=_get(cfg_encoder, "hidden_layers", [512, 512, 256]),
            output_dim=int(_get(cfg_encoder, "output_dim", 256)),
            activation=str(_get(cfg_encoder, "activation", "relu")),
            layer_norm=bool(_get(cfg_encoder, "layer_norm", True)),
            dropout=float(_get(cfg_encoder, "dropout", 0.0)),
        )
    if encoder_type in {"lq", "latent_query_v2"}:
        if observation_schema is None or input_dim != SemanticObservationSchemaV11.flat_dim:
            raise ValueError(
                "LatentQueryEncoderV2 requires semantic v1.1 observation schema and D=2541."
            )
        if bool(_get(cfg_encoder, "residual_gating", False)):
            raise ValueError(
                "Residual gating is not supported by the encoder v1.0 core configuration."
            )
        return LatentQueryEncoderV2(
            schema=observation_schema,
            token_dim=int(_get(cfg_encoder, "token_dim", _get(cfg_encoder, "d_model", 64))),
            num_latents=int(_get(cfg_encoder, "num_latents", 16)),
            latent_dim=int(_get(cfg_encoder, "latent_dim", 128)),
            output_dim=int(_get(cfg_encoder, "output_dim", 256)),
            depth=int(_get(cfg_encoder, "depth", 4)),
            num_heads=int(_get(cfg_encoder, "num_heads", 4)),
            ff_dim=int(_get(cfg_encoder, "ff_dim", 256)),
            pooling=str(_get(cfg_encoder, "pooling", "mean")),
        )
    if encoder_type in {"latent_query_v3", "lq_v3"}:
        if not isinstance(observation_schema, SemanticObservationSchemaV12) or input_dim != 3064:
            raise ValueError(
                "LatentQueryEncoderV3 requires semantic v1.2 observation schema and D=3064."
            )
        if bool(_get(cfg_encoder, "residual_gating", False)):
            raise ValueError(
                "Residual gating is not supported by the encoder v1.1 core configuration."
            )
        return LatentQueryEncoderV3(
            schema=observation_schema,
            token_dim=int(_get(cfg_encoder, "token_dim", _get(cfg_encoder, "d_model", 64))),
            num_latents=int(_get(cfg_encoder, "num_latents", 16)),
            latent_dim=int(_get(cfg_encoder, "latent_dim", 128)),
            output_dim=int(_get(cfg_encoder, "output_dim", 256)),
            depth=int(_get(cfg_encoder, "depth", 4)),
            num_heads=int(_get(cfg_encoder, "num_heads", 4)),
            ff_dim=int(_get(cfg_encoder, "ff_dim", 256)),
            pooling=str(_get(cfg_encoder, "pooling", "mean")),
        )
    raise ValueError(f"Unknown encoder type: {encoder_type}")
