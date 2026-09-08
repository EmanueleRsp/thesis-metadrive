"""Factory for the frozen encoder v1.0 configurations."""

from __future__ import annotations

from typing import Any

from thesis_rl.agent.planners.encoders.base import BaseEncoder
from thesis_rl.agent.planners.encoders.lq_encoder import (
    LatentQueryEncoderV2,
    LatentQueryEncoderV3,
    LatentQueryEncoderV3Lite,
    LatentQueryEncoderV3Micro,
)
from thesis_rl.agent.planners.encoders.lq_lidar_encoder import LatentQueryEncoderLidar
from thesis_rl.agent.planners.encoders.mlp_encoder import FlatMLPEncoder
from thesis_rl.agent.planners.encoders.none_encoder import NoneEncoder
from thesis_rl.contracts.observation_schema import (
    SemanticObservationSchemaV11,
    SemanticObservationSchemaV12,
)


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    return getter(key, default) if callable(getter) else getattr(config, key, default)


# The latent-query architecture is frozen: every setting the constructor accepts
# is validated against its core value there, and the ones below are not accepted
# at all because the module hard-codes them -- ReLU in `_TokenProjection` and
# `_FeedForward`, `dropout=0.0` on both `MultiheadAttention` modules, and type,
# time and slot embeddings that are constructed and added unconditionally.
#
# Listing them in the YAML documents the architecture accurately, but until C24
# nothing enforced it: `agent.planner.encoder.dropout=0.2` or
# `type_embedding=false` parsed, logged, reached the checkpoint's configuration
# record, and changed nothing. An ablation run that way reports "no effect" for a
# knob that was never connected, which is a false negative in a results table
# rather than an inefficiency. Rejecting the value is the same discipline the
# constructor already applies to `num_latents` and `depth`.
_FROZEN_LATENT_QUERY_SETTINGS: tuple[tuple[str, Any], ...] = (
    ("activation", "relu"),
    ("dropout", 0.0),
    ("attention_dropout", 0.0),
    ("type_embedding", True),
    ("time_embedding", True),
    ("slot_embedding", True),
    ("residual_gating", False),
)


def _reject_non_core_latent_query_settings(cfg_encoder: Any, *, encoder_name: str) -> None:
    """Refuse any frozen latent-query setting the configuration tries to change."""

    for key, frozen in _FROZEN_LATENT_QUERY_SETTINGS:
        value = _get(cfg_encoder, key, frozen)
        if isinstance(frozen, bool):
            matches = bool(value) is frozen
        elif isinstance(frozen, float):
            matches = float(value) == frozen
        else:
            matches = str(value).strip().lower() == frozen
        if not matches:
            raise ValueError(
                f"{encoder_name} hard-codes {key}={frozen!r}; configuration requested {value!r}. "
                "The latent-query architecture is frozen, so this setting cannot take effect: "
                "change the encoder module, or drop the override."
            )


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
        _reject_non_core_latent_query_settings(cfg_encoder, encoder_name="LatentQueryEncoderV2")
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
        if (
            not isinstance(observation_schema, SemanticObservationSchemaV12)
            or input_dim != SemanticObservationSchemaV12.flat_dim
        ):
            raise ValueError(
                "LatentQueryEncoderV3 requires semantic v1.2 observation schema and D=3011."
            )
        _reject_non_core_latent_query_settings(cfg_encoder, encoder_name="LatentQueryEncoderV3")
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
    if encoder_type in {"latent_query_v3_lite", "lq_v3_lite"}:
        if (
            not isinstance(observation_schema, SemanticObservationSchemaV12)
            or input_dim != SemanticObservationSchemaV12.flat_dim
        ):
            raise ValueError(
                "LatentQueryEncoderV3Lite requires semantic v1.2 observation schema and D=3011."
            )
        _reject_non_core_latent_query_settings(cfg_encoder, encoder_name="LatentQueryEncoderV3Lite")
        return LatentQueryEncoderV3Lite(
            schema=observation_schema,
            token_dim=int(_get(cfg_encoder, "token_dim", 64)),
            num_latents=int(_get(cfg_encoder, "num_latents", 8)),
            latent_dim=int(_get(cfg_encoder, "latent_dim", 128)),
            output_dim=int(_get(cfg_encoder, "output_dim", 256)),
            depth=int(_get(cfg_encoder, "depth", 2)),
            num_heads=int(_get(cfg_encoder, "num_heads", 4)),
            ff_dim=int(_get(cfg_encoder, "ff_dim", 256)),
            pooling=str(_get(cfg_encoder, "pooling", "mean")),
        )
    if encoder_type in {"latent_query_v3_micro", "lq_v3_micro"}:
        if (
            not isinstance(observation_schema, SemanticObservationSchemaV12)
            or input_dim != SemanticObservationSchemaV12.flat_dim
        ):
            raise ValueError(
                "LatentQueryEncoderV3Micro requires semantic v1.2 observation schema and D=3011."
            )
        _reject_non_core_latent_query_settings(
            cfg_encoder, encoder_name="LatentQueryEncoderV3Micro"
        )
        return LatentQueryEncoderV3Micro(
            schema=observation_schema,
            token_dim=int(_get(cfg_encoder, "token_dim", 64)),
            num_latents=int(_get(cfg_encoder, "num_latents", 4)),
            latent_dim=int(_get(cfg_encoder, "latent_dim", 128)),
            output_dim=int(_get(cfg_encoder, "output_dim", 256)),
            depth=int(_get(cfg_encoder, "depth", 1)),
            num_heads=int(_get(cfg_encoder, "num_heads", 4)),
            ff_dim=int(_get(cfg_encoder, "ff_dim", 256)),
            pooling=str(_get(cfg_encoder, "pooling", "mean")),
        )
    if encoder_type in {"lq_lidar", "latent_query_lidar"}:
        if input_dim != LatentQueryEncoderLidar.input_dim:
            raise ValueError(
                f"LatentQueryEncoderLidar requires stacked_lidar_v2 observation, D="
                f"{LatentQueryEncoderLidar.input_dim}."
            )
        if bool(_get(cfg_encoder, "residual_gating", False)):
            raise ValueError(
                "Residual gating is not supported by the encoder v1.4 core configuration."
            )
        return LatentQueryEncoderLidar(
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
