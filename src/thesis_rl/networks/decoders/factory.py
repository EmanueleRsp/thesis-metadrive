from __future__ import annotations

from typing import Any

from thesis_rl.networks.decoders.mlp_decoder import MLPDecoder


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def build_decoder(cfg_decoder: Any, input_dim: int) -> MLPDecoder:
    dec_type = str(_cfg_get(cfg_decoder, "type", "mlp")).lower()
    if dec_type != "mlp":
        raise ValueError(f"Unknown decoder type: {dec_type}")
    return MLPDecoder(
        input_dim=int(input_dim),
        hidden_layers=list(_cfg_get(cfg_decoder, "hidden_layers", [256, 256])),
        output_dim=_cfg_get(cfg_decoder, "output_dim", None),
        activation=str(_cfg_get(cfg_decoder, "activation", "relu")),
        dropout=float(_cfg_get(cfg_decoder, "dropout", 0.0)),
        layer_norm=bool(_cfg_get(cfg_decoder, "layer_norm", False)),
    )
