"""Audit 2026-09-07, C24: a frozen encoder setting cannot be silently overridden.

`conf/agent/planner/encoder/lq_v3.yaml` lists `activation`, `dropout`,
`attention_dropout`, `type_embedding`, `time_embedding` and `slot_embedding`, but
the factory never passed them and the module hard-codes every one. An ablation
driven from those keys parsed, logged, reached the checkpoint's configuration
record and changed nothing, so it would have reported "no effect" for a knob that
was never connected -- a false negative in a results table, not a slow path.

The constructor already refuses a non-core `num_latents` or `depth`; these keys
now get the same treatment instead of being ignored.
"""

from __future__ import annotations

import pytest
import torch

from thesis_rl.agent.planners.encoders.factory import (
    _FROZEN_LATENT_QUERY_SETTINGS,
    build_encoder,
)
from thesis_rl.agent.planners.encoders.lq_encoder import LatentQueryEncoderV3
from thesis_rl.contracts.observation_schema import SemanticObservationSchemaV12

# The values `conf/agent/planner/encoder/lq_v3.yaml` actually ships.
_PRODUCTION_ENCODER_CONFIG = {
    "name": "latent_query_v3",
    "type": "latent_query_v3",
    "token_dim": 64,
    "num_latents": 16,
    "latent_dim": 128,
    "output_dim": 256,
    "depth": 4,
    "num_heads": 4,
    "ff_dim": 256,
    "activation": "relu",
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "residual_gating": False,
    "type_embedding": True,
    "time_embedding": True,
    "slot_embedding": True,
    "pooling": "mean",
}


def _build(**overrides: object) -> LatentQueryEncoderV3:
    schema = SemanticObservationSchemaV12()
    return build_encoder(
        {**_PRODUCTION_ENCODER_CONFIG, **overrides},
        input_dim=schema.flat_dim,
        observation_schema=schema,
    )


def test_the_shipped_configuration_still_builds_and_runs() -> None:
    """The guard must cost the production path nothing."""

    schema = SemanticObservationSchemaV12()
    encoder = _build()

    assert isinstance(encoder, LatentQueryEncoderV3)
    output = encoder(torch.zeros(2, schema.flat_dim, dtype=torch.float32))
    assert output.shape == (2, 256)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("key", "override"),
    [
        ("activation", "gelu"),
        ("dropout", 0.2),
        ("attention_dropout", 0.1),
        ("type_embedding", False),
        ("time_embedding", False),
        ("slot_embedding", False),
        ("residual_gating", True),
    ],
)
def test_overriding_a_hard_coded_setting_fails_instead_of_being_ignored(
    key: str, override: object
) -> None:
    with pytest.raises(ValueError, match=key):
        _build(**{key: override})


def test_every_frozen_key_present_in_the_shipped_config_is_guarded() -> None:
    """The guard list and the shipped config must not drift apart.

    A key added to the YAML without being added here would be exactly the defect
    this test exists to prevent, one knob later.
    """

    guarded = {key for key, _ in _FROZEN_LATENT_QUERY_SETTINGS}
    constructor_arguments = {
        "token_dim",
        "num_latents",
        "latent_dim",
        "output_dim",
        "depth",
        "num_heads",
        "ff_dim",
        "pooling",
    }
    descriptive = {"name", "type", "architecture_version", "required_observation_schema"}
    shipped = set(_PRODUCTION_ENCODER_CONFIG)

    unaccounted = shipped - guarded - constructor_arguments - descriptive
    assert not unaccounted, (
        f"encoder keys neither guarded nor passed through: {sorted(unaccounted)}"
    )


def test_the_frozen_values_match_what_the_module_implements() -> None:
    """Pin the guard to the module, not to the YAML.

    If someone makes dropout configurable in `lq_encoder.py` the guard must be
    removed rather than left refusing a setting that now works.
    """

    encoder = _build()
    for block in encoder.blocks:
        assert block.cross_attention.dropout == 0.0
        assert block.self_attention.dropout == 0.0
    assert hasattr(encoder, "type_embedding")
    assert hasattr(encoder, "history_time_embedding")
    assert hasattr(encoder, "route_slot_embedding")
    assert dict(_FROZEN_LATENT_QUERY_SETTINGS)["dropout"] == 0.0
    assert dict(_FROZEN_LATENT_QUERY_SETTINGS)["type_embedding"] is True
