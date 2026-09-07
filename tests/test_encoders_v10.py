from __future__ import annotations

import pytest
import torch

from thesis_rl.agent.planners.encoders.factory import build_encoder
from thesis_rl.agent.planners.encoders.lq_encoder import LatentQueryEncoderV2
from thesis_rl.agent.planners.encoders.mlp_encoder import FlatMLPEncoder
from thesis_rl.contracts.observation_schema import SemanticObservationSchemaV11


@pytest.mark.parametrize(
    ("input_dim", "expected_parameters"),
    [(1540, 1_251_840), (2541, 1_764_352)],
)
def test_flat_mlp_v10_shape_backward_and_parameter_contract(
    input_dim: int, expected_parameters: int
) -> None:
    encoder = FlatMLPEncoder(input_dim=input_dim)
    obs = torch.randn(2, input_dim, dtype=torch.float32, requires_grad=True)

    output = encoder(obs)
    output.square().mean().backward()

    assert output.shape == (2, 256)
    assert sum(parameter.numel() for parameter in encoder.parameters()) == expected_parameters
    assert obs.grad is not None and torch.isfinite(obs.grad).all()


def test_flat_mlp_v10_rejects_invalid_input_and_honours_layer_norm_switch() -> None:
    encoder = FlatMLPEncoder(input_dim=1540, layer_norm=False)

    assert not any(isinstance(module, torch.nn.LayerNorm) for module in encoder.modules())
    with pytest.raises(ValueError, match="rank 2"):
        encoder(torch.zeros(1540, dtype=torch.float32))
    with pytest.raises(TypeError, match="float32"):
        encoder(torch.zeros(1, 1540, dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        encoder(torch.full((1, 1540), float("nan"), dtype=torch.float32))


def test_lq_v10_has_122_schema_driven_tokens_and_masks_payload() -> None:
    schema = SemanticObservationSchemaV11()
    encoder = LatentQueryEncoderV2(schema=schema)
    flat = torch.zeros(2, schema.flat_dim, dtype=torch.float32)
    structured = schema.unflatten_torch(flat)

    tokens, mask = encoder.tokenize_structured(structured)

    assert tokens.shape == (2, 122, 64)
    assert mask.shape == (2, 122)
    assert mask[:, 5].all() and mask[:, 104].all() and mask[:, 121].all()
    # Structural, not data-dependent: those three positions are literal
    # `torch.ones` blocks. The tokenizer's runtime assertion of this was a
    # device-to-host synchronisation guarding a branch that cannot be taken.
    assert mask.any(dim=1).all()
    assert torch.count_nonzero(tokens[~mask]).item() == 0


def test_lq_v10_is_invariant_to_masked_payload_and_has_finite_backward() -> None:
    torch.manual_seed(7)
    schema = SemanticObservationSchemaV11()
    encoder = LatentQueryEncoderV2(schema=schema)
    first = torch.zeros(2, schema.flat_dim, dtype=torch.float32, requires_grad=True)
    second = first.detach().clone()
    dynamic = schema.slices["dynamic"]
    second[:, dynamic] = torch.randn_like(second[:, dynamic])

    first_output = encoder(first)
    second_output = encoder(second)
    first_output.sum().backward()

    torch.testing.assert_close(first_output, second_output)
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert len(encoder.blocks) == 4
    assert encoder.latent_queries.shape == (16, 128)


def test_encoder_factory_rejects_lidar_lq_pairing() -> None:
    with pytest.raises(ValueError, match="semantic v1.1"):
        build_encoder({"type": "latent_query_v2"}, input_dim=1540, observation_schema=None)
