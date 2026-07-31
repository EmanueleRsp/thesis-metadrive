from __future__ import annotations

import pytest
import torch

from thesis_rl.agent.planners.encoders.factory import build_encoder
from thesis_rl.agent.planners.encoders.lq_encoder import (
    LatentQueryEncoderV3,
    LatentQueryEncoderV3Lite,
    LatentQueryEncoderV3Micro,
)
from thesis_rl.agent.planners.encoders.mlp_encoder import FlatMLPEncoder
from thesis_rl.contracts.observation_schema import SemanticObservationSchemaV12


def test_flat_mlp_v11_has_3009_input_and_exact_parameter_count() -> None:
    encoder = FlatMLPEncoder(input_dim=3009)
    observation = torch.randn(2, 3009, dtype=torch.float32, requires_grad=True)

    output = encoder(observation)
    output.square().mean().backward()

    assert output.shape == (2, 256)
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 2_003_968
    assert observation.grad is not None and torch.isfinite(observation.grad).all()


def test_lq_v11_has_143_tokens_and_21_step_time_embedding() -> None:
    schema = SemanticObservationSchemaV12()
    encoder = LatentQueryEncoderV3(schema=schema)
    flat = torch.zeros(2, schema.flat_dim, dtype=torch.float32)
    structured = schema.unflatten_torch(flat)

    tokens, mask = encoder.tokenize_structured(structured)

    assert tokens.shape == (2, 143, 64)
    assert mask.shape == (2, 143)
    assert mask[:, 5].all() and mask[:, 104].all() and mask[:, 142].all()
    assert encoder.type_embedding.num_embeddings == 10
    assert encoder.history_time_embedding.num_embeddings == 21
    assert torch.count_nonzero(tokens[~mask]).item() == 0


def test_lq_v11_is_invariant_to_masked_compliance_payload() -> None:
    torch.manual_seed(15)
    schema = SemanticObservationSchemaV12()
    encoder = LatentQueryEncoderV3(schema=schema)
    first = torch.zeros(2, schema.flat_dim, dtype=torch.float32, requires_grad=True)
    second = first.detach().clone()
    compliance = schema.slices["context_history"]
    second[:, compliance] = torch.randn_like(second[:, compliance])

    first_output = encoder(first)
    second_output = encoder(second)
    first_output.sum().backward()

    torch.testing.assert_close(first_output, second_output)
    assert first.grad is not None and torch.isfinite(first.grad).all()


def test_lq_v11_dynamic_output_is_invariant_to_which_slot_holds_an_actor() -> None:
    """Regression for ENC-V1.2/ADR-026: a fixed dynamic_slot_embedding used to
    tag the buffer index itself, so identical actor content produced different
    latents depending on which of the 16 sticky slots it happened to occupy."""

    schema = SemanticObservationSchemaV12()
    encoder = LatentQueryEncoderV3(schema=schema)
    encoder.eval()

    def _flat_with_actor_in_slot(slot: int) -> torch.Tensor:
        flat = torch.zeros(1, schema.flat_dim, dtype=torch.float32)
        dynamic = torch.zeros(1, 16, 5, 22, dtype=torch.float32)
        dynamic_mask = torch.zeros(1, 16, 5, dtype=torch.float32)
        dynamic[0, slot, :, 0] = 0.4
        dynamic[0, slot, :, 1] = -0.2
        dynamic_mask[0, slot, :] = 1.0
        flat[:, schema.slices["dynamic"]] = dynamic.reshape(1, -1)
        flat[:, schema.slices["dynamic_mask"]] = dynamic_mask.reshape(1, -1)
        return flat

    with torch.no_grad():
        output_slot_0 = encoder(_flat_with_actor_in_slot(0))
        output_slot_9 = encoder(_flat_with_actor_in_slot(9))

    torch.testing.assert_close(output_slot_0, output_slot_9)
    assert not hasattr(encoder, "dynamic_slot_embedding")


def test_lq_v3_lite_preserves_semantic_io_contract() -> None:
    schema = SemanticObservationSchemaV12()
    encoder = LatentQueryEncoderV3Lite(schema=schema)
    output = encoder(torch.zeros(2, schema.flat_dim, dtype=torch.float32))

    assert output.shape == (2, 256)
    assert torch.isfinite(output).all()


def test_lq_v3_micro_preserves_semantic_io_contract() -> None:
    schema = SemanticObservationSchemaV12()
    encoder = LatentQueryEncoderV3Micro(schema=schema)
    output = encoder(torch.zeros(2, schema.flat_dim, dtype=torch.float32))

    assert output.shape == (2, 256)
    assert torch.isfinite(output).all()


def test_encoder_factory_rejects_cross_version_lq_pairing() -> None:
    with pytest.raises(ValueError, match="semantic v1.2"):
        build_encoder({"type": "latent_query_v3"}, input_dim=2541, observation_schema=None)
