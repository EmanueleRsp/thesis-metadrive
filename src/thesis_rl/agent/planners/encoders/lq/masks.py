"""Semantic v1.1 LQ global-mask helper."""

import torch

from thesis_rl.contracts.observation_schema import SemanticObservationTensorBatch


def build_global_token_mask(observation: SemanticObservationTensorBatch) -> torch.Tensor:
    batch_size = observation.ego_history.shape[0]
    device = observation.ego_history.device
    return torch.cat(
        (
            observation.ego_history_mask,
            torch.ones((batch_size, 1), device=device, dtype=observation.ego_history_mask.dtype),
            observation.route_mask,
            observation.dynamic_mask.reshape(batch_size, 80),
            observation.static_mask,
            torch.ones((batch_size, 1), device=device, dtype=observation.static_mask.dtype),
            observation.controls_mask,
            observation.interactions_mask,
            torch.ones((batch_size, 1), device=device, dtype=observation.interactions_mask.dtype),
        ),
        dim=1,
    ).bool()
