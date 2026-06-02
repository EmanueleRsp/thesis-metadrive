from __future__ import annotations

import torch

from thesis_rl.agent.planners.encoders.lq.unflatten import StructuredObservation


def build_global_token_mask(obs: StructuredObservation) -> torch.Tensor:
    """Build a [B, T] validity mask for tokenized semantic observations."""

    batch_size, history, _ = obs.ego.shape
    ego_mask = torch.ones((batch_size, history), dtype=obs.ego.dtype, device=obs.ego.device)
    lane_mask = torch.ones((batch_size, 1), dtype=obs.ego.dtype, device=obs.ego.device)

    dynamic_mask = obs.dynamic_mask.reshape(batch_size, -1)
    mask = torch.cat(
        [
            ego_mask,
            obs.route_mask,
            dynamic_mask,
            obs.static_mask,
            obs.control_mask,
            lane_mask,
        ],
        dim=1,
    )
    return mask
