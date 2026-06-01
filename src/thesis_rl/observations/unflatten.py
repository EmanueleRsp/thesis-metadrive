from __future__ import annotations

from dataclasses import dataclass

import torch

from thesis_rl.observations.spec import ObservationSpec


@dataclass
class StructuredObservation:
    ego: torch.Tensor
    route: torch.Tensor
    route_mask: torch.Tensor
    dynamic: torch.Tensor
    dynamic_mask: torch.Tensor
    static: torch.Tensor
    static_mask: torch.Tensor
    controls: torch.Tensor
    control_mask: torch.Tensor
    lane: torch.Tensor


def unflatten_observation(obs_flat: torch.Tensor, spec: ObservationSpec) -> StructuredObservation:
    """Unflatten semantic-state observation into typed tensors."""

    obs = obs_flat
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)
    if obs.dim() != 2:
        raise ValueError(f"Expected obs shape [B, D] or [D], got {tuple(obs_flat.shape)}")
    if obs.shape[-1] != spec.flat_dim:
        raise ValueError(
            f"Unexpected observation dimension: got {obs.shape[-1]}, expected {spec.flat_dim}"
        )

    batch_size = int(obs.shape[0])
    cursor = 0

    ego_n = spec.history * spec.ego_dim
    route_n = spec.num_route * spec.route_dim
    route_mask_n = spec.num_route
    dyn_n = spec.num_dynamic * spec.history * spec.dynamic_dim
    dyn_mask_n = spec.num_dynamic * spec.history
    static_n = spec.num_static * spec.static_dim
    static_mask_n = spec.num_static
    control_n = spec.num_controls * spec.control_dim
    control_mask_n = spec.num_controls
    lane_n = spec.lane_dim

    ego = obs[:, cursor : cursor + ego_n].view(batch_size, spec.history, spec.ego_dim)
    cursor += ego_n
    route = obs[:, cursor : cursor + route_n].view(batch_size, spec.num_route, spec.route_dim)
    cursor += route_n
    route_mask = obs[:, cursor : cursor + route_mask_n].view(batch_size, spec.num_route)
    cursor += route_mask_n
    dynamic = obs[:, cursor : cursor + dyn_n].view(
        batch_size, spec.num_dynamic, spec.history, spec.dynamic_dim
    )
    cursor += dyn_n
    dynamic_mask = obs[:, cursor : cursor + dyn_mask_n].view(
        batch_size, spec.num_dynamic, spec.history
    )
    cursor += dyn_mask_n
    static = obs[:, cursor : cursor + static_n].view(batch_size, spec.num_static, spec.static_dim)
    cursor += static_n
    static_mask = obs[:, cursor : cursor + static_mask_n].view(batch_size, spec.num_static)
    cursor += static_mask_n
    controls = obs[:, cursor : cursor + control_n].view(batch_size, spec.num_controls, spec.control_dim)
    cursor += control_n
    control_mask = obs[:, cursor : cursor + control_mask_n].view(batch_size, spec.num_controls)
    cursor += control_mask_n
    lane = obs[:, cursor : cursor + lane_n].view(batch_size, spec.lane_dim)
    cursor += lane_n

    if cursor != spec.flat_dim:
        raise RuntimeError(f"Unflatten consumed {cursor} dims, expected {spec.flat_dim}")

    return StructuredObservation(
        ego=ego,
        route=route,
        route_mask=route_mask,
        dynamic=dynamic,
        dynamic_mask=dynamic_mask,
        static=static,
        static_mask=static_mask,
        controls=controls,
        control_mask=control_mask,
        lane=lane,
    )
