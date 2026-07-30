"""Latent-query encoder for the 21-frame causal LiDAR observation (`lq_lidar`).

Reuses `LatentQueryEncoder`'s frozen core (16 latents, depth 4, `token_dim=64`,
`output_dim=256`) untouched; only the group projections and the tokenizer
(`lq_lidar/tokenizer.py`) are new. See `docs/implementation/
lidar_arm_temporal_alignment_and_lq_tokenization_v2.0_exec_plan.md` (`DEC-003`,
`DEC-005`, `DEC-008`).
"""

from __future__ import annotations

import torch
from torch import nn

from thesis_rl.agent.planners.encoders.base import BaseEncoder
from thesis_rl.agent.planners.encoders.lq_encoder import _LatentQueryBlock
from thesis_rl.agent.planners.encoders.lq_lidar.tokenizer import (
    FLAT_DIM,
    NUM_NEARBY_SLOTS,
    NUM_SECTORS,
    NUM_TOKENS,
    circular_positional_encoding,
    gather_lidar_tokens,
)


class _GroupProjection(nn.Module):
    """Linear, ReLU, LayerNorm projection to `token_dim`, mirroring `lq_encoder._TokenProjection`."""

    def __init__(self, input_dim: int, token_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class LatentQueryEncoderLidar(BaseEncoder):
    """The `lq_lidar` encoder: 38 tokens over `stacked_lidar_v2` (`D=6489`)."""

    input_dim = FLAT_DIM

    def __init__(
        self,
        *,
        token_dim: int = 64,
        num_latents: int = 16,
        latent_dim: int = 128,
        output_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        ff_dim: int = 256,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if token_dim != 64 or num_latents != 16 or latent_dim != 128 or depth != 4:
            raise ValueError(
                "LatentQueryEncoderLidar only supports the frozen LQ core architecture."
            )
        if num_heads != 4 or ff_dim != 256 or output_dim != 256 or pooling != "mean":
            raise ValueError("LatentQueryEncoderLidar received a non-core architecture setting.")
        self.output_dim = output_dim
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.pooling = pooling

        self.ego_projection = _GroupProjection(147, token_dim)
        self.navigation_projection = _GroupProjection(462, token_dim)
        self.side_projection = _GroupProjection(252, token_dim)
        self.lane_projection = _GroupProjection(252, token_dim)
        # Shared across the 4 neighbor slots and the 30 sector slots: no
        # per-slot identity embedding is added (`DEC-008`), since the k-nearest
        # ranking and the LiDAR ring index are the only stable slot semantics.
        self.neighbor_projection = _GroupProjection(84, token_dim)
        self.sector_projection = _GroupProjection(168, token_dim)

        self.type_embedding = nn.Embedding(6, token_dim)
        for embedding in (self.type_embedding,):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

        self.register_buffer(
            "sector_positional_encoding",
            circular_positional_encoding(NUM_SECTORS, token_dim),
            persistent=False,
        )

        self.latent_queries = nn.Parameter(torch.empty(num_latents, latent_dim))
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)

        self.token_to_latent = nn.Linear(token_dim, latent_dim)
        self.blocks = nn.ModuleList(
            [_LatentQueryBlock(latent_dim, num_heads, ff_dim) for _ in range(depth)]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(latent_dim, output_dim), nn.LayerNorm(output_dim), nn.ReLU()
        )

    def _type(self, values: torch.Tensor, type_id: int) -> torch.Tensor:
        return values + self.type_embedding.weight[type_id].view(1, 1, -1)

    def tokenize_structured(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """Return the 38 scene tokens `[B, 38, token_dim]`."""

        groups, _mask = gather_lidar_tokens(flat_obs)

        ego = self._type(self.ego_projection(groups.ego).unsqueeze(1), 0)
        navigation = self._type(self.navigation_projection(groups.navigation).unsqueeze(1), 1)
        side = self._type(self.side_projection(groups.side).unsqueeze(1), 2)
        lane = self._type(self.lane_projection(groups.lane).unsqueeze(1), 3)
        neighbor = self._type(self.neighbor_projection(groups.neighbor), 4)
        sector = self._type(self.sector_projection(groups.sector), 5)
        sector = sector + self.sector_positional_encoding.unsqueeze(0)

        if neighbor.shape[1] != NUM_NEARBY_SLOTS or sector.shape[1] != NUM_SECTORS:
            raise RuntimeError("lq_lidar tokenizer produced an unexpected slot count")

        tokens = torch.cat((ego, navigation, side, lane, neighbor, sector), dim=1)
        if tokens.shape[1] != NUM_TOKENS:
            raise RuntimeError("lq_lidar tokenizer must emit exactly 38 tokens")
        return tokens

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        self._validate_input(flat_obs)
        tokens = self.tokenize_structured(flat_obs)
        scene_memory = self.token_to_latent(tokens)
        latents = self.latent_queries.unsqueeze(0).expand(flat_obs.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, scene_memory, key_padding_mask=None)
        return self.output_projection(latents.mean(dim=1))
