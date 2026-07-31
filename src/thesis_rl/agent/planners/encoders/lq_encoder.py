"""Schema-driven latent-query encoder for semantic observation v1.1."""

from __future__ import annotations

import torch
from torch import nn

from thesis_rl.agent.planners.encoders.base import BaseEncoder
from thesis_rl.contracts.observation_schema import (
    SemanticObservationSchemaV11,
    SemanticObservationTensorBatch,
    SemanticObservationSchemaV12,
    SemanticObservationTensorBatchV12,
)


class _TokenProjection(nn.Module):
    """Normative token projection: Linear, ReLU, then LayerNorm."""

    def __init__(self, input_dim: int, token_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _LatentQueryBlock(nn.Module):
    """One pre-normalized cross/self-attention latent block."""

    def __init__(self, latent_dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.cross_query_norm = nn.LayerNorm(latent_dim)
        self.cross_memory_norm = nn.LayerNorm(latent_dim)
        self.cross_attention = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.cross_ff_norm = nn.LayerNorm(latent_dim)
        self.cross_ff = _FeedForward(latent_dim, ff_dim)
        self.self_norm = nn.LayerNorm(latent_dim)
        self.self_attention = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.self_ff_norm = nn.LayerNorm(latent_dim)
        self.self_ff = _FeedForward(latent_dim, ff_dim)

    def forward(
        self,
        latents: torch.Tensor,
        scene_memory: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.cross_query_norm(latents)
        memory = self.cross_memory_norm(scene_memory)
        cross, _ = self.cross_attention(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = latents + cross
        latents = latents + self.cross_ff(self.cross_ff_norm(latents))
        normalized = self.self_norm(latents)
        attended, _ = self.self_attention(
            query=normalized,
            key=normalized,
            value=normalized,
            need_weights=False,
        )
        latents = latents + attended
        return latents + self.self_ff(self.self_ff_norm(latents))


class LatentQueryEncoderV2(BaseEncoder):
    """The approved 122-token, four-block semantic v1.1 LQ encoder."""

    observation_schema_version = SemanticObservationSchemaV11.version

    def __init__(
        self,
        *,
        schema: SemanticObservationSchemaV11,
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
        if schema.version != SemanticObservationSchemaV11.version:
            raise ValueError(f"LQ v2 requires schema {SemanticObservationSchemaV11.version}.")
        if token_dim != 64 or num_latents != 16 or latent_dim != 128 or depth != 4:
            raise ValueError(
                "LatentQueryEncoderV2 only supports the frozen v1.0 core architecture."
            )
        if num_heads != 4 or ff_dim != 256 or output_dim != 256 or pooling != "mean":
            raise ValueError("LatentQueryEncoderV2 received a non-core architecture setting.")
        self.schema = schema
        self.input_dim = schema.flat_dim
        self.output_dim = output_dim
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.pooling = pooling

        self.ego_history_projection = _TokenProjection(10, token_dim)
        self.ego_current_projection = _TokenProjection(3, token_dim)
        self.route_projection = _TokenProjection(7, token_dim)
        self.dynamic_projection = _TokenProjection(22, token_dim)
        self.static_projection = _TokenProjection(13, token_dim)
        self.lane_road_projection = _TokenProjection(14, token_dim)
        self.controls_projection = _TokenProjection(17, token_dim)
        self.interactions_projection = _TokenProjection(35, token_dim)
        self.temporal_projection = _TokenProjection(5, token_dim)

        self.type_embedding = nn.Embedding(9, token_dim)
        self.time_embedding = nn.Embedding(5, token_dim)
        self.route_slot_embedding = nn.Embedding(10, token_dim)
        self.dynamic_slot_embedding = nn.Embedding(16, token_dim)
        self.static_slot_embedding = nn.Embedding(8, token_dim)
        self.control_slot_embedding = nn.Embedding(8, token_dim)
        self.interaction_slot_embedding = nn.Embedding(8, token_dim)
        self.latent_queries = nn.Parameter(torch.empty(num_latents, latent_dim))
        for embedding in (
            self.type_embedding,
            self.time_embedding,
            self.route_slot_embedding,
            self.dynamic_slot_embedding,
            self.static_slot_embedding,
            self.control_slot_embedding,
            self.interaction_slot_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)

        self.token_to_latent = nn.Linear(token_dim, latent_dim)
        self.blocks = nn.ModuleList(
            [_LatentQueryBlock(latent_dim, num_heads, ff_dim) for _ in range(depth)]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(latent_dim, output_dim), nn.LayerNorm(output_dim), nn.ReLU()
        )

    @staticmethod
    def _indices(size: int, *, device: torch.device) -> torch.Tensor:
        return torch.arange(size, device=device, dtype=torch.long)

    def _type(self, values: torch.Tensor, type_id: int) -> torch.Tensor:
        return values + self.type_embedding.weight[type_id].view(1, 1, -1)

    def tokenize_structured(
        self, observation: SemanticObservationTensorBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked scene tokens ``[B, 122, 64]`` and valid mask ``[B, 122]``."""

        batch_size = observation.ego_history.shape[0]
        device = observation.ego_history.device
        ego_history = self._type(self.ego_history_projection(observation.ego_history), 0)
        ego_history = ego_history + self.time_embedding(self._indices(5, device=device)).view(
            1, 5, -1
        )

        ego_current = self._type(
            self.ego_current_projection(observation.ego_current).unsqueeze(1), 1
        )
        route = self._type(self.route_projection(observation.route), 2)
        route = route + self.route_slot_embedding(self._indices(10, device=device)).view(1, 10, -1)

        dynamic = self._type(self.dynamic_projection(observation.dynamic), 3)
        dynamic = dynamic + self.time_embedding(self._indices(5, device=device)).view(1, 1, 5, -1)
        dynamic = dynamic + self.dynamic_slot_embedding(self._indices(16, device=device)).view(
            1, 16, 1, -1
        )
        dynamic = dynamic.reshape(batch_size, 80, self.token_dim)

        static = self._type(self.static_projection(observation.static), 4)
        static = static + self.static_slot_embedding(self._indices(8, device=device)).view(1, 8, -1)
        lane_road = self._type(self.lane_road_projection(observation.lane_road).unsqueeze(1), 5)
        controls = self._type(self.controls_projection(observation.controls), 6)
        controls = controls + self.control_slot_embedding(self._indices(8, device=device)).view(
            1, 8, -1
        )
        interactions = self._type(self.interactions_projection(observation.interactions), 7)
        interactions = interactions + self.interaction_slot_embedding(
            self._indices(8, device=device)
        ).view(1, 8, -1)
        temporal = self._type(self.temporal_projection(observation.temporal).unsqueeze(1), 8)

        tokens = torch.cat(
            (
                ego_history,
                ego_current,
                route,
                dynamic,
                static,
                lane_road,
                controls,
                interactions,
                temporal,
            ),
            dim=1,
        )
        valid_mask = torch.cat(
            (
                observation.ego_history_mask,
                torch.ones(
                    (batch_size, 1), dtype=observation.ego_history_mask.dtype, device=device
                ),
                observation.route_mask,
                observation.dynamic_mask.reshape(batch_size, 80),
                observation.static_mask,
                torch.ones((batch_size, 1), dtype=observation.static_mask.dtype, device=device),
                observation.controls_mask,
                observation.interactions_mask,
                torch.ones(
                    (batch_size, 1), dtype=observation.interactions_mask.dtype, device=device
                ),
            ),
            dim=1,
        ).bool()
        if not valid_mask.any(dim=1).all():
            raise ValueError("LQ v2 requires at least one valid token per batch item.")
        return tokens * valid_mask.unsqueeze(-1).to(tokens.dtype), valid_mask

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        self._validate_input(flat_obs)
        structured = self.schema.unflatten_torch(flat_obs)
        tokens, valid_mask = self.tokenize_structured(structured)
        scene_memory = self.token_to_latent(tokens)
        latents = self.latent_queries.unsqueeze(0).expand(flat_obs.shape[0], -1, -1)
        key_padding_mask = ~valid_mask
        for block in self.blocks:
            latents = block(latents, scene_memory, key_padding_mask)
        return self.output_projection(latents.mean(dim=1))


# Compatibility alias for internal callers during the v1.1 replacement.
LQEncoder = LatentQueryEncoderV2


class LatentQueryEncoderV3(BaseEncoder):
    """The approved 143-token, four-block perception-bounded LQ encoder."""

    observation_schema_version = SemanticObservationSchemaV12.version
    _expected_num_latents = 16
    _expected_depth = 4

    def __init__(
        self,
        *,
        schema: SemanticObservationSchemaV12,
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
        if schema.version != SemanticObservationSchemaV12.version:
            raise ValueError(f"LQ v3 requires schema {SemanticObservationSchemaV12.version}.")
        if (
            token_dim != 64
            or num_latents != self._expected_num_latents
            or latent_dim != 128
            or depth != self._expected_depth
        ):
            raise ValueError(
                "LatentQueryEncoderV3 only supports the frozen v1.1 core architecture."
            )
        if num_heads != 4 or ff_dim != 256 or output_dim != 256 or pooling != "mean":
            raise ValueError("LatentQueryEncoderV3 received a non-core architecture setting.")
        self.schema = schema
        self.input_dim = schema.flat_dim
        self.output_dim = output_dim
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.pooling = pooling

        self.ego_history_projection = _TokenProjection(10, token_dim)
        self.ego_current_projection = _TokenProjection(3, token_dim)
        self.route_projection = _TokenProjection(7, token_dim)
        self.dynamic_projection = _TokenProjection(22, token_dim)
        self.static_projection = _TokenProjection(13, token_dim)
        self.lane_road_projection = _TokenProjection(12, token_dim)
        self.controls_projection = _TokenProjection(15, token_dim)
        self.interactions_projection = _TokenProjection(33, token_dim)
        self.context_history_projection = _TokenProjection(23, token_dim)
        self.signal_onset_state_projection = _TokenProjection(3, token_dim)

        self.type_embedding = nn.Embedding(10, token_dim)
        self.history_time_embedding = nn.Embedding(21, token_dim)
        self.route_slot_embedding = nn.Embedding(10, token_dim)
        # No dynamic_slot_embedding (ENC-V1.2, ADR-026): the dynamic buffer index
        # carries no stable meaning under sticky/first-fit slot assignment, so a
        # learned per-slot identity tag can only encode spurious correlations,
        # not signal. Route/static/control/interaction slots keep their
        # embedding because those slot indices ARE semantically stable.
        self.static_slot_embedding = nn.Embedding(8, token_dim)
        self.control_slot_embedding = nn.Embedding(8, token_dim)
        self.interaction_slot_embedding = nn.Embedding(8, token_dim)
        self.latent_queries = nn.Parameter(torch.empty(num_latents, latent_dim))
        for embedding in (
            self.type_embedding,
            self.history_time_embedding,
            self.route_slot_embedding,
            self.static_slot_embedding,
            self.control_slot_embedding,
            self.interaction_slot_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)

        self.token_to_latent = nn.Linear(token_dim, latent_dim)
        self.blocks = nn.ModuleList(
            [_LatentQueryBlock(latent_dim, num_heads, ff_dim) for _ in range(depth)]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(latent_dim, output_dim), nn.LayerNorm(output_dim), nn.ReLU()
        )

    @staticmethod
    def _indices(size: int, *, device: torch.device, start: int = 0) -> torch.Tensor:
        return torch.arange(start, start + size, device=device, dtype=torch.long)

    def _type(self, values: torch.Tensor, type_id: int) -> torch.Tensor:
        return values + self.type_embedding.weight[type_id].view(1, 1, -1)

    def tokenize_structured(
        self, observation: SemanticObservationTensorBatchV12
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked scene tokens ``[B, 143, 64]`` and valid mask ``[B, 143]``."""

        batch_size = observation.ego_history.shape[0]
        device = observation.ego_history.device
        ego_history = self._type(self.ego_history_projection(observation.ego_history), 0)
        ego_history = ego_history + self.history_time_embedding(
            self._indices(5, device=device, start=16)
        ).view(1, 5, -1)
        ego_current = self._type(
            self.ego_current_projection(observation.ego_current).unsqueeze(1), 1
        )
        route = self._type(self.route_projection(observation.route), 2)
        route = route + self.route_slot_embedding(self._indices(10, device=device)).view(1, 10, -1)
        dynamic = self._type(self.dynamic_projection(observation.dynamic), 3)
        dynamic = dynamic + self.history_time_embedding(
            self._indices(5, device=device, start=16)
        ).view(1, 1, 5, -1)
        dynamic = dynamic.reshape(batch_size, 80, self.token_dim)
        static = self._type(self.static_projection(observation.static), 4)
        static = static + self.static_slot_embedding(self._indices(8, device=device)).view(1, 8, -1)
        lane_road = self._type(self.lane_road_projection(observation.lane_road).unsqueeze(1), 5)
        controls = self._type(self.controls_projection(observation.controls), 6)
        controls = controls + self.control_slot_embedding(self._indices(8, device=device)).view(
            1, 8, -1
        )
        interactions = self._type(self.interactions_projection(observation.interactions), 7)
        interactions = interactions + self.interaction_slot_embedding(
            self._indices(8, device=device)
        ).view(1, 8, -1)
        compliance = self._type(
            self.context_history_projection(observation.context_history), 8
        )
        compliance = compliance + self.history_time_embedding(
            self._indices(21, device=device)
        ).view(1, 21, -1)
        yellow = self._type(
            self.signal_onset_state_projection(observation.signal_onset_state).unsqueeze(1), 9
        )
        tokens = torch.cat(
            (
                ego_history,
                ego_current,
                route,
                dynamic,
                static,
                lane_road,
                controls,
                interactions,
                compliance,
                yellow,
            ),
            dim=1,
        )
        valid_mask = torch.cat(
            (
                observation.ego_history_mask,
                torch.ones(
                    (batch_size, 1), dtype=observation.ego_history_mask.dtype, device=device
                ),
                observation.route_mask,
                observation.dynamic_mask.reshape(batch_size, 80),
                observation.static_mask,
                torch.ones((batch_size, 1), dtype=observation.static_mask.dtype, device=device),
                observation.controls_mask,
                observation.interactions_mask,
                observation.context_history_mask,
                torch.ones(
                    (batch_size, 1), dtype=observation.context_history_mask.dtype, device=device
                ),
            ),
            dim=1,
        ).bool()
        if not valid_mask.any(dim=1).all():
            raise ValueError("LQ v3 requires at least one valid token per batch item.")
        return tokens * valid_mask.unsqueeze(-1).to(tokens.dtype), valid_mask

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        self._validate_input(flat_obs)
        structured = self.schema.unflatten_torch(flat_obs)
        tokens, valid_mask = self.tokenize_structured(structured)
        scene_memory = self.token_to_latent(tokens)
        latents = self.latent_queries.unsqueeze(0).expand(flat_obs.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, scene_memory, ~valid_mask)
        return self.output_projection(latents.mean(dim=1))


class LatentQueryEncoderV3Lite(LatentQueryEncoderV3):
    """Diagnostic-only reduced LQ stack with unchanged semantic I/O dimensions."""

    _expected_num_latents = 8
    _expected_depth = 2

    def __init__(
        self,
        *,
        schema: SemanticObservationSchemaV12,
        token_dim: int = 64,
        num_latents: int = 8,
        latent_dim: int = 128,
        output_dim: int = 256,
        depth: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        pooling: str = "mean",
    ) -> None:
        super().__init__(
            schema=schema,
            token_dim=token_dim,
            num_latents=num_latents,
            latent_dim=latent_dim,
            output_dim=output_dim,
            depth=depth,
            num_heads=num_heads,
            ff_dim=ff_dim,
            pooling=pooling,
        )


class LatentQueryEncoderV3Micro(LatentQueryEncoderV3Lite):
    """Diagnostic-only minimal LQ stack with unchanged semantic I/O dimensions."""

    _expected_num_latents = 4
    _expected_depth = 1

    def __init__(
        self,
        *,
        schema: SemanticObservationSchemaV12,
        token_dim: int = 64,
        num_latents: int = 4,
        latent_dim: int = 128,
        output_dim: int = 256,
        depth: int = 1,
        num_heads: int = 4,
        ff_dim: int = 256,
        pooling: str = "mean",
    ) -> None:
        super().__init__(
            schema=schema,
            token_dim=token_dim,
            num_latents=num_latents,
            latent_dim=latent_dim,
            output_dim=output_dim,
            depth=depth,
            num_heads=num_heads,
            ff_dim=ff_dim,
            pooling=pooling,
        )
