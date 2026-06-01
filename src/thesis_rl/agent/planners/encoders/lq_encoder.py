from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from thesis_rl.agent.planners.encoders.base import BaseEncoder
from thesis_rl.observations.masks import build_global_token_mask
from thesis_rl.observations.spec import ObservationSpec
from thesis_rl.observations.unflatten import StructuredObservation, unflatten_observation


def _activation(name: str) -> type[nn.Module]:
    key = str(name).lower()
    if key == "relu":
        return nn.ReLU
    if key == "gelu":
        return nn.GELU
    if key == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


class _ResidualBranch(nn.Module):
    def __init__(self, residual_gating: bool) -> None:
        super().__init__()
        self.residual_gating = bool(residual_gating)
        self.alpha = nn.Parameter(torch.zeros(1)) if self.residual_gating else None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            return x + y
        return x + torch.tanh(self.alpha) * y


class _TokenMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", layer_norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(int(in_dim), int(out_dim)), _activation(activation)()]
        if layer_norm:
            layers.append(nn.LayerNorm(int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, activation: str = "relu", dropout: float = 0.0) -> None:
        super().__init__()
        act = _activation(activation)
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LQBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_heads: int,
        ff_dim: int,
        activation: str,
        dropout: float,
        attention_dropout: float,
        residual_gating: bool,
    ) -> None:
        super().__init__()
        self.cross_ln_q = nn.LayerNorm(latent_dim)
        self.cross_ln_kv = nn.LayerNorm(latent_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.cross_res = _ResidualBranch(residual_gating=residual_gating)
        self.cross_ff_ln = nn.LayerNorm(latent_dim)
        self.cross_ff = _FeedForward(latent_dim, ff_dim, activation=activation, dropout=dropout)
        self.cross_ff_res = _ResidualBranch(residual_gating=residual_gating)

        self.self_ln = nn.LayerNorm(latent_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.self_res = _ResidualBranch(residual_gating=residual_gating)
        self.self_ff_ln = nn.LayerNorm(latent_dim)
        self.self_ff = _FeedForward(latent_dim, ff_dim, activation=activation, dropout=dropout)
        self.self_ff_res = _ResidualBranch(residual_gating=residual_gating)

    def forward(self, latents: torch.Tensor, tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        q = self.cross_ln_q(latents)
        kv = self.cross_ln_kv(tokens)
        cross_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        latents = self.cross_res(latents, cross_out)
        latents = self.cross_ff_res(latents, self.cross_ff(self.cross_ff_ln(latents)))

        self_in = self.self_ln(latents)
        self_out, _ = self.self_attn(
            query=self_in,
            key=self_in,
            value=self_in,
            need_weights=False,
        )
        latents = self.self_res(latents, self_out)
        latents = self.self_ff_res(latents, self.self_ff(self.self_ff_ln(latents)))
        return latents


@dataclass(frozen=True)
class _TokenSlices:
    ego: int
    route: int
    dynamic: int
    static: int
    controls: int
    lane: int


class LQEncoder(BaseEncoder):
    def __init__(
        self,
        obs_spec: ObservationSpec,
        d_model: int = 64,
        num_latents: int = 16,
        latent_dim: int = 128,
        output_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        ff_dim: int = 256,
        activation: str = "relu",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_gating: bool = True,
        pooling: str = "mean",
        type_embedding: bool = True,
        time_embedding: bool = True,
        slot_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.spec = obs_spec
        self.output_dim = int(output_dim)
        self.pooling = str(pooling).lower()
        self.use_type_embedding = bool(type_embedding)
        self.use_time_embedding = bool(time_embedding)
        self.use_slot_embedding = bool(slot_embedding)
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)

        self.ego_embed = _TokenMLP(self.spec.ego_dim, self.d_model, activation=activation, layer_norm=True)
        self.route_embed = _TokenMLP(self.spec.route_dim, self.d_model, activation=activation, layer_norm=True)
        self.dynamic_embed = _TokenMLP(
            self.spec.dynamic_dim, self.d_model, activation=activation, layer_norm=True
        )
        self.static_embed = _TokenMLP(self.spec.static_dim, self.d_model, activation=activation, layer_norm=True)
        self.control_embed = _TokenMLP(
            self.spec.control_dim, self.d_model, activation=activation, layer_norm=True
        )
        self.lane_embed = _TokenMLP(self.spec.lane_dim, self.d_model, activation=activation, layer_norm=True)

        if self.use_type_embedding:
            self.type_embed = nn.Embedding(6, self.d_model)
        else:
            self.type_embed = None

        if self.use_time_embedding:
            self.time_embed = nn.Embedding(self.spec.history, self.d_model)
        else:
            self.time_embed = None

        if self.use_slot_embedding:
            self.route_slot_embed = nn.Embedding(self.spec.num_route, self.d_model)
            self.dynamic_slot_embed = nn.Embedding(self.spec.num_dynamic, self.d_model)
            self.static_slot_embed = nn.Embedding(self.spec.num_static, self.d_model)
            self.control_slot_embed = nn.Embedding(self.spec.num_controls, self.d_model)
        else:
            self.route_slot_embed = None
            self.dynamic_slot_embed = None
            self.static_slot_embed = None
            self.control_slot_embed = None

        self.token_to_latent = nn.Linear(self.d_model, self.latent_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, self.latent_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                _LQBlock(
                    latent_dim=self.latent_dim,
                    num_heads=int(num_heads),
                    ff_dim=int(ff_dim),
                    activation=activation,
                    dropout=float(dropout),
                    attention_dropout=float(attention_dropout),
                    residual_gating=bool(residual_gating),
                )
                for _ in range(int(depth))
            ]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            _activation(activation)(),
        )

        self._type_ids = _TokenSlices(ego=0, route=1, dynamic=2, static=3, controls=4, lane=5)

    def _add_type(self, tokens: torch.Tensor, type_idx: int) -> torch.Tensor:
        if self.type_embed is None:
            return tokens
        type_tokens = self.type_embed.weight[type_idx].view(1, 1, -1)
        return tokens + type_tokens

    def _embed_ego(self, obs: StructuredObservation) -> torch.Tensor:
        bsz = obs.ego.shape[0]
        x = self.ego_embed(obs.ego)
        if self.time_embed is not None:
            t = torch.arange(self.spec.history, device=x.device, dtype=torch.long)
            x = x + self.time_embed(t).view(1, self.spec.history, -1)
        x = self._add_type(x, self._type_ids.ego)
        return x.view(bsz, self.spec.history, self.d_model)

    def _embed_route(self, obs: StructuredObservation) -> torch.Tensor:
        x = self.route_embed(obs.route)
        if self.route_slot_embed is not None:
            s = torch.arange(self.spec.num_route, device=x.device, dtype=torch.long)
            x = x + self.route_slot_embed(s).view(1, self.spec.num_route, -1)
        return self._add_type(x, self._type_ids.route)

    def _embed_dynamic(self, obs: StructuredObservation) -> torch.Tensor:
        bsz = obs.dynamic.shape[0]
        x = self.dynamic_embed(obs.dynamic)
        if self.time_embed is not None:
            t = torch.arange(self.spec.history, device=x.device, dtype=torch.long)
            x = x + self.time_embed(t).view(1, 1, self.spec.history, -1)
        if self.dynamic_slot_embed is not None:
            s = torch.arange(self.spec.num_dynamic, device=x.device, dtype=torch.long)
            x = x + self.dynamic_slot_embed(s).view(1, self.spec.num_dynamic, 1, -1)
        x = self._add_type(x, self._type_ids.dynamic)
        return x.view(bsz, self.spec.num_dynamic * self.spec.history, self.d_model)

    def _embed_static(self, obs: StructuredObservation) -> torch.Tensor:
        x = self.static_embed(obs.static)
        if self.static_slot_embed is not None:
            s = torch.arange(self.spec.num_static, device=x.device, dtype=torch.long)
            x = x + self.static_slot_embed(s).view(1, self.spec.num_static, -1)
        return self._add_type(x, self._type_ids.static)

    def _embed_controls(self, obs: StructuredObservation) -> torch.Tensor:
        x = self.control_embed(obs.controls)
        if self.control_slot_embed is not None:
            s = torch.arange(self.spec.num_controls, device=x.device, dtype=torch.long)
            x = x + self.control_slot_embed(s).view(1, self.spec.num_controls, -1)
        return self._add_type(x, self._type_ids.controls)

    def _embed_lane(self, obs: StructuredObservation) -> torch.Tensor:
        x = self.lane_embed(obs.lane).unsqueeze(1)
        return self._add_type(x, self._type_ids.lane)

    def _tokenize(self, obs: StructuredObservation) -> tuple[torch.Tensor, torch.Tensor]:
        ego_tokens = self._embed_ego(obs)
        route_tokens = self._embed_route(obs)
        dynamic_tokens = self._embed_dynamic(obs)
        static_tokens = self._embed_static(obs)
        control_tokens = self._embed_controls(obs)
        lane_token = self._embed_lane(obs)

        tokens = torch.cat(
            [
                ego_tokens,
                route_tokens,
                dynamic_tokens,
                static_tokens,
                control_tokens,
                lane_token,
            ],
            dim=1,
        )
        mask = build_global_token_mask(obs)
        return tokens, mask

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        obs = unflatten_observation(obs_flat, self.spec)
        tokens, mask = self._tokenize(obs)
        tokens = self.token_to_latent(tokens)

        bsz = tokens.shape[0]
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)
        key_padding_mask = ~mask.bool()

        for block in self.blocks:
            latents = block(latents, tokens, key_padding_mask)

        if self.pooling == "mean":
            pooled = latents.mean(dim=1)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")
        return self.output_proj(pooled)
