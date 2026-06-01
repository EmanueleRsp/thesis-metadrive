from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from thesis_rl.agent.planners.encoders.base import BaseEncoder


def _activation(name: str) -> type[nn.Module]:
    key = str(name).lower()
    if key == "relu":
        return nn.ReLU
    if key == "tanh":
        return nn.Tanh
    if key == "gelu":
        return nn.GELU
    raise ValueError(f"Unsupported activation: {name}")


class MLPEncoder(BaseEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Iterable[int],
        output_dim: int,
        activation: str = "relu",
        layer_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

        act = _activation(activation)
        layers: list[nn.Module] = []
        prev = int(input_dim)

        for width in hidden_layers:
            width_i = int(width)
            layers.append(nn.Linear(prev, width_i))
            if layer_norm:
                layers.append(nn.LayerNorm(width_i))
            layers.append(act())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = width_i

        layers.append(nn.Linear(prev, self.output_dim))
        layers.append(nn.LayerNorm(self.output_dim))
        layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
