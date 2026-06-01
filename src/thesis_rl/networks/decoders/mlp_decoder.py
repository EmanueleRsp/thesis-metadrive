from __future__ import annotations

from typing import Iterable

from torch import nn


def _activation(name: str) -> type[nn.Module]:
    key = str(name).lower()
    if key == "relu":
        return nn.ReLU
    if key == "tanh":
        return nn.Tanh
    if key == "gelu":
        return nn.GELU
    raise ValueError(f"Unsupported activation: {name}")


class MLPDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Iterable[int],
        output_dim: int | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
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
        if output_dim is not None:
            layers.append(nn.Linear(prev, int(output_dim)))
            prev = int(output_dim)

        self.net = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x):
        return self.net(x)
