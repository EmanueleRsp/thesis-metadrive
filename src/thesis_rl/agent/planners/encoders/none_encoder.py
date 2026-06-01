from __future__ import annotations

import torch

from thesis_rl.agent.planners.encoders.base import BaseEncoder


class NoneEncoder(BaseEncoder):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.output_dim = int(input_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs
