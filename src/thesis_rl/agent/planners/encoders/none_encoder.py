from __future__ import annotations

import torch

from thesis_rl.agent.planners.encoders.base import BaseEncoder


class NoneEncoder(BaseEncoder):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(input_dim)

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        self._validate_input(flat_obs)
        return flat_obs
