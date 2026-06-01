from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseEncoder(nn.Module, ABC):
    output_dim: int

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
