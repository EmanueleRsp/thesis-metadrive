from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
from torch import nn


class BaseEncoder(nn.Module, ABC):
    """Versioned encoder contract shared by all approved implementations."""

    observation_schema_version: ClassVar[str | None] = None
    encoder_architecture_version: ClassVar[str] = "1.0-final"
    input_dim: int
    output_dim: int

    def _validate_input(self, flat_obs: torch.Tensor) -> None:
        if flat_obs.ndim != 2:
            raise ValueError(
                f"Thesis encoders require rank 2 input [B, D], got {tuple(flat_obs.shape)}."
            )
        if flat_obs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Thesis encoder received D={flat_obs.shape[-1]}, expected {self.input_dim}."
            )
        if __debug__:
            if flat_obs.dtype != torch.float32:
                raise TypeError(
                    f"Thesis encoders require torch.float32 input, got {flat_obs.dtype}."
                )
            if not torch.isfinite(flat_obs).all():
                raise ValueError("Thesis encoder input must contain only finite values.")

    @abstractmethod
    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
