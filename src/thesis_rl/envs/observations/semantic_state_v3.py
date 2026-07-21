"""Strict runtime adapter for the perception-bounded semantic OBS-V1.2 contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from metadrive.obs.observation_base import BaseObservation

from thesis_rl.contracts.observation_schema import (
    SemanticObservationBatchV12,
    SemanticObservationSchemaV12,
)


class SemanticStateObservationV3(BaseObservation):
    """Expose one schema-owned, perception-bounded OBS-V1.2 observation."""

    FLAT_DIM = SemanticObservationSchemaV12.flat_dim

    def __init__(self, config: Mapping[str, Any] | dict[str, Any]):
        self.schema = SemanticObservationSchemaV12()
        self._batch_builder: Callable[[object], SemanticObservationBatchV12] | None = None
        super().__init__(dict(config))
        self._observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.schema.flat_dim,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def set_batch_builder(self, builder: Callable[[object], SemanticObservationBatchV12]) -> None:
        """Install the environment-owned perception-bounded batch builder."""

        if not callable(builder):
            raise TypeError("Semantic v1.2 batch builder must be callable")
        self._batch_builder = builder

    def reset(self, env: object, vehicle: object | None = None) -> None:
        del env, vehicle

    def observe(self, vehicle: object) -> np.ndarray:
        if self._batch_builder is None:
            raise RuntimeError("SemanticStateObservationV3 requires a perception-bounded batch builder")
        batch = self._batch_builder(vehicle)
        if not isinstance(batch, SemanticObservationBatchV12):
            raise TypeError("Semantic v1.2 batch builder must return SemanticObservationBatchV12")
        self._validate_masks_and_padding(batch)
        flat = self.schema.flatten_numpy(batch)
        if flat.shape != (self.schema.flat_dim,):
            raise RuntimeError("Semantic v1.2 schema produced an invalid flat dimension")
        if not np.all(np.isfinite(flat)):
            raise ValueError("Semantic v1.2 observation must contain finite values")
        if not np.all((-1.0 <= flat) & (flat <= 1.0)):
            raise ValueError("Semantic v1.2 observation must be bounded in [-1, 1]")
        self.current_observation = flat
        return flat

    @staticmethod
    def _validate_masks_and_padding(batch: SemanticObservationBatchV12) -> None:
        schema = SemanticObservationSchemaV12()
        if not any(
            np.any(np.asarray(getattr(batch, name), dtype=np.float32)) for name in schema.mask_order
        ):
            raise ValueError("Semantic v1.2 batch cannot mask all variable tokens")
        for name in schema.mask_order:
            mask = np.asarray(getattr(batch, name), dtype=np.float32)
            if not np.all((mask == 0.0) | (mask == 1.0)):
                raise ValueError(f"Semantic mask '{name}' must be binary")

        masked_groups = {
            "ego_history": "ego_history_mask",
            "route": "route_mask",
            "dynamic": "dynamic_mask",
            "static": "static_mask",
            "controls": "controls_mask",
            "interactions": "interactions_mask",
            "compliance_history": "compliance_history_mask",
        }
        for group_name, mask_name in masked_groups.items():
            values = np.asarray(getattr(batch, group_name), dtype=np.float32)
            mask = np.asarray(getattr(batch, mask_name), dtype=np.float32)
            invalid = mask[..., np.newaxis] == 0.0
            if np.any(np.where(invalid, np.abs(values), 0.0) > 0.0):
                raise ValueError(f"Masked '{group_name}' tokens must be zero")
