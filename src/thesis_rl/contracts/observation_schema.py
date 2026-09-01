"""Authoritative semantic-observation v1.1 schema.

This module is deliberately the only owner of semantic v1.1 group dimensions,
flat slices, and token order. Observation builders and encoders must consume it
instead of reproducing offset arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import ClassVar

import numpy as np
import torch


@dataclass(frozen=True)
class SemanticObservationBatch:
    """Unbatched structured semantic observation payload."""

    ego_history: np.ndarray
    ego_history_mask: np.ndarray
    ego_current: np.ndarray
    route: np.ndarray
    route_mask: np.ndarray
    dynamic: np.ndarray
    dynamic_mask: np.ndarray
    static: np.ndarray
    static_mask: np.ndarray
    lane_road: np.ndarray
    controls: np.ndarray
    controls_mask: np.ndarray
    interactions: np.ndarray
    interactions_mask: np.ndarray
    temporal: np.ndarray


@dataclass(frozen=True)
class SemanticObservationTensorBatch:
    """Batch-first torch representation of semantic observation v1.1."""

    ego_history: torch.Tensor
    ego_history_mask: torch.Tensor
    ego_current: torch.Tensor
    route: torch.Tensor
    route_mask: torch.Tensor
    dynamic: torch.Tensor
    dynamic_mask: torch.Tensor
    static: torch.Tensor
    static_mask: torch.Tensor
    lane_road: torch.Tensor
    controls: torch.Tensor
    controls_mask: torch.Tensor
    interactions: torch.Tensor
    interactions_mask: torch.Tensor
    temporal: torch.Tensor


class SemanticObservationSchemaV11:
    """The frozen v1.1 semantic observation layout and LQ token contract."""

    # Route-derived fields (ego route station, route samples, lane offset,
    # heading error, route-relative actor/feature positions) now source from
    # the mission's canonical trimmed/oriented route and the shared
    # MissionSnapshot station instead of the legacy assigned-route polyline,
    # per DRIVING-MISSION-V1.1 §3/§5. Same tensor widths, new numeric values
    # (DEC-MSN-004): bump the schema identity so old checkpoints are flagged.
    version: ClassVar[str] = "1.1-final-mission-route-v1"
    flat_dim: ClassVar[int] = 2541
    raw_token_count: ClassVar[int] = 122
    group_shapes: ClassVar[dict[str, tuple[int, ...]]] = {
        "ego_history": (5, 10),
        "ego_history_mask": (5,),
        "ego_current": (3,),
        "route": (10, 7),
        "route_mask": (10,),
        "dynamic": (16, 5, 22),
        "dynamic_mask": (16, 5),
        "static": (8, 13),
        "static_mask": (8,),
        "lane_road": (14,),
        "controls": (8, 17),
        "controls_mask": (8,),
        "interactions": (8, 35),
        "interactions_mask": (8,),
        "temporal": (5,),
    }
    token_order: ClassVar[tuple[str, ...]] = (
        "ego_history",
        "ego_current",
        "route",
        "dynamic",
        "static",
        "lane_road",
        "controls",
        "interactions",
        "temporal",
    )
    mask_order: ClassVar[tuple[str, ...]] = (
        "ego_history_mask",
        "route_mask",
        "dynamic_mask",
        "static_mask",
        "controls_mask",
        "interactions_mask",
    )

    def __init__(self) -> None:
        cursor = 0
        slices: dict[str, slice] = {}
        for name, shape in self.group_shapes.items():
            size = int(np.prod(shape))
            slices[name] = slice(cursor, cursor + size)
            cursor += size
        if cursor != self.flat_dim:
            raise RuntimeError(
                f"Semantic v1.1 schema consumed {cursor} dimensions, expected {self.flat_dim}."
            )
        self._slices = slices

    @property
    def slices(self) -> dict[str, slice]:
        """Return copies of canonical flat slices to prevent mutation."""

        return dict(self._slices)

    def flatten_numpy(self, batch: SemanticObservationBatch) -> np.ndarray:
        parts: list[np.ndarray] = []
        for name, shape in self.group_shapes.items():
            value = np.asarray(getattr(batch, name), dtype=np.float32)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"Semantic v1.1 group '{name}' has shape {tuple(value.shape)}, expected {shape}."
                )
            parts.append(value.reshape(-1))
        return np.concatenate(parts, dtype=np.float32)

    def unflatten_numpy(self, flat_obs: np.ndarray) -> SemanticObservationBatch:
        flat = np.asarray(flat_obs, dtype=np.float32)
        if flat.ndim != 1 or flat.shape[0] != self.flat_dim:
            raise ValueError(
                f"Semantic v1.1 flat observation must have shape ({self.flat_dim},), "
                f"got {tuple(flat.shape)}."
            )
        values = {
            name: flat[self._slices[name]].reshape(shape)
            for name, shape in self.group_shapes.items()
        }
        return SemanticObservationBatch(**values)

    def unflatten_torch(self, flat_obs: torch.Tensor) -> SemanticObservationTensorBatch:
        if flat_obs.ndim != 2:
            raise ValueError(
                f"Semantic v1.1 tensors must be batch-first [B, 2541]; got {tuple(flat_obs.shape)}."
            )
        if flat_obs.shape[-1] != self.flat_dim:
            raise ValueError(
                f"Semantic v1.1 tensor has dim {flat_obs.shape[-1]}, expected {self.flat_dim}."
            )
        batch_size = flat_obs.shape[0]
        values = {
            name: flat_obs[:, self._slices[name]].reshape(batch_size, *shape)
            for name, shape in self.group_shapes.items()
        }
        return SemanticObservationTensorBatch(**values)

    def canonical_dict(self) -> dict[str, object]:
        """Return the exact JSON-serializable fingerprint payload."""

        return {
            "schema_version": self.version,
            "flat_order": list(self.group_shapes),
            "group_shapes": {name: list(shape) for name, shape in self.group_shapes.items()},
            "mask_shapes": {name: list(self.group_shapes[name]) for name in self.mask_order},
            "flat_slices": {
                name: [self._slices[name].start, self._slices[name].stop]
                for name in self.group_shapes
            },
            "token_order": list(self.token_order),
        }

    @staticmethod
    def fingerprint_sha256_from_canonical_dict(payload: dict[str, object]) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def fingerprint_sha256(self) -> str:
        return self.fingerprint_sha256_from_canonical_dict(self.canonical_dict())


@dataclass(frozen=True)
class SemanticObservationBatchV12:
    """Unbatched structured payload for OBS-V1.2."""

    ego_history: np.ndarray
    ego_history_mask: np.ndarray
    ego_current: np.ndarray
    route: np.ndarray
    route_mask: np.ndarray
    dynamic: np.ndarray
    dynamic_mask: np.ndarray
    static: np.ndarray
    static_mask: np.ndarray
    lane_road: np.ndarray
    controls: np.ndarray
    controls_mask: np.ndarray
    interactions: np.ndarray
    interactions_mask: np.ndarray
    context_history: np.ndarray
    context_history_mask: np.ndarray
    signal_onset_state: np.ndarray


@dataclass(frozen=True)
class SemanticObservationTensorBatchV12:
    """Batch-first torch representation of OBS-V1.2."""

    ego_history: torch.Tensor
    ego_history_mask: torch.Tensor
    ego_current: torch.Tensor
    route: torch.Tensor
    route_mask: torch.Tensor
    dynamic: torch.Tensor
    dynamic_mask: torch.Tensor
    static: torch.Tensor
    static_mask: torch.Tensor
    lane_road: torch.Tensor
    controls: torch.Tensor
    controls_mask: torch.Tensor
    interactions: torch.Tensor
    interactions_mask: torch.Tensor
    context_history: torch.Tensor
    context_history_mask: torch.Tensor
    signal_onset_state: torch.Tensor


class SemanticObservationSchemaV12:
    """The approved perception-bounded OBS-V1.2 layout and token contract."""

    # See SemanticObservationSchemaV11's mission-route note; the same route
    # consumer swap applies here.
    # OBS-V1.3.1: `lane_road` gains the posted speed limit and its explicit
    # availability flag, required by RULEBOOK-V5.1's `speed_limit` sub-rule.
    # `D` changes from 3009 to 3011 and **checkpoint compatibility is
    # intentionally broken** (`DEC-RB51-001`), which is acceptable because the
    # production runs have not started.
    version: ClassVar[str] = "1.3.1-perception-bounded-mission-route-speed-limit"
    flat_dim: ClassVar[int] = 3011
    raw_token_count: ClassVar[int] = 143
    group_shapes: ClassVar[dict[str, tuple[int, ...]]] = {
        "ego_history": (5, 10),
        "ego_history_mask": (5,),
        "ego_current": (3,),
        "route": (10, 7),
        "route_mask": (10,),
        "dynamic": (16, 5, 22),
        "dynamic_mask": (16, 5),
        "static": (8, 13),
        "static_mask": (8,),
        "lane_road": (14,),
        "controls": (8, 15),
        "controls_mask": (8,),
        "interactions": (8, 33),
        "interactions_mask": (8,),
        "context_history": (21, 23),
        "context_history_mask": (21,),
        "signal_onset_state": (3,),
    }
    token_order: ClassVar[tuple[str, ...]] = (
        "ego_history",
        "ego_current",
        "route",
        "dynamic",
        "static",
        "lane_road",
        "controls",
        "interactions",
        "context_history",
        "signal_onset_state",
    )
    mask_order: ClassVar[tuple[str, ...]] = (
        "ego_history_mask",
        "route_mask",
        "dynamic_mask",
        "static_mask",
        "controls_mask",
        "interactions_mask",
        "context_history_mask",
    )

    def __init__(self) -> None:
        cursor = 0
        slices: dict[str, slice] = {}
        for name, shape in self.group_shapes.items():
            size = int(np.prod(shape))
            slices[name] = slice(cursor, cursor + size)
            cursor += size
        if cursor != self.flat_dim:
            raise RuntimeError(
                f"Semantic v1.2 schema consumed {cursor} dimensions, expected {self.flat_dim}."
            )
        self._slices = slices

    @property
    def slices(self) -> dict[str, slice]:
        """Return copies of canonical flat slices to prevent mutation."""

        return dict(self._slices)

    def flatten_numpy(self, batch: SemanticObservationBatchV12) -> np.ndarray:
        parts: list[np.ndarray] = []
        for name, shape in self.group_shapes.items():
            value = np.asarray(getattr(batch, name), dtype=np.float32)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"Semantic v1.2 group '{name}' has shape {tuple(value.shape)}, expected {shape}."
                )
            parts.append(value.reshape(-1))
        return np.concatenate(parts, dtype=np.float32)

    def unflatten_numpy(self, flat_obs: np.ndarray) -> SemanticObservationBatchV12:
        flat = np.asarray(flat_obs, dtype=np.float32)
        if flat.ndim != 1 or flat.shape[0] != self.flat_dim:
            raise ValueError(
                f"Semantic v1.2 flat observation must have shape ({self.flat_dim},), "
                f"got {tuple(flat.shape)}."
            )
        values = {
            name: flat[self._slices[name]].reshape(shape)
            for name, shape in self.group_shapes.items()
        }
        return SemanticObservationBatchV12(**values)

    def unflatten_torch(self, flat_obs: torch.Tensor) -> SemanticObservationTensorBatchV12:
        if flat_obs.ndim != 2:
            raise ValueError(
                f"Semantic v1.2 tensors must be batch-first [B, {self.flat_dim}]; "
                f"got {tuple(flat_obs.shape)}."
            )
        if flat_obs.shape[-1] != self.flat_dim:
            raise ValueError(
                f"Semantic v1.2 tensor has dim {flat_obs.shape[-1]}, expected {self.flat_dim}."
            )
        batch_size = flat_obs.shape[0]
        values = {
            name: flat_obs[:, self._slices[name]].reshape(batch_size, *shape)
            for name, shape in self.group_shapes.items()
        }
        return SemanticObservationTensorBatchV12(**values)

    def canonical_dict(self) -> dict[str, object]:
        """Return the exact JSON-serializable fingerprint payload."""

        return {
            "schema_version": self.version,
            "flat_order": list(self.group_shapes),
            "group_shapes": {name: list(shape) for name, shape in self.group_shapes.items()},
            "mask_shapes": {name: list(self.group_shapes[name]) for name in self.mask_order},
            "flat_slices": {
                name: [self._slices[name].start, self._slices[name].stop]
                for name in self.group_shapes
            },
            "token_order": list(self.token_order),
        }

    @staticmethod
    def fingerprint_sha256_from_canonical_dict(payload: dict[str, object]) -> str:
        return SemanticObservationSchemaV11.fingerprint_sha256_from_canonical_dict(payload)

    def fingerprint_sha256(self) -> str:
        return self.fingerprint_sha256_from_canonical_dict(self.canonical_dict())
