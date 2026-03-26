from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(slots=True)
class RuleEvalInput:
    """Context payload consumed by rule functions."""

    ego_state: Mapping[str, Any]
    neighbors: Sequence[Mapping[str, Any]]
    drivable_area: Any | None = None
    opposite_carriageway: Any | None = None
    lane_centerline: Any | None = None
    target_region: Any | None = None
    target_point: Any | None = None
    speed_limit: float | None = None
    local_to_global: Any | None = None
    prev_ego_state: Mapping[str, Any] | None = None
    prev_neighbors: Sequence[Mapping[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuleVector:
    """Ordered vector of rule margins."""

    names: list[str]
    values: np.ndarray
    priorities: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuleSpec:
    """Rule specification loaded from config."""

    name: str
    fn: Any
    priority: int
    params: dict[str, Any] = field(default_factory=dict)
    order: int = 0
