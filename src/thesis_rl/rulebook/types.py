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
    allowed_driving_area: Any | None = None
    opposite_carriageway: Any | None = None
    lane_centerline: Any | None = None
    solid_lane_markings: Any | None = None
    dashed_lane_markings: Any | None = None
    lane_boundaries: Any | None = None
    route_progress: float | None = None
    prev_route_progress: float | None = None
    route_checkpoints: Sequence[Any] | None = None
    target_region: Any | None = None
    target_point: Any | None = None
    speed_limit: float | None = None
    local_to_global: Any | None = None
    prev_ego_state: Mapping[str, Any] | None = None
    prev_neighbors: Sequence[Mapping[str, Any]] | None = None
    prev_neighbors_by_id: Mapping[str, Mapping[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuleVector:
    """Ordered vector of rule margins."""

    names: list[str]
    values: np.ndarray
    priorities: list[int]
    results: list["RuleResult"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuleResult:
    """Structured result for one Rulebook v1 rule."""

    name: str
    margin: float
    violated: bool
    severity: float
    available: bool
    fallback_used: bool
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "margin": float(self.margin),
            "violated": bool(self.violated),
            "severity": float(self.severity),
            "available": bool(self.available),
            "fallback_used": bool(self.fallback_used),
            "raw": dict(self.raw),
        }


@dataclass(slots=True)
class RuleSpec:
    """Rule specification loaded from config."""

    name: str
    fn: Any
    priority: int
    params: dict[str, Any] = field(default_factory=dict)
    order: int = 0
