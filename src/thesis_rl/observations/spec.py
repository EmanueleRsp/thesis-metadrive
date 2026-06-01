from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObservationSpec:
    """Canonical semantic-state observation dimensions."""

    history: int = 5
    num_route: int = 5
    num_dynamic: int = 16
    num_static: int = 8
    num_controls: int = 8

    ego_dim: int = 13
    route_dim: int = 5
    dynamic_dim: int = 24
    static_dim: int = 14
    control_dim: int = 16
    lane_dim: int = 12

    @property
    def flat_dim(self) -> int:
        return (
            self.history * self.ego_dim
            + self.num_route * self.route_dim
            + self.num_route
            + self.num_dynamic * self.history * self.dynamic_dim
            + self.num_dynamic * self.history
            + self.num_static * self.static_dim
            + self.num_static
            + self.num_controls * self.control_dim
            + self.num_controls
            + self.lane_dim
        )

    @property
    def num_tokens(self) -> int:
        return (
            self.history
            + self.num_route
            + self.num_dynamic * self.history
            + self.num_static
            + self.num_controls
            + 1
        )

    @classmethod
    def from_obs_config(cls, obs_cfg: Any | None) -> "ObservationSpec":
        cfg = obs_cfg or {}
        return cls(
            history=int(cfg.get("history_length", 5)),
            num_route=int(cfg.get("max_route_points", 5)),
            num_dynamic=int(cfg.get("max_dynamic_objects", 16)),
            num_static=int(cfg.get("max_static_objects", 8)),
            num_controls=int(cfg.get("max_control_objects", 8)),
        )
