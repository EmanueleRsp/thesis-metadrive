"""Directed mission-gate geometry using the canonical Rulebook crossing primitive."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from shapely.geometry import LineString, Polygon

from thesis_rl.rulebook.v2.geometry.footprint import front_bumper_segment, swept_front_bumper
from thesis_rl.rulebook.v2.geometry.vertical import VERTICAL_COMPATIBILITY_TOLERANCE_M


GATE_CROSSING_EPSILON_M = 0.05


@dataclass(frozen=True, slots=True)
class GateGeometry:
    line_xy: tuple[tuple[float, float], tuple[float, float]]
    forward_tangent_xy: tuple[float, float]
    elevation_m: float

    def __post_init__(self) -> None:
        values = (*self.line_xy[0], *self.line_xy[1], *self.forward_tangent_xy, self.elevation_m)
        if not all(isfinite(value) for value in values):
            raise ValueError("gate geometry values must be finite")
        if hypot(*self.forward_tangent_xy) <= 0.0 or self.line_xy[0] == self.line_xy[1]:
            raise ValueError("gate geometry must have a non-zero line and tangent")

    @property
    def line(self) -> LineString:
        return LineString(self.line_xy)


def directed_gate_crossed(
    gate: GateGeometry,
    *,
    pre_footprint: Polygon,
    post_footprint: Polygon,
    pre_heading_rad: float,
    post_heading_rad: float,
    post_ego_z_m: float,
) -> bool:
    """Return true only for a forward, level-compatible swept-bumper crossing."""
    if (
        not isfinite(post_ego_z_m)
        or abs(post_ego_z_m - gate.elevation_m) > VERTICAL_COMPATIBILITY_TOLERANCE_M
    ):
        return False
    pre_front = front_bumper_segment(pre_footprint, heading_rad=pre_heading_rad).centroid
    post_front = front_bumper_segment(post_footprint, heading_rad=post_heading_rad).centroid
    anchor_x, anchor_y = gate.line_xy[0]
    # The signed-distance normal points upstream so the canonical Rulebook
    # predicate is ``pre >= -epsilon`` then ``post < -epsilon``.
    normal = (-gate.forward_tangent_xy[0], -gate.forward_tangent_xy[1])
    pre_signed = (pre_front.x - anchor_x) * normal[0] + (pre_front.y - anchor_y) * normal[1]
    post_signed = (post_front.x - anchor_x) * normal[0] + (post_front.y - anchor_y) * normal[1]
    forward = (post_front.x - pre_front.x) * gate.forward_tangent_xy[0] + (
        post_front.y - pre_front.y
    ) * gate.forward_tangent_xy[1]
    return (
        pre_signed >= -GATE_CROSSING_EPSILON_M
        and post_signed < -GATE_CROSSING_EPSILON_M
        and forward > 0.0
        and swept_front_bumper(
            pre_footprint,
            post_footprint,
            pre_heading_rad=pre_heading_rad,
            post_heading_rad=post_heading_rad,
        ).intersects(gate.line)
    )
