"""Canonical route-progress evaluator."""

from __future__ import annotations

from math import isfinite

from thesis_rl.rulebook.v2.geometry.route import GEOMETRY_EPSILON_M, RoutePolyline
from thesis_rl.rulebook.v2.types import ActorSnapshot, CacheDelta, ComponentStatus, MemoryDelta, RuleComponentResult


def evaluate_progress(*, pre_ego: ActorSnapshot, post_ego: ActorSnapshot, route: RoutePolyline,
                      previous_route_s_m: float, delta_t_s: float) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Project pre/post ego positions and return raw delta plus normalized margin."""
    cap = post_ego.configured_speed_cap_mps
    if cap is None or not isfinite(cap) or cap <= 0.0 or not isfinite(delta_t_s) or delta_t_s <= 0.0:
        raise ValueError("Progress requires positive configured speed cap and timestep")
    if not isfinite(previous_route_s_m):
        raise ValueError("Previous route coordinate must be finite")
    pre = route.project(pre_ego.position_xy, position_z=pre_ego.position_z, previous_s_m=previous_route_s_m)
    if abs(pre.s_m - previous_route_s_m) > GEOMETRY_EPSILON_M:
        raise ValueError("Pre-state route projection is discontinuous with memory")
    post = route.project(post_ego.position_xy, position_z=post_ego.position_z, previous_s_m=pre.s_m)
    raw_delta = post.s_m - pre.s_m
    margin = min(max(raw_delta / (cap * delta_t_s), -1.0), 1.0)
    result = RuleComponentResult(
        "progress", margin, {"route_delta_m": raw_delta, "pre_s_m": pre.s_m, "post_s_m": post.s_m},
        True, True, ComponentStatus.SATISFIED, {"speed_cap_mps": cap, "delta_t_s": delta_t_s},
    )
    delta = MemoryDelta(writer="progress", writes=(("previous_route_s_m", post.s_m),))
    return result, delta, CacheDelta()
