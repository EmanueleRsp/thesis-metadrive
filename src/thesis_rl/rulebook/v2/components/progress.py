"""Canonical route-progress evaluator."""

from __future__ import annotations

from math import isfinite

from thesis_rl.rulebook.v2.geometry.route import GEOMETRY_EPSILON_M, RoutePolyline
from thesis_rl.rulebook.v2.types import (
    ActorSnapshot,
    CacheDelta,
    ComponentStatus,
    MemoryDelta,
    RuleComponentResult,
)


# OPEN-EF-03: derived implementation constant, not a scientific parameter.  The
# physical bound on one step's route advance is v_max * delta_t, but cutting the
# inside of a curve makes the centerline coordinate advance faster than the ego
# itself, so a raw factor of 1.0 would deprioritize legitimate motion.  The
# bound is a preference inside ``project``, never a hard gate.  Recorded here so
# it is visible rather than buried in an expression.
ROUTE_CONTINUITY_JUMP_FACTOR = 2.0


def route_outside_fraction(ego_footprint, task_corridor) -> float | None:
    """Return the ego footprint area fraction outside the task corridor.

    REQ-EF-15, diagnostic only.  ``R4`` currently credits projected
    longitudinal progress with no lateral cut-off, and the drivable surface
    used by R3 off-road is the union of *every* map lane, so advancing along a
    legal parallel road scores positive progress with zero off-road cost.
    This measures how often that actually happens, so a corridor gate can be
    decided from data rather than assumed (DEC-EF-06 / M6a).  It deliberately
    does not affect any cost or margin.
    """

    if task_corridor is None or task_corridor.is_empty:
        return None
    area = ego_footprint.area
    if not isfinite(area) or area <= 0.0:
        return None
    outside = ego_footprint.difference(task_corridor).area
    if not isfinite(outside) or outside < 0.0:
        return None
    return min(max(outside / area, 0.0), 1.0)


def evaluate_progress(
    *,
    pre_ego: ActorSnapshot,
    post_ego: ActorSnapshot,
    route: RoutePolyline,
    previous_route_s_m: float,
    delta_t_s: float,
    task_corridor=None,
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Project pre/post ego positions and return raw delta plus normalized margin."""
    cap = post_ego.configured_speed_cap_mps
    if (
        cap is None
        or not isfinite(cap)
        or cap <= 0.0
        or not isfinite(delta_t_s)
        or delta_t_s <= 0.0
    ):
        raise ValueError("Progress requires positive configured speed cap and timestep")
    if not isfinite(previous_route_s_m):
        raise ValueError("Previous route coordinate must be finite")
    pre = route.project(
        pre_ego.position_xy, position_z=pre_ego.position_z, previous_s_m=previous_route_s_m
    )
    if abs(pre.s_m - previous_route_s_m) > GEOMETRY_EPSILON_M:
        raise ValueError("Pre-state route projection is discontinuous with memory")
    # OPEN-EF-03: the pre-state projection is already protected by the
    # discontinuity check above; the post-state one was not, so a
    # self-intersecting or closely parallel route (a roundabout, in practice)
    # could jump to a far branch and report spurious progress.  The bound reuses
    # v_max * delta_t, the same normalizer §8.2 already uses for the margin,
    # scaled by ROUTE_CONTINUITY_JUMP_FACTOR: cutting the inside of a curve
    # legitimately advances the centerline coordinate faster than the ego's own
    # displacement, so the raw product would be too tight.
    post = route.project(
        post_ego.position_xy,
        position_z=post_ego.position_z,
        previous_s_m=pre.s_m,
        max_s_jump_m=ROUTE_CONTINUITY_JUMP_FACTOR * cap * delta_t_s,
    )
    raw_delta = post.s_m - pre.s_m
    margin = min(max(raw_delta / (cap * delta_t_s), -1.0), 1.0)
    outside_fraction = route_outside_fraction(post_ego.footprint, task_corridor)
    diagnostics: dict = {"speed_cap_mps": cap, "delta_t_s": delta_t_s}
    if outside_fraction is not None:
        # REQ-EF-15: diagnostic only; the margin above is unchanged.
        diagnostics["route_outside_fraction"] = outside_fraction
        diagnostics["route_adherence"] = 1.0 - outside_fraction
    result = RuleComponentResult(
        "progress",
        margin,
        {"route_delta_m": raw_delta, "pre_s_m": pre.s_m, "post_s_m": post.s_m},
        True,
        True,
        ComponentStatus.SATISFIED,
        diagnostics,
    )
    delta = MemoryDelta(writer="progress", writes=(("previous_route_s_m", post.s_m),))
    return result, delta, CacheDelta()
