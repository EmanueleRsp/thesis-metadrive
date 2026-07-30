"""Canonical route-progress evaluator."""

from __future__ import annotations

from math import isfinite

from thesis_rl.rulebook.v2.errors import (
    RuntimeScenarioNotEvaluableError,
    RuntimeScenarioNotEvaluableReason,
)
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
    # OPEN-EF-03: on a self-intersecting or closely parallel route (a
    # roundabout, in practice), an unbounded nearest-point search can pick a
    # strictly-closer far branch over the point next to ``previous_s_m``,
    # producing a spurious jump. ``route.project`` treats ``max_s_jump_m`` as
    # a preference, not a hard gate: it narrows the candidate set only when a
    # plausible one already exists nearby, and falls back to the unbounded
    # search otherwise, so applying it here cannot mask a genuinely
    # discontinuous memory -- it can only stop a false positive when a
    # continuous candidate was available all along. The bound reuses
    # v_max * delta_t, the same normalizer Sec.8.2 already uses for the
    # margin, scaled by ROUTE_CONTINUITY_JUMP_FACTOR: cutting the inside of a
    # curve legitimately advances the centerline coordinate faster than the
    # ego's own displacement, so the raw product would be too tight.
    max_route_jump_m = ROUTE_CONTINUITY_JUMP_FACTOR * cap * delta_t_s
    pre = route.project(
        pre_ego.position_xy,
        position_z=pre_ego.position_z,
        previous_s_m=previous_route_s_m,
        max_s_jump_m=max_route_jump_m,
    )
    if abs(pre.s_m - previous_route_s_m) > GEOMETRY_EPSILON_M:
        # DEC-EF-XX (user-directed, session of 2026-07-30): treated as a
        # runtime scenario data defect eligible for a single-episode data
        # abort, mirroring INVALID_SIGNAL_TRANSITION, rather than a fatal
        # ValueError. The OPEN-EF-03 jump bound above already prefers a
        # continuous candidate near ``previous_s_m`` when one is plausible;
        # reaching this branch means no such candidate existed within one
        # timestep's travel, which on live ScenarioNet/Waymo routes reflects
        # map/route-annotation ambiguity (self-intersecting or branching
        # geometry) rather than a recoverable programming error. Diagnostics
        # are attached in full so root-cause analysis does not require
        # reproducing a live crash. Root cause not yet confirmed; revisit if
        # data-abort frequency turns out to be high.
        cause = ValueError("Pre-state route projection is discontinuous with memory")
        raise RuntimeScenarioNotEvaluableError(
            RuntimeScenarioNotEvaluableReason.ROUTE_PROJECTION_DISCONTINUOUS,
            str(cause),
            diagnostics={
                "pre_xy": pre_ego.position_xy,
                "previous_route_s_m": previous_route_s_m,
                "projected_s_m": pre.s_m,
                "segment_index": pre.segment_index,
                "max_route_jump_m": max_route_jump_m,
                "configured_speed_cap_mps": cap,
                "delta_t_s": delta_t_s,
                "route_point_count": len(route.points_xyz),
                "route_points": route.points_xyz,
            },
        ) from cause
    post = route.project(
        post_ego.position_xy,
        position_z=post_ego.position_z,
        previous_s_m=pre.s_m,
        max_s_jump_m=max_route_jump_m,
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
