"""Canonical route-progress evaluator."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from thesis_rl.rulebook.v2.types import (
    CacheDelta,
    ComponentStatus,
    MemoryDelta,
    RuleComponentResult,
)

if TYPE_CHECKING:
    from thesis_rl.mission.types import MissionSnapshot


# OPEN-EF-03: derived implementation constant, not a scientific parameter.  The
# physical bound on one step's route advance is v_max * delta_t, but cutting the
# inside of a curve makes the centerline coordinate advance faster than the ego
# itself, so a raw factor of 1.0 would deprioritize legitimate motion.  The
# bound is a preference inside ``project``, never a hard gate.  Recorded here so
# it is visible rather than buried in an expression.


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


MISSION_PROGRESS_REFERENCE_SPEED_MPS = 22.2222222222


def evaluate_progress(
    *,
    pre_mission: MissionSnapshot,
    post_mission: MissionSnapshot,
    delta_t_s: float,
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate R4 exclusively from the exact canonical-route ``delta_s``."""
    from thesis_rl.mission.types import MissionSnapshot

    if not isinstance(pre_mission, MissionSnapshot) or not isinstance(
        post_mission, MissionSnapshot
    ):
        raise ValueError("Progress requires pre/post mission context")
    if pre_mission.mission_hash != post_mission.mission_hash:
        raise ValueError("Progress mission snapshot identity must match")
    terminal_unreachable_noop = (
        pre_mission.mission_unreachable
        and post_mission.mission_unreachable
        and post_mission.step_index == pre_mission.step_index
    )
    if not terminal_unreachable_noop and post_mission.step_index != pre_mission.step_index + 1:
        raise ValueError(
            "Progress mission snapshots must be consecutive: "
            f"pre_step={pre_mission.step_index}, post_step={post_mission.step_index}, "
            f"pre_s_m={pre_mission.s_m}, post_s_m={post_mission.s_m}"
        )
    if not isfinite(delta_t_s) or delta_t_s <= 0.0:
        raise ValueError("Progress requires a positive finite timestep")
    if pre_mission.s_m is None or post_mission.s_m is None:
        raise ValueError("R4 requires exact canonical route stations")
    raw_delta = post_mission.s_m - pre_mission.s_m
    margin = min(max(raw_delta / (MISSION_PROGRESS_REFERENCE_SPEED_MPS * delta_t_s), -1.0), 1.0)
    diagnostics = {
        "reference_speed_mps": MISSION_PROGRESS_REFERENCE_SPEED_MPS,
        "delta_t_s": delta_t_s,
        "terminal_mission_unreachable_noop": terminal_unreachable_noop,
    }
    result = RuleComponentResult(
        "progress",
        margin,
        {
            "route_delta_m": raw_delta,
            "delta_s_m": raw_delta,
            "mission_distance_delta_m": raw_delta,
        },
        True,
        True,
        ComponentStatus.SATISFIED,
        diagnostics,
    )
    return result, MemoryDelta(writer="progress"), CacheDelta()
