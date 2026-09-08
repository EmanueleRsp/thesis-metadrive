"""At-fault classification for contact onsets (ADR-071), following nuPlan.

`evaluate_collision_impact` charges every contact the MAIS3+F injury risk of the
pre-state closing speed and the actor class alone. Whoever caused the contact is
not consulted, so a stopped ego struck from behind pays **full cost at the
highest priority level** -- the canonical case the RSS literature uses to
introduce blame, and the level at which nuPlan actually applies at-fault logic.

The pressure that creates is the opposite of the degeneracy this project has been
chasing: punishing an unavoidable event teaches the agent to avoid *legitimate*
stops, at lights and at crossings, which would be a new degeneracy rather than a
fix for the old one.

This module is pure and shared. The reward reads it to decide what to charge, and
the episode contract reads it to decide whether the episode terminates or
truncates; those must never be able to disagree.
"""

from __future__ import annotations

from enum import Enum
from math import cos, hypot, sin

from thesis_rl.rulebook.v2.types import ActorSnapshot


# nuPlan's `_get_collision_type(..., stopped_speed_threshold: float = 5e-02)`,
# the same published constant ADR-070's gate uses.
STOPPED_SPEED_THRESHOLD_MPS = 5e-02


class CollisionFault(str, Enum):
    """nuPlan's collision taxonomy, restricted to what is decidable here."""

    STOPPED_EGO = "stopped_ego_collision"
    ACTIVE_REAR = "active_rear_collision"
    STOPPED_TRACK = "stopped_track_collision"
    ACTIVE_FRONT = "active_front_collision"
    ACTIVE_LATERAL = "active_lateral_collision"


# `STOPPED_EGO` and `ACTIVE_REAR` are excluded, exactly as nuPlan excludes them.
# `ACTIVE_LATERAL` is conditional and is resolved by the caller's lane-containment
# input before it reaches this set.
_UNCONDITIONALLY_AT_FAULT = frozenset({CollisionFault.STOPPED_TRACK, CollisionFault.ACTIVE_FRONT})


def classify_contact(
    *,
    pre_ego: ActorSnapshot,
    actor: ActorSnapshot,
) -> CollisionFault:
    """Classify one contact onset from the pre-transition state.

    The order is nuPlan's and it matters: a stopped ego is not at fault whatever
    the geometry, which is checked first; then a stopped track, which the ego
    drove into whatever its own geometry; then whether the other agent is behind
    the ego, which is the rear-impact exclusion; and only then front versus
    lateral.
    """

    if hypot(*pre_ego.velocity_xy) <= STOPPED_SPEED_THRESHOLD_MPS:
        return CollisionFault.STOPPED_EGO
    if hypot(*actor.velocity_xy) <= STOPPED_SPEED_THRESHOLD_MPS:
        return CollisionFault.STOPPED_TRACK
    if _is_agent_behind(pre_ego, actor):
        return CollisionFault.ACTIVE_REAR
    if _is_agent_ahead(pre_ego, actor):
        return CollisionFault.ACTIVE_FRONT
    return CollisionFault.ACTIVE_LATERAL


def is_at_fault(
    fault: CollisionFault,
    *,
    ego_within_single_lane: bool | None,
) -> bool:
    """Whether the ego is to blame, resolving the conditional lateral branch.

    ``ego_within_single_lane`` is nuPlan's condition for the lateral case: a
    lateral impact is the ego's fault when its footprint is **not** fully within
    a single lane or lane connector, i.e. when the ego was the one changing
    lanes or straddling.

    ``None`` means the containment could not be determined, and it resolves to
    **at fault**. That direction is deliberate: an undeterminable input must
    never buy an exculpation, because the failure would be silent and would
    reward exactly the states where the geometry is hardest to resolve.
    """

    if fault in _UNCONDITIONALLY_AT_FAULT:
        return True
    if fault is CollisionFault.ACTIVE_LATERAL:
        return ego_within_single_lane is not True
    return False


def _relative_position(pre_ego: ActorSnapshot, actor: ActorSnapshot) -> tuple[float, float]:
    """The actor's centre in the ego's frame: +x ahead, +y left."""

    dx = float(actor.position_xy[0]) - float(pre_ego.position_xy[0])
    dy = float(actor.position_xy[1]) - float(pre_ego.position_xy[1])
    heading = float(pre_ego.heading_rad)
    return (
        dx * cos(heading) + dy * sin(heading),
        -dx * sin(heading) + dy * cos(heading),
    )


def _is_agent_behind(pre_ego: ActorSnapshot, actor: ActorSnapshot) -> bool:
    longitudinal, _ = _relative_position(pre_ego, actor)
    return longitudinal < 0.0


def _is_agent_ahead(pre_ego: ActorSnapshot, actor: ActorSnapshot) -> bool:
    """Ahead and within the ego's own width band, i.e. a front impact.

    The band is the ego's half width taken from its footprint, so the test is
    "would the front bumper sweep meet it", not "is it somewhere forward".
    """

    longitudinal, lateral = _relative_position(pre_ego, actor)
    if longitudinal <= 0.0:
        return False
    return abs(lateral) <= _footprint_half_width(pre_ego)


def _footprint_half_width(actor: ActorSnapshot) -> float:
    """Half of the footprint's *shorter* oriented side, i.e. half its width.

    Read from the minimum rotated rectangle so the value does not depend on the
    heading: axis-aligned bounds of a rotated rectangle inflate with the angle.
    An earlier revision took the *longer* side of the axis-aligned bounds, i.e.
    half the vehicle's length (~2.4 m), which classified lateral impacts inside
    that band as frontal and therefore unconditionally at fault
    (audit 2026-09-06, A6).
    """

    rectangle = actor.footprint.minimum_rotated_rectangle
    coords = list(rectangle.exterior.coords) if hasattr(rectangle, "exterior") else []
    if len(coords) < 4:
        bounds = actor.footprint.bounds
        return min(bounds[3] - bounds[1], bounds[2] - bounds[0]) / 2.0
    side_a = hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    side_b = hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    return min(side_a, side_b) / 2.0
