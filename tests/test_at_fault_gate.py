"""`T-RB51-10`: ADR-070's at-fault gate on the L2 interaction sub-rules.

The gate exists because `clearance`, `ttc` and `rss_lateral` have their cost set
by another agent's state, so from a stopped ego the region they define is not
controlled-invariant -- the same test that rejected `rss` longitudinal. It makes
them **inapplicable**, which is a different statement from satisfied-at-zero and
is the one the specification requires: the rule has nothing to say about that
state, it is not being met.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.aggregation import aggregate_max_component
from thesis_rl.rulebook.v2.components.at_fault_gate import (
    AT_FAULT_GATE_SPEED_MPS,
    AT_FAULT_GATED_SUB_RULES,
    at_fault_gated_result,
    ego_is_stopped,
)
from thesis_rl.rulebook.v2.components.clearance import evaluate_clearance
from thesis_rl.rulebook.v2.components.progress import MISSION_PROGRESS_REFERENCE_SPEED_MPS
from thesis_rl.rulebook.v2.components.road import evaluate_offroad, evaluate_solid_line
from thesis_rl.rulebook.v2.components.rss_lateral import evaluate_rss_lateral
from thesis_rl.rulebook.v2.components.ttc import evaluate_ttc
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, ComponentStatus


_EGO = Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5)))
_ROADWAY = Polygon(((-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)))
_STOPPED = (0.0, 0.0)
_MOVING = (5.0, 0.0)


def _vru(actor_id: str, x: float) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id,
        ActorClass.CYCLIST,
        (x, 0.0),
        0.0,
        0.0,
        (0.0, 0.0),
        Polygon(((x, -0.5), (x + 1.0, -0.5), (x + 1.0, 0.5), (x, 0.5))),
        None,
        20.0,
    )


def _closing_vehicle(actor_id: str, x: float) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id,
        ActorClass.VEHICLE,
        (x, 0.0),
        0.0,
        0.0,
        (-1.0, 0.0),
        Polygon(((x, -0.5), (x + 1.0, -0.5), (x + 1.0, 0.5), (x, 0.5))),
        None,
        20.0,
    )


def test_the_gate_threshold_is_below_any_progress_the_agent_could_want() -> None:
    """Why an evaluation threshold may be transplanted into a reward.

    A planner being scored does not optimise the scorer; an RL agent optimises
    the reward and will find any hole in it. The only defence here is magnitude:
    at the gate speed the L4 margin is a rounding error, so crawling under the
    gate to become immune to the three sub-rules earns nothing. A gate at any of
    the larger swept thresholds (0.5, 1.0, 2.0 m/s) would not have this property
    and must not be used.
    """

    assert AT_FAULT_GATE_SPEED_MPS / MISSION_PROGRESS_REFERENCE_SPEED_MPS < 0.005


def test_gate_boundary_is_inclusive_and_tight() -> None:
    assert ego_is_stopped((AT_FAULT_GATE_SPEED_MPS, 0.0)) is True
    assert ego_is_stopped((AT_FAULT_GATE_SPEED_MPS * 1.001, 0.0)) is False
    # Speed, not a signed component: reversing at the same rate is not stopped.
    assert ego_is_stopped((-AT_FAULT_GATE_SPEED_MPS * 2.0, 0.0)) is False
    # And it is the magnitude of the whole vector, not of one axis.
    assert ego_is_stopped((0.0, AT_FAULT_GATE_SPEED_MPS * 2.0)) is False


def test_gate_covers_exactly_the_three_interaction_sub_rules() -> None:
    assert AT_FAULT_GATED_SUB_RULES == {"clearance", "ttc", "rss_lateral"}
    with pytest.raises(ValueError, match="not an at-fault gated sub-rule"):
        at_fault_gated_result("offroad")


def test_stopped_ego_makes_clearance_inapplicable_not_satisfied() -> None:
    """The hazard is real and unchanged; only the ego's ability to act differs."""

    moving, _, _ = evaluate_clearance(
        ego_footprint=_EGO,
        actors=(_vru("cyclist", 1.5),),
        vertically_compatible_actor_ids=frozenset({"cyclist"}),
        drivable_surface=_ROADWAY,
        post_ego_velocity_xy=_MOVING,
    )
    stopped, _, _ = evaluate_clearance(
        ego_footprint=_EGO,
        actors=(_vru("cyclist", 1.5),),
        vertically_compatible_actor_ids=frozenset({"cyclist"}),
        drivable_surface=_ROADWAY,
        post_ego_velocity_xy=_STOPPED,
    )

    assert moving.cost == pytest.approx(0.5)
    assert moving.status is ComponentStatus.VIOLATED
    assert stopped.cost == 0.0
    assert stopped.applicable is False
    assert stopped.status is ComponentStatus.NOT_APPLICABLE
    assert stopped.diagnostics["at_fault_gated"] is True


def test_stopped_ego_makes_ttc_inapplicable() -> None:
    moving, _, _ = evaluate_ttc(
        ego_footprint=_EGO,
        ego_velocity_xy=(1.0, 0.0),
        actors=(_closing_vehicle("other", 2.0),),
        vertically_compatible_actor_ids=frozenset({"other"}),
        post_ego_velocity_xy=_MOVING,
    )
    stopped, _, _ = evaluate_ttc(
        ego_footprint=_EGO,
        ego_velocity_xy=(1.0, 0.0),
        actors=(_closing_vehicle("other", 2.0),),
        vertically_compatible_actor_ids=frozenset({"other"}),
        post_ego_velocity_xy=_STOPPED,
    )

    assert moving.applicable is True
    assert stopped.applicable is False
    assert stopped.status is ComponentStatus.NOT_APPLICABLE


def test_stopped_ego_makes_rss_lateral_inapplicable() -> None:
    stopped, _, _ = evaluate_rss_lateral(candidates=(), post_ego_velocity_xy=_STOPPED)
    assert stopped.applicable is False
    assert stopped.diagnostics["at_fault_gated"] is True


def test_position_sub_rules_are_not_gated_at_any_speed() -> None:
    """The asymmetry is deliberate, and nuPlan draws the same line.

    From a state stopped astride a lane marking an action that leaves it exists,
    so the charge is a normative disagreement rather than an unavoidable cost.
    These evaluators take no ego velocity at all, which is the strongest possible
    statement of that: the gate cannot reach them even by mistake.
    """

    offroad, _, _ = evaluate_offroad(
        ego_footprint=Polygon(((0, 0), (2, 0), (2, 1), (0, 1))),
        drivable_surface=Polygon(((0, 0), (1, 0), (1, 1), (0, 1))),
    )
    assert offroad.applicable is True
    assert offroad.cost > 0.0

    marking = Polygon(((0.4, -5.0), (0.6, -5.0), (0.6, 5.0), (0.4, 5.0)))
    solid, _, _ = evaluate_solid_line(ego_footprint=_EGO, solid_boundaries=(marking,))
    assert solid.applicable is True
    assert solid.cost > 0.0


def test_a_gated_component_contributes_nothing_to_its_channel() -> None:
    """`aggregate_max_component` filters on applicability, so the channel is 0
    because nothing is charged -- not because a charge was dropped."""

    gated, _, _ = at_fault_gated_result("clearance")
    aggregated = aggregate_max_component(name="interaction_risk", components=(gated,))
    assert aggregated.applicable is False
    assert aggregated.cost == 0.0
