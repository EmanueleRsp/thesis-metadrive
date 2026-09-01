from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.components.ttc import evaluate_ttc
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot

# ADR-070 gates the L2 interaction sub-rules on a stopped ego. These fixtures
# are about the rules themselves, so they drive the ego well above the gate; the
# gate has its own tests below.
_MOVING_EGO = (5.0, 0.0)


def _actor(actor_id: str, x: float) -> ActorSnapshot:
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


def test_ttc_uses_relative_motion_and_thresholded_cost() -> None:
    result, _, _ = evaluate_ttc(
        post_ego_velocity_xy=_MOVING_EGO,
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        ego_velocity_xy=(1.0, 0.0),
        actors=(_actor("other", 4.0),),
        vertically_compatible_actor_ids=frozenset({"other"}),
    )
    assert result.raw["worst_ttc_s"] == pytest.approx(1.5)
    assert result.cost == pytest.approx(0.0)


def test_ttc_reports_imminent_overlap_and_excludes_incompatible_actor() -> None:
    result, _, _ = evaluate_ttc(
        post_ego_velocity_xy=_MOVING_EGO,
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        ego_velocity_xy=(1.0, 0.0),
        actors=(_actor("other", 1.5),),
        vertically_compatible_actor_ids=frozenset({"other"}),
    )
    assert result.cost > 0.0
    excluded, _, _ = evaluate_ttc(
        post_ego_velocity_xy=_MOVING_EGO,
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        ego_velocity_xy=(1.0, 0.0),
        actors=(_actor("other", 1.5),),
        vertically_compatible_actor_ids=frozenset(),
    )
    assert excluded.status.value == "not_applicable"
