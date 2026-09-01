from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.components.clearance import evaluate_clearance
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot

# ADR-070 gates the L2 interaction sub-rules on a stopped ego. These fixtures
# are about the rules themselves, so they drive the ego well above the gate; the
# gate has its own tests below.
_MOVING_EGO = (5.0, 0.0)


# ADR-067 scopes the rule to VRU whose centroid lies on the drivable surface.
# These fixtures predate that scoping and are about class eligibility, vertical
# compatibility and the static diagnostic, so they hold every actor on a roadway
# wide enough not to interfere with what they assert. The scoping itself is
# asserted separately below.
_ROADWAY = Polygon(((-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (-50.0, 50.0)))


def _actor(actor_id: str, actor_class: ActorClass, x: float) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id,
        actor_class,
        (x, 0.0),
        0.0,
        0.0,
        (0.0, 0.0),
        Polygon(((x, -0.5), (x + 1.0, -0.5), (x + 1.0, 0.5), (x, 0.5))),
        None,
        20.0,
    )


def test_clearance_iterates_live_candidates_and_uses_class_thresholds() -> None:
    result, memory_delta, cache_delta = evaluate_clearance(
        post_ego_velocity_xy=_MOVING_EGO,
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        actors=(
            _actor("cyclist", ActorClass.CYCLIST, 1.5),
            _actor("ped", ActorClass.PEDESTRIAN, 5.0),
        ),
        vertically_compatible_actor_ids=frozenset({"cyclist", "ped"}),
        drivable_surface=_ROADWAY,
    )
    assert result.raw["worst_actor_id"] == "cyclist"
    assert result.cost == pytest.approx(0.5)
    assert memory_delta.writes == ()
    assert cache_delta.new_conflict_zones == ()


def test_clearance_excludes_vehicle_class_per_rulebook_v4_8() -> None:
    """REQ-R2-01: vehicle clearance is replaced by the scoped lateral-RSS
    metric; a VEHICLE actor must never contribute to the clearance cost,
    even when very close to ego."""
    result, _, _ = evaluate_clearance(
        post_ego_velocity_xy=_MOVING_EGO,
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        actors=(_actor("vehicle", ActorClass.VEHICLE, 1.05),),
        vertically_compatible_actor_ids=frozenset({"vehicle"}),
        drivable_surface=_ROADWAY,
    )
    assert result.status.value == "not_applicable"
    assert result.cost == 0.0
    assert result.raw["actors"] == ()


def test_clearance_static_distance_is_diagnostic_only_per_rulebook_v4_8() -> None:
    """REQ-R2-02/DEC-R2-04: a close static obstacle is logged in
    ``diagnostics`` as ``static_polygon_distance_m`` but never contributes
    to the clearance cost or its ``raw`` candidate set."""
    result, _, _ = evaluate_clearance(
        post_ego_velocity_xy=_MOVING_EGO,
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        actors=(_actor("cone", ActorClass.STATIC_COLLIDABLE, 1.05),),
        vertically_compatible_actor_ids=frozenset({"cone"}),
        drivable_surface=_ROADWAY,
    )
    assert result.status.value == "not_applicable"
    assert result.cost == 0.0
    assert result.raw["actors"] == ()
    assert result.diagnostics["static_polygon_distance_m"] == pytest.approx(0.05)


def test_clearance_excludes_incompatible_and_non_collidable_actors() -> None:
    result, _, _ = evaluate_clearance(
        post_ego_velocity_xy=_MOVING_EGO,
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        actors=(
            _actor("vehicle", ActorClass.VEHICLE, 1.1),
            _actor("infra", ActorClass.INFRASTRUCTURE_NON_COLLIDABLE, 0.0),
        ),
        vertically_compatible_actor_ids=frozenset({"infra"}),
        drivable_surface=_ROADWAY,
    )
    assert result.status.value == "not_applicable"
    assert result.cost == 0.0


def test_clearance_ignores_a_vru_standing_off_the_roadway() -> None:
    """ADR-067. A person on the kerb is in conflict with nobody.

    The unscoped rule charged any VRU within 1 m of the footprint with no test of
    where that VRU was standing, which priced normal driving down a narrow street
    (0.327 % -> 0.199 % of expert steps). The criterion is centre-entry, the same
    one `evaluate_wrong_carriageway` uses, so the rulebook applies one geometric
    convention rather than two.
    """

    # A roadway that ends at x = 1.2, so the cyclist's centroid at x = 2.0 is off
    # it while the geometry and the 0.5 m gap are untouched.
    kerbside = Polygon(((-50.0, -50.0), (1.2, -50.0), (1.2, 50.0), (-50.0, 50.0)))
    on_road, _, _ = evaluate_clearance(
        post_ego_velocity_xy=_MOVING_EGO,
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        actors=(_actor("cyclist", ActorClass.CYCLIST, 1.5),),
        vertically_compatible_actor_ids=frozenset({"cyclist"}),
        drivable_surface=_ROADWAY,
    )
    off_road, _, _ = evaluate_clearance(
        post_ego_velocity_xy=_MOVING_EGO,
        ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
        actors=(_actor("cyclist", ActorClass.CYCLIST, 1.5),),
        vertically_compatible_actor_ids=frozenset({"cyclist"}),
        drivable_surface=kerbside,
    )

    # Same geometry, same distance: only where the VRU is standing differs.
    assert on_road.cost == pytest.approx(0.5)
    assert off_road.cost == 0.0
    assert off_road.status.value == "not_applicable"
    assert off_road.diagnostics["off_roadway_vru_count"] == 1


def test_clearance_refuses_an_unusable_drivable_surface() -> None:
    """An empty surface would empty the candidate set and report cost 0.

    Failing closed matters more here than elsewhere: a scoping change whose
    degenerate case is "no candidates" is indistinguishable from "no hazard".
    `evaluate_offroad` validates the same object the same way.
    """

    with pytest.raises(ValueError, match="valid drivable surface"):
        evaluate_clearance(
            post_ego_velocity_xy=_MOVING_EGO,
            ego_footprint=Polygon(((0.0, -0.5), (1.0, -0.5), (1.0, 0.5), (0.0, 0.5))),
            actors=(_actor("cyclist", ActorClass.CYCLIST, 1.5),),
            vertically_compatible_actor_ids=frozenset({"cyclist"}),
            drivable_surface=Polygon(),
        )
