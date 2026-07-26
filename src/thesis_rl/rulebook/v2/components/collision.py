"""R1 collision-impact evaluator, independent of environment side effects."""

from __future__ import annotations

from collections import defaultdict
from math import hypot, isfinite
from typing import Mapping, NoReturn

from thesis_rl.rulebook.v2.components.injury_risk import (
    INJURY_RISK_SOURCE,
    INJURY_SEVERITY,
    REFERENCE_AGE_YEARS,
    injury_risk_cost,
    model_for,
)
from thesis_rl.rulebook.v2.errors import EvaluationFailure, RulebookEvaluationError
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    CacheDelta,
    ComponentStatus,
    ContactOnsetRecord,
    MemoryDelta,
    RuleComponentResult,
)


# Rulebook v4.9 §4.5: retained from v4.7 §5.5 as a defensive numerical net.  It
# is structurally inert now, because the injury-risk curve is strictly positive
# at every closing speed (min P(0) ~ 1.85e-3 across the mapped classes).
COLLISION_FLOOR = 1.0e-6
CENTER_COINCIDENCE_TOLERANCE_M = 1.0e-6


def _fail(scenario_id: str, step_index: int, cause: str) -> NoReturn:
    raise RulebookEvaluationError(
        EvaluationFailure(scenario_id, step_index, "collision", cause)
    )


def _canonical_footprint_center(
    actor: ActorSnapshot, *, scenario_id: str, step_index: int
) -> tuple[float, float]:
    centroid = actor.footprint.centroid
    if centroid.is_empty:
        _fail(scenario_id, step_index, f"canonical footprint for actor {actor.actor_id!r} is empty")
    center = (float(centroid.x), float(centroid.y))
    if not all(isfinite(value) for value in center):
        _fail(
            scenario_id,
            step_index,
            f"canonical footprint center for actor {actor.actor_id!r} is not finite",
        )
    return center


def evaluate_collision_impact(
    *,
    scenario_id: str,
    step_index: int,
    ego_configured_speed_cap_mps: float | None,
    pre_ego: ActorSnapshot,
    pre_actors_by_id: Mapping[str, ActorSnapshot],
    onset_records: tuple[ContactOnsetRecord, ...],
    previous_contact_ids: frozenset[str],
    post_active_contact_ids: frozenset[str],
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate only new contact onsets and atomically propose contact memory.

    The cost of each new contact is the MAIS3+F injury-risk probability of
    Rulebook v4.9 §4.1, a function of the pre-state normal closing speed and
    the actor class alone.  ``ego_configured_speed_cap_mps`` and the per-vehicle
    caps no longer normalize the cost -- that was the v4.7 §5.4 formulation
    whose scenario dependence ADR-027 removes -- but they remain validated here
    because they are scenario-eligibility preconditions under v4.7 §5.4, which
    v4.9 deliberately leaves in scope (DEC-R1-06).
    """

    if not all(isfinite(value) for value in pre_ego.velocity_xy):
        _fail(scenario_id, step_index, "ego pre-state velocity is not finite")
    if (
        ego_configured_speed_cap_mps is None
        or not isfinite(ego_configured_speed_cap_mps)
        or ego_configured_speed_cap_mps <= 0.0
    ):
        _fail(scenario_id, step_index, "ego configured speed normalization cap is invalid")
    assert ego_configured_speed_cap_mps is not None
    onset_by_actor: dict[str, list[ContactOnsetRecord]] = defaultdict(list)
    for record in onset_records:
        if record.actor_id not in previous_contact_ids:
            onset_by_actor[record.actor_id].append(record)
    if not onset_by_actor:
        result = RuleComponentResult(
            name="collision",
            cost=0.0,
            raw={"new_collision": False, "actors": ()},
            applicable=False,
            evaluable=True,
            status=ComponentStatus.NOT_APPLICABLE,
            diagnostics={"new_collision": False},
        )
        return (
            result,
            MemoryDelta("collision", (("previous_contact_ids", post_active_contact_ids),)),
            CacheDelta(),
        )

    actor_costs: list[tuple[str, ActorSnapshot, float, float, tuple[float, float]]] = []
    missing_pre_state_ids: list[str] = []
    for actor_id, records in onset_by_actor.items():
        actor = pre_actors_by_id.get(actor_id)
        if actor is None:
            # Actors may appear between two simulator snapshots.  They cannot
            # contribute a pre-state closing-speed estimate on this transition;
            # the transition memory will include them from the current snapshot.
            missing_pre_state_ids.append(actor_id)
            continue
        assert actor is not None
        if actor.actor_class == ActorClass.STATIC_COLLIDABLE:
            other_velocity = (0.0, 0.0)
        elif actor.actor_class in {ActorClass.VEHICLE, ActorClass.PEDESTRIAN, ActorClass.CYCLIST}:
            other_velocity = actor.velocity_xy
            if actor.actor_class == ActorClass.VEHICLE:
                # v4.9 DEC-R1-06: still a scenario-eligibility precondition,
                # even though it no longer takes part in the cost.
                if actor.configured_speed_cap_mps is None or actor.configured_speed_cap_mps <= 0.0:
                    _fail(scenario_id, step_index, f"vehicle {actor_id!r} speed cap is invalid")
        else:
            _fail(scenario_id, step_index, f"actor class {actor.actor_class.value!r} is not collidable")
        del records
        ego_center = _canonical_footprint_center(
            pre_ego, scenario_id=scenario_id, step_index=step_index
        )
        actor_center = _canonical_footprint_center(
            actor, scenario_id=scenario_id, step_index=step_index
        )
        separation_x = actor_center[0] - ego_center[0]
        separation_y = actor_center[1] - ego_center[1]
        separation_norm = hypot(separation_x, separation_y)
        if not isfinite(separation_norm) or separation_norm <= CENTER_COINCIDENCE_TOLERANCE_M:
            _fail(
                scenario_id,
                step_index,
                f"pre-state canonical footprint centers for actor {actor_id!r} coincide",
            )
        normal_x = separation_x / separation_norm
        normal_y = separation_y / separation_norm
        relative_x = pre_ego.velocity_xy[0] - other_velocity[0]
        relative_y = pre_ego.velocity_xy[1] - other_velocity[1]
        raw_speed = max(0.0, relative_x * normal_x + relative_y * normal_y)
        try:
            risk = injury_risk_cost(
                actor_class=actor.actor_class,
                normal_closing_speed_mps=raw_speed,
            )
        except ValueError as error:
            _fail(scenario_id, step_index, str(error))
        actor_costs.append(
            (actor_id, actor, raw_speed, max(COLLISION_FLOOR, risk), (normal_x, normal_y))
        )
    if not actor_costs:
        return (
            RuleComponentResult(
                name="collision",
                cost=0.0,
                raw={"new_collision": False, "actors": ()},
                applicable=False,
                evaluable=True,
                status=ComponentStatus.NOT_APPLICABLE,
                diagnostics={
                    "new_collision": False,
                    "ignored_missing_pre_state_actor_ids": tuple(sorted(missing_pre_state_ids)),
                },
            ),
            MemoryDelta("collision", (("previous_contact_ids", post_active_contact_ids),)),
            CacheDelta(),
        )
    worst_actor, _, worst_speed, cost, _ = max(actor_costs, key=lambda item: (item[3], item[0]))
    result = RuleComponentResult(
        name="collision",
        cost=cost,
        raw={
            "new_collision": True,
            "worst_actor_id": worst_actor,
            "worst_closing_speed_mps": worst_speed,
            "actors": tuple(
                {
                    "actor_id": actor_id,
                    "actor_class": actor.actor_class.value,
                    "closing_speed_mps": raw_speed,
                    "injury_risk_curve": model_for(actor.actor_class).curve_id,
                    "cost": actor_cost,
                    "normal_source": "pre_state_canonical_footprint_centers",
                    "normal_ego_to_other_xy": normal,
                }
                for actor_id, actor, raw_speed, actor_cost, normal in actor_costs
            ),
        },
        applicable=True,
        evaluable=True,
        status=ComponentStatus.VIOLATED,
        diagnostics={
            "onset_actor_ids": tuple(sorted(onset_by_actor)),
            "ignored_missing_pre_state_actor_ids": tuple(sorted(missing_pre_state_ids)),
            "normal_source": "pre_state_canonical_footprint_centers",
            "injury_risk_model": {
                "severity": INJURY_SEVERITY,
                "age_years": REFERENCE_AGE_YEARS,
                "source": INJURY_RISK_SOURCE,
            },
        },
    )
    return (
        result,
        MemoryDelta("collision", (("previous_contact_ids", post_active_contact_ids),)),
        CacheDelta(),
    )
