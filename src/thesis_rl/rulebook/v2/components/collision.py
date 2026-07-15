"""R1 collision-impact evaluator, independent of environment side effects."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite, sqrt
from typing import Mapping

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


COLLISION_FLOOR = 1.0e-6
NORMAL_TOLERANCE = 1.0e-6


def _fail(scenario_id: str, step_index: int, cause: str) -> None:
    raise RulebookEvaluationError(
        EvaluationFailure(scenario_id, step_index, "collision", cause)
    )


def evaluate_collision_impact(
    *,
    scenario_id: str,
    step_index: int,
    ego_velocity_xy: tuple[float, float],
    ego_configured_speed_cap_mps: float | None,
    pre_actors_by_id: Mapping[str, ActorSnapshot],
    onset_records: tuple[ContactOnsetRecord, ...],
    previous_contact_ids: frozenset[str],
    post_active_contact_ids: frozenset[str],
) -> tuple[RuleComponentResult, MemoryDelta, CacheDelta]:
    """Evaluate only new contact onsets and atomically propose contact memory."""

    if not all(isfinite(value) for value in ego_velocity_xy):
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

    actor_costs: list[tuple[str, float, float]] = []
    for actor_id, records in onset_by_actor.items():
        actor = pre_actors_by_id.get(actor_id)
        if actor is None:
            _fail(scenario_id, step_index, f"dynamic actor {actor_id!r} has no pre-state")
        assert actor is not None
        if actor.actor_class == ActorClass.STATIC_COLLIDABLE:
            other_velocity = (0.0, 0.0)
            cap = ego_configured_speed_cap_mps
        elif actor.actor_class in {ActorClass.VEHICLE, ActorClass.PEDESTRIAN, ActorClass.CYCLIST}:
            other_velocity = actor.velocity_xy
            if actor.actor_class == ActorClass.VEHICLE:
                if actor.configured_speed_cap_mps is None or actor.configured_speed_cap_mps <= 0.0:
                    _fail(scenario_id, step_index, f"vehicle {actor_id!r} speed cap is invalid")
                assert actor.configured_speed_cap_mps is not None
                cap = ego_configured_speed_cap_mps + actor.configured_speed_cap_mps
            else:
                cap = ego_configured_speed_cap_mps
        else:
            _fail(scenario_id, step_index, f"actor class {actor.actor_class.value!r} is not collidable")
        assert cap is not None
        normal_closing_speeds: list[float] = []
        for record in records:
            normal_norm = sqrt(
                record.normal_ego_to_other_xy[0] ** 2 + record.normal_ego_to_other_xy[1] ** 2
            )
            if not isfinite(normal_norm) or abs(normal_norm - 1.0) > NORMAL_TOLERANCE:
                _fail(scenario_id, step_index, f"contact normal for actor {actor_id!r} is not unit")
            relative_x = ego_velocity_xy[0] - other_velocity[0]
            relative_y = ego_velocity_xy[1] - other_velocity[1]
            normal_closing_speeds.append(
                max(
                    0.0,
                    relative_x * record.normal_ego_to_other_xy[0]
                    + relative_y * record.normal_ego_to_other_xy[1],
                )
            )
        raw_speed = max(normal_closing_speeds)
        bounded_ratio = min(raw_speed, cap) / cap
        actor_costs.append((actor_id, raw_speed**2, max(COLLISION_FLOOR, bounded_ratio**2)))
    worst_actor, raw, cost = max(actor_costs, key=lambda item: (item[2], item[0]))
    result = RuleComponentResult(
        name="collision",
        cost=cost,
        raw={
            "new_collision": True,
            "worst_actor_id": worst_actor,
            "worst_raw_closing_speed_squared": raw,
            "actors": tuple(
                {"actor_id": actor_id, "raw_speed_squared": raw_speed, "cost": actor_cost}
                for actor_id, raw_speed, actor_cost in actor_costs
            ),
        },
        applicable=True,
        evaluable=True,
        status=ComponentStatus.VIOLATED,
        diagnostics={"onset_actor_ids": tuple(sorted(onset_by_actor))},
    )
    return (
        result,
        MemoryDelta("collision", (("previous_contact_ids", post_active_contact_ids),)),
        CacheDelta(),
    )
