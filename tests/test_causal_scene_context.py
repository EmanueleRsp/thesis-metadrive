from __future__ import annotations

import pytest
from shapely.geometry import box

from thesis_rl.contracts.causal_scene_context import CausalSceneContext
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorSnapshot,
    EnvSnapshot,
    EpisodeCache,
    RulebookMemory,
    TaskRouteRecord,
)


def _snapshot(scenario_id: str) -> EnvSnapshot:
    ego = ActorSnapshot(
        actor_id="ego",
        actor_class=ActorClass.VEHICLE,
        position_xy=(0.0, 0.0),
        position_z=0.0,
        heading_rad=0.0,
        velocity_xy=(0.0, 0.0),
        footprint=box(-1.0, -0.5, 1.0, 0.5),
        live_lane_id="lane-0",
        configured_speed_cap_mps=20.0,
    )
    return EnvSnapshot(scenario_id, 0, 0.0, ego, (ego,), (), frozenset(), {})


def test_causal_scene_context_is_bound_to_one_committed_scenario() -> None:
    route = TaskRouteRecord("scene-a", ("lane-0",), "task", "v1", "hash")
    context = CausalSceneContext(
        EpisodeCache("scene-a", route), _snapshot("scene-a"), RulebookMemory()
    )

    assert context.task_route is route
    with pytest.raises(ValueError, match="same scenario"):
        CausalSceneContext(EpisodeCache("scene-a", route), _snapshot("scene-b"), RulebookMemory())
