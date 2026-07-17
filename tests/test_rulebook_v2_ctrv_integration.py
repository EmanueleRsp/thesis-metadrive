from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from thesis_rl.rulebook.v2.context.pg_static_adapter import build_pg_static_adapter_result
from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result
from thesis_rl.rulebook.v2.geometry.ctrv import predict_conflict_zone_occupancy_intervals
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box
from thesis_rl.rulebook.v2.types import (
    ActorClass,
    ActorMotionHistory,
    ActorMotionSample,
    ActorSnapshot,
)


def _scenario(provider: str) -> dict[str, object]:
    lane = {
        "type": "LANE_SURFACE_STREET",
        "polyline": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        "exit_lanes": [],
    }
    scenario: dict[str, object] = {
        "metadata": {
            "sdc_id": "ego",
            "assigned_route_lane_ids": ["lane-a"],
            "assigned_route_source": f"{provider}_sdc_offline_task_annotation",
        },
        "tracks": {
            "ego": {
                "state": {
                    "position": [[0.0, 0.0, 0.0]],
                    "heading": [0.0],
                    "valid": [True],
                }
            }
        },
        "map_features": {"lane-a": lane},
    }
    if provider == "waymo":
        scenario["id"] = "smoke"
        scenario["length"] = 1
        lane["width"] = [3.5, 3.5]
    return scenario


def _actor(
    actor_id: str, position: tuple[float, float], velocity: tuple[float, float]
) -> ActorSnapshot:
    return ActorSnapshot(
        actor_id,
        ActorClass.VEHICLE,
        position,
        0.0,
        0.0,
        velocity,
        oriented_bounding_box(center_xy=position, heading_rad=0.0, length_m=0.5, width_m=0.3),
        "lane-a",
        20.0,
    )


@pytest.mark.parametrize("provider", ("pg", "waymo"))
def test_fixed_pg_waymo_straight_and_curved_conflict_smoke(provider: str) -> None:
    scenario = _scenario(provider)
    static_result = (
        build_pg_static_adapter_result(scenario, scenario_uid=f"{provider}-smoke")
        if provider == "pg"
        else build_waymo_static_adapter_result(scenario, scenario_uid=f"{provider}-smoke")
    )
    assert static_result.task_route.lane_ids == ("lane-a",)
    zone = Polygon(((0.6, 0.1), (1.1, 0.1), (1.1, 0.6), (0.6, 0.6)))
    ego = _actor("ego", (0.0, 0.0), (1.0, 0.0))
    other = _actor("other", (0.0, 0.0), (1.0, 0.0))
    history = ActorMotionHistory(
        "other",
        tuple(
            ActorMotionSample(index * 0.25, (0.0, 0.0), index * 0.25, (1.0, 0.0))
            for index in range(3)
        ),
    )
    ego_interval, actor_intervals, diagnostics = predict_conflict_zone_occupancy_intervals(
        ego=ego,
        actors=(other,),
        histories=(history,),
        sim_time_s=0.5,
        zone=zone,
    )
    assert ego_interval is not None
    assert actor_intervals and actor_intervals[0][0] == "other"
    assert diagnostics["other"]["motion_model"] == "CTRV"
    assert math.isfinite(actor_intervals[0][1].start_s)
