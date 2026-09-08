"""REQ-AF-01 integration check on a real ``ScenarioOnlineEnv``.

A synthetic ScenarioNet descriptor spawns one replayed vehicle 15 m ahead of
the ego.  The LiDAR admission set and the live snapshot set must share at least
that actor; before OBS-AUDIT-FIX-001 the sweep carried MetaDrive's random
object name and the snapshot the scenario id ``"77"``, so the intersection was
always empty and the semantic observation never contained another road user.
"""

from __future__ import annotations

import numpy as np
from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.scenario.scenario_description import ScenarioDescription

from thesis_rl.envs.observations.perception import first_hit_lidar_sweep
from thesis_rl.rulebook.v2.context.metadrive_live import live_actor_snapshots

_TS = 0.1


def _rows(values) -> np.ndarray:
    return np.asarray(tuple(values), dtype=np.float32)


def _lane(x0: float, x1: float) -> dict:
    return {
        "type": "LANE_SURFACE_STREET",
        "polyline": _rows(((x0, 0.0, 0.0), (x1, 0.0, 0.0))),
        "polygon": np.asarray(((x0, -1.75), (x1, -1.75), (x1, 1.75), (x0, 1.75)), dtype=np.float32),
        "width": np.full(2, 3.5, dtype=np.float32),
        "exit_lanes": (),
    }


def _track(n: int, x0: float, speed: float, object_id: str) -> dict:
    positions = _rows(tuple((x0 + float(i) * speed * _TS, 0.0, 0.0) for i in range(n)))
    velocity = np.zeros((n, 2), dtype=np.float32)
    velocity[:, 0] = speed
    return {
        "type": "VEHICLE",
        "state": {
            "position": positions,
            "heading": np.zeros(n, dtype=np.float32),
            "velocity": velocity,
            "length": np.full(n, 4.5, dtype=np.float32),
            "width": np.full(n, 2.0, dtype=np.float32),
            "height": np.full(n, 1.5, dtype=np.float32),
            "valid": np.ones(n, dtype=bool),
        },
        "metadata": {"type": "VEHICLE", "object_id": object_id, "track_length": n},
    }


def _scenario(n: int = 60) -> dict:
    return {
        "id": "perception_ids",
        "version": "MetaDrive v0.3.0.1",
        "length": n,
        "metadata": {
            "metadrive_processed": True,
            "coordinate": "metadrive",
            "ts": np.arange(n, dtype=np.float32) * _TS,
            "sdc_id": "ego",
            "scenario_id": "perception_ids",
            "dataset": "thesis_test",
        },
        "tracks": {"ego": _track(n, 0.0, 5.0, "ego"), "77": _track(n, 15.0, 0.0, "77")},
        "dynamic_map_states": {},
        "map_features": {"lane-a": _lane(-10.0, 200.0)},
    }


def test_scenario_online_env_sweep_ids_match_live_snapshot_ids() -> None:
    env = ScenarioOnlineEnv(
        {
            "agent_policy": EnvInputPolicy,
            "use_render": False,
            "log_level": 50,
            "store_data": False,
            "store_map": False,
            "reactive_traffic": False,
            "no_light": True,
            "filter_overlapping_car": False,
            "horizon": 1000,
        }
    )
    try:
        env.set_scenario(ScenarioDescription(_scenario()))
        env.reset(seed=0)
        for _ in range(3):
            env.step(np.asarray([0.0, 0.3], dtype=np.float32))

        sweep = first_hit_lidar_sweep(env.agent)
        snapshot_ids = {snapshot.actor_id for snapshot in live_actor_snapshots(env)}

        assert sweep.hit_count > 0
        assert "77" in snapshot_ids
        assert "77" in sweep.actor_ids
        assert env.agent.id not in sweep.actor_ids
        assert "ego" not in sweep.actor_ids
    finally:
        env.close()
