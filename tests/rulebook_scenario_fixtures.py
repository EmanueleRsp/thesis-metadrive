"""Deterministic, repository-owned ScenarioDescription fixtures for Rulebook tests.

The generated files are intentionally small MetaDrive descriptors.  They are
not ScenarioNet records and must never be used as training data.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "rulebook-synthetic-scenarios-v1"
TIMESTEP_S = 0.1


def _array_rows(values: Iterable[tuple[float, float, float]]) -> np.ndarray:
    return np.asarray(tuple(values), dtype=np.float32)


def _array_xy(values: Iterable[tuple[float, float]]) -> np.ndarray:
    return np.asarray(tuple(values), dtype=np.float32)


def _track(
    *,
    object_id: str,
    object_type: str,
    positions: tuple[tuple[float, float, float], ...],
    heading_rad: float = 0.0,
    length_m: float = 4.5,
    width_m: float = 2.0,
    height_m: float = 1.5,
) -> dict[str, Any]:
    position = _array_rows(positions)
    length = len(position)
    velocity = np.zeros((length, 2), dtype=np.float32)
    if length > 1:
        velocity[:-1] = (position[1:, :2] - position[:-1, :2]) / TIMESTEP_S
        velocity[-1] = velocity[-2]
    return {
        "type": object_type,
        "state": {
            "position": position,
            "heading": np.full(length, heading_rad, dtype=np.float32),
            "velocity": velocity,
            "length": np.full(length, length_m, dtype=np.float32),
            "width": np.full(length, width_m, dtype=np.float32),
            "height": np.full(length, height_m, dtype=np.float32),
            "valid": np.ones(length, dtype=bool),
        },
        "metadata": {"type": object_type, "object_id": object_id},
    }


def _lane(lane_id: str, y_m: float = 0.0) -> dict[str, Any]:
    return {
        "type": "LANE_SURFACE_STREET",
        "polyline": _array_rows(((0.0, y_m, 0.0), (50.0, y_m, 0.0))),
        "polygon": _array_xy(
            ((0.0, y_m - 1.75), (50.0, y_m - 1.75), (50.0, y_m + 1.75), (0.0, y_m + 1.75))
        ),
        "width": np.full(2, 3.5, dtype=np.float32),
        "exit_lanes": (),
    }


def _base_scenario(
    *,
    scenario_id: str,
    ego_positions: tuple[tuple[float, float, float], ...],
) -> dict[str, Any]:
    length = len(ego_positions)
    return {
        "id": scenario_id,
        "version": "MetaDrive v0.3.0.1",
        "length": length,
        "metadata": {
            "metadrive_processed": True,
            "coordinate": "metadrive",
            "ts": np.arange(length, dtype=np.float32) * TIMESTEP_S,
            "sdc_id": "ego",
            "scenario_id": scenario_id,
            "dataset": "synthetic_rulebook_test",
            "assigned_route_lane_ids": ["lane-ego"],
            "assigned_route_source": "synthetic_test_route",
        },
        "tracks": {"ego": _track(object_id="ego", object_type="VEHICLE", positions=ego_positions)},
        "dynamic_map_states": {},
        "map_features": {"lane-ego": _lane("lane-ego")},
    }


def red_light_scenario() -> dict[str, Any]:
    """Ego approaches the route's red signal without relying on future state."""

    scenario = _base_scenario(
        scenario_id="rb_syn_red_light_v1",
        ego_positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
    )
    scenario["dynamic_map_states"]["lane-ego"] = {
        "type": "TRAFFIC_LIGHT",
        "lane": "lane-ego",
        "stop_point": np.asarray((10.0, 0.0, 0.0), dtype=np.float32),
        "state": {"object_state": np.asarray(["LANE_STATE_STOP"] * 4)},
        "metadata": {"type": "TRAFFIC_LIGHT", "object_id": "lane-ego"},
    }
    return scenario


def yellow_light_scenario() -> dict[str, Any]:
    """A green-to-yellow transition for frozen signal-obligation coverage."""

    scenario = red_light_scenario()
    scenario["id"] = "rb_syn_yellow_light_v1"
    scenario["metadata"]["scenario_id"] = scenario["id"]
    scenario["dynamic_map_states"]["lane-ego"]["state"] = {
        "object_state": np.asarray(
            ["LANE_STATE_GO", "LANE_STATE_CAUTION", "LANE_STATE_CAUTION", "LANE_STATE_STOP"]
        )
    }
    return scenario


def crosswalk_pedestrian_scenario() -> dict[str, Any]:
    """Ego and one live pedestrian approach a route-intersecting crosswalk."""

    scenario = _base_scenario(
        scenario_id="rb_syn_crosswalk_pedestrian_v1",
        ego_positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
    )
    scenario["tracks"]["pedestrian"] = _track(
        object_id="pedestrian",
        object_type="PEDESTRIAN",
        positions=((10.0, -3.0, 0.0), (10.0, -2.0, 0.0), (10.0, -1.0, 0.0), (10.0, 0.0, 0.0)),
        heading_rad=np.pi / 2.0,
        length_m=0.6,
        width_m=0.6,
        height_m=1.7,
    )
    scenario["map_features"]["crosswalk-main"] = {
        "type": "CROSSWALK",
        "polygon": _array_xy(((9.0, -3.0), (11.0, -3.0), (11.0, 3.0), (9.0, 3.0))),
    }
    return scenario


def vehicle_pedestrian_collision_scenario() -> dict[str, Any]:
    """A pre-state separated pedestrian crosses into the ego path."""

    scenario = _base_scenario(
        scenario_id="rb_syn_vehicle_pedestrian_collision_v1",
        ego_positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
    )
    scenario["tracks"]["pedestrian"] = _track(
        object_id="pedestrian",
        object_type="PEDESTRIAN",
        positions=((3.0, 2.0, 0.0), (2.0, 1.0, 0.0), (2.0, 0.0, 0.0), (2.0, -1.0, 0.0)),
        heading_rad=-np.pi / 2.0,
        length_m=0.6,
        width_m=0.6,
        height_m=1.7,
    )
    return scenario


def vehicle_cyclist_collision_scenario() -> dict[str, Any]:
    """A pre-state separated cyclist crosses into the ego path."""

    scenario = vehicle_pedestrian_collision_scenario()
    scenario["id"] = "rb_syn_vehicle_cyclist_collision_v1"
    scenario["metadata"]["scenario_id"] = scenario["id"]
    pedestrian = scenario["tracks"].pop("pedestrian")
    pedestrian["type"] = "CYCLIST"
    pedestrian["metadata"] = {"type": "CYCLIST", "object_id": "cyclist"}
    pedestrian["state"]["length"] = np.full(4, 1.8, dtype=np.float32)
    pedestrian["state"]["width"] = np.full(4, 0.6, dtype=np.float32)
    scenario["tracks"]["cyclist"] = pedestrian
    return scenario


def rss_front_vehicle_scenario() -> dict[str, Any]:
    """Ego and a slower front vehicle share one canonical route lane."""

    scenario = _base_scenario(
        scenario_id="rb_syn_rss_front_vehicle_v1",
        ego_positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
    )
    scenario["tracks"]["front-vehicle"] = _track(
        object_id="front-vehicle",
        object_type="VEHICLE",
        positions=((12.0, 0.0, 0.0), (12.2, 0.0, 0.0), (12.4, 0.0, 0.0), (12.6, 0.0, 0.0)),
    )
    return scenario


def stop_sign_scenario() -> dict[str, Any]:
    """A route-associated stop sign with geometry sufficient for a canonical line."""

    scenario = _base_scenario(
        scenario_id="rb_syn_stop_sign_v1",
        ego_positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
    )
    scenario["map_features"]["stop-main"] = {
        "type": "STOP_SIGN",
        "position": np.asarray((10.0, 0.0, 0.0), dtype=np.float32),
        "lane": ("lane-ego",),
    }
    return scenario


def wrong_way_scenario() -> dict[str, Any]:
    """A reversed ego route reference for the R3 wrong-way applicability path."""

    scenario = _base_scenario(
        scenario_id="rb_syn_wrong_way_v1",
        ego_positions=((3.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    scenario["tracks"]["ego"] = _track(
        object_id="ego",
        object_type="VEHICLE",
        positions=((3.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        heading_rad=np.pi,
    )
    return scenario


def offroad_scenario() -> dict[str, Any]:
    """Ego starts outside the only route-lane drivable polygon."""

    return _base_scenario(
        scenario_id="rb_syn_offroad_v1",
        ego_positions=((1.0, 5.0, 0.0), (2.0, 5.0, 0.0), (3.0, 5.0, 0.0), (4.0, 5.0, 0.0)),
    )


def solid_line_scenario() -> dict[str, Any]:
    """A solid centre marking intersecting the ego's canonical footprint."""

    scenario = _base_scenario(
        scenario_id="rb_syn_solid_line_v1",
        ego_positions=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
    )
    scenario["map_features"]["solid-centre"] = {
        "type": "ROAD_LINE_SOLID_SINGLE_WHITE",
        "polyline": _array_rows(((0.0, 0.0, 0.0), (50.0, 0.0, 0.0))),
    }
    return scenario


def dashed_line_scenario() -> dict[str, Any]:
    """A dashed marking retained across live transitions for timer coverage."""

    scenario = solid_line_scenario()
    scenario["id"] = "rb_syn_dashed_line_v1"
    scenario["metadata"]["scenario_id"] = scenario["id"]
    scenario["map_features"]["solid-centre"]["type"] = "ROAD_LINE_BROKEN_SINGLE_WHITE"
    return scenario


def build_scenarios() -> dict[str, dict[str, Any]]:
    """Return fresh fixture mappings so callers cannot share mutable arrays."""

    return {
        "red_light": red_light_scenario(),
        "yellow_light": yellow_light_scenario(),
        "crosswalk_pedestrian": crosswalk_pedestrian_scenario(),
        "vehicle_pedestrian_collision": vehicle_pedestrian_collision_scenario(),
        "vehicle_cyclist_collision": vehicle_cyclist_collision_scenario(),
        "rss_front_vehicle": rss_front_vehicle_scenario(),
        "stop_sign": stop_sign_scenario(),
        "wrong_way": wrong_way_scenario(),
        "offroad": offroad_scenario(),
        "solid_line": solid_line_scenario(),
        "dashed_line": dashed_line_scenario(),
    }


def write_persistent_fixtures(root: Path) -> Path:
    """Write the descriptor files and a deterministic manifest under ``root``."""

    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "fixtures": []}
    for fixture_id, scenario in build_scenarios().items():
        file_name = f"{fixture_id}.pkl"
        with (root / file_name).open("wb") as handle:
            pickle.dump(scenario, handle, protocol=pickle.HIGHEST_PROTOCOL)
        manifest["fixtures"].append(
            {
                "id": fixture_id,
                "file": file_name,
                "scenario_id": scenario["id"],
                "target_transition": 1,
            }
        )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path
