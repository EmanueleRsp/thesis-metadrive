#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir

from thesis_rl.common.paths import default_output_path_str
from thesis_rl.runtime.wiring.builders import build_env


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "conf" / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate repository root containing conf/config.yaml. "
        f"Start path: {start}"
    )


def _xy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 2:
        return None
    return arr[:2]


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _get_ego_vehicle(base_env: Any) -> Any | None:
    agents = getattr(base_env, "agents", None)
    if isinstance(agents, dict) and agents:
        for vehicle in agents.values():
            if vehicle is not None:
                return vehicle
    return getattr(base_env, "vehicle", None)


def _set_position(vehicle: Any, pos_xy: np.ndarray) -> bool:
    if hasattr(vehicle, "set_position"):
        try:
            vehicle.set_position(pos_xy)
            return True
        except Exception:
            pass
    if hasattr(vehicle, "position"):
        try:
            vehicle.position = pos_xy
            return True
        except Exception:
            pass
    return False


def _set_yaw(vehicle: Any, yaw: float) -> bool:
    for method_name in ("set_heading_theta", "set_heading", "set_yaw"):
        if hasattr(vehicle, method_name):
            try:
                getattr(vehicle, method_name)(float(yaw))
                return True
            except Exception:
                continue
    for attr_name in ("heading_theta", "yaw", "heading"):
        if hasattr(vehicle, attr_name):
            try:
                setattr(vehicle, attr_name, float(yaw))
                return True
            except Exception:
                continue
    return False


def _find_nearest_neighbor_state(info: dict[str, Any]) -> dict[str, Any] | None:
    neighbors = info.get("neighbors")
    ego_state = info.get("ego_state")
    if not isinstance(neighbors, list) or not isinstance(ego_state, dict):
        return None
    ego_pos = _xy(ego_state.get("position"))
    if ego_pos is None:
        return None

    best: tuple[float, dict[str, Any]] | None = None
    for n in neighbors:
        if not isinstance(n, dict):
            continue
        n_pos = _xy(n.get("position"))
        if n_pos is None:
            continue
        dist = float(np.linalg.norm(ego_pos - n_pos))
        if best is None or dist < best[0]:
            best = (dist, n)
    return None if best is None else best[1]


def _centroid_xy(geom: Any) -> np.ndarray | None:
    if geom is None:
        return None
    if hasattr(geom, "centroid"):
        try:
            c = geom.centroid
            return np.array([float(c.x), float(c.y)], dtype=np.float32)
        except Exception:
            return None
    return None


def _extract_rule_snapshot(label: str, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "rule_components": dict(info.get("rule_components", {})),
        "rule_input_available": dict(info.get("rule_input_available", {})),
        "ego_position": info.get("ego_state", {}).get("position") if isinstance(info.get("ego_state"), dict) else None,
        "neighbors_count": len(info.get("neighbors", [])) if isinstance(info.get("neighbors"), list) else 0,
    }


def _step_zero(env: Any):
    action_space = env.action_space
    if hasattr(action_space, "shape") and action_space.shape:
        action = np.zeros(action_space.shape, dtype=np.float32)
    else:
        action = 0
    return env.step(action)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=default_output_path_str("forced_rule_scenarios.json"))
    parser.add_argument("--traffic-density", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map", dest="map_id", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=10000)
    parser.add_argument("--num-scenarios", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=300)
    args = parser.parse_args()

    repo_root = _find_repo_root(Path(__file__).resolve())
    conf_dir = repo_root / "conf"

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "reward=scalar_reward",
                "curriculum=stages",
                f"seed={int(args.seed)}",
                f"reward.rule_margin_log_path={default_output_path_str('debug_rule_margins_forced_scenarios.jsonl')}",
            ],
        )

    env = build_env(
        cfg,
        env_overrides={
            "map": int(args.map_id),
            "traffic_density": float(args.traffic_density),
            "start_seed": int(args.start_seed),
            "num_scenarios": int(args.num_scenarios),
            "horizon": int(args.horizon),
        },
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"scenarios": []}
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = _step_zero(env)
    result["scenarios"].append(_extract_rule_snapshot("baseline_after_reset", info))

    base = _base_env(env)
    ego_vehicle = _get_ego_vehicle(base)
    if ego_vehicle is None:
        raise RuntimeError("Could not locate ego vehicle in base env.")

    # Scenario 1: force ego into nearest neighbor position -> collision rule should activate.
    nearest = _find_nearest_neighbor_state(info)
    if nearest is None:
        raise RuntimeError("Could not find a valid nearest neighbor with position.")
    n_pos = _xy(nearest.get("position"))
    if n_pos is None:
        raise RuntimeError("Nearest neighbor has no usable position.")

    if not _set_position(ego_vehicle, n_pos):
        raise RuntimeError("Could not teleport ego vehicle to neighbor position.")
    n_yaw = nearest.get("yaw")
    if isinstance(n_yaw, (int, float)):
        _set_yaw(ego_vehicle, float(n_yaw))

    obs, reward, terminated, truncated, info = _step_zero(env)
    snap_collision = _extract_rule_snapshot("forced_collision", info)
    snap_collision["target_neighbor_entity_id"] = nearest.get("entity_id")
    snap_collision["target_neighbor_position"] = nearest.get("position")
    result["scenarios"].append(snap_collision)

    # Scenario 2a: place ego into opposite carriageway centroid -> wrong_way should activate.
    opposite = info.get("opposite_carriageway")
    opposite_center = _centroid_xy(opposite)
    if opposite_center is None:
        raise RuntimeError("Could not derive opposite_carriageway centroid.")
    if not _set_position(ego_vehicle, opposite_center):
        raise RuntimeError("Could not teleport ego vehicle to opposite carriageway.")

    obs, reward, terminated, truncated, info = _step_zero(env)
    snap_wrong_way = _extract_rule_snapshot("forced_wrong_way", info)
    snap_wrong_way["forced_position"] = opposite_center.tolist()
    result["scenarios"].append(snap_wrong_way)

    # Scenario 2b: place ego far from drivable area -> drivable_area should activate.
    drivable = info.get("drivable_area")
    drivable_center = _centroid_xy(drivable)
    if drivable_center is None:
        raise RuntimeError("Could not derive drivable_area centroid.")
    far_point = drivable_center + np.array([300.0, 300.0], dtype=np.float32)
    if not _set_position(ego_vehicle, far_point):
        raise RuntimeError("Could not teleport ego vehicle outside drivable area.")

    obs, reward, terminated, truncated, info = _step_zero(env)
    snap_outside = _extract_rule_snapshot("forced_out_of_drivable", info)
    snap_outside["forced_position"] = far_point.tolist()
    result["scenarios"].append(snap_outside)

    out_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    env.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
