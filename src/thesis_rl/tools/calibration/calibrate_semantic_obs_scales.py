#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from thesis_rl.runtime.wiring.builders import build_env


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "conf" / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate repository root containing conf/config.yaml. "
        f"Start path: {start}"
    )


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _get_ego_vehicle(base_env: Any) -> Any | None:
    agents = getattr(base_env, "agents", None)
    if isinstance(agents, dict) and agents:
        for vehicle in agents.values():
            if vehicle is not None:
                return vehicle
    return getattr(base_env, "vehicle", None)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _object_type(obj: Any) -> str:
    obj_type = getattr(obj, "metadrive_type", None)
    if isinstance(obj_type, str):
        return obj_type
    return str(obj_type) if obj_type is not None else ""


def _object_position(obj: Any) -> np.ndarray | None:
    pos = getattr(obj, "position", None)
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return None
    return arr[:2]


def _sym_norm(value: float, scale: float) -> float:
    if scale <= 1e-6:
        return 0.0
    return float(np.clip(value / scale, -1.0, 1.0))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def _max_nonzero(a: float, b: float, min_value: float) -> float:
    return max(a, b, min_value)


def _sample_stats(values: list[float], percentile: float) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p_target": 0.0,
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": float(arr.size),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "p_target": float(np.percentile(arr, percentile)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate semantic observation normalization scales from MetaDrive rollouts. "
            "Outputs percentile-based recommendations and estimated clipping rates."
        )
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map", dest="map_id", type=int, default=5)
    parser.add_argument("--traffic-density", type=float, default=0.2)
    parser.add_argument("--start-seed", type=int, default=10000)
    parser.add_argument("--num-scenarios", type=int, default=5000)
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Target percentile used for recommended scales.",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=1e-3,
        help="Lower bound for each recommended scale.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/semantic_obs_scale_calibration.json",
    )
    parser.add_argument(
        "--ego-policy",
        type=str,
        default="idm",
        choices=["expert", "idm", "random"],
        help=(
            "Policy used to drive the ego during calibration. "
            "'idm' uses MetaDrive's built-in IDM policy and is the recommended default; "
            "'random' samples actions. "
            "When using 'expert', the script automatically switches env control observation to "
            "LiDAR-state (required by MetaDrive ExpertPolicy) while still calibrating semantic scales."
        ),
    )
    return parser.parse_args()


def _resolve_ego_policy_override(policy_name: str) -> str | None:
    name = str(policy_name).strip().lower()
    if name == "expert":
        return "expert_policy"
    if name == "idm":
        return "idm_policy"
    if name == "random":
        return None
    raise ValueError(f"Unsupported ego policy '{policy_name}'.")


def _zero_action(env: Any):
    action_space = env.action_space
    if hasattr(action_space, "shape") and action_space.shape:
        return np.zeros(action_space.shape, dtype=np.float32)
    return 0


def main() -> None:
    args = _parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if not (50.0 <= float(args.percentile) <= 100.0):
        raise ValueError("--percentile must be in [50, 100]")

    repo_root = _find_repo_root(Path(__file__).resolve())
    conf_dir = repo_root / "conf"

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "reward=native",
                "curriculum=disabled",
                "obs=semantic_state",
                f"seed={int(args.seed)}",
            ],
        )

    # Keep semantic normalization/radius parameters from `obs=semantic_state`, but if we
    # drive with MetaDrive ExpertPolicy we must use LiDAR-state as env control observation.
    control_observation_type = str(cfg.obs.get("type", "semantic_state"))
    if str(args.ego_policy).strip().lower() == "expert":
        cfg.obs.type = "lidar_state"
        control_observation_type = "lidar_state"

    ego_policy_override = _resolve_ego_policy_override(args.ego_policy)
    env_overrides: dict[str, Any] = {
        "map": int(args.map_id),
        "traffic_density": float(args.traffic_density),
        "start_seed": int(args.start_seed),
        "num_scenarios": int(args.num_scenarios),
        "horizon": int(max(args.max_steps, int(cfg.env.config.horizon))),
    }
    if ego_policy_override is not None:
        env_overrides["agent_policy"] = ego_policy_override
    if str(args.ego_policy).strip().lower() == "expert":
        # MetaDrive ExpertPolicy internally asserts a fixed 275-dim LiDAR-state observation.
        # This requires random_agent_model=False in the global env config.
        env_overrides["random_agent_model"] = False

    env = build_env(
        cfg,
        env_overrides=env_overrides,
    )
    base = _base_env(env)

    dynamic_radius = float(cfg.obs.dynamic_radius_m)
    static_radius = float(cfg.obs.static_radius_m)
    control_radius = float(cfg.obs.control_radius_m)

    current_scales = {
        "speed_norm_kmh": float(cfg.obs.speed_norm_kmh),
        "accel_norm_mps2": float(cfg.obs.accel_norm_mps2),
        "yaw_rate_norm_rad_s": float(cfg.obs.yaw_rate_norm_rad_s),
        "length_norm_m": float(cfg.obs.length_norm_m),
        "width_norm_m": float(cfg.obs.width_norm_m),
        "lane_width_norm_m": float(cfg.obs.lane_width_norm_m),
    }

    physics_step = float(cfg.env.config.physics_world_step_size)
    decision_repeat = float(cfg.env.config.decision_repeat)
    dt = max(physics_step * decision_repeat, 1e-6)

    values: dict[str, list[float]] = {
        "ego_speed_kmh_abs": [],
        "obj_speed_kmh_abs": [],
        "ego_accel_mps2_abs": [],
        "ego_yaw_rate_rad_s_abs": [],
        "obj_length_m_abs": [],
        "obj_width_m_abs": [],
        "lane_width_m_abs": [],
        "dynamic_rel_vx_mps_abs": [],
        "dynamic_rel_vy_mps_abs": [],
    }
    saturation_counts: dict[str, int] = {
        "ego_speed": 0,
        "obj_speed": 0,
        "ego_accel": 0,
        "ego_yaw_rate": 0,
        "obj_length": 0,
        "obj_width": 0,
        "lane_width": 0,
        "dynamic_rel_vx": 0,
        "dynamic_rel_vy": 0,
    }
    total_counts: dict[str, int] = {k: 0 for k in saturation_counts.keys()}

    dynamic_types = {"VEHICLE", "PEDESTRIAN", "CYCLIST", "OTHER", "UNSET"}
    static_types = {"TRAFFIC_CONE", "TRAFFIC_BARRIER", "TRAFFIC_OBJECT", "BUILDING", "INVISIBLE_WALL"}
    control_types = {"TRAFFIC_LIGHT", "STOP_SIGN", "CROSSWALK", "SPEED_BUMP"}

    prev_speed_mps: float | None = None
    prev_heading: float | None = None
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        episodes_task = progress.add_task("Episodes", total=int(args.episodes))
        steps_task = progress.add_task(
            f"Episode 1/{int(args.episodes)}",
            total=int(args.max_steps),
        )

        for episode_idx in range(int(args.episodes)):
            obs, info = env.reset(seed=int(args.start_seed) + episode_idx)
            del obs, info
            prev_speed_mps = None
            prev_heading = None

            progress.update(
                steps_task,
                description=f"Episode {episode_idx + 1}/{int(args.episodes)}",
                completed=0,
                total=int(args.max_steps),
            )

            for _step_idx in range(1, int(args.max_steps) + 1):
                vehicle = _get_ego_vehicle(base)
                if vehicle is None:
                    break

                # Ego features
                ego_speed_mps = _safe_float(getattr(vehicle, "speed", None), 0.0)
                ego_speed_kmh = abs(ego_speed_mps) * 3.6
                values["ego_speed_kmh_abs"].append(ego_speed_kmh)
                total_counts["ego_speed"] += 1
                if abs(_sym_norm(ego_speed_kmh, current_scales["speed_norm_kmh"])) >= 1.0:
                    saturation_counts["ego_speed"] += 1

                if prev_speed_mps is not None:
                    accel_mps2 = abs((ego_speed_mps - prev_speed_mps) / dt)
                    values["ego_accel_mps2_abs"].append(accel_mps2)
                    total_counts["ego_accel"] += 1
                    if abs(_sym_norm(accel_mps2, current_scales["accel_norm_mps2"])) >= 1.0:
                        saturation_counts["ego_accel"] += 1
                prev_speed_mps = ego_speed_mps

                heading = _safe_float(getattr(vehicle, "heading_theta", None), 0.0)
                if prev_heading is not None:
                    yaw_rate = abs(_wrap_to_pi(heading - prev_heading) / dt)
                    values["ego_yaw_rate_rad_s_abs"].append(yaw_rate)
                    total_counts["ego_yaw_rate"] += 1
                    if abs(_sym_norm(yaw_rate, current_scales["yaw_rate_norm_rad_s"])) >= 1.0:
                        saturation_counts["ego_yaw_rate"] += 1
                prev_heading = heading

                lane = getattr(vehicle, "lane", None)
                if lane is not None:
                    lane_width = _safe_float(getattr(lane, "width", None), 0.0)
                    if lane_width <= 0.0 and hasattr(lane, "width_at"):
                        try:
                            lane_width = _safe_float(lane.width_at(0.0), 0.0)
                        except Exception:
                            lane_width = 0.0
                    if lane_width > 0.0:
                        lane_width = abs(lane_width)
                        values["lane_width_m_abs"].append(lane_width)
                        total_counts["lane_width"] += 1
                        if abs(_sym_norm(lane_width, current_scales["lane_width_norm_m"])) >= 1.0:
                            saturation_counts["lane_width"] += 1

                ego_position = np.asarray(getattr(vehicle, "position", (0.0, 0.0)), dtype=np.float32)
                ego_velocity = np.asarray(getattr(vehicle, "velocity", (0.0, 0.0)), dtype=np.float32)

                get_objects = getattr(base.engine, "get_objects", None)
                objects = get_objects() if callable(get_objects) else {}
                if not isinstance(objects, dict):
                    objects = {}

                for obj_id, obj in objects.items():
                    if str(obj_id) == str(getattr(vehicle, "id", None)):
                        continue
                    obj_type = _object_type(obj)
                    if obj_type not in (dynamic_types | static_types | control_types):
                        continue

                    obj_pos = _object_position(obj)
                    if obj_pos is None:
                        continue

                    rel_local = np.asarray(
                        vehicle.convert_to_local_coordinates(obj_pos, ego_position),
                        dtype=np.float32,
                    )[:2]
                    dist = float(np.linalg.norm(rel_local))

                    length = abs(_safe_float(getattr(obj, "LENGTH", None), 0.0))
                    width = abs(_safe_float(getattr(obj, "WIDTH", None), 0.0))
                    if length > 0.0:
                        values["obj_length_m_abs"].append(length)
                        total_counts["obj_length"] += 1
                        if abs(_sym_norm(length, current_scales["length_norm_m"])) >= 1.0:
                            saturation_counts["obj_length"] += 1
                    if width > 0.0:
                        values["obj_width_m_abs"].append(width)
                        total_counts["obj_width"] += 1
                        if abs(_sym_norm(width, current_scales["width_norm_m"])) >= 1.0:
                            saturation_counts["obj_width"] += 1

                    if obj_type in dynamic_types and dist <= dynamic_radius:
                        obj_speed_kmh = abs(_safe_float(getattr(obj, "speed_km_h", None), 0.0))
                        values["obj_speed_kmh_abs"].append(obj_speed_kmh)
                        total_counts["obj_speed"] += 1
                        if abs(_sym_norm(obj_speed_kmh, current_scales["speed_norm_kmh"])) >= 1.0:
                            saturation_counts["obj_speed"] += 1

                        obj_velocity = np.asarray(getattr(obj, "velocity", (0.0, 0.0)), dtype=np.float32)
                        rel_vel_world = obj_velocity - ego_velocity
                        rel_vel_local = np.asarray(
                            vehicle.convert_to_local_coordinates(ego_position + rel_vel_world, ego_position),
                            dtype=np.float32,
                        )[:2]
                        rel_vx = abs(float(rel_vel_local[0]))
                        rel_vy = abs(float(rel_vel_local[1]))
                        values["dynamic_rel_vx_mps_abs"].append(rel_vx)
                        values["dynamic_rel_vy_mps_abs"].append(rel_vy)
                        total_counts["dynamic_rel_vx"] += 1
                        total_counts["dynamic_rel_vy"] += 1
                        speed_scale_mps = current_scales["speed_norm_kmh"] / 3.6
                        if abs(_sym_norm(rel_vx, speed_scale_mps)) >= 1.0:
                            saturation_counts["dynamic_rel_vx"] += 1
                        if abs(_sym_norm(rel_vy, speed_scale_mps)) >= 1.0:
                            saturation_counts["dynamic_rel_vy"] += 1

                if str(args.ego_policy).lower() == "random":
                    action = env.action_space.sample()
                else:
                    action = _zero_action(env)
                obs, reward, terminated, truncated, info = env.step(action)
                del obs, reward, info
                progress.advance(steps_task, 1)

                if terminated or truncated:
                    break

            progress.advance(episodes_task, 1)

    env.close()

    p = float(args.percentile)
    recommended = {
        "speed_norm_kmh": _max_nonzero(
            _percentile(values["ego_speed_kmh_abs"], p),
            _percentile(values["obj_speed_kmh_abs"], p),
            args.min_scale,
        ),
        "accel_norm_mps2": _max_nonzero(
            _percentile(values["ego_accel_mps2_abs"], p),
            0.0,
            args.min_scale,
        ),
        "yaw_rate_norm_rad_s": _max_nonzero(
            _percentile(values["ego_yaw_rate_rad_s_abs"], p),
            0.0,
            args.min_scale,
        ),
        "length_norm_m": _max_nonzero(
            _percentile(values["obj_length_m_abs"], p),
            0.0,
            args.min_scale,
        ),
        "width_norm_m": _max_nonzero(
            _percentile(values["obj_width_m_abs"], p),
            0.0,
            args.min_scale,
        ),
        "lane_width_norm_m": _max_nonzero(
            _percentile(values["lane_width_m_abs"], p),
            0.0,
            args.min_scale,
        ),
    }

    clipping_rates = {}
    for name, total in total_counts.items():
        if total <= 0:
            clipping_rates[name] = 0.0
        else:
            clipping_rates[name] = float(saturation_counts[name]) / float(total)

    stats = {k: _sample_stats(v, p) for k, v in values.items()}

    payload = {
        "config": {
            "episodes": int(args.episodes),
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
            "map": int(args.map_id),
            "traffic_density": float(args.traffic_density),
            "start_seed": int(args.start_seed),
            "num_scenarios": int(args.num_scenarios),
            "percentile": p,
            "ego_policy": str(args.ego_policy).lower(),
            "control_observation_type": str(control_observation_type),
            "random_agent_model_forced": bool(env_overrides.get("random_agent_model", cfg.env.config.random_agent_model)),
            "dt": dt,
            "dynamic_radius_m": dynamic_radius,
            "static_radius_m": static_radius,
            "control_radius_m": control_radius,
        },
        "current_scales": current_scales,
        "recommended_scales": recommended,
        "stats": stats,
        "estimated_clipping_rates_with_current_scales": clipping_rates,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    print("Recommended semantic observation scales:")
    for key in [
        "speed_norm_kmh",
        "accel_norm_mps2",
        "yaw_rate_norm_rad_s",
        "length_norm_m",
        "width_norm_m",
        "lane_width_norm_m",
    ]:
        print(f"  {key}: {recommended[key]:.6f}")


if __name__ == "__main__":
    main()
