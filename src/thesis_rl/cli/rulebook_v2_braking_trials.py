"""Collect the Rulebook v2 ego-braking trials on a deterministic straight road."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

TARGET_SPEEDS_MPS = (5.0, 10.0, 15.0, 20.0)
TARGET_TOLERANCE_MPS = 0.2
DEFAULT_TRIALS_PER_TARGET = 10
DEFAULT_ACCELERATION_STEPS = 500
DEFAULT_BRAKING_STEPS = 300


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("ego calibration config is not readable JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("ego calibration config must be a JSON object")
    return payload


def validate_calibration_config(config: dict[str, Any]) -> None:
    """Reject configurations that do not describe the normative track."""

    map_config = config.get("map_config")
    if not isinstance(map_config, dict) or map_config.get("type") != "block_sequence":
        raise ValueError("calibration config requires map_config.type=block_sequence")
    sequence = map_config.get("config")
    if not isinstance(sequence, str) or not sequence or set(sequence) != {"S"}:
        raise ValueError("calibration config requires a straight S-only block sequence")
    if float(config.get("traffic_density", 0.0)) != 0.0:
        raise ValueError("calibration config requires traffic_density=0")
    if bool(config.get("random_agent_model", False)):
        raise ValueError("calibration config requires random_agent_model=false")
    if int(config.get("num_agents", 1)) != 1:
        raise ValueError("calibration config requires num_agents=1")
    vehicle_config = config.get("vehicle_config")
    if not isinstance(vehicle_config, dict):
        raise ValueError("calibration config requires vehicle_config")


def _speed(vehicle: Any) -> float:
    value = float(vehicle.speed)
    if not isfinite(value) or value < 0.0:
        raise RuntimeError("ego speed became non-finite or negative")
    return value


def _trial(
    *,
    env: Any,
    target_speed_mps: float,
    seed: int,
    acceleration_steps: int,
    braking_steps: int,
    decision_dt_s: float,
) -> dict[str, Any]:
    env.reset(seed=seed)
    vehicle = env.agent
    reached_speed = 0.0
    reached = False
    left_lane = not bool(vehicle.on_lane)
    collided = False
    for _ in range(acceleration_steps):
        current_speed = _speed(vehicle)
        reached_speed = max(reached_speed, current_speed)
        if current_speed >= target_speed_mps:
            reached_speed = current_speed
            reached = True
            break
        # The MetaDrive decision repeat advances several physics frames at
        # once.  Taper the throttle in the final approach so the measured
        # speed remains inside the calibration tolerance instead of jumping
        # past the target by a full repeated action.
        remaining = target_speed_mps - current_speed
        throttle = 1.0
        if remaining <= 0.25:
            throttle = max(0.1, min(1.0, remaining / 0.25))
        _, _, terminated, truncated, _ = env.step(
            np.asarray([0.0, throttle], dtype=np.float32)
        )
        left_lane = left_lane or not bool(vehicle.on_lane)
        collided = collided or any(
            bool(getattr(vehicle, name, False))
            for name in ("crash_vehicle", "crash_object", "crash_human", "crash_building")
        )
        post_step_speed = _speed(vehicle)
        if abs(post_step_speed - target_speed_mps) <= TARGET_TOLERANCE_MPS:
            reached_speed = post_step_speed
            reached = True
            break
        if terminated or truncated:
            break

    brake_speeds = [reached_speed]
    for _ in range(braking_steps):
        _, _, terminated, truncated, _ = env.step(np.asarray([0.0, -1.0], dtype=np.float32))
        current_speed = _speed(vehicle)
        brake_speeds.append(current_speed)
        left_lane = left_lane or not bool(vehicle.on_lane)
        collided = collided or any(
            bool(getattr(vehicle, name, False))
            for name in ("crash_vehicle", "crash_object", "crash_human", "crash_building")
        )
        if current_speed <= 0.1 * max(reached_speed, 1.0) or terminated or truncated:
            break

    if not reached:
        reached_speed = max(brake_speeds[0], max((_speed(vehicle),), default=0.0))
    if reached_speed <= 0.0:
        raise RuntimeError(f"target {target_speed_mps} m/s was never reached at seed {seed}")
    under_90 = next(
        (index for index, speed in enumerate(brake_speeds) if speed <= 0.9 * reached_speed),
        None,
    )
    under_10 = next(
        (index for index, speed in enumerate(brake_speeds) if speed <= 0.1 * reached_speed),
        None,
    )
    if under_90 is None or under_10 is None or under_10 <= under_90:
        raise RuntimeError(f"target {target_speed_mps} m/s did not produce a complete braking trace at seed {seed}")
    elapsed = (under_10 - under_90) * decision_dt_s
    mean_deceleration = (brake_speeds[under_90] - brake_speeds[under_10]) / elapsed
    if not isfinite(mean_deceleration) or mean_deceleration <= 0.0:
        raise RuntimeError(f"target {target_speed_mps} m/s produced an invalid deceleration at seed {seed}")
    return {
        "target_speed_mps": target_speed_mps,
        "reached_speed_mps": reached_speed,
        "collided": collided,
        "left_lane": left_lane,
        "mean_deceleration_mps2": mean_deceleration,
        "seed": seed,
        "braking_samples": len(brake_speeds),
        "braking_interval_s": elapsed,
    }


def collect_trials(
    *,
    config: dict[str, Any],
    trials_per_target: int = DEFAULT_TRIALS_PER_TARGET,
    seed_start: int = 0,
    acceleration_steps: int = DEFAULT_ACCELERATION_STEPS,
    braking_steps: int = DEFAULT_BRAKING_STEPS,
) -> tuple[dict[str, Any], ...]:
    """Run all target trials and return JSON-compatible measurements."""

    if trials_per_target < 1:
        raise ValueError("trials_per_target must be positive")
    validate_calibration_config(config)
    runtime_config = dict(config)
    runtime_config.update(
        {
            "use_render": False,
            "num_scenarios": max(seed_start + len(TARGET_SPEEDS_MPS) * trials_per_target, 1),
            "start_seed": 0,
            "horizon": max(int(config.get("horizon", 2000)), 2000),
            "out_of_road_done": False,
            "out_of_route_done": False,
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "crash_human_done": False,
        }
    )
    physics_step = float(runtime_config.get("physics_world_step_size", 0.02))
    decision_repeat = int(runtime_config.get("decision_repeat", 5))
    if not isfinite(physics_step) or physics_step <= 0.0 or decision_repeat < 1:
        raise ValueError("calibration config has invalid physics step or decision_repeat")
    decision_dt_s = physics_step * decision_repeat
    try:
        from metadrive.envs import MetaDriveEnv
    except ImportError as error:  # pragma: no cover - exercised only outside the project image
        raise RuntimeError("MetaDrive is required to collect braking trials") from error

    records: list[dict[str, Any]] = []
    env = MetaDriveEnv(runtime_config)
    try:
        for target_index, target in enumerate(TARGET_SPEEDS_MPS):
            for trial_index in range(trials_per_target):
                seed = seed_start + target_index * trials_per_target + trial_index
                records.append(
                    _trial(
                        env=env,
                        target_speed_mps=target,
                        seed=seed,
                        acceleration_steps=acceleration_steps,
                        braking_steps=braking_steps,
                        decision_dt_s=decision_dt_s,
                    )
                )
    finally:
        env.close()
    return tuple(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials-per-target", type=int, default=DEFAULT_TRIALS_PER_TARGET)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--acceleration-steps", type=int, default=DEFAULT_ACCELERATION_STEPS)
    parser.add_argument("--braking-steps", type=int, default=DEFAULT_BRAKING_STEPS)
    args = parser.parse_args()
    config = _load_config(args.config.expanduser().resolve())
    records = collect_trials(
        config=config,
        trials_per_target=args.trials_per_target,
        seed_start=args.seed_start,
        acceleration_steps=args.acceleration_steps,
        braking_steps=args.braking_steps,
    )
    payload = {
        "schema": "rulebook-v2-braking-trials-v1",
        "target_speeds_mps": TARGET_SPEEDS_MPS,
        "trials_per_target": args.trials_per_target,
        "trials": list(records),
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "trial_count": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
