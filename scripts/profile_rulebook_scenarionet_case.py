"""Profile one fixed ScenarioNet Rulebook rollout without changing its behavior."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf

from thesis_rl.rulebook.v2 import transition as transition_module
from thesis_rl.rulebook.v2.geometry import ctrv as ctrv_module
from thesis_rl.rulebook.v2.wrapper import RulebookV2MonitorWrapper
from thesis_rl.runtime.wiring.builders import build_train_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-index", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--trace-boundaries", action="store_true")
    return parser.parse_args()


def _build_config(*, data_root: Path, scenario_index: int):
    catalog_path = data_root / "catalog" / "scenario_catalog.parquet"
    runtime_path = data_root / "runtime" / "train"
    if not catalog_path.is_file() or not (runtime_path / "dataset_summary.pkl").is_file():
        raise FileNotFoundError("Prepared canonical ScenarioNet runtime is required.")
    return OmegaConf.create(
        {
            "seed": 0,
            "paths": {"logs_dir": str(data_root / "logs")},
            "reward": {"behavior": "monitor_only"},
            "rulebook": {"version": "4.7-final-implementation-complete"},
            "env": {
                "name": "scenarionet",
                "env_id": "ThesisScenarioEnv",
                "split": "train",
                "catalog_path": str(catalog_path.resolve()),
                "global_seed": 0,
                "config": {
                    "data_directory": str(runtime_path.resolve()),
                    "num_scenarios": 1,
                    "start_scenario_index": scenario_index,
                    "worker_index": 0,
                    "num_workers": 1,
                    "agent_policy": "env_input_policy",
                    "horizon": None,
                    "allowed_more_steps": None,
                    "truncate_as_terminate": False,
                    "reactive_traffic": True,
                    "store_data": False,
                    "store_map": False,
                },
                "episode_control": {"extra_steps_after_scenario": 0},
                "provider": {"kind": "fixed_sequence", "repeat": True},
            },
        }
    )


def _find_rulebook_wrapper(env: object) -> RulebookV2MonitorWrapper:
    current = env
    while not isinstance(current, RulebookV2MonitorWrapper):
        current = getattr(current, "env", None)
        if current is None:
            raise RuntimeError("RulebookV2MonitorWrapper is missing from the environment chain.")
    return current


def _install_boundary_trace(env: object) -> None:
    """Emit timing boundaries around the existing wrapper callbacks only."""

    wrapper = _find_rulebook_wrapper(env)
    original_env_step = wrapper.env.step
    original_snapshotter = wrapper._snapshotter
    original_evaluator = wrapper._transition_evaluator

    def timed_env_step(action):
        print("boundary=wrapped_env_step_start", flush=True)
        result = original_env_step(action)
        print("boundary=wrapped_env_step_end", flush=True)
        return result

    def timed_snapshotter(runtime_env):
        print("boundary=snapshot_start", flush=True)
        result = original_snapshotter(runtime_env)
        print("boundary=snapshot_end", flush=True)
        return result

    def timed_evaluator(*args, **kwargs):
        print("boundary=transition_evaluator_start", flush=True)
        result = original_evaluator(*args, **kwargs)
        print("boundary=transition_evaluator_end", flush=True)
        return result

    wrapper.env.step = timed_env_step
    wrapper._snapshotter = timed_snapshotter
    wrapper._transition_evaluator = timed_evaluator


def _install_transition_trace() -> None:
    """Print entry/exit for the existing transition subphases without replacing logic."""

    for name in (
        "_crosswalk_inputs",
        "drivable_surface_for_ego",
        "_vehicle_yield_inputs",
        "_rss_candidates",
        "build_vehicle_conflict_zone_candidates",
        "attach_route_intervals",
        "predict_conflict_zone_occupancy_intervals",
        "evaluate_registered_transition",
    ):
        original = getattr(transition_module, name)

        def timed(*args, _name=name, _original=original, **kwargs):
            print(f"boundary={_name}_start", flush=True)
            result = _original(*args, **kwargs)
            print(f"boundary={_name}_end", flush=True)
            return result

        setattr(transition_module, name, timed)

    original_predict_actor = ctrv_module.predict_actor_occupancy_interval

    def timed_predict_actor(*args, **kwargs):
        actor = kwargs["actor"]
        zone = kwargs["zone"]
        print(
            f"boundary=predict_actor_start actor={actor.actor_id} "
            f"zone_vertices={len(zone.exterior.coords)} zone_holes={len(zone.interiors)}",
            flush=True,
        )
        result = original_predict_actor(*args, **kwargs)
        print(f"boundary=predict_actor_end actor={actor.actor_id}", flush=True)
        return result

    ctrv_module.predict_actor_occupancy_interval = timed_predict_actor


def main() -> None:
    args = _parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    data_root = Path(os.environ.get("SCENARIONET_DATA_ROOT", "data/scenarionet"))
    config = _build_config(data_root=data_root, scenario_index=args.scenario_index)

    started = time.perf_counter()
    env = build_train_env(config)
    print(f"build_seconds={time.perf_counter() - started:.6f}", flush=True)
    try:
        started = time.perf_counter()
        observation, _ = env.reset()
        print(f"reset_seconds={time.perf_counter() - started:.6f}", flush=True)
        if args.trace_boundaries:
            _install_boundary_trace(env)
            _install_transition_trace()
        if not np.isfinite(np.asarray(observation)).all():
            raise ValueError("Reset observation contains non-finite values.")
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        for step in range(args.steps):
            started = time.perf_counter()
            observation, _reward, terminated, truncated, info = env.step(action)
            print(
                f"step={step} seconds={time.perf_counter() - started:.6f} "
                f"terminated={terminated} truncated={truncated}",
                flush=True,
            )
            if not np.isfinite(np.asarray(observation)).all():
                raise ValueError("Step observation contains non-finite values.")
            if not info.get("rulebook", {}).get("complete_evaluation"):
                raise RuntimeError("Rulebook evaluation is incomplete.")
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
