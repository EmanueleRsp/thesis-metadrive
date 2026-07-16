"""Run a deterministic offline Rulebook v2 adapter/task-route pilot.

The report is deliberately limited to static adapter conversion and the
``TaskRouteEligibility`` contract.  It is not a substitute for reset-smoke
validation or the final scenario eligibility artifact.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from thesis_rl.rulebook.v2.config import RULEBOOK_V2_VERSION
from thesis_rl.rulebook.v2.context.map_matching import TaskRouteMapMatchError
from thesis_rl.rulebook.v2.context.pg_static_adapter import build_pg_static_adapter_result
from thesis_rl.rulebook.v2.context.task_route import validate_task_route
from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result

_PG_PROFILES = (
    "P0_simple",
    "P1_vehicle_interaction",
    "P2_merge_or_roundabout",
    "P3_intersection",
    "P5_complex_mixed",
)


def _scenario_files(data_root: Path, *, pg_per_profile: int, waymo_count: int) -> tuple[tuple[str, str | None, Path], ...]:
    selected: list[tuple[str, str | None, Path]] = []
    for profile in _PG_PROFILES:
        paths = sorted((data_root / "pg" / "database" / profile).glob("*/*sd_pg*.pkl"))
        selected.extend(("pg", profile, path) for path in paths[:pg_per_profile])
    waymo_paths = sorted((data_root / "waymo" / "database").glob("*/sd_waymo*.pkl"))
    selected.extend(("waymo", None, path) for path in waymo_paths[:waymo_count])
    return tuple(selected)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(probability * len(ordered)))
    return float(ordered[index])


def _run_one(
    *,
    source: str,
    profile: str | None,
    path: Path,
    adapter: Callable[..., Any],
    geometry_config_hash: str | None,
    calibration_hash: str | None,
    data_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    relative_path = str(path.relative_to(data_root))
    scenario_uid = f"{source}:{path.stem}"
    try:
        with path.open("rb") as handle:
            scenario = pickle.load(handle)
        scenario_uid = f"{source}:{scenario.get('id', path.stem)}"
        result = adapter(scenario, scenario_uid=scenario_uid)
        errors = list(result.validation_errors)
        status = "adapter_excluded" if errors else "converted"
        if not errors and geometry_config_hash is not None and calibration_hash is not None:
            eligibility = validate_task_route(
                result.task_route,
                available_lane_ids={lane.lane_id: lane for lane in result.route_lanes},
                rulebook_version=RULEBOOK_V2_VERSION,
                geometry_config_hash=geometry_config_hash,
                calibration_hash=calibration_hash,
            )
            errors = list(eligibility.validation_errors)
            status = "task_route_eligible" if eligibility.rulebook_eligible else "task_route_excluded"
        elif not errors:
            status = "task_route_deferred_missing_hash"
    except TaskRouteMapMatchError as error:
        status = "adapter_excluded"
        errors = [error.validation_error]
    except Exception as error:
        status = "adapter_exception"
        errors = [f"{type(error).__name__}:{error}"]
    return {
        "source": source,
        "profile": profile,
        "scenario_uid": scenario_uid,
        "path": relative_path,
        "status": status,
        "errors": errors,
        "elapsed_s": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pg-per-profile", type=int, default=2)
    parser.add_argument("--waymo-count", type=int, default=10)
    parser.add_argument("--geometry-config-hash")
    parser.add_argument("--calibration-hash")
    args = parser.parse_args()
    if args.pg_per_profile < 1 or args.waymo_count < 1:
        parser.error("sample sizes must be positive")
    if (args.geometry_config_hash is None) != (args.calibration_hash is None):
        parser.error("geometry and calibration hashes must be supplied together")

    data_root = args.data_root.expanduser().resolve()
    selected = _scenario_files(data_root, pg_per_profile=args.pg_per_profile, waymo_count=args.waymo_count)
    records = tuple(
        _run_one(
            source=source,
            profile=profile,
            path=path,
            adapter=build_pg_static_adapter_result if source == "pg" else build_waymo_static_adapter_result,
            geometry_config_hash=args.geometry_config_hash,
            calibration_hash=args.calibration_hash,
            data_root=data_root,
        )
        for source, profile, path in selected
    )
    counts = Counter(record["status"] for record in records)
    causes = Counter(error for record in records for error in record["errors"])
    elapsed = [float(record["elapsed_s"]) for record in records]
    payload = {
        "schema": "rulebook-v2-offline-pilot-v1",
        "rulebook_version": RULEBOOK_V2_VERSION,
        "sample": {"pg_per_profile": args.pg_per_profile, "waymo_count": args.waymo_count},
        "hashes_supplied": args.geometry_config_hash is not None,
        "requested": len(selected),
        "counts_by_status": dict(sorted(counts.items())),
        "causes": dict(sorted(causes.items())),
        "elapsed_s": {
            "mean": sum(elapsed) / len(elapsed) if elapsed else 0.0,
            "p95_lower": _percentile(elapsed, 0.95),
        },
        "records": records,
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "counts_by_status": payload["counts_by_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
