#!/usr/bin/env python3
"""Measure the lane-to-lane endpoint gaps behind `assigned_route_invalid`.

`build_assigned_route_polyline` (`src/thesis_rl/rulebook/v2/geometry/route.py:319`)
rejects an assigned route whenever two consecutive lane centerlines fail to meet
within `GEOMETRY_EPSILON_M = 1e-2` m in XY. That single gate is the largest
exclusion in the dataset funnel: 32,829 of 66,854 catalogued scenarios trip it,
and 18,290 of them trip nothing else.

The funnel report can say how many records the gate removes but not whether the
gate is *right*: a 4 cm seam between two lane polylines that a human would call
contiguous and a 40 m jump to an unrelated lane both surface as the same error
string. This script measures the actual geometry, so the dataset chapter can
state the tolerance's effect instead of asserting it.

Read-only: it opens frozen scenario files and the frozen eligibility artifact,
and writes nothing but its own JSON report.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from math import hypot
from pathlib import Path
from typing import Any

from thesis_rl.rulebook.v2.context.pg_static_adapter import build_pg_static_adapter_result
from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result
from thesis_rl.rulebook.v2.geometry.route import GEOMETRY_EPSILON_M
from thesis_rl.rulebook.v2.geometry.vertical import VERTICAL_COMPATIBILITY_TOLERANCE_M

# Candidate tolerances, in metres, for the "how many records would a wider
# epsilon readmit" table. `0.10` is the value ADR-046 measured and adopted for
# the analogous drivable-surface seam problem.
CANDIDATE_EPSILONS_M = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0)
ROUTE_ERROR_PREFIX = "assigned_route_invalid"


def _max_xy_gap(lane_ids: tuple[str, ...], lanes_by_id: dict[str, Any]) -> dict[str, Any] | None:
    """Return the per-seam XY gaps of one assigned route, or None if unbuildable."""

    try:
        lanes = [lanes_by_id[lane_id] for lane_id in lane_ids]
    except KeyError:
        return None
    if len(lanes) < 2:
        return {"seams": 0, "gaps_m": [], "max_gap_m": 0.0, "max_z_gap_m": 0.0}
    gaps: list[float] = []
    z_gaps: list[float] = []
    for previous, following in zip(lanes, lanes[1:]):
        end = previous.centerline.points_xyz[-1]
        start = following.centerline.points_xyz[0]
        gaps.append(hypot(end[0] - start[0], end[1] - start[1]))
        z_gaps.append(abs(end[2] - start[2]))
    return {
        "seams": len(gaps),
        "gaps_m": gaps,
        "max_gap_m": max(gaps),
        "max_z_gap_m": max(z_gaps),
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def q(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return {
        "min": ordered[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": ordered[-1],
    }


def _measure(
    records: list[dict[str, Any]],
    *,
    data_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    measured: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for record in records:
        path = data_root / record["relative_path"]
        try:
            with path.open("rb") as handle:
                scenario = pickle.load(handle)
        except Exception as error:  # noqa: BLE001 - diagnostic script
            skipped[f"load:{type(error).__name__}"] = skipped.get(f"load:{type(error).__name__}", 0) + 1
            continue
        builder = (
            build_pg_static_adapter_result
            if record["source"] == "pg"
            else build_waymo_static_adapter_result
        )
        try:
            result = builder(scenario, scenario_uid=record["scenario_uid"])
        except Exception as error:  # noqa: BLE001 - diagnostic script
            key = f"adapter:{type(error).__name__}"
            skipped[key] = skipped.get(key, 0) + 1
            continue
        lanes_by_id = {lane.lane_id: lane for lane in result.route_lanes}
        lane_ids = tuple(result.task_route.lane_ids)
        geometry = _max_xy_gap(lane_ids, lanes_by_id)
        if geometry is None:
            skipped["missing_route_lane"] = skipped.get("missing_route_lane", 0) + 1
            continue
        measured.append(
            {
                "scenario_uid": record["scenario_uid"],
                "source": record["source"],
                "lane_count": len(lane_ids),
                **{key: value for key, value in geometry.items() if key != "gaps_m"},
                "breaks_over_epsilon": sum(1 for gap in geometry["gaps_m"] if gap > GEOMETRY_EPSILON_M),
            }
        )
    return measured, skipped


def _summarize(measured: list[dict[str, Any]]) -> dict[str, Any]:
    max_gaps = [item["max_gap_m"] for item in measured]
    readmitted = {
        f"{epsilon:g}": sum(
            1
            for item in measured
            if item["max_gap_m"] <= epsilon and item["max_z_gap_m"] <= VERTICAL_COMPATIBILITY_TOLERANCE_M
        )
        for epsilon in CANDIDATE_EPSILONS_M
    }
    return {
        "measured_records": len(measured),
        "max_gap_quantiles_m": _quantiles(max_gaps),
        "records_buildable_at_epsilon_m": readmitted,
        "records_with_vertical_break": sum(
            1 for item in measured if item["max_z_gap_m"] > VERTICAL_COMPATIBILITY_TOLERANCE_M
        ),
        "mean_seams_per_route": (
            sum(item["seams"] for item in measured) / len(measured) if measured else 0.0
        ),
        "mean_breaks_per_route": (
            sum(item["breaks_over_epsilon"] for item in measured) / len(measured) if measured else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-source", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    eligibility = json.loads(
        (data_root / "rulebook_v2" / "catalog_eligibility.json").read_text(encoding="utf-8")
    )

    failing: dict[str, list[dict[str, Any]]] = {"waymo": [], "pg": []}
    passing: dict[str, list[dict[str, Any]]] = {"waymo": [], "pg": []}
    for record in eligibility["records"]:
        errors = record.get("validation_errors") or []
        bucket = (
            failing
            if any(error.startswith(ROUTE_ERROR_PREFIX) for error in errors)
            else (passing if record.get("rulebook_eligible") else None)
        )
        if bucket is None:
            continue
        bucket[record["source"]].append(record)

    rng = random.Random(args.seed)

    def sample(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(pool, key=lambda item: item["scenario_uid"])
        if len(ordered) <= args.per_source:
            return ordered
        return rng.sample(ordered, args.per_source)

    report: dict[str, Any] = {
        "schema": "assigned-route-contiguity-gaps-v1",
        "geometry_epsilon_m": GEOMETRY_EPSILON_M,
        "vertical_tolerance_m": VERTICAL_COMPATIBILITY_TOLERANCE_M,
        "population": {
            "failing": {source: len(pool) for source, pool in failing.items()},
            "eligible": {source: len(pool) for source, pool in passing.items()},
        },
        "sampling": {"per_source": args.per_source, "seed": args.seed},
        "groups": {},
    }
    for label, pools in (("route_rejected", failing), ("rulebook_eligible", passing)):
        for source, pool in pools.items():
            chosen = sample(pool)
            measured, skipped = _measure(chosen, data_root=data_root)
            report["groups"][f"{label}:{source}"] = {
                "sampled": len(chosen),
                "skipped": skipped,
                **_summarize(measured),
            }
            print(f"{label}:{source} measured={len(measured)} skipped={skipped}", flush=True)

    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
