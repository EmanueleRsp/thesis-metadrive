#!/usr/bin/env python3
"""Classify why each assigned-route seam fails the contiguity gate.

`measure_route_contiguity_gaps.py` establishes that the rejected seams are tens
of metres wide, so `GEOMETRY_EPSILON_M` is not a tunable that decides the
dataset. It does not establish *what* those seams are, and the two candidate
answers have opposite consequences for the thesis:

- the source map has no lane connecting the two matched lanes, so the scenario
  genuinely cannot carry a lane-topology route -- the exclusion is a property of
  the data;
- the map does connect them, and the offline map-matcher simply did not look --
  the exclusion is a property of our tooling, and the 32,829 excluded records
  are recoverable.

`map_match_sdc_track_to_task_route` (`context/map_matching.py:104-118`)
associates every track sample independently and appends the lane id whenever it
changes, with no continuity constraint. Meanwhile `RouteLaneRecord` already
carries `successor_lane_ids` (WOMD `exit_lanes`) and `lateral_lane_ids`
(`left_neighbor`/`right_neighbor`), which `build_assigned_route_polyline` never
consults. This script consults them, per broken seam:

- `lateral_neighbour`  B is a declared side neighbour of A: the ego changed
  lane. A concatenation of whole lane centerlines cannot represent this, at any
  tolerance -- a representational limit, not a data defect.
- `successor_1`        B is a declared successor of A but their endpoints are
  apart: a hole in the source geometry.
- `successor_k`        B is reachable from A through k>=2 successors: the
  matcher skipped the intermediate lanes; a graph search would repair it.
- `predecessor`        A is reachable from B: the matched order is inverted.
- `unrelated`          no directed path either way within the hop budget.

Read-only.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import deque
from math import hypot
from pathlib import Path
from typing import Any

from thesis_rl.rulebook.v2.context.pg_static_adapter import build_pg_static_adapter_result
from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result
from thesis_rl.rulebook.v2.geometry.route import GEOMETRY_EPSILON_M

MAX_HOPS = 8
ROUTE_ERROR_PREFIX = "assigned_route_invalid"


def _hops_to(
    start: str, target: str, successors: dict[str, tuple[str, ...]], *, max_hops: int
) -> int | None:
    """Shortest directed successor-hop count from `start` to `target`, or None."""

    if start == target:
        return 0
    seen = {start}
    frontier = deque([(start, 0)])
    while frontier:
        lane_id, depth = frontier.popleft()
        if depth >= max_hops:
            continue
        for following in successors.get(lane_id, ()):
            if following == target:
                return depth + 1
            if following not in seen:
                seen.add(following)
                frontier.append((following, depth + 1))
    return None


def _classify_seam(
    previous: Any,
    following: Any,
    successors: dict[str, tuple[str, ...]],
    laterals: dict[str, frozenset[str]],
) -> tuple[str, int | None]:
    if following.lane_id in laterals.get(previous.lane_id, ()) or previous.lane_id in laterals.get(
        following.lane_id, ()
    ):
        return "lateral_neighbour", None
    forward = _hops_to(previous.lane_id, following.lane_id, successors, max_hops=MAX_HOPS)
    if forward == 1:
        return "successor_1", 1
    if forward is not None:
        return "successor_k", forward
    backward = _hops_to(following.lane_id, previous.lane_id, successors, max_hops=MAX_HOPS)
    if backward is not None:
        return "predecessor", backward
    return "unrelated", None


def _analyze(record: dict[str, Any], *, data_root: Path) -> dict[str, Any] | None:
    path = data_root / record["relative_path"]
    try:
        with path.open("rb") as handle:
            scenario = pickle.load(handle)
    except Exception:  # noqa: BLE001 - diagnostic script
        return None
    builder = (
        build_pg_static_adapter_result
        if record["source"] == "pg"
        else build_waymo_static_adapter_result
    )
    try:
        result = builder(scenario, scenario_uid=record["scenario_uid"])
    except Exception:  # noqa: BLE001 - diagnostic script
        return None

    lanes_by_id = {lane.lane_id: lane for lane in result.route_lanes}
    successors = {
        lane.lane_id: tuple(lane.successor_lane_ids) for lane in result.route_lanes
    }
    laterals = {
        lane.lane_id: frozenset(lane.lateral_lane_ids) for lane in result.route_lanes
    }
    lane_ids = tuple(result.task_route.lane_ids)
    try:
        lanes = [lanes_by_id[lane_id] for lane_id in lane_ids]
    except KeyError:
        return None

    breaks: list[dict[str, Any]] = []
    for previous, following in zip(lanes, lanes[1:]):
        end = previous.centerline.points_xyz[-1]
        start = following.centerline.points_xyz[0]
        gap = hypot(end[0] - start[0], end[1] - start[1])
        if gap <= GEOMETRY_EPSILON_M:
            continue
        kind, hops = _classify_seam(previous, following, successors, laterals)
        breaks.append({"gap_m": gap, "kind": kind, "hops": hops})
    if not breaks:
        return None
    kinds = {item["kind"] for item in breaks}
    return {
        "source": record["source"],
        "lane_count": len(lane_ids),
        "break_count": len(breaks),
        "kinds": sorted(kinds),
        "breaks": breaks,
        # A route is repairable by graph search only if every one of its breaks
        # is a skipped successor chain: a lane change cannot be repaired by
        # inserting lanes, and an unrelated pair has nothing to insert.
        "repairable_by_successor_search": kinds <= {"successor_k"},
        "has_lateral": "lateral_neighbour" in kinds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-source", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    eligibility = json.loads(
        (data_root / "rulebook_v2" / "catalog_eligibility.json").read_text(encoding="utf-8")
    )
    failing: dict[str, list[dict[str, Any]]] = {"waymo": [], "pg": []}
    for record in eligibility["records"]:
        errors = record.get("validation_errors") or []
        if any(error.startswith(ROUTE_ERROR_PREFIX) for error in errors):
            failing[record["source"]].append(record)

    rng = random.Random(args.seed)
    report: dict[str, Any] = {
        "schema": "assigned-route-break-classification-v1",
        "geometry_epsilon_m": GEOMETRY_EPSILON_M,
        "max_hops": MAX_HOPS,
        "population": {source: len(pool) for source, pool in failing.items()},
        "sampling": {"per_source": args.per_source, "seed": args.seed},
        "by_source": {},
    }
    for source, pool in failing.items():
        ordered = sorted(pool, key=lambda item: item["scenario_uid"])
        chosen = ordered if len(ordered) <= args.per_source else rng.sample(ordered, args.per_source)
        analyzed = [item for item in (_analyze(r, data_root=data_root) for r in chosen) if item]
        seam_kinds: dict[str, int] = {}
        seam_gap_by_kind: dict[str, list[float]] = {}
        record_kinds: dict[str, int] = {}
        hop_histogram: dict[str, int] = {}
        for item in analyzed:
            for entry in item["breaks"]:
                seam_kinds[entry["kind"]] = seam_kinds.get(entry["kind"], 0) + 1
                seam_gap_by_kind.setdefault(entry["kind"], []).append(entry["gap_m"])
                if entry["hops"] is not None:
                    key = str(entry["hops"])
                    hop_histogram[key] = hop_histogram.get(key, 0) + 1
            for kind in item["kinds"]:
                record_kinds[kind] = record_kinds.get(kind, 0) + 1
        report["by_source"][source] = {
            "analyzed_records": len(analyzed),
            "total_breaks": sum(item["break_count"] for item in analyzed),
            "seams_by_kind": seam_kinds,
            "median_gap_by_kind_m": {
                kind: sorted(gaps)[len(gaps) // 2] for kind, gaps in seam_gap_by_kind.items()
            },
            "records_touching_kind": record_kinds,
            "records_repairable_by_successor_search": sum(
                1 for item in analyzed if item["repairable_by_successor_search"]
            ),
            "records_with_lane_change": sum(1 for item in analyzed if item["has_lateral"]),
            "successor_hop_histogram": hop_histogram,
        }
        print(f"{source}: {report['by_source'][source]}", flush=True)

    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
