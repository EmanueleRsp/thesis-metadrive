#!/usr/bin/env python3
"""Measure how long the offline map-matcher stays in each lane it reports.

`map_match_sdc_track_to_task_route` associates every SDC track sample to a lane
independently (`geometry/lanes.py:associate_route_lane`, ranked by heading
misalignment then lateral distance) and appends the lane id whenever it changes.
Nothing constrains consecutive samples to agree, so an ego travelling near a
lane boundary can be reported as alternating between two parallel lanes. The
stored `assigned_route_lane_ids` cannot show this because consecutive duplicates
are collapsed before it is written.

This script recovers the per-visit dwell. A lane visit lasting a handful of
10 Hz samples is not a manoeuvre -- at 10 m/s, 0.3 s covers 3 m, less than a
lane width -- so it is an association artifact. The fraction of route seams
produced by such visits bounds how much of the `assigned_route_invalid`
exclusion is our matcher rather than the data.

Read-only.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_rl.rulebook.v2.context.waymo_static_adapter import (
    build_waymo_static_adapter_result,
)
from thesis_rl.rulebook.v2.geometry.lanes import associate_route_lane

# A visit this short cannot be a lane change: at 10 m/s it spans 3 m.
SPURIOUS_DWELL_SAMPLES = 3


def _track_samples(track: Any, z_origin: float) -> list[tuple[tuple[float, float], float, float]]:
    state = track["state"]
    positions = state["position"]
    headings = state["heading"]
    valid = state.get("valid")
    out = []
    for index in range(len(positions)):
        if valid is not None and not valid[index]:
            continue
        point = positions[index]
        out.append(
            ((float(point[0]), float(point[1])), float(point[2]) - z_origin, float(headings[index]))
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    eligibility = json.loads(
        (data_root / "rulebook_v2" / "catalog_eligibility.json").read_text(encoding="utf-8")
    )
    pool = [
        record
        for record in eligibility["records"]
        if record["source"] == "waymo"
        and any(
            error.startswith("assigned_route_invalid")
            for error in (record.get("validation_errors") or [])
        )
    ]
    pool.sort(key=lambda item: item["scenario_uid"])
    chosen = random.Random(args.seed).sample(pool, min(args.sample, len(pool)))

    dwell_histogram: Counter[int] = Counter()
    records_with_spurious = 0
    analyzed = 0
    visits_total = 0
    visits_spurious = 0
    for position, record in enumerate(chosen):
        try:
            with (data_root / record["relative_path"]).open("rb") as handle:
                scenario = pickle.load(handle)
            result = build_waymo_static_adapter_result(
                scenario, scenario_uid=record["scenario_uid"]
            )
        except Exception:  # noqa: BLE001 - diagnostic script
            continue
        metadata = scenario["metadata"]
        sdc = scenario["tracks"][str(metadata["sdc_id"])]
        z_origin = 0.0
        try:
            z_origin = float(sdc["state"]["position"][0][2])
        except Exception:  # noqa: BLE001
            pass
        lanes = tuple(result.route_lanes)
        visits: list[tuple[str, int]] = []
        for position_xy, position_z, heading in _track_samples(sdc, z_origin):
            association = associate_route_lane(
                position_xy=position_xy,
                position_z=position_z,
                heading_rad=heading,
                route_lanes=lanes,
            )
            if association is None:
                continue
            if visits and visits[-1][0] == association.lane_id:
                visits[-1] = (association.lane_id, visits[-1][1] + 1)
            else:
                visits.append((association.lane_id, 1))
        if not visits:
            continue
        analyzed += 1
        spurious = 0
        for _lane_id, dwell in visits:
            dwell_histogram[min(dwell, 50)] += 1
            visits_total += 1
            if dwell <= SPURIOUS_DWELL_SAMPLES:
                visits_spurious += 1
                spurious += 1
        if spurious:
            records_with_spurious += 1
        if (position + 1) % 50 == 0:
            print(f"analyzed {position + 1}/{len(chosen)}", flush=True)

    report = {
        "schema": "lane-visit-dwell-v1",
        "spurious_dwell_samples": SPURIOUS_DWELL_SAMPLES,
        "population_rejected_waymo": len(pool),
        "sampled": len(chosen),
        "analyzed": analyzed,
        "visits_total": visits_total,
        "visits_at_or_below_spurious_dwell": visits_spurious,
        "records_with_at_least_one_spurious_visit": records_with_spurious,
        "dwell_histogram_samples": dict(sorted(dwell_histogram.items())),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "dwell_histogram_samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
