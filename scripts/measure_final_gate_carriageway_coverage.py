"""Measure how much of the terminal carriageway the frozen final gate covers.

`mission_success` is a swept front-bumper crossing of one frozen, **finite**
segment (`mission/gates.py`), while `R4` credits arc-length advance of an
unconstrained nearest-point projection. Nothing links the two, so the question
this instrument answers is where the two can come apart: **at the goal
cross-section, is there same-direction drivable surface that the gate does not
cover?**

The measurement is not a new geometry. `mission/builder.py` already decomposes
the goal cross-section: it cuts a +/-100 m line along the route normal, intersects
it with every vertically compatible, same-direction lane polygon, merges the
resulting offset intervals at `FINAL_GATE_BUILDER_EPSILON_M`, and freezes **the
single merged component containing offset 0**. Every *other* merged component is,
by construction, same-direction drivable surface at the goal that the gate does
not intersect. This script re-runs that decomposition and keeps what the builder
discarded.

Because it re-executes the builder's own arithmetic, it must reproduce the frozen
gate exactly. It checks that per record and refuses to count any record whose
host component disagrees with the frozen segment by more than the builder's own
tolerance, so a reconstruction error is reported rather than silently counted.

Nothing is simulated and no policy is involved: this is a property of the frozen
map and the frozen mission record.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pickle
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from shapely.geometry import LineString, Point

# The builder's own constants, imported rather than restated so this instrument
# cannot drift from the construction it is auditing.
from thesis_rl.mission.builder import FINAL_GATE_BUILDER_EPSILON_M, _line_parts
from thesis_rl.mission.types import DrivingMissionRecord
from thesis_rl.rulebook.v2.geometry.drivable import DIRECTION_ALIGNMENT_COS_THRESHOLD

# MetaDrive `DefaultVehicle`: the swept front bumper is a band of full vehicle
# width, so an ego centred at lateral offset `d` presents `[d - w/2, d + w/2]`
# to the gate segment.
EGO_WIDTH_M = 1.852
EGO_LENGTH_M = 4.515
# The width both static adapters fall back to when the source declares none, and
# twice the half-width the PG adapter buffers a centreline by. Used only as the
# physical reading of "one lane over", never as a tuning knob.
LANE_WIDTH_M = 3.5
CROSS_SECTION_HALF_LENGTH_M = 100.0


@dataclass
class RecordResult:
    scenario_uid: str
    source: str
    split: str
    outcome: str
    detail: str = ""
    gate_lo_m: float | None = None
    gate_hi_m: float | None = None
    reconstruction_residual_m: float | None = None
    components_forward: int = 0
    components_aligned: int = 0
    nearest_gap_m: float | None = None
    nearest_interval: tuple[float, float] | None = None
    nearest_lane_types: tuple[str, ...] = ()
    nearest_is_route_lane: bool = False
    lane_change_left_crosses: bool | None = None
    lane_change_right_crosses: bool | None = None


def _lane_record(source: str, lane_id: str, lane: Mapping[str, Any], z_origin_m: float):
    if source == "pg":
        from thesis_rl.rulebook.v2.context.pg_static_adapter import _lane_record as build_lane
    elif source == "waymo":
        from thesis_rl.rulebook.v2.context.waymo_static_adapter import _lane_record as build_lane
    else:
        raise ValueError(f"unsupported source: {source!r}")
    return build_lane(lane_id, lane, z_origin_m=z_origin_m)


def _merge(intervals: list[tuple[float, float, str, str]]) -> list[list[Any]]:
    """Merge offset intervals exactly as `builder.py` does."""

    merged: list[list[Any]] = []
    for lo, hi, lane_id, lane_type in sorted(intervals):
        if not merged or lo > merged[-1][1] + FINAL_GATE_BUILDER_EPSILON_M:
            merged.append([lo, hi, [lane_id], [lane_type]])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
            merged[-1][2].append(lane_id)
            merged[-1][3].append(lane_type)
    return merged


def _decompose(
    *,
    goal_xyz: tuple[float, float, float],
    tangent: tuple[float, float],
    lanes: Mapping[str, Any],
    lane_types: Mapping[str, str],
    final_route_lane_id: str,
    cos_threshold: float,
) -> list[list[Any]]:
    normal = (-tangent[1], tangent[0])
    cross = LineString(
        (
            (
                goal_xyz[0] - CROSS_SECTION_HALF_LENGTH_M * normal[0],
                goal_xyz[1] - CROSS_SECTION_HALF_LENGTH_M * normal[1],
            ),
            (
                goal_xyz[0] + CROSS_SECTION_HALF_LENGTH_M * normal[0],
                goal_xyz[1] + CROSS_SECTION_HALF_LENGTH_M * normal[1],
            ),
        )
    )
    intervals: list[tuple[float, float, str, str]] = []
    for lane_id, lane in lanes.items():
        if lane.centerline is None or lane.polygon_xy is None:
            continue
        try:
            projection = lane.centerline.project(goal_xyz[:2], position_z=goal_xyz[2])
        except ValueError as error:
            if "vertically compatible" in str(error):
                continue
            raise
        candidate_tangent = projection.tangent_xy
        if lane_id == final_route_lane_id:
            candidate_tangent = tangent
        alignment = candidate_tangent[0] * tangent[0] + candidate_tangent[1] * tangent[1]
        if alignment <= cos_threshold:
            continue
        for part in _line_parts(cross.intersection(lane.polygon_xy)):
            first, last = part.coords[0], part.coords[-1]
            lo = cross.project(Point(first)) - CROSS_SECTION_HALF_LENGTH_M
            hi = cross.project(Point(last)) - CROSS_SECTION_HALF_LENGTH_M
            lo, hi = min(lo, hi), max(lo, hi)
            if hi - lo <= FINAL_GATE_BUILDER_EPSILON_M:
                continue
            intervals.append((lo, hi, lane_id, lane_types.get(lane_id, "")))
    return _merge(intervals)


def _crosses(d: float, gate_lo: float, gate_hi: float) -> bool:
    """Can an ego centred at lateral offset ``d`` sweep the finite gate segment?"""

    near, far = d - EGO_WIDTH_M / 2.0, d + EGO_WIDTH_M / 2.0
    return not (near > gate_hi or far < gate_lo)


def evaluate_record(data_root: Path, record: Mapping[str, Any]) -> RecordResult:
    uid = str(record.get("scenario_uid", ""))
    source = str(record.get("source", ""))
    split = str(record.get("split", ""))

    def failure(outcome: str, detail: str) -> RecordResult:
        return RecordResult(uid, source, split, outcome, detail)

    payload = record.get("driving_mission")
    if not isinstance(payload, Mapping):
        return failure("missing_mission", "record carries no driving_mission")
    try:
        mission = DrivingMissionRecord.from_dict(dict(payload))
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return failure("unloadable_mission", f"{type(exc).__name__}: {exc}")
    if mission.final_gate_segment is None or not mission.route_lane_ids:
        return failure("not_route_mission", "no frozen final gate or no route lane ids")

    source_path = data_root / str(record.get("relative_path", ""))
    if not source_path.is_file():
        return failure("missing_file", f"source file does not exist: {source_path}")
    try:
        with source_path.open("rb") as handle:
            scenario = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001
        return failure("unloadable_scenario", f"{type(exc).__name__}: {exc}")

    features = scenario.get("map_features")
    metadata = scenario.get("metadata")
    tracks = scenario.get("tracks")
    if not isinstance(features, Mapping) or not isinstance(metadata, Mapping):
        return failure("invalid_description", "map_features or metadata missing")
    try:
        position = tracks[metadata["sdc_id"]]["state"]["position"][0]
        z_origin_m = float(position[2]) if len(position) > 2 else 0.0
    except Exception as exc:  # noqa: BLE001
        return failure("invalid_description", f"no SDC origin: {type(exc).__name__}: {exc}")

    lanes: dict[str, Any] = {}
    lane_types: dict[str, str] = {}
    for lane_id, feature in features.items():
        if not isinstance(feature, Mapping):
            continue
        lane_type = str(feature.get("type", ""))
        if not lane_type.startswith("LANE_"):
            continue
        try:
            lanes[str(lane_id)] = _lane_record(source, str(lane_id), feature, z_origin_m)
        except ValueError:
            # The static adapters swallow invalid lane geometry the same way, so
            # the lane set here equals the one the runtime would build.
            continue
        lane_types[str(lane_id)] = lane_type
    if not lanes:
        return failure("no_lanes", "scenario declares no usable LANE_* features")

    # The frozen gate, and the goal frame it was built in.
    gate = mission.final_gate_segment
    goal_point = mission.canonical_route_points_xyz[-1]
    tangent = tuple(float(value) for value in gate.static_tangent_xy)
    normal = (-tangent[1], tangent[0])
    offsets = sorted(
        (x - goal_point[0]) * normal[0] + (y - goal_point[1]) * normal[1] for x, y in gate.line_xy
    )
    gate_lo, gate_hi = offsets[0], offsets[1]

    goal_xyz = (float(goal_point[0]), float(goal_point[1]), float(goal_point[2]))
    final_route_lane_id = str(mission.route_lane_ids[-1])

    try:
        forward = _decompose(
            goal_xyz=goal_xyz,
            tangent=tangent,
            lanes=lanes,
            lane_types=lane_types,
            final_route_lane_id=final_route_lane_id,
            cos_threshold=0.0,
        )
        aligned = _decompose(
            goal_xyz=goal_xyz,
            tangent=tangent,
            lanes=lanes,
            lane_types=lane_types,
            final_route_lane_id=final_route_lane_id,
            cos_threshold=DIRECTION_ALIGNMENT_COS_THRESHOLD,
        )
    except Exception as exc:  # noqa: BLE001
        return failure("decomposition_failed", f"{type(exc).__name__}: {exc}")

    host = [
        item
        for item in forward
        if item[0] - FINAL_GATE_BUILDER_EPSILON_M <= 0.0 <= item[1] + FINAL_GATE_BUILDER_EPSILON_M
    ]
    if len(host) != 1:
        return failure("host_component_not_unique", f"{len(host)} components contain offset 0")
    residual = max(abs(host[0][0] - gate_lo), abs(host[0][1] - gate_hi))
    if residual > FINAL_GATE_BUILDER_EPSILON_M * 10.0:
        return RecordResult(
            uid,
            source,
            split,
            "unreconstructible",
            f"host [{host[0][0]:.3f},{host[0][1]:.3f}] vs frozen [{gate_lo:.3f},{gate_hi:.3f}]",
            gate_lo_m=gate_lo,
            gate_hi_m=gate_hi,
            reconstruction_residual_m=residual,
        )

    route_ids = set(mission.route_lane_ids)
    others = [item for item in forward if item is not host[0]]
    nearest_gap: float | None = None
    nearest: list[Any] | None = None
    for item in others:
        gap = item[0] - gate_hi if item[0] > gate_hi else gate_lo - item[1]
        if gap < 0.0:
            gap = 0.0
        if nearest_gap is None or gap < nearest_gap:
            nearest_gap, nearest = gap, item

    result = RecordResult(
        uid,
        source,
        split,
        "ok",
        gate_lo_m=gate_lo,
        gate_hi_m=gate_hi,
        reconstruction_residual_m=residual,
        components_forward=len(forward),
        components_aligned=len(aligned),
        # Can the ego still sweep the gate one lane to either side of its goal?
        lane_change_left_crosses=_crosses(-LANE_WIDTH_M, gate_lo, gate_hi),
        lane_change_right_crosses=_crosses(+LANE_WIDTH_M, gate_lo, gate_hi),
    )
    if nearest is not None and nearest_gap is not None:
        result.nearest_gap_m = nearest_gap
        result.nearest_interval = (nearest[0], nearest[1])
        result.nearest_lane_types = tuple(sorted(set(nearest[3])))
        result.nearest_is_route_lane = bool(route_ids.intersection(nearest[2]))
    return result


_DATA_ROOT: Path | None = None


def _init_worker(data_root: str) -> None:
    global _DATA_ROOT
    _DATA_ROOT = Path(data_root)
    # Shapely is single-threaded; every record is independent, so keep BLAS from
    # oversubscribing the pool. Same reasoning as the expert replay instrument.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"


def _work(record: Mapping[str, Any]) -> RecordResult:
    assert _DATA_ROOT is not None
    try:
        return evaluate_record(_DATA_ROOT, record)
    except Exception as exc:  # noqa: BLE001 - a crash must not lose the pass
        return RecordResult(
            str(record.get("scenario_uid", "")),
            str(record.get("source", "")),
            str(record.get("split", "")),
            "crashed",
            f"{type(exc).__name__}: {exc}",
        )


def selected_records(
    payload: Mapping[str, Any], *, split: str, source: str, limit: int | None
) -> list[Mapping[str, Any]]:
    records = [
        record
        for record in payload["records"]
        if (source == "all" or record.get("source") == source)
        and (split == "all" or record.get("split") == split)
    ]
    records.sort(key=lambda record: str(record.get("scenario_uid", "")))
    return records[:limit] if limit else records


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(q: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[index]

    return {
        "min": ordered[0],
        "p05": at(0.05),
        "p25": at(0.25),
        "median": statistics.median(ordered),
        "p75": at(0.75),
        "p95": at(0.95),
        "max": ordered[-1],
    }


def summarize(results: Sequence[RecordResult]) -> dict[str, Any]:
    ok = [result for result in results if result.outcome == "ok"]
    outcomes: dict[str, int] = {}
    for result in results:
        outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1

    gate_lengths = [
        result.gate_hi_m - result.gate_lo_m
        for result in ok
        if result.gate_hi_m is not None and result.gate_lo_m is not None
    ]
    with_other = [result for result in ok if result.nearest_gap_m is not None]
    gaps = [result.nearest_gap_m for result in with_other if result.nearest_gap_m is not None]

    def count(predicate) -> int:
        return sum(1 for result in ok if predicate(result))

    total = len(ok) or 1
    return {
        "records_evaluated": len(results),
        "records_ok": len(ok),
        "outcomes": outcomes,
        "reconstruction_residual_m": _quantiles(
            [
                result.reconstruction_residual_m
                for result in ok
                if result.reconstruction_residual_m is not None
            ]
        ),
        "gate_length_m": _quantiles(gate_lengths),
        "uncovered_same_direction_component": {
            "records": len(with_other),
            "fraction": len(with_other) / total,
            "gap_to_gate_m": _quantiles(gaps),
            "within_ego_width": count(
                lambda r: r.nearest_gap_m is not None and r.nearest_gap_m <= EGO_WIDTH_M
            ),
            "within_ego_length": count(
                lambda r: r.nearest_gap_m is not None and r.nearest_gap_m <= EGO_LENGTH_M
            ),
            "bike_lane_only": count(
                lambda r: bool(r.nearest_lane_types)
                and all("BIKE" in t for t in r.nearest_lane_types)
            ),
            "is_a_route_lane": count(lambda r: r.nearest_is_route_lane),
        },
        "gate_narrower_than_carriageway": {
            "one_lane_change_either_side_leaves_gate": count(
                lambda r: r.lane_change_left_crosses is False
                and r.lane_change_right_crosses is False
            ),
            "one_lane_change_some_side_leaves_gate": count(
                lambda r: r.lane_change_left_crosses is False
                or r.lane_change_right_crosses is False
            ),
        },
        "component_counts": {
            "forward_cone_gt_0": _quantiles([float(r.components_forward) for r in ok]),
            "aligned_cone_ge_0p5": _quantiles([float(r.components_aligned) for r in ok]),
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", default="all")
    parser.add_argument("--source", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = json.loads(args.frozen_index.read_text(encoding="utf-8"))
    if payload.get("schema") != "scenarionet_frozen_selection_v1":
        raise SystemExit(f"unexpected frozen index schema: {payload.get('schema')!r}")
    records = selected_records(
        payload, split=args.split, source=args.source, limit=args.limit
    )
    if not records:
        raise SystemExit("no records selected; the measurement would be vacuous")
    print(f"evaluating {len(records)} records on {args.workers} workers", flush=True)

    results: list[RecordResult] = []
    context = multiprocessing.get_context("fork")
    with context.Pool(
        processes=args.workers, initializer=_init_worker, initargs=(str(args.data_root),)
    ) as pool:
        for index, result in enumerate(pool.imap_unordered(_work, records, chunksize=2), 1):
            results.append(result)
            if index % 250 == 0:
                print(f"  {index}/{len(records)}", flush=True)

    summary = summarize(results)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "scope": {
                        "split": args.split,
                        "source": args.source,
                        "limit": args.limit,
                        "records": len(records),
                    },
                    "constants": {
                        "final_gate_builder_epsilon_m": FINAL_GATE_BUILDER_EPSILON_M,
                        "direction_alignment_cos_threshold": DIRECTION_ALIGNMENT_COS_THRESHOLD,
                        "ego_width_m": EGO_WIDTH_M,
                        "ego_length_m": EGO_LENGTH_M,
                        "lane_width_m": LANE_WIDTH_M,
                    },
                    "summary": summary,
                    "records": [
                        {
                            "scenario_uid": result.scenario_uid,
                            "source": result.source,
                            "split": result.split,
                            "outcome": result.outcome,
                            "detail": result.detail,
                            "gate_lo_m": result.gate_lo_m,
                            "gate_hi_m": result.gate_hi_m,
                            "reconstruction_residual_m": result.reconstruction_residual_m,
                            "components_forward": result.components_forward,
                            "components_aligned": result.components_aligned,
                            "nearest_gap_m": result.nearest_gap_m,
                            "nearest_interval": list(result.nearest_interval)
                            if result.nearest_interval
                            else None,
                            "nearest_lane_types": list(result.nearest_lane_types),
                            "nearest_is_route_lane": result.nearest_is_route_lane,
                            "lane_change_left_crosses": result.lane_change_left_crosses,
                            "lane_change_right_crosses": result.lane_change_right_crosses,
                        }
                        for result in sorted(results, key=lambda item: item.scenario_uid)
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
