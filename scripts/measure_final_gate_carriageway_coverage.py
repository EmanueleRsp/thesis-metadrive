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

`--walk-spacing-m` adds `D14`'s backwards route walk, which the question above
deliberately does not answer: whether that uncovered surface is **continuous back
along the route**, so that an ego could drive it while `R4` credits the advance of
its projection. It calls the same decomposition at stations along the route rather
than only at the goal, so it inherits the reconciliation against the frozen gate
above. Omitted by default, and the goal cross-section figures are unaffected by
it. See `_walk_backwards`.
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
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from shapely.geometry import LineString, Point

# The builder's own constants, imported rather than restated so this instrument
# cannot drift from the construction it is auditing.
from thesis_rl.mission.builder import FINAL_GATE_BUILDER_EPSILON_M, _line_parts
from thesis_rl.mission.types import DrivingMissionRecord
from thesis_rl.rulebook.v2.geometry.drivable import DIRECTION_ALIGNMENT_COS_THRESHOLD

if TYPE_CHECKING:
    from thesis_rl.rulebook.v2.geometry.route import RoutePolyline

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
    # `D14`'s backwards route walk, populated only when `--walk-spacing-m` is
    # given. Everything above is the goal cross-section alone and is unchanged.
    walk_stations: int = 0
    walk_unusable_stations: int = 0
    walk_span_m: float | None = None
    # The corridor that reaches the goal cross-section, walked backwards from it:
    # same-direction drivable surface outside the route's own carriageway, wide
    # enough for the ego, continuous station to station.
    corridor_to_goal_m: float | None = None
    corridor_to_goal_offset_m: float | None = None
    corridor_to_goal_clears_gate: bool | None = None
    corridor_entry_gap_m: float | None = None
    corridor_to_goal_is_route_lane: bool = False
    corridor_to_goal_lane_types: tuple[str, ...] = ()


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


def _walk_backwards(
    *,
    route: "RoutePolyline",
    s_goal_m: float,
    spacing_m: float,
    gate_lo: float,
    gate_hi: float,
    lanes: Mapping[str, Any],
    lane_types: Mapping[str, str],
    route_lane_ids: set[str],
) -> dict[str, Any]:
    """Walk the route backwards from the goal, following parallel surface.

    `D14` records that the goal cross-section measurement "does not show that an
    ego could drive such surface *while banking the `R4` budget* — that needs the
    backwards route walk, which is designed but not run". This is that walk.

    The question it answers is not whether uncovered surface exists at the goal —
    that is already measured — but whether it is **continuous back along the
    route**, because `R4` credits the arc-length advance of the projection and an
    ego on a parallel carriageway projects onto the route and advances. A corridor
    that runs `X` metres and ends outside the final gate is `X / D_REF` channel
    units an ego can bank and still fail the mission, with `offroad = 0` because
    the drivable surface is the union of every vertically compatible lane, and
    `wrong_carriageway = 0` because it charges only opposing surface.

    Continuity is the crux, and each of its two conditions makes the reported
    corridor a **lower** bound rather than an upper one. A station contributes
    only if its component is at least one ego width wide, since a corridor the ego
    does not fit in is not one it can drive; and consecutive components must
    **overlap** in offset, which is what makes them the same physical carriageway
    rather than two unrelated strips at similar distances.

    Every qualifying candidate at the goal is walked, not only the nearest one,
    and the longest corridor wins. Taking the nearest would have been cheaper and
    would have under-reported: a corridor one lane out that stops after 5 m would
    hide one three lanes out that runs the length of the route. Under-reporting is
    the wrong direction for an audit, and it is also why a candidate qualifies on
    containing a position from which the ego misses the gate rather than on its
    centre being such a position.

    `corridor_entry_gap_m` is the separation between the corridor and the route's
    own carriageway at the station the walk stops at. Zero means the ego changes
    into it for free; positive means it must cross surface that is not drivable,
    which `offroad` does charge, so the exposure is priced rather than free. That
    distinction is the one neither the goal cross-section measurement nor `D14`
    could make.
    """

    cache: dict[float, list[list[Any]] | None] = {}

    def at(station_m: float) -> list[list[Any]] | None:
        """Same-direction components at one station, or None if unusable."""

        if station_m in cache:
            return cache[station_m]
        point = route.point_at(station_m)
        try:
            projection = route.project((point[0], point[1]), position_z=point[2])
            components = _decompose(
                goal_xyz=(float(point[0]), float(point[1]), float(point[2])),
                tangent=projection.tangent_xy,
                lanes=lanes,
                lane_types=lane_types,
                # No override: along the walk the reference direction is the
                # route polyline's own tangent, and the alignment filter does the
                # rest. The goal cross-section keeps the gate's static tangent.
                final_route_lane_id="",
                cos_threshold=DIRECTION_ALIGNMENT_COS_THRESHOLD,
            )
        except Exception:  # noqa: BLE001 - counted by the caller, not raised
            cache[station_m] = None
            return None
        cache[station_m] = components
        return components

    def host_of(components: list[list[Any]]) -> list[Any] | None:
        hosts = [
            item
            for item in components
            if item[0] - FINAL_GATE_BUILDER_EPSILON_M
            <= 0.0
            <= item[1] + FINAL_GATE_BUILDER_EPSILON_M
        ]
        return hosts[0] if len(hosts) == 1 else None

    def drivable_others(components: list[list[Any]], host: list[Any]) -> list[list[Any]]:
        return [
            item for item in components if item is not host and item[1] - item[0] >= EGO_WIDTH_M
        ]

    stations_visited: set[float] = set()
    unusable = 0

    goal_components = at(s_goal_m)
    if goal_components is None:
        return {
            "walk_stations": 0,
            "walk_unusable_stations": 1,
            "walk_span_m": s_goal_m,
        }
    stations_visited.add(s_goal_m)
    goal_host = host_of(goal_components)
    candidates: list[list[Any]] = []
    if goal_host is not None:
        candidates = [
            item
            for item in drivable_others(goal_components, goal_host)
            # A corridor is an exposure only if it CONTAINS a position from which
            # the ego misses the gate. Testing the component's centre instead
            # would under-report: a wide component whose centre still sweeps the
            # gate can have an end that does not. An ego centred at `d` presents
            # `[d - w/2, d + w/2]`, and `d` must itself lie a half-width inside
            # the component, so such a position exists exactly when the component
            # runs a full ego width past either end of the gate.
            if item[1] > gate_hi + EGO_WIDTH_M or item[0] < gate_lo - EGO_WIDTH_M
        ]

    best: dict[str, Any] | None = None
    for candidate in candidates:
        previous = candidate
        previous_station = s_goal_m
        length_m = 0.0
        station = max(0.0, s_goal_m - spacing_m)
        while previous_station > 0.0:
            components = at(station)
            stations_visited.add(station)
            if components is None:
                unusable += 1
                break
            host = host_of(components)
            if host is None:
                unusable += 1
                break
            overlapping = [
                item
                for item in drivable_others(components, host)
                if item[0] <= previous[1] and previous[0] <= item[1]
            ]
            if not overlapping:
                break
            centre = (previous[0] + previous[1]) / 2.0
            chosen = min(overlapping, key=lambda item: abs((item[0] + item[1]) / 2.0 - centre))
            length_m += previous_station - station
            previous = chosen
            previous_station = station
            if station == 0.0:
                break
            station = max(0.0, station - spacing_m)

        # The gap is read where the corridor ends, because that is where the ego
        # would have to enter it.
        end_components = cache.get(previous_station)
        entry_gap: float | None = None
        if end_components is not None:
            end_host = host_of(end_components)
            if end_host is not None:
                gap = (
                    previous[0] - end_host[1]
                    if previous[0] > end_host[1]
                    else end_host[0] - previous[1]
                )
                entry_gap = max(0.0, gap)
        summary = {
            "corridor_to_goal_m": length_m,
            "corridor_to_goal_offset_m": (candidate[0] + candidate[1]) / 2.0,
            "corridor_to_goal_clears_gate": True,
            "corridor_entry_gap_m": entry_gap,
            "corridor_to_goal_is_route_lane": bool(route_lane_ids.intersection(candidate[2])),
            "corridor_to_goal_lane_types": tuple(sorted(set(candidate[3]))),
        }
        if best is None or length_m > float(best["corridor_to_goal_m"]):
            best = summary

    result: dict[str, Any] = {
        "walk_stations": len(stations_visited),
        "walk_unusable_stations": unusable,
        "walk_span_m": s_goal_m,
    }
    if best is not None:
        result.update(best)
    return result


def _crosses(d: float, gate_lo: float, gate_hi: float) -> bool:
    """Can an ego centred at lateral offset ``d`` sweep the finite gate segment?"""

    near, far = d - EGO_WIDTH_M / 2.0, d + EGO_WIDTH_M / 2.0
    return not (near > gate_hi or far < gate_lo)


def evaluate_record(
    data_root: Path, record: Mapping[str, Any], *, walk_spacing_m: float | None = None
) -> RecordResult:
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

    if walk_spacing_m is not None:
        from thesis_rl.rulebook.v2.geometry.route import RoutePolyline

        # The walk runs only after the host component has been reconciled with
        # the frozen gate above, so a record whose reconstruction is wrong is
        # refused before any of it is measured.
        route = RoutePolyline(
            points_xyz=tuple(
                tuple(float(value) for value in point)
                for point in mission.canonical_route_points_xyz
            )
        )
        try:
            walked = _walk_backwards(
                route=route,
                s_goal_m=route.length_m,
                spacing_m=float(walk_spacing_m),
                gate_lo=gate_lo,
                gate_hi=gate_hi,
                lanes=lanes,
                lane_types=lane_types,
                route_lane_ids=set(route_ids),
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            result.detail = f"walk failed: {type(exc).__name__}: {exc}"
            return result
        for name, value in walked.items():
            setattr(result, name, value)
    return result


_DATA_ROOT: Path | None = None
_WALK_SPACING_M: float | None = None


def _init_worker(data_root: str, walk_spacing_m: float | None = None) -> None:
    global _DATA_ROOT, _WALK_SPACING_M
    _DATA_ROOT = Path(data_root)
    _WALK_SPACING_M = walk_spacing_m
    # Shapely is single-threaded; every record is independent, so keep BLAS from
    # oversubscribing the pool. Same reasoning as the expert replay instrument.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"


def _work(record: Mapping[str, Any]) -> RecordResult:
    assert _DATA_ROOT is not None
    try:
        return evaluate_record(_DATA_ROOT, record, walk_spacing_m=_WALK_SPACING_M)
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
                lambda r: (
                    bool(r.nearest_lane_types) and all("BIKE" in t for t in r.nearest_lane_types)
                )
            ),
            "is_a_route_lane": count(lambda r: r.nearest_is_route_lane),
        },
        "gate_narrower_than_carriageway": {
            "one_lane_change_either_side_leaves_gate": count(
                lambda r: (
                    r.lane_change_left_crosses is False and r.lane_change_right_crosses is False
                )
            ),
            "one_lane_change_some_side_leaves_gate": count(
                lambda r: (
                    r.lane_change_left_crosses is False or r.lane_change_right_crosses is False
                )
            ),
        },
        "component_counts": {
            "forward_cone_gt_0": _quantiles([float(r.components_forward) for r in ok]),
            "aligned_cone_ge_0p5": _quantiles([float(r.components_aligned) for r in ok]),
        },
        "backwards_route_walk": _summarize_walk(ok),
    }


def _summarize_walk(ok: Sequence[RecordResult]) -> dict[str, Any]:
    """`D14`: is the parallel surface at the goal *continuous back along the route*?

    Only present when `--walk-spacing-m` ran. The statistic the decision needs is
    `corridor_to_goal_m`: the arc length over which an ego could drive
    same-direction surface outside its own carriageway and still be outside the
    final gate when it gets there. Divided by `D_REF = 2.2222 m` it is the `R4`
    budget bankable on a trajectory that fails the mission, and reported beside
    it is `corridor_entry_gap_m`, which says whether reaching that corridor costs
    `offroad` or is free.
    """

    walked = [result for result in ok if result.walk_stations > 0]
    if not walked:
        return {"records_walked": 0}
    with_corridor = [result for result in walked if result.corridor_to_goal_m is not None]
    lengths = [
        result.corridor_to_goal_m
        for result in with_corridor
        if result.corridor_to_goal_m is not None
    ]
    # Reported in bands rather than against one threshold, for the reason V3's
    # audit reports its proximity bands: choosing a cut-off here would be
    # choosing how much `offroad` an ego is allowed to pay to enter the corridor,
    # which is a parameter this work is not allowed to add and would not want to.
    # 0.05 m is below every lane-merge artefact the 2026-09-08 run found (the
    # tightest were 0.010-0.011 m, filed as `C48`), so that band is surface the
    # map representation separates and physical geometry does not.
    entry_bands: dict[str, Any] = {}
    for label, bound in (
        ("le_0p05m_representation_artefact", 0.05),
        ("le_0p5m", 0.5),
        ("le_one_ego_width", EGO_WIDTH_M),
        ("any", float("inf")),
    ):
        banded = [
            result
            for result in with_corridor
            if result.corridor_entry_gap_m is not None and result.corridor_entry_gap_m <= bound
        ]
        banded_lengths = [
            result.corridor_to_goal_m for result in banded if result.corridor_to_goal_m is not None
        ]
        entry_bands[label] = {
            "records": len(banded),
            "fraction_of_walked": len(banded) / len(walked),
            "corridor_to_goal_m": _quantiles(banded_lengths),
            "longest_m": max(banded_lengths) if banded_lengths else None,
            "bike_lane_only": sum(
                1
                for result in banded
                if result.corridor_to_goal_lane_types
                and all("BIKE" in kind for kind in result.corridor_to_goal_lane_types)
            ),
            "is_a_route_lane": sum(1 for result in banded if result.corridor_to_goal_is_route_lane),
        }
    return {
        "records_walked": len(walked),
        "stations_per_record": _quantiles([float(r.walk_stations) for r in walked]),
        "records_with_unusable_station": sum(
            1 for result in walked if result.walk_unusable_stations > 0
        ),
        "records_with_corridor_to_goal": len(with_corridor),
        "fraction_with_corridor_to_goal": len(with_corridor) / len(walked),
        "corridor_to_goal_m": _quantiles(lengths),
        "corridor_offset_m": _quantiles(
            [
                abs(result.corridor_to_goal_offset_m)
                for result in with_corridor
                if result.corridor_to_goal_offset_m is not None
            ]
        ),
        "corridor_entry_gap_m": _quantiles(
            [
                result.corridor_entry_gap_m
                for result in with_corridor
                if result.corridor_entry_gap_m is not None
            ]
        ),
        # How much `offroad` entering the corridor costs, as bands rather than a
        # verdict. The first band is surface only the map representation
        # separates; the last is every corridor however far the median is.
        "by_entry_gap": entry_bands,
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
    parser.add_argument(
        "--walk-spacing-m",
        type=float,
        default=None,
        help=(
            "run D14's backwards route walk at this station spacing. Omitted by "
            "default, so the goal cross-section measurement this instrument was "
            "built for is unchanged. 5.0 m is about one and a half vehicle lengths"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = json.loads(args.frozen_index.read_text(encoding="utf-8"))
    if payload.get("schema") != "scenarionet_frozen_selection_v1":
        raise SystemExit(f"unexpected frozen index schema: {payload.get('schema')!r}")
    records = selected_records(payload, split=args.split, source=args.source, limit=args.limit)
    if not records:
        raise SystemExit("no records selected; the measurement would be vacuous")
    print(f"evaluating {len(records)} records on {args.workers} workers", flush=True)

    results: list[RecordResult] = []
    context = multiprocessing.get_context("fork")
    with context.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(str(args.data_root), args.walk_spacing_m),
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
                            "walk_stations": result.walk_stations,
                            "walk_unusable_stations": result.walk_unusable_stations,
                            "walk_span_m": result.walk_span_m,
                            "corridor_to_goal_m": result.corridor_to_goal_m,
                            "corridor_to_goal_offset_m": result.corridor_to_goal_offset_m,
                            "corridor_entry_gap_m": result.corridor_entry_gap_m,
                            "corridor_to_goal_is_route_lane": result.corridor_to_goal_is_route_lane,
                            "corridor_to_goal_lane_types": list(result.corridor_to_goal_lane_types),
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
