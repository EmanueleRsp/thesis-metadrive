"""Read-only measurement of Rulebook v2 sub-rule costs along the expert trajectory.

The question this answers is empirical, not architectural: driving the *logged
human SDC* through the exact Rulebook v2 adapter, how often and how strongly is
each sub-rule violated?

A compliance metric that declares expert human driving non-compliant at almost
every step cannot discriminate between policies, and — because the scalarized
reward weights R2 nine times the progress channel — makes standing still the
optimal policy. This script measures that directly instead of arguing about it.

Nothing is simulated: the ego pose/velocity comes from the recorded SDC track,
so the measured costs are a property of the *metric definition*, not of any
policy. The script imports the same private candidate-selection helpers the
runtime uses (`_rss_candidates`, `_vertical_actor_ids`) rather than
reimplementing them, so a divergence between measurement and runtime is
impossible by construction.

RSS response-time sensitivity (`--rho-sweep`) recomputes only the longitudinal
safe distance with alternative response times, from the same candidate gaps and
speeds, to answer "which rho makes the expert compliant?" in one pass.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import pickle
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from thesis_rl.rulebook.v2.components.road import (
    evaluate_offroad,
    evaluate_solid_line,
    evaluate_wrong_carriageway,
)
from thesis_rl.rulebook.v2.components.rss import (
    FRONT_MAX_BRAKE_MPS2,
    MAX_RESPONSE_ACCEL_MPS2,
    RSSCalibrationArtifact,
    RSS_STANDSTILL_SPEED_MPS,
    evaluate_rss,
)
from thesis_rl.rulebook.v2.components.ttc import evaluate_ttc
from thesis_rl.rulebook.v2.context.waymo_static_adapter import build_waymo_static_adapter_result
from thesis_rl.rulebook.v2.geometry.drivable import (
    DrivableLaneRecord,
    carriageway_surfaces_for_ego,
    drivable_surface_for_ego,
)
from thesis_rl.rulebook.v2.geometry.footprint import oriented_bounding_box
from thesis_rl.rulebook.v2.geometry.vertical import VERTICAL_COMPATIBILITY_TOLERANCE_M
from thesis_rl.rulebook.v2.transition import (
    _rss_candidates,
    _vertical_actor_ids,
    build_episode_cache,
)
from thesis_rl.rulebook.v2.types import ActorClass, ActorSnapshot, MapFeatureClass

# MetaDrive's per-vehicle-type constant (`pg_space.py`: max_speed_km_h=80).
# The runtime reads it from the live vehicle; offline it is a fixed constant.
DEFAULT_SPEED_CAP_MPS = 80.0 / 3.6
# ADR-047: b_meas ~= 10.7 m/s^2 clamped by the physical bound to 8.0.
DEFAULT_EGO_BRAKE_MPS2 = 8.0
DELTA_T_S = 0.1
_CALIBRATION_HASH = "expert-cost-measurement"

_ACTOR_CLASSES = {
    "VEHICLE": ActorClass.VEHICLE,
    "PEDESTRIAN": ActorClass.PEDESTRIAN,
    "CYCLIST": ActorClass.CYCLIST,
    "TRAFFIC_CONE": ActorClass.STATIC_COLLIDABLE,
    "TRAFFIC_BARRIER": ActorClass.STATIC_COLLIDABLE,
    "TRAFFIC_OBJECT": ActorClass.STATIC_COLLIDABLE,
}


def parametric_safe_distance_m(
    *, ego_speed_mps: float, front_speed_mps: float, ego_brake_mps2: float, response_time_s: float
) -> float:
    """`components.rss.safe_distance_m` with the response time left free.

    Deliberately duplicated, and only for the sensitivity sweep: the production
    formula freezes `RESPONSE_TIME_S` as a module constant, and a measurement
    must not mutate it. `response_time_s == 1.0` reproduces production exactly,
    which the accompanying test asserts.
    """

    ego_speed = max(0.0, ego_speed_mps)
    front_speed = max(0.0, front_speed_mps)
    response_distance = ego_speed * response_time_s
    response_acceleration = 0.5 * MAX_RESPONSE_ACCEL_MPS2 * response_time_s**2
    ego_braking = (ego_speed + response_time_s * MAX_RESPONSE_ACCEL_MPS2) ** 2 / (
        2.0 * ego_brake_mps2
    )
    front_braking = front_speed**2 / (2.0 * FRONT_MAX_BRAKE_MPS2)
    return max(0.0, response_distance + response_acceleration + ego_braking - front_braking)


@dataclass
class CostSamples:
    """Streaming accumulator for one sub-rule's per-step cost distribution."""

    applicable_steps: int = 0
    violated_steps: int = 0
    positive_costs: list[float] = field(default_factory=list)

    def add(self, cost: float, *, applicable: bool) -> None:
        if not applicable:
            return
        self.applicable_steps += 1
        if cost > 0.0:
            self.violated_steps += 1
            self.positive_costs.append(float(cost))

    def merge(self, other: "CostSamples") -> None:
        """Absorb a worker's partial accumulator; the statistics are additive."""

        self.applicable_steps += other.applicable_steps
        self.violated_steps += other.violated_steps
        self.positive_costs.extend(other.positive_costs)

    def summary(self, total_steps: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "applicable_steps": self.applicable_steps,
            "violated_steps": self.violated_steps,
            "violated_fraction_of_applicable": (
                self.violated_steps / self.applicable_steps if self.applicable_steps else None
            ),
            "violated_fraction_of_all_steps": (
                self.violated_steps / total_steps if total_steps else None
            ),
        }
        if self.positive_costs:
            values = sorted(self.positive_costs)
            payload["positive_cost_percentiles"] = {
                label: round(_percentile(values, quantile), 4)
                for label, quantile in (
                    ("p50", 0.50),
                    ("p75", 0.75),
                    ("p90", 0.90),
                    ("p99", 0.99),
                    ("max", 1.0),
                )
            }
            payload["positive_cost_mean"] = round(statistics.fmean(values), 4)
        return payload


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, int(round(quantile * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


def _sdc_z_origin(scenario: Mapping[str, Any]) -> float:
    """Mirror `waymo_static_adapter._sdc_z_origin` so both frames agree."""

    metadata = scenario.get("metadata", {})
    tracks = scenario.get("tracks", {})
    sdc_id = str(metadata.get("sdc_id", ""))
    if not isinstance(tracks, Mapping) or sdc_id not in tracks:
        return 0.0
    state = tracks[sdc_id].get("state")
    positions = state.get("position") if isinstance(state, Mapping) else None
    if positions is None:
        return 0.0
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3 or len(values) == 0 or not np.isfinite(values[0, 2]):
        return 0.0
    return float(values[0, 2])


def _snapshot_at(
    track_id: str, track: Mapping[str, Any], step: int, *, z_origin_m: float
) -> ActorSnapshot | None:
    """Build one canonical actor snapshot from a logged track sample."""

    actor_class = _ACTOR_CLASSES.get(str(track.get("type", "")))
    if actor_class is None:
        return None
    state = track.get("state")
    if not isinstance(state, Mapping):
        return None
    valid = np.asarray(state.get("valid"), dtype=bool)
    if step >= len(valid) or not bool(valid[step]):
        return None
    position = np.asarray(state["position"], dtype=float)[step]
    heading = float(np.asarray(state["heading"], dtype=float)[step])
    velocity = np.asarray(state["velocity"], dtype=float)[step]
    length = float(np.asarray(state["length"], dtype=float)[step])
    width = float(np.asarray(state["width"], dtype=float)[step])
    if length <= 0.0 or width <= 0.0:
        return None
    values = (position[0], position[1], position[2], heading, velocity[0], velocity[1])
    if not all(math.isfinite(float(value)) for value in values):
        return None
    center = (float(position[0]), float(position[1]))
    return ActorSnapshot(
        actor_id=str(track_id),
        actor_class=actor_class,
        position_xy=center,
        position_z=float(position[2]) - z_origin_m,
        heading_rad=heading,
        velocity_xy=(float(velocity[0]), float(velocity[1])),
        footprint=oriented_bounding_box(
            center_xy=center, heading_rad=heading, length_m=length, width_m=width
        ),
        live_lane_id=None,
        configured_speed_cap_mps=(
            DEFAULT_SPEED_CAP_MPS if actor_class is ActorClass.VEHICLE else None
        ),
    )


@dataclass
class Measurement:
    """Aggregate over every measured scenario."""

    rho_values: tuple[float, ...]
    scenarios_measured: int = 0
    scenarios_skipped: Counter = field(default_factory=Counter)
    total_steps: int = 0
    rss: CostSamples = field(default_factory=CostSamples)
    ttc: CostSamples = field(default_factory=CostSamples)
    offroad: CostSamples = field(default_factory=CostSamples)
    solid_line: CostSamples = field(default_factory=CostSamples)
    wrong_carriageway: CostSamples = field(default_factory=CostSamples)
    r2_macro: CostSamples = field(default_factory=CostSamples)
    rho_sweep: dict[float, CostSamples] = field(default_factory=dict)
    ego_speed_mps: list[float] = field(default_factory=list)
    front_gap_m: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        for rho in self.rho_values:
            self.rho_sweep.setdefault(rho, CostSamples())

    def merge(self, other: "Measurement") -> None:
        """Absorb one worker's result. Every field is order-independent.

        Percentiles are computed from the merged sample lists at report time,
        never from per-worker percentiles, so the parallel result is identical
        to the sequential one regardless of completion order.
        """

        if other.rho_values != self.rho_values:
            raise ValueError("Cannot merge measurements with different response-time sweeps")
        self.scenarios_measured += other.scenarios_measured
        self.scenarios_skipped.update(other.scenarios_skipped)
        self.total_steps += other.total_steps
        for name in ("rss", "ttc", "offroad", "solid_line", "wrong_carriageway", "r2_macro"):
            getattr(self, name).merge(getattr(other, name))
        for rho, samples in other.rho_sweep.items():
            self.rho_sweep[rho].merge(samples)
        self.ego_speed_mps.extend(other.ego_speed_mps)
        self.front_gap_m.extend(other.front_gap_m)

    def summary(self) -> dict[str, Any]:
        speeds = sorted(self.ego_speed_mps)
        gaps = sorted(self.front_gap_m)
        return {
            "scenarios_measured": self.scenarios_measured,
            "scenarios_skipped": dict(self.scenarios_skipped),
            "measured_steps": self.total_steps,
            "expert_speed_mps": {
                "p50": round(_percentile(speeds, 0.50), 2) if speeds else None,
                "p90": round(_percentile(speeds, 0.90), 2) if speeds else None,
            },
            "expert_front_gap_m": {
                "p10": round(_percentile(gaps, 0.10), 2) if gaps else None,
                "p50": round(_percentile(gaps, 0.50), 2) if gaps else None,
            },
            "sub_rules": {
                "rss": self.rss.summary(self.total_steps),
                "ttc": self.ttc.summary(self.total_steps),
                "offroad": self.offroad.summary(self.total_steps),
                "solid_line": self.solid_line.summary(self.total_steps),
                "wrong_carriageway": self.wrong_carriageway.summary(self.total_steps),
            },
            "r2_macro_max": self.r2_macro.summary(self.total_steps),
            "rss_response_time_sweep": {
                f"rho_{rho:g}s": self.rho_sweep[rho].summary(self.total_steps)
                for rho in self.rho_values
            },
        }


def measure_scenario(
    scenario: Mapping[str, Any],
    *,
    scenario_uid: str,
    measurement: Measurement,
    ego_brake_mps2: float,
    step_stride: int,
    road_rules: bool,
) -> None:
    """Accumulate one scenario's per-step expert costs into `measurement`."""

    static = build_waymo_static_adapter_result(scenario, scenario_uid=scenario_uid)
    if static.validation_errors:
        measurement.scenarios_skipped[
            f"validation:{static.validation_errors[0].split(':')[0]}"
        ] += 1
        return
    cache = build_episode_cache(static)
    route = cache.route_polyline
    if route is None:
        measurement.scenarios_skipped["no_route_polyline"] += 1
        return
    z_origin_m = _sdc_z_origin(scenario)
    metadata = scenario["metadata"]
    tracks = scenario["tracks"]
    sdc_id = str(metadata.get("sdc_id", ""))
    if sdc_id not in tracks:
        measurement.scenarios_skipped["no_sdc_track"] += 1
        return
    calibration = RSSCalibrationArtifact(
        config_hash=_CALIBRATION_HASH, ego_min_brake_mps2=ego_brake_mps2
    )
    drivable_lanes = tuple(
        DrivableLaneRecord(lane.lane_id, lane.centerline, lane.polygon_xy, None)
        for lane in cache.route_lanes
    )
    length = int(scenario.get("length", 0))
    measured_any = False
    for step in range(0, length, max(1, step_stride)):
        ego = _snapshot_at(sdc_id, tracks[sdc_id], step, z_origin_m=z_origin_m)
        if ego is None:
            continue
        actors = tuple(
            snapshot
            for track_id, track in tracks.items()
            if str(track_id) != sdc_id
            and (snapshot := _snapshot_at(track_id, track, step, z_origin_m=z_origin_m)) is not None
        )
        measured_any = True
        measurement.total_steps += 1
        measurement.ego_speed_mps.append(float(math.hypot(*ego.velocity_xy)))
        costs: list[float] = []

        candidates = _rss_candidates(
            ego=ego, actors=actors, route=route, route_lanes=cache.route_lanes
        )
        moving = tuple(
            candidate
            for candidate in candidates
            if candidate.ego_speed_mps > RSS_STANDSTILL_SPEED_MPS
            or candidate.front_speed_mps > RSS_STANDSTILL_SPEED_MPS
        )
        if candidates:
            measurement.front_gap_m.append(min(item.gap_m for item in candidates))
        result, _, _ = evaluate_rss(
            scenario_id=scenario_uid,
            step_index=step,
            candidates=candidates,
            calibration=calibration,
            expected_config_hash=_CALIBRATION_HASH,
        )
        measurement.rss.add(result.cost, applicable=result.applicable)
        if result.applicable:
            costs.append(result.cost)
        for rho, samples in measurement.rho_sweep.items():
            worst = 0.0
            for candidate in moving:
                safe = parametric_safe_distance_m(
                    ego_speed_mps=candidate.ego_speed_mps,
                    front_speed_mps=candidate.front_speed_mps,
                    ego_brake_mps2=ego_brake_mps2,
                    response_time_s=rho,
                )
                if safe > 0.0:
                    worst = max(worst, max(0.0, 1.0 - candidate.gap_m / safe))
            samples.add(worst, applicable=bool(moving))

        ttc_result, _, _ = evaluate_ttc(
            ego_footprint=ego.footprint,
            ego_velocity_xy=ego.velocity_xy,
            actors=actors,
            vertically_compatible_actor_ids=_vertical_actor_ids(ego, actors),
        )
        measurement.ttc.add(ttc_result.cost, applicable=ttc_result.applicable)
        if ttc_result.applicable:
            costs.append(ttc_result.cost)
        measurement.r2_macro.add(max(costs, default=0.0), applicable=bool(costs))

        if not road_rules:
            continue
        drivable = drivable_surface_for_ego(
            ego_footprint=ego.footprint,
            ego_position_xy=ego.position_xy,
            ego_position_z=ego.position_z,
            lanes=drivable_lanes,
        )
        if drivable.is_empty or not drivable.is_valid:
            continue
        offroad_result, _, _ = evaluate_offroad(
            ego_footprint=ego.footprint, drivable_surface=drivable
        )
        measurement.offroad.add(offroad_result.cost, applicable=offroad_result.applicable)
        surfaces = carriageway_surfaces_for_ego(
            ego_position_xy=ego.position_xy,
            ego_position_z=ego.position_z,
            route_tangent_xy=route.project(ego.position_xy, position_z=ego.position_z).tangent_xy,
            lanes=drivable_lanes,
        )
        carriageway_result, _, _ = evaluate_wrong_carriageway(
            ego_footprint=ego.footprint,
            aligned_surface=surfaces.aligned,
            opposing_surface=surfaces.opposing,
        )
        measurement.wrong_carriageway.add(
            carriageway_result.cost, applicable=carriageway_result.applicable
        )
        solid_boundaries = tuple(
            feature
            for feature in cache.map_feature_catalog.values()
            if feature.feature_class is MapFeatureClass.LANE_MARKING_SOLID
            and feature.elevation_m is not None
            and abs(feature.elevation_m - ego.position_z) <= VERTICAL_COMPATIBILITY_TOLERANCE_M
        )
        solid_result, _, _ = evaluate_solid_line(
            ego_footprint=ego.footprint, solid_boundaries=solid_boundaries
        )
        measurement.solid_line.add(solid_result.cost, applicable=solid_result.applicable)
    if measured_any:
        measurement.scenarios_measured += 1
    else:
        measurement.scenarios_skipped["no_valid_sdc_step"] += 1


@dataclass(frozen=True)
class WorkItem:
    """One record's work unit, sized to survive process serialization."""

    relative_path: str
    scenario_uid: str
    rho_values: tuple[float, ...]
    ego_brake_mps2: float
    step_stride: int
    road_rules: bool


_WORKER_DATA_ROOT: Path | None = None


def _init_worker(data_root: Path) -> None:
    global _WORKER_DATA_ROOT
    _WORKER_DATA_ROOT = data_root


def measure_work_item(item: WorkItem) -> Measurement:
    """Measure one record in isolation and return its partial accumulator.

    Each record is fully independent — the adapter, cache and per-step
    evaluations share no state across scenarios — so the work parallelizes
    exactly, and `Measurement.merge` recombines the partials without changing
    the result.
    """

    assert _WORKER_DATA_ROOT is not None, "worker was not initialized with a data root"
    measurement = Measurement(rho_values=item.rho_values)
    path = _WORKER_DATA_ROOT / item.relative_path
    if not path.exists():
        measurement.scenarios_skipped["missing_source_file"] += 1
        return measurement
    with path.open("rb") as handle:
        scenario = pickle.load(handle)
    try:
        measure_scenario(
            scenario,
            scenario_uid=item.scenario_uid,
            measurement=measurement,
            ego_brake_mps2=item.ego_brake_mps2,
            step_stride=item.step_stride,
            road_rules=item.road_rules,
        )
    except (ValueError, KeyError) as error:
        measurement.scenarios_skipped[f"error:{type(error).__name__}"] += 1
    return measurement


def selected_records(
    payload: Mapping[str, Any], *, split: str, source: str, limit: int | None
) -> list[Mapping[str, Any]]:
    records = [
        record
        for record in payload["records"]
        if record.get("source") == source
        and (split == "all" or record.get("split") == split)
        and record.get("rulebook_eligible", True)
    ]
    records.sort(key=lambda record: str(record.get("scenario_uid")))
    return records if limit is None else records[:limit]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--source", default="waymo")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--ego-brake-mps2", type=float, default=DEFAULT_EGO_BRAKE_MPS2)
    parser.add_argument(
        "--rho-sweep",
        default="1.0,0.75,0.5,0.4,0.3",
        help="Comma-separated RSS response times for the sensitivity sweep.",
    )
    parser.add_argument(
        "--no-road-rules",
        action="store_true",
        help="Measure only R2 (skips the per-step drivable-surface unions).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Worker processes; 1 runs in-process. Records are independent.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = json.loads(args.frozen_index.read_text(encoding="utf-8"))
    records = selected_records(payload, split=args.split, source=args.source, limit=args.limit)
    rho_values = tuple(float(value) for value in str(args.rho_sweep).split(",") if value.strip())
    measurement = Measurement(rho_values=rho_values)
    items = [
        WorkItem(
            relative_path=str(record["relative_path"]),
            scenario_uid=str(record["scenario_uid"]),
            rho_values=rho_values,
            ego_brake_mps2=float(args.ego_brake_mps2),
            step_stride=int(args.step_stride),
            road_rules=not args.no_road_rules,
        )
        for record in records
    ]
    workers = max(1, int(args.workers))
    if workers == 1:
        _init_worker(args.data_root)
        results: Iterable[Measurement] = (measure_work_item(item) for item in items)
        for index, partial in enumerate(results, start=1):
            measurement.merge(partial)
            if index % 100 == 0:
                print(f"... {index}/{len(items)} records", flush=True)
    else:
        # Shapely is single-threaded and each record is independent, so the work
        # scales with processes. Cap the per-process BLAS/OpenMP pools first:
        # 72 workers each spawning their own pool would oversubscribe the host.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(variable, "1")
        with multiprocessing.get_context("fork").Pool(
            processes=workers, initializer=_init_worker, initargs=(args.data_root,)
        ) as pool:
            for index, partial in enumerate(
                pool.imap_unordered(measure_work_item, items, chunksize=4), start=1
            ):
                measurement.merge(partial)
                if index % 100 == 0:
                    print(f"... {index}/{len(items)} records", flush=True)

    summary = {
        "configuration": {
            "split": args.split,
            "source": args.source,
            "records_selected": len(records),
            "step_stride": int(args.step_stride),
            "ego_brake_mps2": float(args.ego_brake_mps2),
            "production_response_time_s": 1.0,
            "road_rules_measured": not args.no_road_rules,
            "workers": workers,
        },
        "result": measurement.summary(),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
