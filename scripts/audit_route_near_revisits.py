"""Re-audit the frozen canonical routes for projection jumps the R4 clip under-charges.

## Why this script exists

`docs/implementation/route_coordinate_driving_mission_v1.1_exec_plan.md` §11.1
records that the runtime's global-nearest route projection is equivalent to a
contiguous cursor **on the frozen population only**, and makes the re-audit a
standing constraint: *"Any regenerated or extended frozen index must re-run the
near-revisit audit ... before the runtime is trusted on it."* Until now that
constraint was not executable, because no script implemented the audit.

## What it measures, and why not the original threshold

The 2026-09-05 run asked whether two route portions with `|delta_s| > 15 m` come
within one lane width of each other, and answered no for all 3,500 missions.
That threshold was chosen for a different question — whether the projection can
skip a whole mission section — and it is the wrong one for the progress channel.

`R4` charges `Delta q = clip(Delta s / D_REF, -1, +1)` with
`D_REF = MISSION_PROGRESS_REFERENCE_SPEED_MPS * delta_t` = 2.2222 m at 10 Hz
(`rulebook/v2/components/progress.py`). The clip therefore starts mis-stating
the arc-length change at **one clip width**, not at 15 m, and the interval
between the two is uncovered: tight hairpins and roundabout entries live there.

Substituting 2.2222 m for 15 m in the original criterion would be degenerate.
Arc length is never shorter than the chord it subtends, so on a straight route
two points 3 m apart in plane are also 3 m apart in `s`, and a criterion of
`|delta_s| > 2.2222 m within one lane width` fires on every route in the index
while describing nothing. The threshold has to be paired with the planar
separation, and the pairing that has a physical reading is this one:

* two route points are `d` apart in plane and `a` apart in arc length;
* an ego at the first can reach the second in `n = max(1, ceil(d / D_REF))`
  steps, because `D_REF` is exactly one step of travel at the reference speed
  and a projection jump is a change between two consecutive states, so it costs
  at least one step however small `d` is;
* over `n` steps the channel can charge at most `n` in magnitude, since every
  step is clipped to `[-1, +1]`;
* the arc-length change the projection actually makes is worth `a / D_REF`.

So the quantity this script reports is

    on-route under-charge (channel units) = a / D_REF - max(1, ceil(d / D_REF))

and it is positive exactly when the channel cannot express the arc-length change
that the projection made. On a straight route `a == d`, so the under-charge is
never positive and the criterion is silent. On a fold it is `a / D_REF - 1` with
`a` unbounded, which is the ratchet
(`docs/audits/rulebook_architecture_2026-09-09/g6_ratchet.py`, `C50`): the
backward jump is charged `-1` and the ego keeps the difference as banked
progress it never undid.

### The second regime, and why the on-route figure alone would overstate safety

That first quantity assumes the ego travels the planar distance `d` between the
two portions, which is the honest way to get from one to the other. It is not
the only way, and it is not the one `g6_ratchet.py` demonstrates.

The projection is global-nearest, so between two route portions there is a
switch surface — the medial axis — on which the two are equidistant. An ego
sitting next to it changes branch by moving an arbitrarily small amount, and the
jump is then charged over **one** step whatever `d` is. What it costs is not
travel between the portions but a lateral excursion of about `d / 2` off the
route, plus the honest steps spent getting there and back.

So the same pair yields a second, larger figure:

    branch-switch under-charge = a / D_REF - 1     at a lateral excursion d / 2

Both are reported, per planar band, and the band is labelled with the excursion
it implies. Neither is the whole answer, because whether an ego may sit `d / 2`
off the route is a property of the drivable surface rather than of the route
geometry this script reads, and it is the open question `D14` records. This
instrument bounds the reward-side exposure and says what lateral reach each
bound costs; it does not decide reachability.

Reading the sign matters. A positive under-charge on the **forward** side loses
the ego credit it earned and is conservative; on the **backward** side it is a
gain the ego did not earn. The route geometry alone cannot say which direction a
policy would traverse a fold in, so this script reports the magnitude and the
geometry, and leaves the direction to the trajectory-level statistic
(`C51`).

## What it emits

A distribution, not a verdict. The standing constraint wants a pass/fail, but a
pass/fail cannot be read: ordinary curvature produces small positive
under-charges everywhere — cutting the inside of a bend genuinely advances the
centerline coordinate faster than the ego moves, which is the whole reason
ADR-035 sized its continuity bound at a *factor* of `2.0` rather than `1.0` —
while a fold produces a large one. Only the distribution separates the two, and
its shape is also what prices any threshold a continuity gate would need.

## Cost model

Read-only over the committed frozen index. Nothing is simulated, no map is
loaded, and no policy is involved: `canonical_route_points_xyz` is stored in the
index itself. Routes are rebuilt through `RoutePolyline`, so the consolidation
and the XY arc length are the runtime's own, not a re-derivation.

## Exactness

For two line segments in the plane the minimum distance is attained at an
endpoint of one of them unless they cross, so scanning every vertex against
every segment is exact for all non-crossing pairs. Where two segments do cross
the true distance is `0` and this scan reports at most one segment length
instead. That error is one-sided in the safe direction — it can only *under*
report exposure — and it is bounded by the longest segment in the index, which
the report prints.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from thesis_rl.rulebook.v2.components.progress import MISSION_PROGRESS_REFERENCE_SPEED_MPS
from thesis_rl.rulebook.v2.geometry.route import RoutePolyline

# The control interval the reference speed is normalised over. `progress.py`
# receives `delta_t_s` from the transition rather than owning it, so it is stated
# here with the configuration it comes from rather than imported.
CONTROL_TIMESTEP_S = 0.1
# One clip width: the arc-length advance that saturates `Delta q` in one step.
D_REF_M = MISSION_PROGRESS_REFERENCE_SPEED_MPS * CONTROL_TIMESTEP_S

# Cumulative planar-separation bands the under-charge distribution is reported
# over. `D_REF_M` is one step of travel; 3.5 m is the lane width both static
# adapters fall back to, and the "one lane width" of the original criterion;
# 6.0 m is the original criterion's own proximity threshold, kept so the two
# runs are comparable.
PROXIMITY_BANDS_M = (D_REF_M, 3.5, 6.0, 10.0, math.inf)

# The original criterion, restated so this instrument can be validated against
# the run it replaces.
LEGACY_ARC_THRESHOLD_M = 15.0
LEGACY_PROXIMITY_BANDS_M = (3.5, 6.0)

# Multiples of one clip width the arc separation is bucketed at. `2.0` is
# ADR-035's `ROUTE_CONTINUITY_JUMP_FACTOR`, i.e. the largest jump that decision
# considered legitimate; anything past it is geometry that bound would have
# rejected.
ARC_MULTIPLES = (1.0, 2.0, 3.0, 5.0, 10.0)

# Vertex-by-segment pairs held in memory at once. Bounds the peak allocation of
# the scan independently of route length.
CHUNK_PAIRS = 2_000_000

# An under-charge is called positive only above this, so that a route whose
# worst pair is exactly at the clip width is not reported as exposed on rounding
# alone. Well below any physically meaningful figure: one thousandth of a clip
# width is 2.2 mm of arc length.
UNDERCHARGE_EPSILON = 1.0e-3


@dataclass
class RouteAudit:
    """Per-route scalars. Pairs are reduced during the scan and never kept."""

    scenario_uid: str
    source: str
    split: str
    route_length_m: float
    point_count: int
    max_segment_length_m: float
    # Maximum on-route under-charge in each cumulative planar band, and the
    # (d, a) that attains it in the one-step band.
    band_max_undercharge: tuple[float, ...]
    # Fold excess `a - d` in metres, per band: how much longer the route is
    # between two portions than the straight line between them. Zero on a
    # straight route, small under ordinary curvature, and the whole loop on a
    # fold. This is the band-free discriminator; the arc and planar separation
    # at the same pair are kept so the branch-switch under-charge can be read
    # off it.
    band_max_fold_excess_m: tuple[float, ...]
    band_fold_excess_arc_m: tuple[float, ...]
    band_fold_excess_planar_m: tuple[float, ...]
    one_step_max_arc_m: float
    one_step_max_undercharge_d_m: float
    # Arc-separation histogram at one step of planar travel, as counts of
    # vertex/segment pairs whose arc separation exceeds each multiple of D_REF.
    one_step_arc_multiple_counts: tuple[int, ...]
    # Smallest planar separation between two portions more than
    # `LEGACY_ARC_THRESHOLD_M` apart in arc length. `inf` when no such pair
    # exists. This is the original criterion's quantity.
    legacy_min_planar_m: float
    error: str = ""


def _route_arrays(points_xyz: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Return consolidated XY vertices and their XY arc length.

    The polyline is built through `RoutePolyline`, so consolidation, the
    collapsed-segment rejection and the vertical-compatibility rejection are the
    runtime's. The arc length is then recomputed from the consolidated vertices
    with the same `hypot` accumulation the constructor uses, which reproduces its
    private `_segment_starts_m` without reaching into it.
    """

    polyline = RoutePolyline(points_xyz=tuple(tuple(float(v) for v in p) for p in points_xyz))
    xy = np.asarray([(p[0], p[1]) for p in polyline.points_xyz], dtype=np.float64)
    steps = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    s = np.concatenate(([0.0], np.cumsum(steps)))
    return xy, s


def _audit_route(
    *,
    scenario_uid: str,
    source: str,
    split: str,
    route_length_m: float,
    points_xyz: Sequence[Sequence[float]],
) -> RouteAudit:
    try:
        xy, s = _route_arrays(points_xyz)
    except ValueError as exc:
        return RouteAudit(
            scenario_uid=scenario_uid,
            source=source,
            split=split,
            route_length_m=route_length_m,
            point_count=len(points_xyz),
            max_segment_length_m=float("nan"),
            band_max_undercharge=tuple(-math.inf for _ in PROXIMITY_BANDS_M),
            band_max_fold_excess_m=tuple(-math.inf for _ in PROXIMITY_BANDS_M),
            band_fold_excess_arc_m=tuple(0.0 for _ in PROXIMITY_BANDS_M),
            band_fold_excess_planar_m=tuple(0.0 for _ in PROXIMITY_BANDS_M),
            one_step_max_arc_m=0.0,
            one_step_max_undercharge_d_m=float("nan"),
            one_step_arc_multiple_counts=tuple(0 for _ in ARC_MULTIPLES),
            legacy_min_planar_m=math.inf,
            error=str(exc),
        )

    start = xy[:-1]
    end = xy[1:]
    delta = end - start
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    unit = delta / lengths[:, None]
    s_start = s[:-1]

    band_max = [-math.inf] * len(PROXIMITY_BANDS_M)
    band_fold_max = [-math.inf] * len(PROXIMITY_BANDS_M)
    band_fold_arc = [0.0] * len(PROXIMITY_BANDS_M)
    band_fold_planar = [0.0] * len(PROXIMITY_BANDS_M)
    one_step_max_arc = 0.0
    one_step_arg_d = float("nan")
    one_step_best_undercharge = -math.inf
    multiple_counts = [0] * len(ARC_MULTIPLES)
    legacy_min_planar = math.inf

    vertices = xy.shape[0]
    segments = lengths.shape[0]
    rows = max(1, CHUNK_PAIRS // max(1, segments))
    for lo in range(0, vertices, rows):
        hi = min(vertices, lo + rows)
        probe = xy[lo:hi]
        # Closest point on every segment to every probe vertex, exact for all
        # non-crossing pairs (see the module docstring on exactness).
        offset_x = probe[:, 0][:, None] - start[:, 0][None, :]
        offset_y = probe[:, 1][:, None] - start[:, 1][None, :]
        along = np.clip(
            (offset_x * unit[:, 0][None, :] + offset_y * unit[:, 1][None, :]),
            0.0,
            lengths[None, :],
        )
        near_x = start[:, 0][None, :] + along * unit[:, 0][None, :]
        near_y = start[:, 1][None, :] + along * unit[:, 1][None, :]
        planar = np.hypot(probe[:, 0][:, None] - near_x, probe[:, 1][:, None] - near_y)
        arc = np.abs(s[lo:hi][:, None] - (s_start[None, :] + along))

        # Steps needed to cover the planar separation, and therefore the largest
        # magnitude the clipped channel can charge over the traversal. The floor
        # of one step is not cosmetic: a projection jump is a change between two
        # consecutive states, so it always costs at least one step and is always
        # charged at least once, however small the planar separation is.
        chargeable = np.maximum(1.0, np.ceil(planar / D_REF_M))
        undercharge = arc / D_REF_M - chargeable

        # How much longer the route is between two portions than the straight
        # line between them. This is what separates a fold from curvature, and
        # it needs no threshold of its own: it is identically zero on a straight
        # route and equal to the whole loop on a closed fold.
        fold_excess = arc - planar

        for index, bound in enumerate(PROXIMITY_BANDS_M):
            within = None if bound == math.inf else planar <= bound
            selected = undercharge if within is None else np.where(within, undercharge, -math.inf)
            candidate = float(selected.max()) if selected.size else -math.inf
            if candidate > band_max[index]:
                band_max[index] = candidate
            folded = fold_excess if within is None else np.where(within, fold_excess, -math.inf)
            if folded.size:
                flat_fold = int(folded.argmax())
                candidate_fold = float(folded.flat[flat_fold])
                if candidate_fold > band_fold_max[index]:
                    band_fold_max[index] = candidate_fold
                    band_fold_arc[index] = float(arc.flat[flat_fold])
                    band_fold_planar[index] = float(planar.flat[flat_fold])

        one_step = planar <= D_REF_M
        if one_step.any():
            arc_one_step = np.where(one_step, arc, 0.0)
            candidate_arc = float(arc_one_step.max())
            if candidate_arc > one_step_max_arc:
                one_step_max_arc = candidate_arc
            under_one_step = np.where(one_step, undercharge, -math.inf)
            flat = int(under_one_step.argmax())
            candidate_under = float(under_one_step.flat[flat])
            if candidate_under > one_step_best_undercharge:
                one_step_best_undercharge = candidate_under
                one_step_arg_d = float(planar.flat[flat])
            for index, multiple in enumerate(ARC_MULTIPLES):
                multiple_counts[index] += int(np.count_nonzero(arc_one_step > multiple * D_REF_M))

        legacy = arc > LEGACY_ARC_THRESHOLD_M
        if legacy.any():
            candidate_legacy = float(np.where(legacy, planar, math.inf).min())
            if candidate_legacy < legacy_min_planar:
                legacy_min_planar = candidate_legacy

    return RouteAudit(
        scenario_uid=scenario_uid,
        source=source,
        split=split,
        route_length_m=route_length_m,
        point_count=vertices,
        max_segment_length_m=float(lengths.max()),
        band_max_undercharge=tuple(band_max),
        band_max_fold_excess_m=tuple(band_fold_max),
        band_fold_excess_arc_m=tuple(band_fold_arc),
        band_fold_excess_planar_m=tuple(band_fold_planar),
        one_step_max_arc_m=one_step_max_arc,
        one_step_max_undercharge_d_m=one_step_arg_d,
        one_step_arc_multiple_counts=tuple(multiple_counts),
        legacy_min_planar_m=legacy_min_planar,
    )


_RECORDS: list[Mapping[str, Any]] = []


def _initializer(records: list[Mapping[str, Any]]) -> None:
    global _RECORDS
    _RECORDS = records


def _audit_index(position: int) -> RouteAudit:
    record = _RECORDS[position]
    mission = record["driving_mission"]
    return _audit_route(
        scenario_uid=str(record["scenario_uid"]),
        source=str(record.get("source", "")),
        split=str(record.get("split", "")),
        route_length_m=float(record.get("route_length_m", float("nan"))),
        points_xyz=mission["canonical_route_points_xyz"],
    )


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _describe(values: Sequence[float]) -> str:
    if not values:
        return "no routes"
    return (
        f"p50 {_quantile(values, 0.50):+8.3f}  p90 {_quantile(values, 0.90):+8.3f}  "
        f"p99 {_quantile(values, 0.99):+8.3f}  max {max(values):+8.3f}"
    )


def _report(audits: Sequence[RouteAudit], *, index_path: Path, top: int) -> None:
    failed = [audit for audit in audits if audit.error]
    usable = [audit for audit in audits if not audit.error]

    print(f"frozen index      {index_path}")
    print(f"records audited   {len(audits)}  ({len(usable)} usable, {len(failed)} rejected)")
    print(
        f"D_REF             {D_REF_M:.4f} m  "
        f"= {MISSION_PROGRESS_REFERENCE_SPEED_MPS:.4f} m/s * {CONTROL_TIMESTEP_S} s"
    )
    if usable:
        longest_segment = max(audit.max_segment_length_m for audit in usable)
        print(
            f"longest segment   {longest_segment:.4f} m  "
            "(bounds the planar over-report on a crossing pair; see docstring)"
        )
    for audit in failed:
        print(f"  REJECTED {audit.scenario_uid}: {audit.error}")
    print()

    print("Cross-validation against the 2026-09-05 run")
    print(
        f"  criterion: two portions more than {LEGACY_ARC_THRESHOLD_M:.0f} m apart in arc length,"
    )
    print("             closer than the stated planar separation")
    for bound in LEGACY_PROXIMITY_BANDS_M:
        hits = [audit for audit in usable if audit.legacy_min_planar_m < bound]
        print(f"    within {bound:5.2f} m   {len(hits):5d} routes")
    with_any = [audit for audit in usable if math.isfinite(audit.legacy_min_planar_m)]
    if with_any:
        closest = min(audit.legacy_min_planar_m for audit in with_any)
        print(f"    routes with any such pair {len(with_any)}, closest approach {closest:.3f} m")
    print()

    print("Under-charge  a/D_REF - max(1, ceil(d/D_REF))  in channel units, by planar band")
    print("  a positive value is arc-length change the clipped channel cannot express")
    for index, bound in enumerate(PROXIMITY_BANDS_M):
        label = "any" if bound == math.inf else f"<= {bound:5.2f} m"
        values = [
            audit.band_max_undercharge[index]
            for audit in usable
            if math.isfinite(audit.band_max_undercharge[index])
        ]
        positive = sum(1 for value in values if value > UNDERCHARGE_EPSILON)
        print(
            f"  {label:>12}   routes {len(values):5d}   positive {positive:5d}   {_describe(values)}"
        )
    print()

    print("Fold excess  a - d  in metres, by planar band, and the branch-switch cost it buys")
    print("  an ego near the medial axis between two portions changes branch in one step,")
    print("  so the whole arc separation is charged once; the price is the lateral excursion")
    print("  and the fold excess is what distinguishes a fold from ordinary curvature")
    print(
        f"  {'band':>12} {'excursion':>12} {'p50':>9} {'p99':>9} {'max':>9} "
        f"{'at a':>9} {'at d':>8} {'a/D_REF-1':>10}"
    )
    for index, bound in enumerate(PROXIMITY_BANDS_M):
        label = "any" if bound == math.inf else f"<= {bound:5.2f} m"
        excursion = "whole route" if bound == math.inf else f"{bound / 2.0:.2f} m"
        values = [
            audit.band_max_fold_excess_m[index]
            for audit in usable
            if math.isfinite(audit.band_max_fold_excess_m[index])
        ]
        if not values:
            continue
        worst = max(
            usable,
            key=lambda audit: (
                audit.band_max_fold_excess_m[index]
                if math.isfinite(audit.band_max_fold_excess_m[index])
                else -math.inf
            ),
        )
        arc_at_worst = worst.band_fold_excess_arc_m[index]
        print(
            f"  {label:>12} {excursion:>12} {_quantile(values, 0.50):9.3f} "
            f"{_quantile(values, 0.99):9.3f} {max(values):9.3f} "
            f"{arc_at_worst:9.3f} {worst.band_fold_excess_planar_m[index]:8.3f} "
            f"{arc_at_worst / D_REF_M - 1.0:10.3f}"
        )
    print()

    print("At one step of planar travel (d <= D_REF), arc separation in clip widths")
    print("  pair counts, summed over routes; the ADR-035 continuity factor is 2.0")
    for index, multiple in enumerate(ARC_MULTIPLES):
        pairs = sum(audit.one_step_arc_multiple_counts[index] for audit in usable)
        routes = sum(1 for audit in usable if audit.one_step_arc_multiple_counts[index] > 0)
        print(f"    a > {multiple:5.1f} * D_REF   pairs {pairs:9d}   routes {routes:5d}")
    print()

    for key, label in (("source", "source"), ("split", "split")):
        print(f"Worst one-step under-charge by {label}")
        groups: dict[str, list[RouteAudit]] = {}
        for audit in usable:
            groups.setdefault(getattr(audit, key), []).append(audit)
        for name in sorted(groups):
            values = [
                audit.band_max_undercharge[0]
                for audit in groups[name]
                if math.isfinite(audit.band_max_undercharge[0])
            ]
            positive = sum(1 for value in values if value > UNDERCHARGE_EPSILON)
            print(
                f"  {name:12s} routes {len(groups[name]):5d}  positive {positive:5d}  "
                f"{_describe(values)}"
            )
        print()

    ranked = sorted(
        (audit for audit in usable if math.isfinite(audit.band_max_undercharge[0])),
        key=lambda audit: audit.band_max_undercharge[0],
        reverse=True,
    )[:top]
    print(f"Top {len(ranked)} routes by one-step under-charge")
    print(
        f"  {'scenario_uid':44s} {'source':7s} {'split':11s} "
        f"{'under':>8s} {'arc_m':>9s} {'d_m':>7s} {'len_m':>8s}"
    )
    for audit in ranked:
        print(
            f"  {audit.scenario_uid:44s} {audit.source:7s} {audit.split:11s} "
            f"{audit.band_max_undercharge[0]:+8.3f} {audit.one_step_max_arc_m:9.3f} "
            f"{audit.one_step_max_undercharge_d_m:7.3f} {audit.route_length_m:8.2f}"
        )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-index",
        type=Path,
        default=Path("data/scenarionet/frozen/scenario_selection_index.json"),
    )
    parser.add_argument("--split", default="all")
    parser.add_argument("--source", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    index = json.loads(args.frozen_index.read_text(encoding="utf-8"))
    records = [
        record
        for record in index["records"]
        if (args.split == "all" or record.get("split") == args.split)
        and (args.source == "all" or record.get("source") == args.source)
        and record.get("driving_mission", {}).get("canonical_route_points_xyz")
    ]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print("no records selected", file=sys.stderr)
        return 2

    if args.workers > 1:
        with multiprocessing.Pool(
            processes=args.workers, initializer=_initializer, initargs=(records,)
        ) as pool:
            audits = pool.map(_audit_index, range(len(records)), chunksize=8)
    else:
        _initializer(records)
        audits = [_audit_index(position) for position in range(len(records))]

    _report(audits, index_path=args.frozen_index, top=args.top)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "frozen_index": str(args.frozen_index),
                    "selection_hash": index.get("selection_hash"),
                    "catalog_hash": index.get("catalog_hash"),
                    "d_ref_m": D_REF_M,
                    "control_timestep_s": CONTROL_TIMESTEP_S,
                    "reference_speed_mps": MISSION_PROGRESS_REFERENCE_SPEED_MPS,
                    "proximity_bands_m": [
                        None if bound == math.inf else bound for bound in PROXIMITY_BANDS_M
                    ],
                    "arc_multiples": list(ARC_MULTIPLES),
                    "legacy_arc_threshold_m": LEGACY_ARC_THRESHOLD_M,
                    "routes": [
                        {
                            "scenario_uid": audit.scenario_uid,
                            "source": audit.source,
                            "split": audit.split,
                            "route_length_m": audit.route_length_m,
                            "point_count": audit.point_count,
                            "max_segment_length_m": audit.max_segment_length_m,
                            "band_max_undercharge": [
                                None if not math.isfinite(value) else value
                                for value in audit.band_max_undercharge
                            ],
                            "band_max_fold_excess_m": [
                                None if not math.isfinite(value) else value
                                for value in audit.band_max_fold_excess_m
                            ],
                            "band_fold_excess_arc_m": list(audit.band_fold_excess_arc_m),
                            "band_fold_excess_planar_m": list(audit.band_fold_excess_planar_m),
                            "one_step_max_arc_m": audit.one_step_max_arc_m,
                            "one_step_max_undercharge_d_m": (
                                None
                                if not math.isfinite(audit.one_step_max_undercharge_d_m)
                                else audit.one_step_max_undercharge_d_m
                            ),
                            "one_step_arc_multiple_counts": list(
                                audit.one_step_arc_multiple_counts
                            ),
                            "legacy_min_planar_m": (
                                None
                                if not math.isfinite(audit.legacy_min_planar_m)
                                else audit.legacy_min_planar_m
                            ),
                            "error": audit.error,
                        }
                        for audit in sorted(audits, key=lambda item: item.scenario_uid)
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
