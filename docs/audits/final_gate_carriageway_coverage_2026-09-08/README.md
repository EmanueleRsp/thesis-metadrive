# Final-gate carriageway coverage, 2026-09-08

**Question.** `mission_success` is a swept front-bumper crossing of one frozen,
**finite** segment, while `R4` credits arc-length advance of an unconstrained
nearest-point projection. Nothing links the two. So: **at the goal cross-section,
is there same-direction drivable surface the gate does not cover?** If there is
not, the gap between reward and success is theoretical and needs no remedy.

**Scope.** All **3500** frozen records (1805 Waymo, 1695 PG), every split.
Nothing is simulated and no policy is involved: this is a property of the frozen
map and the frozen mission record.

## How it was produced

```
docker compose -p thesis-metadrive run --rm -T dev uv run --no-sync \
  python scripts/measure_final_gate_carriageway_coverage.py \
  --data-root /workspace/data/scenarionet \
  --frozen-index /workspace/thesis-metadrive/data/scenarionet/frozen/scenario_selection_index.json \
  --workers 24 \
  --output /workspace/outputs/final_gate_carriageway_coverage.json
```

About four minutes on 24 workers.

**The instrument is the builder's own arithmetic, not a reimplementation.**
`mission/builder.py` decomposes the goal cross-section — a ±100 m line along the
route normal, intersected with every vertically compatible same-direction lane
polygon, offsets merged at `FINAL_GATE_BUILDER_EPSILON_M = 0.01 m` — and freezes
**the single merged component containing offset 0**. Every *other* merged
component is, by construction, same-direction drivable surface the gate does not
intersect. The script re-runs that decomposition and keeps what the builder threw
away, importing the builder's constants and `_line_parts` rather than restating
them.

**It validates itself per record.** The reconstructed host component must equal
the frozen `final_gate_segment` to within the builder's own tolerance; a record
that fails is reported `unreconstructible` and excluded rather than counted.
**All 3500 reconstructed, maximum residual 3.5 × 10⁻¹² m** — machine precision.
So no result here can be an artifact of re-deriving the geometry differently.

## What it found

**The gate is normally wide enough, and always contains the goal.**

| gate segment length | min | p05 | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| metres | 2.68 | 3.97 | 7.46 | **10.26** | 14.06 | 65.95 | 200.0 |

The goal point is inside the gate on every record (`d_lo ≤ −1.337 m` always,
`d_hi ≥ +1.263 m` always). The median 10.26 m is about three lanes, so an
ordinary lane change usually still crosses.

**Same-direction surface the gate does not cover is common, but mostly far away.**

- **1534 / 3500 (43.8 %)** have at least one uncovered same-direction component;
- gap to the gate edge: median **16.55 m**, p25 1.63 m, p05 0.046 m, min 0.010 m;
- within one ego width (1.852 m): **426 (12.2 %)**; within one ego length
  (4.515 m): **532 (15.2 %)**;
- **zero** of the nearest components are bike-lane-only, so these are traffic
  lanes rather than a filter artifact of `LANE_*` admitting `LANE_BIKE_LANE`.

Lane types of the nearest uncovered component, over the 1534: `LANE_SURFACE_STREET`
1390, `LANE_SURFACE_UNSTRUCTURE` 101, `LANE_FREEWAY` 21, mixtures 22. Among the
426 adjacent ones: 350 street, 68 unstructured, 8 mixed or freeway. The
**unstructured** share matters for interpretation — `LANE_SURFACE_UNSTRUCTURE` is
drivable surface without lane structure (parking aprons, forecourts), so calling
those "a parallel road" would overstate them, while they do still count for the
out-of-road predicate, which admits any `LANE_*` feature.

**The tightest gaps are a merge-tolerance artifact, not a parallel road.** The
six smallest are all PG at **0.010–0.011 m**, i.e. a hair over
`FINAL_GATE_BUILDER_EPSILON_M`, and they sit beside gates that are *already* 11.5
to 34.7 m wide. Two adjacent lane polygons that are physically contiguous fail to
merge because the source geometry leaves a centimetre between them, so the gate
stops one strip short of a carriageway it otherwise spans. That is a bounded
defect of the tolerance, and it is not the failure mode the audit was looking for.

**The genuinely narrow gates are a small minority.**

| | records | share |
|---|---:|---:|
| a one-lane change to **either** side leaves the gate | **289** | 8.3 % |
| a one-lane change to **some** side leaves the gate | 1996 | 57.0 % |
| whole gate ≤ one lane width (3.5 m) | 95 | 2.7 % |
| less than one lane on **both** sides of the goal | 412 | 11.8 % |

## What it does not establish

**This measures the goal cross-section only.** It says that uncovered
same-direction surface exists *at the goal*; it does **not** say how far back
along the route that surface accompanies the mission, and therefore does **not**
establish that an ego could drive it while still banking the `R4` budget. That
requires walking the route backwards from `s_goal` and testing lateral offset
stability at each station — designed, not run.

It also says nothing about whether a trained policy ever goes there. The
diagnostic that would answer that, `route_outside_fraction`, is computed every
step and aggregated nowhere (`D14`).

## One more thing the audit surfaced

MetaDrive already ships a lateral route-departure termination —
`out_of_route_done`, which ends the episode when
`abs(navigation.current_lateral) > 10` m
(`third_party/metadrive/metadrive/envs/scenario_env.py:386-387`) — and **every
configuration in this repository sets it to `false`**
(`conf/env/scenarionet.yaml:44`, `conf/env/metadrive.yaml:12`, both native
presets, `conf/curriculum/scenario_acl.yaml:39`). That is deliberate and correct
as things stand: `current_lateral` comes from ScenarioNet's `TrajectoryNavigation`
reference lanes, not from the frozen mission route, and
`src/thesis_rl/envs/scene_context.py:83-91` demotes exactly those fields to
diagnostics because they describe deviation from a different route authority.
Worth recording because any future route-departure termination should be built on
the frozen mission, not by flipping this flag.

## Reading

`summary.json` carries the aggregate and the constants used; `per_record.csv` has
one row per record: `gate_lo_m`,
`gate_hi_m`, `reconstruction_residual_m`, component counts under both direction
cones (`cos > 0`, the builder's own, and `cos ≥ 0.5`, the repository's "aligned
carriageway" threshold from `drivable.py`), and for the nearest uncovered
component its interval, gap, lane types, and whether it contains an assigned-route
lane.
