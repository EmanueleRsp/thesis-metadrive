# Specification amendment: the posted speed limit enters the LiDAR observation

## Metadata

- Feature: `obs_lidar_posted_speed_limit`
- Specification ID: `OBS-LIDAR-V2.0.2`
- Version: `2.0.2`
- Status: `APPROVED`
- Date: `2026-09-01`
- Amends: `docs/specifications/observation_lidar_v2.0_specification.md`, the
  causal frame contract and the stacked dimensionality. Frame stacking, the
  21-step window and its derivation from `DASHED_TCAP_S`, the masking rule, the
  ray counts (240/12/12), the nearby-vehicle count and the `ENC-V1.4`
  tokenization contract apart from the frame width are unchanged.
  `OBS-LIDAR-V2.0.1`'s traffic-light limitation stands unaffected.
- Required by: `RULEBOOK-V5.1` §6, which carries `RULEBOOK-V5.0` §7
  (`REQ-RB5-OBS-01`) unchanged.
- Related ADR: `docs/decisions/ADR-068-speed-limit-normative-sub-rule.md`
- ExecPlan: `docs/implementation/rulebook_v5.1_six_level_hierarchy_exec_plan.md`
  (`RB51`, milestone `M4`)
- Approval evidence: `DEC-RB51-001`, approved 2026-08-20, option (a).
- Authoritative: `YES` for §2 and §3 below; not authoritative for anything else.

## 1. Context

`REQ-RB5-OBS-01` requires the posted limit in **both** observation paths. The
LiDAR arm is a baseline the comparison depends on, and a baseline that cannot see
a rule it is charged for is not measuring the same task.

## 2. Amended frame contract

The causal frame goes from the previously frozen **308** to **310** values. The
two appended sit immediately after the six ego-state values, before the route
navigation block:

| index | value | encoding |
|---|---|---|
| 6 | posted speed limit of the ego's **associated route lane** | `clip(v_limit_kmh / 120, 0, 1)`, the same scale the ego speed channel uses |
| 7 | availability flag | `1.0` when a posted limit is present, `0.0` otherwise |

Consequently the stacked dimension changes from `308 * 21 + 21 = 6489` to
`310 * 21 + 21 = **6531**`, and the `ENC-V1.4` frame projection widens
accordingly. **Checkpoint compatibility is intentionally broken.**

The rationale for two values rather than a sentinel, and for normalizing by the
speed scale already in use, is `OBS-V1.3.1` §2 and is not restated.

## 3. The limit is read from the map record, never from the simulator

MetaDrive's own `lane.speed_limit` is **not** read, and this is the load-bearing
part of the amendment rather than an implementation note.

On PG that attribute is whichever default the lane constructor happened to hold —
`1000` for lanes built directly, `20` for lanes built through
`create_pg_block_utils` — exported verbatim under a `speed_limit_kmh` key while
MetaDrive's own blocks document those numbers in **m/s**. On Waymo it arrives
through `ScenarioLane`, whose `MAX_SPEED_LIMIT` cap ADR-068 also prohibits.

The LiDAR frame builder therefore resolves the limit through the **rulebook's**
`associate_route_lane` + `associated_speed_limit_mps`, from the static adapter's
route lanes, exactly as the semantic path does. Absent route lanes, an
unresolvable association, or a lane without real-map provenance all produce the
same unavailable encoding, which is the condition that makes the sub-rule
inapplicable.

On the PG panel the feature is unavailable on every step.

## 4. What this does not change

`OBS-LIDAR-V2.0.1` stands: the LiDAR arm remains blind to traffic lights, and
this amendment neither worsens nor repairs that. The ray-noise wrapper, the
native-noise prohibition, and the zero-fill-and-flag treatment of absent frames
at episode start are untouched.
