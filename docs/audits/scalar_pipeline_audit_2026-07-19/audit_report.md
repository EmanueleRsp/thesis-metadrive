# Frozen ScenarioNet Audit and Golden-Suite Proposal

Status: `FINAL ALL-ON TD3 INTEGRATION VERIFIED — SCIENTIFIC PERFORMANCE NOT CLAIMED`

## Immutable Inputs

- Frozen index: `/home/e.respino/main/thesis/thesis-metadrive/data/scenarionet/frozen/scenario_selection_index.json`
- Frozen index SHA-256: `a37996eb9875524104c3b5423baf95d6dd53e16975ec8b3469c91e11e558285a`
- Index schema: `scenarionet_frozen_selection_v1`
- Index-created timestamp: `2026-07-18T11:09:10.685468+00:00`
- Deterministic catalog fingerprint: `919872129dccd1bfdbe3ac9b61c6a73fd33fb3eea6d92e31f9a42dc3b1405180`
- Live-validation root (read-only Docker mount): `/workspace/data/scenarionet`
- Protected-data policy: no source file, ScenarioDescription, split, tag, or selected reference was modified.

## Canonical Replay Alignment

The frozen index is the sole selection authority. Its declared catalog hash
`8d8a72b81db34fe41595114883adaddc1be3f3c1fc4e55754ec427f07b942801` equals the
SHA-256 of the mounted canonical `catalog/scenario_catalog.parquet`. The
catalog UID set equals the 3,500 index records, and the canonical runtime
mapping matches the catalog for every split: train 2,000, validation 500, and
test 1,000. The obsolete `runtime/frozen/`,
`catalog/scenario_catalog_frozen.parquet`, and
`splits/split_manifest_frozen.yaml` artifacts are absent. This alignment was
verified read-only in Docker; no source scenario was rewritten.

## Population And Targets

| split | source | actual | target | result |
| --- | --- | --- | --- | --- |
| train | waymo | 1000 | 1000 | match |
| train | pg | 1000 | 1000 | match |
| validation | waymo | 250 | 250 | match |
| validation | pg | 250 | 250 | match |
| test | waymo | 500 | 500 | match |
| test | pg | 500 | 500 | match |

| split | source | count |
| --- | --- | --- |
| train | waymo | 1000 |
| train | pg | 1000 |
| validation | waymo | 250 |
| validation | pg | 250 |
| test | waymo | 500 |
| test | pg | 500 |

| primary arm | Waymo | PG | total |
| --- | --- | --- | --- |
| A0_simple_low_traffic | 35 | 548 | 583 |
| A1_traffic | 140 | 444 | 584 |
| A2_junction | 141 | 443 | 584 |
| A3_complex_junction | 459 | 124 | 583 |
| A4_vru | 583 | 0 | 583 |
| A5_critical_mixed | 392 | 191 | 583 |

The selected population contains `3500` records. The approved near-uniform arm totals are 583 or 584; observed totals are represented above. No target deviation was found.

## Source Identity, Duplicates, And Leakage

- Duplicate scenario UIDs: `0`; duplicate scenario IDs: `0`.
- Waymo selected-shard entries recorded by the frozen index: `768`; PG generation identities recorded: `1750`.
- Duplicate Waymo original source-segment identities: `0`; duplicate PG `(profile, seed)` identities: `0`.
- Cross-split duplicate original source scenario identities: `0`; cross-split duplicate PG scenario identities: `0`.
- Waymo provenance shard/log IDs crossing splits: `370`. This is reported as provenance reuse, not split leakage: ADR-001 permits `source_log_id` grouping only when it proves a shared source group; the frozen metadata does not establish that proof.

## Metadata Validation Evidence

- Validation statuses: `{'valid': 3500}`; validation-warning records: `0`; selected exclusion/error records: `0`.
- Catalog-declared Rulebook eligibility: `3500/3500`; this is not a live Rulebook or source-content validation result.
- Signal reliability: `{'complete': 689, 'not_applicable': 2811}`.
- Catalog route metadata lists missing: `0`; route-source identities: `{'pg_sdc_offline_task_annotation': 1750, 'waymo_sdc_offline_task_annotation': 1750}`. This is not live assigned-route validity verification.
- Scenario horizons in catalog metadata: min `64`, max `501` decision steps.
- Frozen-source existence: `3,500/3,500` index references verified under the read-only Docker mount.
- Dataset-wide read-only structural validation passed for `3,500/3,500` records: pickle loadability, catalog/description horizon equality, nonempty tracks/maps, SDC trajectory length and finite values, nonempty valid mask, and assigned-route lane membership. The external issue CSV is empty. Live Rulebook eligibility, simulator semantics, and policy behavior remain pending.

## Final Golden-Suite Reference Manifest

The suite contains exactly eight train references per arm. Source allocation uses largest-remainder rounding over actual train presence, with the required at-least-one-source guard when feasible.

| primary arm | Waymo quota | PG quota | total |
| --- | --- | --- | --- |
| A0_simple_low_traffic | 0 | 8 | 8 |
| A1_traffic | 0 | 8 | 8 |
| A2_junction | 0 | 8 | 8 |
| A3_complex_junction | 8 | 0 | 8 |
| A4_vru | 8 | 0 | 8 |
| A5_critical_mixed | 8 | 0 | 8 |

The candidate was finalized as a reference-only suite after read-only inspection of all 48 selected `ScenarioDescription` pickle mappings. Every reference exists, has matching catalog length, nonempty tracks and map features, an SDC track, and assigned-route lanes present in map features. The final manifest and content evidence are under `golden_suite_content_validated/`; no source file was copied or modified. The final suite contains 24 PG and 24 Waymo references.

The companion candidate CSV remains the required selection coverage matrix. The content-evidence CSV adds actual map-feature types, traffic controls, and lane markings. These checks do not prove `ScenarioEnv` reset/step behavior, live Rulebook evaluation, policy termination/truncation, or semantic-policy behavior.

The canonical five-test ScenarioNet integration smoke subsequently passed in the
Docker GPU container, covering all split/runtime equality checks and PG/Waymo
vectorized reset/step paths. Rulebook v4.7 learner wiring now reaches the
deferred live adapter after reset; no learner stage is claimed under a silent
fallback.
The first isolated live-adapter increments now normalize finite MetaDrive actor
snapshots and Bullet contact onset records for one real PG and one real Waymo
fixed-sequence scenario. The isolated source-bound provider also mapped 14
current Waymo traffic-light states by their ScenarioNet physical IDs; unknown
states remain explicit. The live adapter increment now also includes persistent
Bullet-manifold contact
reconstruction, successor-graph preservation, fail-closed movement-key
derivation, an immutable episode-cache builder, and a source-neutral complete
transition evaluator. These are implementation-level Rulebook contracts and
are covered by the focused suite; the deferred adapter is installed on
`ThesisScenarioEnv` after reset. A direct canonical-runtime PG reset/step probe
reached the complete adapter and failed closed at RSS because no approved
calibration artifact was configured. This is an expected prerequisite failure,
not a dataset defect.
After loading the existing read-only calibration artifact from the mounted data
root and aligning the source elevation datum to MetaDrive's live `z=0` spawn
convention, the same complete Rulebook path passed one real PG and one real
Waymo reset/step (`2 passed`). The alignment translates only the common datum;
relative elevation and topology remain unchanged.
The live object-registry probe on a Waymo scenario tagged with VRU content
produced 12 unique finite other-actor snapshots (11 vehicles and one cyclist)
after one ego exclusion. This does not yet establish complete
pedestrian/static-object parity across the dataset; unknown public-registry
objects are rejected rather than silently dropped.
Read-only static-adapter probes on the actually loaded canonical content had
zero validation errors for one PG record, one Waymo record, and one Waymo
traffic-light record (two route-relevant signal controls). These are
representative probes, not dataset-wide live Rulebook validation.

The Docker learner diagnostics subsequently passed S1 (TD3/MLP, PER and
ACL off), S2 (TD3/LQ, PER and ACL off), S3 (TD3/LQ with the approved ACL core,
PER off), S4 (TD3/LQ with PER on, ACL off), S5 (TD3/LQ with ACL and PER on),
and diagnostic S6 (PPO/LQ with replay off). Each run used the canonical root,
finite source-backed observations/actions, short updates, evaluation, and
checkpoint publication. S3/S5 also emitted ACL arm probabilities and persisted
curriculum state. These are diagnostic runtime smokes only; they do not claim
policy quality, convergence, dataset-wide Rulebook eligibility, or thesis
performance.

The GPU overlay was then exercised explicitly with an S2 TD3/LQ smoke: the
planner reported `device=cuda`, Torch `2.9.1+cu128`, and GH200 execution; the
20-step evaluation and final checkpoint passed.

The short learner matrix was then repeated at 200 steps in the GPU overlay:
S1 TD3/MLP with ACL/PER off, S2 TD3/LQ with ACL/PER off, S3 TD3/LQ with the
approved ACL core and PER off, S4 TD3/LQ with PER on and ACL off, and S6
PPO/LQ with replay off. Every run completed with finite losses/actions,
evaluation, and checkpoints; S3 persisted ACL/MAB state. These remain
diagnostic runtime checks, not policy-quality or scientific results.

An additional 200-step S5 diagnostic was executed with the GPU overlay using
TD3/LQ, the approved ACL core, PER, and transition replay `n_steps=3`. The
run completed 200 environment steps with finite actor/critic updates,
evaluation, ACL/MAB state, replay configuration, and final/latest checkpoints.
The same run was resumed from `latest.zip` with RNG restoration and completed
the next 100 steps (global step 300), again producing finite updates,
evaluation, ACL state, and checkpoints. This is checkpoint/resume evidence for
one scalar configuration only; it is not a full algorithm/source parity or
scientific performance result.

The smokes exposed and fixed three repository defects: non-ego and node-only
Bullet callbacks are filtered/deferred to the current manifold; Bullet
single-point manifold access supports both indexed and no-argument bindings;
and off-road geometry unions every vertically compatible drivable lane rather
than returning an empty surface after the ego leaves a lane. Semantic builders
now consume the elevation-aligned Rulebook route, and ACL buffer persistence
creates its parent directory. No protected source file was changed.

## Reproducibility

The catalog fingerprint hashes the sorted tuple `(scenario_uid, source, split, primary_arm, runtime_index, relative_path)`. The golden-suite manifest includes the frozen-index digest, selection policy, source quotas, and the 48 source references. Re-running this command on the same index produces the same reference set.

## Final All-On Integration Verification

The final effective Hydra configuration used `run_profile=smoke`: canonical
`env=scenarionet`; strict provider with `allow_fallback=false`; semantic v1.1;
LQ encoder v1.0; Rulebook v4.7; `bounded_satisfaction_rank`; ScenarioNet ACL;
TD3-SB3; PER; transition replay `n_steps=3`; replay persistence; checkpoint and
RNG persistence. The ACL driver requires sequential execution, so
`vectorized=false` was used. Diagnostic-only overrides set `learning_starts=100`
and `batch_size=64` to exercise TD3/PER updates.

The corrected run completed 2,000 steps, 1,901 TD3 updates, PG and Waymo
episodes, finite `(2541,)` float32 observations, `(2,)` float32 actions, finite
Rulebook scalar rewards, finite algorithm-specific learning potential, ACL
MAB probabilities, checkpoints, and final evaluation. No provider fallback,
mutation, or mutation-generated child was observed.

Read-only Docker inspection found `PrioritizedNStepReplayBuffer`, `n_steps=3`,
2,000 stored transitions, finite raw priorities and sum-tree values, valid
`final.zip`, replay buffers, checkpoint pairs, and RNG state. Resume restored
ACL/MAB/buffer state, replay/PER and RNG, completed 500 additional steps to
global step 2,500, and recorded 11 scenario-buffer replay episodes plus 13
generated episodes before final evaluation. These are integration results only;
they do not claim convergence or scientific performance. SAC was not run because
TD3 is the requested primary learner and no authoritative final configuration
requires SAC.

## Step-Level Rulebook Trace Correction (2026-07-20)

The initial trace attempt exposed an implementation gap: Rulebook v4.7 uses
`RulebookV2MonitorWrapper`, but the diagnostic paths were previously wired only
to the legacy reward wrapper. The v2 wrapper now writes Rulebook margins,
component values, scalarization details, and scalar reward at every transition;
it also publishes the `scalar_rule_reward` key consumed by aggregate metrics.

The corrected all-on smoke run is stored under
`outputs/EXP_final_scalar_acl_td3_smoke_rulebook_trace_fixed_RP_smoke_CUR_scenario_acl_scenarionet_REW_scalar_reward/`.
It completed 2,000 steps and produced 2,072 records in both the Rulebook trace
and runtime trace. The records contain 1,819 PG transitions and 253 Waymo
transitions. Finiteness, four-margin shape, complete evaluation, and exact
scalarization recomputation checks passed with zero mismatches. Training and
final-evaluation CSV scalar reward fields are now populated.

The same trace then exposed a collision observability defect: five environment
episodes had `collision=true`, but no Rulebook collision onset. The cause was
ego resolution through `traffic_manager.ego_vehicle` instead of ScenarioEnv's
`env.agents` ownership boundary. This is corrected and covered by a live
adapter regression test. The earlier smoke remains valid for scalarization
logging but is not collision-conformance evidence; a post-fix smoke is pending.
