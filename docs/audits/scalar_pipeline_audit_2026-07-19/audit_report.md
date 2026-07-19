# Frozen ScenarioNet Audit and Golden-Suite Proposal

Status: `METADATA AUDIT COMPLETE — LIVE DATASET VALIDATION PENDING`

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
vectorized reset/step paths. Rulebook v4.7 learner wiring remains intentionally
fail-closed because `ThesisScenarioEnv` does not yet expose the required live
`rulebook_v2_adapter`; no learner stage was started under a silent fallback.
The first isolated live-adapter increments now normalize finite MetaDrive actor
snapshots and Bullet contact onset records for one real PG and one real Waymo
fixed-sequence scenario. The isolated source-bound provider also mapped 14
current Waymo traffic-light states by their ScenarioNet physical IDs; unknown
states remain explicit. Persistent-contact wiring, static zones, and the
transition evaluator remain pending.
Read-only static-adapter probes on the actually loaded canonical content had
zero validation errors for one PG record, one Waymo record, and one Waymo
traffic-light record (two route-relevant signal controls). These are
representative probes, not dataset-wide live Rulebook validation.

## Reproducibility

The catalog fingerprint hashes the sorted tuple `(scenario_uid, source, split, primary_arm, runtime_index, relative_path)`. The golden-suite manifest includes the frozen-index digest, selection policy, source quotas, and the 48 source references. Re-running this command on the same index produces the same reference set.
