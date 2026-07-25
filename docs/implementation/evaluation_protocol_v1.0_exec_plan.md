# ExecPlan: Evaluation And Algorithm Comparison Protocol v1.0

## 1. Metadata

- Feature: Implementation of the evaluation and algorithm comparison protocol
- Plan ID: `EVAL-PROTOCOL-EXEC-001`
- Authoritative specification: `docs/specifications/evaluation_protocol_v1.0_specification.md`,
  ID `EVAL-PROTOCOL`, version `1.0`, `Status: APPROVED`, `Authoritative: YES`
  (approved by the user on 2026-07-24)
- Status: `IN_PROGRESS` (Milestones 1--6, 8, 9, 10 fully implemented and
  tested, including Milestone 5's PPO overshoot path now live-verified
  against a real SB3 PPO run, Milestone 2's real validation/test manifests
  generated against the real ScenarioNet catalog with per-run hash
  recording, and Milestone 10's applicability-aware R1--R3 seed-level
  aggregation (`REQ-008`); one requirement-level gap remains out of this
  ExecPlan's original milestone scope: `REQ-016`'s full end-to-end analysis
  regeneration against real aggregated multi-condition artifacts (not run);
  see §15 for the full reconciliation)
- Created: 2026-07-24
- Last updated: 2026-07-25 (Milestone 10, `REQ-008` applicability-aware
  aggregation, added as a new milestone outside the original scope; full
  repository test suite re-run, 970/974 passing, the same 4 pre-existing
  unrelated failures as the prior pass)
- Branch: `scenarionet-implementation` (current working branch at plan creation;
  confirm before starting a milestone)
- Related ADRs: `docs/decisions/ADR-018-parallel-evaluation-and-test.md`,
  `docs/decisions/ADR-019-asynchronous-evaluation-queue.md`,
  `docs/decisions/ADR-020-evaluation-video-diagnostics.md`,
  `docs/decisions/ADR-021-source-bounded-reactive-traffic.md`,
  `docs/decisions/ADR-024-runtime-scenario-data-abort.md`
- Owner: unassigned

## 2. Objective And Scope

### Objective

Make the current repository conform to `EVAL-PROTOCOL` v1.0 so that an official
`thesis`-profile run produces, and the analysis pipeline consumes, exactly the
artifacts the specification requires: a target-budget-aware PPO stop condition
that never discards a partial rollout (`DEC-015`), a frozen/hashed evaluation
panel, persisted checkpoint identity, persisted data-abort coverage, complete
reproducibility metadata, an explicit three-way run disposition, and a core
comparison report that reports raw seed values/mean/SD only (no 95% CI, no
10-seed assumption) with the R1--R3/R4 metric split intact.

Success is recognized when: (a) every `TEST-*` case in §9 passes; (b) a
representative `run_profile=smoke` PPO/TD3/SAC run each produce a conformant
`run_metadata.yaml`, panel manifest reference, and checkpoint identity; (c)
`make analysis` (or the chosen canonical command, `DEC-EP-002`) regenerates the
core `REQ-016` report deterministically from a small fixture run registry
without computing a confidence interval; (d) the mandatory regression matrix
in §9 passes; (e) `docs/project_index.md` is updated to `VERIFIED` only after
this reconciliation, per `docs/engineering_workflow.md`.

### In Scope

- `REQ-002`/`DEC-015`: PPO end-of-budget atomic-rollout-boundary stop condition.
- `REQ-004`/`DEC-005`: persisted, hashed, deduplicated validation/test panel
  manifest.
- `REQ-006`: checkpoint path/hash/role persistence for the official `final.zip`
  consumption event.
- `REQ-007`/`DEC-013`: primary-metric reconciliation, R1--R3 vs. R4 split, in
  the analysis pipeline.
- `REQ-009`/`DEC-003`: removal of the `1.96 * s / sqrt(n)` CI convention and any
  10-seed assumption from the analysis pipeline; raw-values/mean/SD-only
  reporting.
- `REQ-011`: explicit three-way run disposition (infrastructure failure /
  non-convergent / condition-attributable reproducible failure) in the run
  registry.
- `REQ-012`: persistence of the already-computed `data_abort_coverage` to
  per-run artifacts.
- `REQ-014`: reconciliation of the existing qualitative-manifest/video-selection
  code against the post-hoc, four-category scheme.
- `REQ-015`: completion of reproducibility metadata (dirty-tree flag,
  dependency versions, additional specification identities).
- `REQ-016`/`DEC-010`: reconciliation of the existing `run_analysis.py`
  pipeline against the minimal core-report requirement, including relabeling
  or gating the pre-existing factor-effect (old reward x curriculum) tables as
  optional/ablation-only per `REQ-001`/`REQ-018`.
- `REQ-018`: explicit ablation-block tagging for the pre-existing
  `conf/presets/td3/*_curr*/*_no_curr.yaml` matrix.

### Out Of Scope

- Any change to PPO, TD3, or SAC algorithmic semantics beyond the exact
  end-of-budget stop condition required by `DEC-015`.
- Any change to Rulebook, scalarization, observation, encoder, ACL, or
  transition-replay formulas or semantics (`EVAL-PROTOCOL` §2 Out Of Scope).
- `REQ-010` (paired seed differences) and any other item `EVAL-PROTOCOL`
  classifies as `Optional Or Deferred` — implemented only if explicitly
  requested later, and then as a separate milestone.
- Creating a new lexicographic/distributional algorithm or `EXTENSION-ALGORITHM-01`
  condition (`DEC-006` explicitly defers this).
- Deleting `docs/protocols/algorithm_comparison_protocol.md`,
  `csv_evaluation_objectives.md`, or `live_eval_video_protocol.md` — they
  remain retained historical/supporting material per `DEC-008`.

### Compatibility Constraints

- `run_metadata.yaml`, `final_eval.csv`, and `eval_episodes.csv` schemas may
  only gain fields (additive-only), per the existing convention documented in
  `docs/protocols/csv_evaluation_objectives.md` ("Principles" section) and
  preserved by `DEC-008`.
- Existing historical runs are not reclassified as official v1.0 runs
  (`EVAL-PROTOCOL` §11.9); this plan does not migrate or backfill old
  `run_metadata.yaml`/CSV artifacts.
- Checkpoint compatibility validation (`REQ-RLB-019`/`REQ-RLB-020` in
  `RL-BASELINES`) must not be weakened by adding new manifest fields.

## 3. Authoritative Requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-002` | Target budget; algorithm-valid atomic collection boundary; PPO overshoot to 1,501,184 transitions (`DEC-015`) | `EVAL-PROTOCOL` §6 REQ-002, §8.5, §9 |
| `REQ-004` | Frozen, hashed, deduplicated validation (100) and test (300) panels, identical across conditions/seeds | `EVAL-PROTOCOL` §6 REQ-004 |
| `REQ-006` | `final.zip` is the sole official checkpoint; identity (path, hash, role) recorded | `EVAL-PROTOCOL` §6 REQ-006, §7.8 |
| `REQ-007` | Primary metrics: success/route-completion/collision/out-of-road plus R1--R3 macro-rule aggregates and the R4 progress margin, kept structurally distinct | `EVAL-PROTOCOL` §6 REQ-007, §7.2, §7.3 |
| `REQ-008` | Seed-level metrics computed only over valid episodes and, for R1--R3, only over applicable steps | `EVAL-PROTOCOL` §6 REQ-008, §7.2 |
| `REQ-009` | Cross-seed reporting: raw values, mean, SD only; no CI/bootstrap/significance | `EVAL-PROTOCOL` §6 REQ-009, §7.4 |
| `REQ-011` | Three-way manual run disposition (infrastructure failure / non-convergent / condition-attributable reproducible failure) | `EVAL-PROTOCOL` §6 REQ-011 |
| `REQ-012` | Data-abort attempted/valid/aborted/coverage persisted to run artifacts | `EVAL-PROTOCOL` §6 REQ-012 |
| `REQ-014` | Post-hoc qualitative selection, four categories, shared `scenario_uid` for direct comparisons | `EVAL-PROTOCOL` §6 REQ-014 |
| `REQ-015` | Reproducibility metadata completeness (dirty-tree, dependency versions, spec identities) | `EVAL-PROTOCOL` §6 REQ-015, §11.11 |
| `REQ-016` | Minimal core comparison report; optional outputs clearly separated | `EVAL-PROTOCOL` §6 REQ-016 |
| `REQ-018` | Ablations kept out of core tables, distinct block IDs | `EVAL-PROTOCOL` §6 REQ-018 |
| `DEC-009` | Official runs require a clean, tracked Git working tree | `EVAL-PROTOCOL` §15 `DEC-009` |
| `DEC-010` | Single documented canonical analysis entry point | `EVAL-PROTOCOL` §15 `DEC-010` |
| `DEC-015` | PPO atomic collection unit = full 2,048-transition rollout; approved bounded overshoot | `EVAL-PROTOCOL` §15 `DEC-015` |

`REQ-001`, `REQ-003`, `REQ-005`, `REQ-013`, `REQ-017`, and `REQ-019` are
already conformant per the repository verification recorded in
`EVAL-PROTOCOL` itself (seeds, panel provider determinism, deterministic
evaluation, learning-curve cadence, test-split isolation, and future-algorithm
gating) and are not re-scoped here; they are re-checked only incidentally by
the regression matrix in §9.

## 4. Current Repository Analysis

Every statement below is `VERIFIED` against the repository state at plan
creation (2026-07-24) unless marked otherwise.

### 4.1 PPO end-of-budget boundary (`REQ-002`/`DEC-015`)

- `Agent.train()` runs a hard `for step in range(1, chunk_timesteps + 1)` loop
  (`src/thesis_rl/agent/agent.py:511`) and unconditionally reports
  `"chunk_steps_actual": int(chunk_timesteps)` (`agent.py:832`).
- The outer loop in `src/thesis_rl/runtime/loops/train_loop.py` computes
  `chunk_steps = min(eval_interval, remaining)` (`train_loop.py:1252`) inside
  `while remaining > 0:` (`train_loop.py:1248`), and decrements `remaining` by
  exactly `actual_chunk_steps` (`train_loop.py:1322-1324`), which equals
  `chunk_timesteps` for the single-environment path. The loop therefore
  terminates at exactly `total_timesteps` env steps regardless of PPO
  rollout-buffer fullness.
- PPO's policy update fires only when `self.model.rollout_buffer.full`
  (`src/thesis_rl/agent/planners/algorithms/ppo_sb3.py:461-469`,
  `maybe_update`); no code path was found that forces a flush/update of a
  partially filled buffer.
- `Agent.train_vectorized()` (non-ACL, `n_envs=4` profile) has a parallel
  `collected_steps`-based `"chunk_steps_actual"` at `agent.py:1586`; its exact
  stop-condition interaction with the PPO buffer boundary is `AWAITING_CONFIRMATION`
  and must be verified before Milestone 5, because `RL-BASELINES` §3.4 lists
  `non_acl_vectorized` as a supported profile even though it is not the
  primary `thesis` runtime.
- TD3/SAC train every collected transition (`train_freq=1`,
  `gradient_steps=auto`) and are unaffected.

### 4.2 Panel manifest (`REQ-004`)

- `FixedSequenceScenarioProvider` (`src/thesis_rl/scenarios/provider.py:240-290`)
  walks an in-memory ordered `records` sequence, rejecting duplicates
  (`provider.py:254-255`) and exhaustion without `repeat=True`
  (`provider.py:271-274`). It performs no hashing or persistence.
- Construction sites: `src/thesis_rl/envs/factory.py:298-304` (single-env path)
  and `factory.py:504-509` (vectorized path), both deriving `sequence_records`
  from `records` filtered/reordered by an `eligible_scenario_uids` window
  (`factory.py:487-501`, `factory.py:293-301`).
- No panel-UID-list manifest file (path/hash) exists anywhere under `src/`.

### 4.3 Checkpoint identity (`REQ-006`)

- `final.zip` is written unconditionally at the end of training, guarded by
  `checkpoint.save_final`, via plain `agent.save(...)` calls in
  `train_loop.py` (verified call sites at lines `1001, 1055, 1101, 1140, 1198,
  1480, 1967, 2466`). None of these calls record a hash or a role string.
- A parallel, currently unused mechanism exists:
  `src/thesis_rl/sb3_extensions/checkpointing.py` defines
  `CheckpointGeneration`, `sha256_file()` (`checkpointing.py:44-51`), and
  `publish_checkpoint_generation`, writing `model.zip` + `manifest.json` under
  a generation directory. `Agent.save_generation()`
  (`src/thesis_rl/agent/agent.py:2538-2562`) wraps it but is never called from
  `train_loop.py`.
- `src/thesis_rl/contracts/checkpoint_manifest.py` defines `CheckpointManifest`
  with `sb3_version`/`sb3_commit`/`git_commit`/rulebook/scalarization identity
  fields (`checkpoint_manifest.py:20-58`), also unused in the production path.
- `csv_recorder.py` already has a `checkpoint_path` column
  (`src/thesis_rl/runtime/io/csv_recorder.py:239`) but no `checkpoint_hash` or
  `checkpoint_role` column.

### 4.4 Reproducibility metadata (`REQ-015`)

- `save_run_metadata()` (`src/thesis_rl/runtime/io/metadata.py:72-200`)
  already captures: `git.branch`/`git.commit` (`metadata.py:177-180`, via
  `get_git_commit()`/`get_git_branch()`, `metadata.py:12-32`); `device`,
  `cuda_device_name` (`metadata.py:174-176`, `torch` already imported at
  `metadata.py:10`, but `torch.__version__` itself is not captured);
  `rulebook.specification_id`/`version` (`metadata.py:125-131`);
  `scalarization.specification_id`/`version`/... (`metadata.py:132-157`);
  ScenarioNet dataset/split/catalog artifact hashes via
  `_snapshot_scenarionet_artifacts()` (`metadata.py:44-69`, called at
  `metadata.py:192-195`).
- Not captured: working-tree dirty/clean state for the training run itself (a
  dirty flag exists only in the separate ScenarioNet dataset-bootstrap
  provenance manifest, `src/thesis_rl/scenarios/bootstrap.py:71-74`,
  `288`, `"project_worktree_dirty"`, unrelated to per-run metadata);
  `torch.__version__`; populated `sb3_version`/`sb3_commit` (schema exists in
  `checkpoint_manifest.py` but is never invoked); MetaDrive/ScenarioNet
  package versions; observation/encoder/ACL/transition-replay/this-protocol's
  own specification identity.
- `update_run_metadata()` (`metadata.py:203-219`) already supports additive
  patching, so new fields can be added without breaking the schema.

### 4.5 Data-abort coverage (`REQ-012`)

- `Agent._evaluate_parallel()` already computes
  `metrics["data_abort_coverage"] = {"attempted": ..., "valid": ...,
  "invalid": ..., "invalid_episodes": [...]}` (`src/thesis_rl/agent/agent.py:2279-2284`),
  covered by `tests/test_parallel_evaluation_data_abort.py`.
- A repository-wide search found zero references to `data_abort_coverage` in
  `train_loop.py` or `src/thesis_rl/runtime/io/csv_recorder.py`: the computed
  dict is never persisted.
- The raw forensic per-abort record path is
  `src/thesis_rl/runtime/data_abort.py` (`RuntimeScenarioQuarantine`,
  `append_data_abort_record()`), called from `agent.py:1092`.

### 4.6 Analysis pipeline (`REQ-007`, `REQ-009`, `REQ-014`, `REQ-016`, `DEC-010`)

- `src/thesis_rl/analysis/run_analysis.py` is a real, already-existing
  canonical entry point (CLI, `argparse`-based) that calls
  `aggregate_runs()`, `make_comparison_views()`, and table builders for
  final/curriculum/rulebook/sample-efficiency/generalization/factor-effect
  outputs, plus `build_qualitative_manifest()`. `DEC-010`'s requirement
  ("a single documented entry point") is therefore largely already satisfied
  at the code level; the remaining work is to verify it is documented (a
  Makefile target or `README`/`AGENTS.md` command) and to reconcile its
  *content* with `EVAL-PROTOCOL`, not to build a new entry point.
- **The prohibited `1.96 * s / sqrt(n)` CI formula is currently implemented in
  four places**: `src/thesis_rl/analysis/common_stats.py:21`,
  `src/thesis_rl/analysis/tables/make_final_tables.py:64`,
  `src/thesis_rl/analysis/plots/make_plots.py:65`, and
  `src/thesis_rl/analysis/tables/make_factor_effect_tables.py:54`. Per
  `REQ-009`/`DEC-003`, none of these may compute or emit a CI for the core
  `EVAL-PROTOCOL` report.
- `src/thesis_rl/analysis/tables/make_factor_effect_tables.py` and
  `make_generalization_tables.py` are `INFERRED` (not yet read in full) to
  implement the historical reward-setting x curriculum factorial design
  referenced by `REQ-001`'s supersession note and the
  `conf/presets/td3/*_curr*/*_no_curr.yaml` preset matrix; this must be
  confirmed and, if so, these outputs relabeled as `REQ-018` ablation-only,
  excluded from the `REQ-016` core report path.
- `make_curriculum_tables.py` and `make_rulebook_tables.py` already compute
  `n_seeds` per condition (`make_curriculum_tables.py:135,152`,
  `make_rulebook_tables.py:93,99`); their exact primary-metric selection
  (R1--R3 vs. R4 split; macro-rule vs. sub-rule) is `AWAITING_CONFIRMATION`
  and must be read in full before Milestone 6.
- `src/thesis_rl/analysis/videos/make_qualitative_manifest.py` and
  `select_video_episodes.py` exist; their category taxonomy and
  predeclaration semantics relative to `REQ-014`'s four fixed categories
  (`representative_success`, `representative_failure`,
  `severe_rule_violation`, `algorithm_disagreement`) are
  `AWAITING_CONFIRMATION` and must be read in full before Milestone 9.

### 4.7 Run registry / disposition (`REQ-011`)

- No existing run-registry field for a manual infrastructure-failure /
  non-convergent / condition-attributable-failure disposition was found
  during the specification review; `AWAITING_CONFIRMATION` on whether any
  registry file already exists beyond `run_metadata.yaml`'s `status` field
  (`metadata.py:181`, currently only `"running"`/`"completed"`).

## 5. Assumptions And Invariants

- Units/frames: environment steps are the primary sample-budget unit
  (`SPECIFIED`, `EVAL-PROTOCOL` §3.1). PPO rollout size is `n_steps * n_envs =
  2,048` for the `thesis`-profile single-environment ACL runtime (`SPECIFIED`,
  `RL-BASELINES` §3.4, `VERIFIED` against `ppo_sb3.py:49`).
- State/reset: this plan must not change episode-boundary reset, termination,
  or truncation semantics (`SPECIFIED`, `EVAL-PROTOCOL` §2 Out Of Scope).
- Determinism: any new metadata/manifest write must not introduce
  nondeterministic ordering into `run_metadata.yaml`, CSV headers, or the
  panel manifest (`SPECIFIED`, `EVAL-PROTOCOL` REQ-016 invariant "rerunning
  analysis on the same canonical inputs produces the same output values and
  ordering").
- Versions: `EVAL-PROTOCOL` itself is version `1.0`; this plan targets exactly
  that version and does not anticipate a `1.1` amendment.
- The three-seed list `[0, 1, 2]` and `run_profile=thesis` values are already
  correct in `conf/run_profile/thesis.yaml` (`VERIFIED`, no change required).
- A dirty flag added to `run_metadata.yaml` must exclude Git-ignored/generated
  files from the dirty determination, per `DEC-009`'s "Git-ignored/generated
  files do not count as dirty" clause (`SPECIFIED`).

## 6. Decisions And Approval Gates

No decision below blocks implementation start; all are implementation details
within Codex's decision authority per `AGENTS.md` ("Decision And Change
Control"). They are recorded here because they affect artifact schema and are
non-obvious, not because they are open scientific questions — every scientific
question was already closed in `EVAL-PROTOCOL` §15.

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-EP-001` | Implementation detail | Where does the panel manifest live and what does it hash? | (a) a new `panels/<split>_panel_manifest.json` under the run/dataset artifacts root, containing the ordered UID list and its SHA-256; (b) reuse `_snapshot_scenarionet_artifacts()`'s existing hashing pattern in `metadata.py` | (a): a dedicated, protocol-owned artifact, generated once per frozen panel (not per run) and referenced by path+hash from each run's `run_metadata.yaml`, following the existing `_snapshot_scenarionet_artifacts()` hashing pattern for consistency | Determines panel-manifest file location and reuse across runs | Resolved by Codex |
| `DEC-EP-002` | Implementation detail | How is checkpoint hash/role recorded — via the existing unused `CheckpointGeneration` mechanism, or by adding fields directly to `run_metadata.yaml`/`final_eval.csv`? | (a) wire `publish_checkpoint_generation`/`Agent.save_generation()` into `train_loop.py`'s save call sites; (b) keep the existing plain `agent.save()` calls and add `sha256_file()` + a role string directly to `run_metadata.yaml` and `final_eval.csv` at consumption time | (b): lower-risk and narrower in scope than switching the production checkpoint path to the currently-unused generation mechanism, which was seemingly deferred for reasons not documented here; revisit (a) only if a future spec requires full manifest-based checkpoint compatibility validation | Determines whether the dead `CheckpointGeneration` code path is activated or bypassed | Resolved by Codex; flagged for user awareness because it leaves `checkpointing.py`/`checkpoint_manifest.py` unused |
| `DEC-EP-003` | Implementation detail | How does the PPO stop condition become atomic-boundary-aware without changing the outer chunk-based training loop's eval-cadence bookkeeping? | (a) after each chunk, if `remaining <= 0` and the algorithm is PPO with a non-full rollout buffer, run one additional bounded step batch until `rollout_buffer.full`, then stop; (b) restructure the loop to always request one more transition after the nominal target until the current algorithm's `maybe_update` reports a completed update | (a): minimal change confined to the loop-termination check; avoids altering `chunk_steps` sizing (`eval_interval`-based) used for periodic-validation cadence bookkeeping | Confines the `DEC-015` fix to a bounded, well-tested extension step rather than a larger loop rewrite | Resolved by Codex |
| `DEC-EP-004` | Implementation detail | Where is the three-way run disposition (`REQ-011`) recorded? | (a) new `disposition` and `disposition_rationale` fields in `run_metadata.yaml`, set by an operator/CLI step, not inferred automatically; (b) a separate `run_registry.yaml` file outside per-run artifacts | (a): keeps the disposition co-located with the rest of the run's reproducibility record and reuses `update_run_metadata()`'s additive-patch mechanism | Determines where manual run-disposition data lives | Resolved by Codex |

## 7. Proposed Design

### 7.1 PPO atomic-boundary stop condition (`REQ-002`/`DEC-015`, `DEC-EP-003`)

Affected module: `src/thesis_rl/runtime/loops/train_loop.py` (training loop),
`src/thesis_rl/agent/agent.py` (`Agent.train`, chunk-level step loop),
`src/thesis_rl/agent/planners/algorithms/ppo_sb3.py` (`maybe_update`,
`rollout_buffer.full`).

- Add an algorithm-reported "atomic collection unit" concept: for PPO, this is
  `n_steps * n_envs`; for TD3/SAC, it is `1`. Expose it from the planner
  (e.g. a `PPOSb3Planner.atomic_collection_unit` property already derivable
  from `self.global_rollout_size`, `ppo_sb3.py:49`) so the training loop does
  not hardcode algorithm identity.
- After the existing `while remaining > 0:` loop's final chunk reaches
  `remaining <= 0`, check whether the current algorithm's rollout buffer (if
  any) is not yet full. If so, continue stepping in bounded increments,
  updating `overshoot_steps`, until `maybe_update` reports a completed update
  (buffer became full and was consumed) or a configured safety bound
  (`atomic_collection_unit - 1` extra steps) is reached — never open-ended.
- Record `target_timesteps`, `actual_completed_timesteps`, `atomic_collection_unit`,
  and `overshoot_steps` in `run_metadata.yaml` via `update_run_metadata()`.
- Errors/fallbacks: if the safety bound is exceeded without the buffer
  becoming full (should not happen given `atomic_collection_unit` is exact),
  fail fast with a descriptive error rather than silently truncating —
  consistent with `EVAL-PROTOCOL`'s prohibition on discarding a partial
  rollout.
- Logging: log the overshoot amount at chunk-loop completion, matching the
  existing `train_logger.info("Chunk finished | ...")` style
  (`train_loop.py:1327-1342`).

### 7.2 Panel manifest (`REQ-004`, `DEC-EP-001`)

Affected module: `src/thesis_rl/scenarios/` (new manifest builder/reader),
`src/thesis_rl/envs/factory.py` (provider construction call sites),
`src/thesis_rl/runtime/io/metadata.py` (reference the manifest hash per run).

- Build the manifest once from the same eligible-UID-window logic already
  used at `factory.py:298-304`/`504-509`, writing an ordered, deduplicated UID
  list plus its SHA-256 to a versioned artifact (e.g.
  `data/scenarionet/panels/<split>_panel_manifest_v1.json`), separate from
  per-run outputs so every condition/seed in a comparison block references the
  identical file.
- `FixedSequenceScenarioProvider` construction reads the frozen manifest
  instead of re-deriving order from `eligible_scenario_uids` config each time,
  when the manifest exists; falls back to today's derivation with a fatal
  error if the derived order does not match the frozen manifest (fail closed,
  per `REQ-004`'s "no fallback sampling" invariant).
- `save_run_metadata()` records the manifest path and hash for the
  validation and test panels actually used by the run.

**Update (2026-07-25, tracked-subset sizing/diversity revision)**: `PanelManifest`
gained a `tracked_subset_uids: tuple[str, ...] = ()` field (REQ-014/DEC-014),
computed once at build time in `build_balanced_panel` (never at runtime --
`DEC-004`/`DEC-005` no-redraw) from feature-diversity greedy selection
within each arm's already-drawn panel candidates (never a separate catalog
draw). Approved sizing: **4 scenarios/arm for the validation split** (24
total across 6 arms), **10 scenarios/arm for the test split** (60 total) --
two different counts for two different purposes (validation is rendered
often during training progression; test only once, but with broader
coverage). Diversity criterion: greedy selection over
`_TRACKED_SUBSET_FEATURE_KEYS` (`has_intersection`,
`has_merge_or_roundabout`, `has_route_traffic_light`, `has_route_stop_sign`,
`has_route_crosswalk`, `has_vehicle`, `has_pedestrian`, `has_cyclist`,
`low_traffic`, `dense_traffic`, `vru_interaction`, `topology_tag`) sourced
from `ScenarioCatalogEntry.features.to_dict()`, so the tracked subset covers
traffic lights, intersections/roundabouts, VRU interactions, dense/light
traffic, and simpler/more complex scenarios rather than an arbitrary prefix.
`scripts/build_panel_manifest.py` builds a `scenario_uid -> features` lookup
from the catalog and defaults `--tracked-subset-count-per-arm` to 4
(validation) / 10 (test) unless overridden. Both committed manifest
artifacts (`data/scenarionet/panels/{validation,test}_panel_manifest_v1.json`,
outside `src/`, in the data volume, not git-tracked) were regenerated with
`seed=20260725`:
  - `validation_panel_manifest_v1.json`: `sha256=24bc974a792c3106f956b91b26f98c2ec45e77c31b21162bffe6a6fd19760f88`,
    `tracked_subset_uids` count=24 (4/arm x 6 arms);
  - `test_panel_manifest_v1.json`: `sha256=35e0af6373794a7c370118e93b5d0e1cd847442d1c8e0bc03c99515109bedd10`,
    `tracked_subset_uids` count=60 (10/arm x 6 arms).
`save_panel_manifest`/`load_panel_manifest` persist/read the new key (optional
on load -- absent for pre-existing manifests, defaulting to `()`).
`PanelManifest.verify_self_consistent` additionally fails closed if
`tracked_subset_uids` is not a subset of `scenario_uids` or contains
duplicates. The old `select_tracked_subset_uids(manifest, count=5)` helper
(round-robin, one-per-arm, no feature awareness) is kept only for manifests
predating this field and is not called by any production code path anymore
-- `train_loop.py` reads `manifest.tracked_subset_uids` directly (§7.7/§10
Milestone 9).

### 7.3 Checkpoint identity (`REQ-006`, `DEC-EP-002`)

Affected module: `src/thesis_rl/runtime/loops/train_loop.py` (checkpoint save
call sites), `src/thesis_rl/runtime/io/metadata.py`,
`src/thesis_rl/runtime/io/csv_recorder.py`.

- At the `checkpoint.save_final`-guarded `final.zip` save call
  (`train_loop.py:1967`) and at official final-evaluation checkpoint
  consumption, compute `sha256_file()` (reusing
  `checkpointing.py:44-51`, without activating the rest of the generation
  mechanism) and record `checkpoint_path`, `checkpoint_hash`, and
  `checkpoint_role="final"` via `update_run_metadata()`.
- Add `checkpoint_hash` and `checkpoint_role` columns (additive) to
  `final_eval.csv`, alongside the existing `checkpoint_path`
  (`csv_recorder.py:239`).
- `best_*` checkpoints may optionally record the same fields with their own
  role string for diagnostic traceability, but this is not required by
  `REQ-006` and is deferred unless requested.

### 7.4 Reproducibility metadata completion (`REQ-015`)

Affected module: `src/thesis_rl/runtime/io/metadata.py`.

- Add a `git.dirty` boolean computed the same way `git_state`/
  `project_worktree_dirty` is computed in `bootstrap.py:71-74`/`288`
  (tracked-file diff only, excluding Git-ignored/generated files), but scoped
  to the training run's own `save_run_metadata()` call, not the dataset
  bootstrap path.
- Add `dependencies.torch_version = torch.__version__` (trivial, `torch`
  already imported).
- Populate `dependencies.sb3_version`/`sb3_commit` using the same resolution
  the (currently unused) `CheckpointManifest` expects, without adopting the
  rest of that manifest mechanism.
- Add `dependencies.metadrive_version`/`scenarionet_version` via
  `importlib.metadata.version(...)`, matching the style already used for
  `torch`.
- Add `observation`, `encoder`, `acl`, `transition_replay`, and
  `evaluation_protocol` blocks mirroring the existing `rulebook`/
  `scalarization` blocks (`metadata.py:125-157`) — same shape, new spec IDs.

### 7.5 Data-abort coverage persistence (`REQ-012`)

Affected module: `src/thesis_rl/agent/agent.py` (caller of `evaluate`),
`src/thesis_rl/runtime/loops/train_loop.py`,
`src/thesis_rl/runtime/io/csv_recorder.py`.

- At every call site that already receives `metrics["data_abort_coverage"]`
  from `Agent.evaluate()`/`_evaluate_parallel()`, write it through
  `csv_recorder` as new additive columns on `evals.csv`/`final_eval.csv`
  (`attempted`, `valid`, `invalid`, `coverage = valid/attempted`) and persist
  `invalid_episodes` (UID + reason code) to a per-evaluation JSON/JSONL
  sidecar, reusing the existing `data_abort.py` forensic-record style.
- No change to `Agent._evaluate_parallel()`'s computation itself — it is
  already correct (`REQ-008`/`REQ-012` computation requirement is already
  satisfied).

### 7.6 Run disposition (`REQ-011`, `DEC-EP-004`)

Affected module: `src/thesis_rl/runtime/io/metadata.py` (schema),
new small CLI helper for recording disposition post hoc (out of the
training loop's hot path, since disposition is a manual, reviewed judgment,
never inferred automatically per `REQ-011`'s invariant).

- Add `disposition: <unset|infrastructure_failure|non_convergent|condition_attributable_failure>`
  and `disposition_rationale: <free text>` fields, defaulting to `unset` and
  patchable via `update_run_metadata()`.
- Provide a minimal CLI/script entry point for a human to set these fields
  post hoc for a given run's artifacts directory, rather than auto-deriving
  them from metric thresholds (this would violate `REQ-011`'s ban on
  automatic inference).

### 7.7 Analysis pipeline reconciliation (`REQ-007`, `REQ-009`, `REQ-014`, `REQ-016`, `REQ-018`, `DEC-010`)

Affected module: `src/thesis_rl/analysis/*`.

- Remove the `1.96 * s / sqrt(n)` computation from `common_stats.py:21`,
  `make_final_tables.py:64`, `make_plots.py:65`, and
  `make_factor_effect_tables.py:54`; replace with raw values + mean + sample
  SD only, per `REQ-009`/`DEC-003`. Any caller currently consuming the
  removed `ci` field must be updated in the same milestone (no partial
  removal).
- Read `make_curriculum_tables.py`, `make_rulebook_tables.py`,
  `make_generalization_tables.py`, and `make_factor_effect_tables.py` in full
  (resolving the `AWAITING_CONFIRMATION` items in §4.6) and: (a) ensure the
  primary-metric table reports R1--R3 macro-rule violation rate/episode-min
  margin and R4 mean progress margin/`negative_progress_rate` as distinct
  columns, never merged or labeled as a shared "violation rate"; (b) if
  `make_factor_effect_tables.py`/`make_generalization_tables.py` implement the
  historical reward x curriculum factorial, gate their invocation in
  `run_analysis.py` behind an explicit `--include-ablations` flag (mirroring
  the existing `include_effects_tables` parameter already visible in
  `run_analysis.py:_build_tables_for_root`) so the default `REQ-016` core
  report path never includes them.
- `run_analysis.py` becomes (or is confirmed to already be) the `DEC-010`
  canonical entry point; document its invocation (Makefile target or
  `python -m thesis_rl.analysis.run_analysis`) in `AGENTS.md`'s Verified
  Commands section once confirmed working end-to-end on a fixture.
- Read `make_qualitative_manifest.py`/`select_video_episodes.py` in full and
  reconcile their category taxonomy with `REQ-014`'s four fixed categories and
  post-hoc selection (no mandatory predeclaration, no mandatory
  pre-rendering).
- (2026-07-25, fourth Milestone 9 pass) The periodic tracked-subset GIF
  *pixel-rendering* call itself -- as opposed to the analysis-pipeline
  reconciliation above, which concerns post-hoc manifest-driven rendering --
  lives in `runtime/io/eval_artifacts.py` and `runtime/loops/train_loop.py`,
  not `analysis/videos/*`; it is now closed. See §10 Milestone 9 for the
  full account.

### 7.8 Ablation tagging (`REQ-018`)

- Tag `conf/presets/td3/td3_native_curr.yaml`, `td3_native_no_curr.yaml`,
  `td3_monitor_only_curr.yaml`, `td3_monitor_only_no_curr.yaml`,
  `td3_scalar_reward_curr.yaml`, `td3_scalar_reward_no_curr.yaml` with an
  explicit `analysis.experiment_group`/comparison-block identity distinct from
  `BASELINE-SCALAR-01`/`EXTENSION-ALGORITHM-01` (the `experiment_group` field
  already exists in `metadata.py:165`), so `REQ-018`'s "distinct
  comparison-block IDs" invariant is enforced by configuration, not by
  convention alone.

### 7.9 Applicability-aware R1--R3 seed-level aggregation (`REQ-008`)

Affected modules: `src/thesis_rl/agent/agent.py`, `src/thesis_rl/runtime/io/csv_recorder.py`,
`src/thesis_rl/runtime/loops/eval_loop.py`, `src/thesis_rl/runtime/loops/train_loop.py`,
`src/thesis_rl/analysis/tables/make_rulebook_tables.py`.

**Verified repository fact (2026-07-25)**: the authoritative per-step
applicability signal `REQ-008`/spec §7.2 requires (`RuleComponentResult
.applicable`, per macro rule) was already computed and exposed on every
step's `info_dict["rule_components"][macro_rule_name]["applicable"]` by
`RulebookV2MonitorWrapper.step` (`src/thesis_rl/rulebook/v2/wrapper.py:220`),
built directly from `aggregate_max_component`/`aggregate_rulebook_result`
(`src/thesis_rl/rulebook/v2/aggregation.py`) -- i.e. the gap was not in the
Rulebook v2 aggregation module (which was correctly out of touch-scope for
this milestone; it is concurrently being modified by an unrelated R2
lateral-RSS session) but entirely in `Agent`'s episode/run-level metric
aggregation, which never read `rule_components` and instead fed every
step's margin (`rule_reward_vector`) into episode-minimum-margin and
violation counting unconditionally, regardless of applicability. Since
`aggregate_max_component` returns `cost=0.0` (a "satisfied"-looking margin)
for a NOT_APPLICABLE step, a step where the Rulebook found a rule
inapplicable was previously silently treated as satisfied rather than
excluded, and `violation_rate` was computed as
"episodes with >=1 violated step" divided by the *total* episode count
(not `N'_s`, the applicable-step-having episode count spec §7.2 requires).
R4 (`route_progress`) has no `RuleComponentResult` entry in
`aggregate_rulebook_result`'s output (verified: only the three R1--R3 macro
components are added to `all_components`), so it never appears in
`rule_components` and is correctly treated as always-applicable by the new
code, preserving its all-step aggregation per §7.3 without any R4-specific
branch.

Design:

- New `Agent._extract_rule_applicability(step_info) -> dict[str, bool]`
  reads `step_info["rule_components"][name]["applicable"]` for every rule
  name present; a rule absent from `rule_components` (R4) is treated as
  always-applicable by every caller (`applicability.get(rule_name, True)`),
  which is what keeps R4 untouched without special-casing it.
- `_ParallelEvaluationEpisode` (used by `_evaluate_parallel`) and the serial
  `Agent.evaluate()` loop (duplicated logic, `count_envs(env) == 1` path)
  both gained per-episode `ep_rule_applicable_step_count`/
  `ep_rule_violated_step_count` counters, populated only on applicable
  steps; `ep_rule_min_margin` is now `min()` over applicable steps only
  (falls out of the same `continue`-on-inapplicable branch). Both
  `finalize()`/episode-completion blocks compute `EpisodeViolationRate_e =
  violated_count_e / applicable_count_e` (spec §7.2) only for rules with
  `applicable_count_e > 0`; a rule with zero applicable steps this episode
  has no entry, so it is not looked up (and not silently zero-filled)
  downstream.
- `_aggregate_parallel_evaluation` and its serial-path twin now compute
  `RuleViolationRate_(i,s) = mean(EpisodeViolationRate_e over episodes with
  an entry for rule i)` -- i.e. over the `N'_s` episodes with >=1
  applicable step, per spec §7.2 -- rather than the previous "any-episode-
  violated / total-episode-count" ratio. `per_rule` rows gained two
  additive fields: `applicable_episode_count` (`N'_s` for that rule) and
  `excluded_episode_count` (`N_s - N'_s`), satisfying the "the exclusion
  count is reported alongside the seed-level value" invariant.
  `mean_margin`/`min_margin`/`max_margin` (already computed from the list
  of per-episode minimum margins, i.e. `RuleMinimumMarginMean`/its
  secondary min/max diagnostics) are unaffected in formula, only in the
  underlying per-episode values now correctly excluding episodes with zero
  applicable steps.
- `rule_metrics.csv`'s schema (`csv_recorder.py`) gained the same two
  additive columns; both `_append_rule_metrics_rows` helpers
  (`eval_loop.py`, `train_loop.py`, intentionally duplicated pre-existing
  code, not refactored in this narrowly-scoped change) pass them through.
  `aggregate_runs.py` unions CSV fieldnames dynamically per file
  (`_extend_unique(fieldnames, list(reader.fieldnames))`), so
  `rule_metrics_all_runs.csv` picks up the new columns with no code change
  there.
- `make_rulebook_tables.py`'s R1--R3 table (`rule_violation_by_rule.csv`/
  `.md`) gained `applicable_episode_count_mean`/`_sd` and
  `excluded_episode_count_mean`/`_sd` columns (cross-seed mean/SD of the
  per-seed counts, consistent with every other metric in that table); the
  R4 table (`rulebook_r4_progress_margin.csv`/`.md`, `REQ-007`) is
  untouched, since these two counts have no meaning for a rule with no
  applicability concept.
- Zero-applicable-steps edge case: when a rule has zero applicable steps
  across every episode in a run (`N'_s = 0`), `violation_rate`/
  `mean_margin`/`min_margin`/`max_margin` fall back to `0.0` (the
  pre-existing "no data" placeholder for this already-existing branch, kept
  for CSV/consumer type-stability rather than switched to `null`), while
  `applicable_episode_count = 0` and `excluded_episode_count = N_s` make the
  exclusion fully visible rather than silently indistinguishable from a
  genuinely-zero violation rate. This has not been observed in practice for
  any of the three R1--R3 macro rules (each has broad applicability), so no
  additional invalidation behavior beyond visible reporting was implemented
  for it; escalate if a real run ever hits this branch.

Out of scope, confirmed unaffected: `rulebook/v2/aggregation.py`,
`rulebook/v2/types.py`, `rulebook/v2/wrapper.py` (read-only, per-step
applicability/margin computation already correct); R4 progress-margin
aggregation semantics (`REQ-007`); the optional-CI/R1--R3-vs-R4 table split
(`REQ-007`/`REQ-009`, Milestone 7, already done); `REQ-012`'s data-abort
coverage persistence (Milestone 4, already done).

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-002`/`DEC-015` | `AC-002` | `train_loop.py`, `agent.py`, `ppo_sb3.py` (§7.1) | `TEST-EP-001`..`003` | Planned |
| `REQ-004` | `AC-004` | `envs/factory.py`, new `scenarios/panel_manifest.py`, `metadata.py` (§7.2) | `TEST-EP-004`..`006` | Planned |
| `REQ-006` | `AC-006` | `train_loop.py`, `metadata.py`, `csv_recorder.py` (§7.3) | `TEST-EP-007`..`008` | Planned |
| `REQ-007`/`REQ-008` | `AC-007`, `AC-008` | `analysis/tables/make_rulebook_tables.py`, `make_curriculum_tables.py` (§7.7) | `TEST-EP-009`..`011` | Planned |
| `REQ-009` | `AC-008` | `analysis/common_stats.py`, `make_final_tables.py`, `make_plots.py`, `make_factor_effect_tables.py` (§7.7) | `TEST-EP-012`..`013` | Planned |
| `REQ-011` | `AC-009` | `metadata.py`, new disposition CLI (§7.6) | `TEST-EP-014`..`015` | Planned |
| `REQ-012` | `AC-010` | `agent.py` call sites, `csv_recorder.py`, `data_abort.py` (§7.5) | `TEST-EP-016`..`017` | Planned |
| `REQ-014` | `AC-012` | `analysis/videos/make_qualitative_manifest.py`, `select_video_episodes.py` (§7.7) | `TEST-EP-018`..`019` | Planned |
| `REQ-015` | `AC-013` | `metadata.py` (§7.4) | `TEST-EP-020`..`021` | Planned |
| `REQ-016`/`DEC-010` | `AC-014` | `analysis/run_analysis.py` and callees (§7.7) | `TEST-EP-022`..`023` | Planned |
| `REQ-018` | `AC-016` | `conf/presets/td3/*.yaml` (§7.8) | `TEST-EP-024` | Planned |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-EP-001` | Unit | PPO stops at the atomic rollout boundary, not mid-rollout | Fixture PPO planner, `n_steps=4`, `n_envs=1`, `total_timesteps` set to a non-multiple of 4 (e.g. 10) | Loop runs exactly 12 steps (3 full rollouts of 4), not 10; `overshoot_steps == 2`; the final rollout triggers exactly one `maybe_update` call | `REQ-002`, `DEC-015` |
| `TEST-EP-002` | Regression | A `thesis`-profile PPO smoke run completes at `1,501,184` transitions, not `1,500,000` | `run_profile=thesis` config overridden to a tiny fixture dataset for CI speed, or a dedicated small-budget fixture with the same `2048`-multiple arithmetic | `run_metadata.yaml` records `target_timesteps=1_500_000`, `actual_completed_timesteps=1_501_184`, `atomic_collection_unit=2048`, `overshoot_steps=1184` | `REQ-002`, `DEC-015` |
| `TEST-EP-003` | Unit | TD3/SAC are unaffected (exact stop, no overshoot) | Fixture TD3 planner, `total_timesteps=10` | Loop stops at exactly 10 transitions, `overshoot_steps == 0` | `REQ-002`, `DEC-015` |
| `TEST-EP-004` | Unit | Panel manifest is deduplicated, ordered, hashed | Synthetic 5-UID list with one intentional duplicate | Manifest builder raises before writing; a clean list produces a stable SHA-256 across two builds | `REQ-004` |
| `TEST-EP-005` | Integration | `FixedSequenceScenarioProvider` matches the frozen manifest | Frozen manifest + matching/mismatching `eligible_scenario_uids` config | Match: provider order equals manifest order. Mismatch: fails closed with no fallback sampling | `REQ-004` |
| `TEST-EP-006` | Regression | Validation and test panel identity are recorded per run | Fixture run | `run_metadata.yaml` contains both manifest paths and hashes | `REQ-004`, `REQ-015` |
| `TEST-EP-007` | Unit | `final.zip` consumption records path/hash/role | Fixture checkpoint file | `checkpoint_hash` matches an independently computed SHA-256; `checkpoint_role == "final"` | `REQ-006` |
| `TEST-EP-008` | Regression | `best_*` checkpoints never populate the official checkpoint fields used by final evaluation | Fixture run with both `final.zip` and `best_lexicographic.zip` present | Final evaluation loads and records only `final.zip`'s identity | `REQ-006`, `EVAL-PROTOCOL` `AC-006` |
| `TEST-EP-009` | Unit | R1--R3 macro-rule metrics aggregate only over applicable steps | Synthetic per-step margin/applicability sequence with some inapplicable steps | `EpisodeViolationRate`/`EpisodeMinimumMargin` computed only over applicable steps; episode excluded from the rule's aggregate when zero steps are applicable | `REQ-008`, `EVAL-PROTOCOL` §7.2 |
| `TEST-EP-010` | Unit | R4 metrics aggregate over all steps and are never labeled "violation rate" | Synthetic progress-margin sequence including negative values | `EpisodeMeanProgressMargin`/`NegativeProgressRate` computed over every step; output table column names contain no "violation" wording for R4 | `REQ-007`, `EVAL-PROTOCOL` §7.3 |
| `TEST-EP-011` | Regression | Primary table has 4+5 distinct primary columns (success, route completion, collision, out-of-road, R1--R3 x 2, R4 x 2) | Fixture run registry | Table schema matches exactly; no collapsed/merged column | `REQ-007`, `EVAL-PROTOCOL` `AC-007` |
| `TEST-EP-012` | Regression | No CI/confidence-interval field appears anywhere in core report output | Fixture 3-seed run registry through the full `run_analysis.py` pipeline | Grep of generated CSV/JSON/Markdown/LaTeX output contains no `ci_95`/`confidence` field for the core report | `REQ-009`, `DEC-003` |
| `TEST-EP-013` | Unit | Cross-seed aggregation with exactly 3 seeds reports raw values + mean + SD | Synthetic 3-seed metric list | Output matches manually computed mean/SD; no interval computed | `REQ-009` |
| `TEST-EP-014` | Unit | Disposition defaults to `unset` and is never auto-derived from metrics | Fixture run with poor metrics | `disposition` remains `unset` until explicitly patched | `REQ-011` |
| `TEST-EP-015` | Regression | Setting disposition via the CLI/script updates `run_metadata.yaml` additively | Fixture run | Other existing fields unchanged; `disposition`/`disposition_rationale` present | `REQ-011` |
| `TEST-EP-016` | Regression | `data_abort_coverage` is persisted to `evals.csv`/`final_eval.csv` | Fixture evaluation containing a synthetic data-abort episode | CSV rows contain `attempted`, `valid`, `invalid`, `coverage` matching the in-memory computation | `REQ-012` |
| `TEST-EP-017` | Regression | Aggregate numerators/denominators exclude aborted episodes without backfill | Same fixture as `TEST-EP-016` | Primary metrics computed over valid episodes only; count matches `valid`, not `attempted` | `REQ-008`, `REQ-012` |
| `TEST-EP-018` | Unit | Qualitative case categorization accepts only the four approved categories | Attempted case with an invalid category string | Rejected with an explicit error | `REQ-014` |
| `TEST-EP-019` | Regression | A cross-condition qualitative comparison requires an identical `scenario_uid` | Two cases with mismatched UIDs marked as a "comparison" | Rejected/flagged, not silently accepted | `REQ-014` |
| `TEST-EP-020` | Unit | `run_metadata.yaml` records `git.dirty`, `torch_version`, and the new specification identities | Fixture run in a clean and in a dirty tree | Fields present and correct in both cases; Git-ignored files do not flip `dirty=true` | `REQ-015` |
| `TEST-EP-021` | Regression | Missing required `REQ-015` metadata is fatal for official-run validity checking | Fixture `run_metadata.yaml` missing a required field | Validity check fails closed | `REQ-015` |
| `TEST-EP-022` | Integration | The canonical analysis command regenerates the core report deterministically | Fixture run registry, two consecutive invocations | Byte-identical (or value-identical, order-identical) core outputs | `REQ-016`, `DEC-010` |
| `TEST-EP-023` | Regression | Core report path never includes factor-effect/ablation tables by default | Same fixture, default invocation vs. `--include-ablations` | Default output has no factor-effect table; flagged invocation does | `REQ-016`, `REQ-018` |
| `TEST-EP-024` | Regression | `td3_*_curr`/`*_no_curr` presets carry a distinct `experiment_group` from the core blocks | Config resolution of each preset | `experiment_group` differs from `BASELINE-SCALAR-01`/`EXTENSION-ALGORITHM-01` | `REQ-018` |

Available repository commands for this matrix: `make test` (full suite via the
primary environment), `uv run --no-sync python -m pytest -q` (inside an
already provisioned container), `make lint`, `make format-check` (scoped via
`PYTHON_QUALITY_PATHS` to newly modified files per `docs/engineering_workflow.md`),
`make rulebook-v2-check` (not expected to be touched by this plan, run only if
`RuleComponentResult`/`aggregation.py` reading code is added). No canonical
`make analysis`/`make evaluate` command currently exists; Milestone 8 must
either confirm one exists under a name not yet found, or add a `Makefile`
target and record it here and in `AGENTS.md`'s Verified Commands before this
plan can reach `VERIFIED`.

## 10. Milestones

### Milestone 1 — Reproducibility metadata completion (`REQ-015`)

- Status: `IMPLEMENTED`; unit-tested and verified live (see §14). `git.dirty`
  uses `git status --porcelain` (Git-ignored files excluded by Git itself);
  `sb3_commit`/`metadrive_version`/`scenarionet_version` resolve to
  `"unknown"`/`None` gracefully when unavailable in a given environment
  (verified in the live smoke run: `sb3_commit: unknown`,
  `git.commit: unknown` inside the container, since `.git` context differs
  there) rather than failing the run — this degrades information richness but
  does not violate any invariant.
- Expected files: `src/thesis_rl/runtime/io/metadata.py`,
  `tests/test_run_metadata.py`
- Tasks: add `git.dirty`, `dependencies.torch_version`,
  `dependencies.sb3_version`/`sb3_commit`, `dependencies.metadrive_version`/
  `scenarionet_version`, and `observation`/`encoder`/`acl`/
  `transition_replay`/`evaluation_protocol` identity blocks per §7.4.
- Tests/commands: `TEST-EP-020`, `TEST-EP-021`; `uv run --no-sync python -m
  pytest -q tests/test_run_metadata.py`.
- Completion evidence: passing focused tests; a real `run_profile=smoke` run
  producing a `run_metadata.yaml` with all new fields populated (not `null`).
- Decision dependencies: none.

### Milestone 2 — Panel manifest (`REQ-004`)

- Status: `COMPLETE` (2026-07-25, second pass). Gaps (b) and (c) below closed:
  `save_run_metadata()`'s `_snapshot_scenarionet_artifacts()` now also
  snapshots+hashes `env.provider.panel_manifest_path` when configured
  (`panel_manifest_<stem>` / `panel_manifest_<stem>_sha256` in
  `scenarionet.artifact_snapshots`), tested by
  `test_run_metadata_records_panel_manifest_hash`
  (`tests/test_run_metadata.py`, PASS). Real manifests were generated end to
  end against the real ScenarioNet catalog at
  `/scratch/e.respino/thesis-metadrive/data/scenarionet` (via the
  `dataset-pipeline` compose service, which mounts data read-write, unlike
  `dev`) and committed to the data volume:
  `data/scenarionet/panels/validation_panel_manifest_v1.json` (split=validation,
  size=100, seed=20260725, per-arm counts 17/17/17/17/16/16,
  sha256=`24bc974a...19760f88`) and
  `data/scenarionet/panels/test_panel_manifest_v1.json` (split=test, size=300,
  seed=20260725, per-arm counts 50 each, sha256=`35e0af63...09bedd10`).
  Remaining gap (a) — no Hydra preset yet sets `provider.panel_manifest_path`
  to these files — is an explicit follow-up config change (Milestone 8/
  preset wiring), not a code or manifest gap.
  Implemented and unit-tested:
  `src/thesis_rl/scenarios/panel_manifest.py`
  (`PanelManifest`/`build_balanced_panel`/`save_panel_manifest`/
  `load_panel_manifest`/`select_tracked_subset_uids`/
  `default_panel_manifest_path`) provides a deterministic seed-driven
  balanced draw across the six scenario arms (`A0`--`A5`), per-arm
  deterministic shuffling seeded from `sha256(seed:split:arm_index:arm)`
  (never the raw training seed), a SHA-256 order-sensitive content hash, and
  fail-closed self-consistency verification on load (tampered
  hash/duplicate UIDs/inconsistent `per_arm_counts` all raise). Wired into
  `src/thesis_rl/envs/factory.py`'s `fixed_sequence` provider construction at
  both call sites (`scenario_evaluation_runtime_indices` and `make_env`) via
  a new `_resolve_frozen_panel_uids()` helper: when
  `provider.panel_manifest_path` is configured, it is the authoritative
  source of the evaluation panel's order (loaded + hash-verified); if
  `provider.scenario_uids_file` is *also* configured, its UID set must agree
  exactly with the manifest's or construction fails closed (no silent
  preference between the two sources, per REQ-004's "no fallback sampling"
  invariant). When `panel_manifest_path` is not configured, behavior is
  byte-identical to before (today's catalog-order/`scenario_uids_file`
  derivation). A one-time operator script,
  `scripts/build_panel_manifest.py`, builds and persists the manifest from a
  real ScenarioNet catalog. **Not done**: no Hydra preset currently sets
  `provider.panel_manifest_path` to the two committed manifest files, so the
  frozen-manifest path is implemented, tested, and its artifacts exist, but
  is not yet the default for any run profile — enabling it for `thesis` runs
  is a follow-up config change (Milestone 8), not a code or data gap.
- Expected files: `src/thesis_rl/scenarios/panel_manifest.py` (new),
  `src/thesis_rl/envs/factory.py` (modified),
  `scripts/build_panel_manifest.py` (new), `tests/test_panel_manifest.py`
  (new), `tests/test_env_factory.py` (extended).
  `src/thesis_rl/runtime/io/metadata.py` per-run recording remains open.
- Tasks: implement manifest build/read/hash per §7.2 (done); wire
  `FixedSequenceScenarioProvider` construction to prefer the frozen manifest
  with fail-closed mismatch handling (done); record manifest identity per
  run (open); generate and check in the real validation/test manifests
  (open, blocked on real dataset access).
- Tests/commands: `TEST-EP-004`, `TEST-EP-005` implemented as
  `tests/test_panel_manifest.py` (13 tests) and `tests/test_env_factory.py`
  (5 tests, 4 new); `docker compose run --rm dev uv run --no-sync python -m
  pytest -q tests/test_panel_manifest.py tests/test_env_factory.py` ->
  `PASS` (18/18). `TEST-EP-006` (a real generated manifest artifact) not
  produced, see above.
- Completion evidence: passing focused tests (done); a generated manifest
  file for the current validation/test splits, checked into the appropriate
  data artifact location (not `src/`) — **not done**, real dataset access
  required.
- Decision dependencies: `DEC-EP-001` (resolved).

### Milestone 3 — Checkpoint identity persistence (`REQ-006`)

- Status: `IMPLEMENTED`; unit-tested (`TEST-EP-007`/`008` equivalents) and
  verified live: the smoke run's `run_metadata.yaml` and `final_eval.csv`
  both record `checkpoint_path=checkpoints/final.zip`,
  `checkpoint_hash=a9b0c...` (SHA-256), `checkpoint_role=final`. `best_*`
  checkpoints are untouched (still recorded with no hash/role, confirming
  they remain outside the official path, per `DEC-EP-002`).
- Expected files: `src/thesis_rl/runtime/loops/train_loop.py`,
  `src/thesis_rl/runtime/io/metadata.py`,
  `src/thesis_rl/runtime/io/csv_recorder.py`,
  extension of `tests/test_checkpointing.py`.
- Tasks: compute and record `checkpoint_hash`/`checkpoint_role` at the
  `final.zip` save/consumption sites per §7.3.
- Tests/commands: `TEST-EP-007`, `TEST-EP-008`.
- Completion evidence: passing focused tests; a real smoke run's
  `run_metadata.yaml`/`final_eval.csv` showing the new fields for `final.zip`
  only.
- Decision dependencies: `DEC-EP-002` (resolved).

### Milestone 4 — Data-abort coverage persistence (`REQ-012`)

- Status: `IMPLEMENTED` (2026-07-25 pass). All four `evals.csv` `append_row`
  call sites in `train_loop.py` now wire `_data_abort_coverage_fields(metrics)`
  identically: the three previously-unwired periodic-validation sites
  (async curriculum-eval block, no-curriculum block, staged-curriculum-gate
  block) plus the pre-existing final-test site. A new regression test,
  `test_every_evals_csv_append_row_call_wires_data_abort_coverage_fields`,
  statically scans `train_loop.py`'s source for every `"evals.csv"`
  `append_row(` call and asserts `_data_abort_coverage_fields(metrics)`
  appears in its call block, guarding against a future call site being
  added without the wiring (or the wiring being reverted).
- Expected files: `src/thesis_rl/runtime/loops/train_loop.py` (modified: 3
  new `**_data_abort_coverage_fields(metrics)` call sites),
  `tests/test_train_loop_eval_protocol_helpers.py` (extended with the
  static-scan regression test).
- Tasks: persist the already-computed `data_abort_coverage` dict to
  `evals.csv`/`final_eval.csv` (done for all sites); a per-evaluation JSON
  sidecar (per §7.5's "invalid_episodes to a per-evaluation JSON/JSONL
  sidecar") was **not** added in this pass — `invalid_episodes` remains
  in-memory only, not persisted to a sidecar file; this is a narrower
  residual gap than the original CSV-column gap this milestone targeted.
- Tests/commands: `docker compose run --rm dev uv run --no-sync python -m
  pytest -q tests/test_train_loop_eval_protocol_helpers.py` -> `PASS` (6/6).
  `ruff check src/thesis_rl/runtime/loops/train_loop.py` -> `PASS`.
- Completion evidence: passing focused tests; CSV-column wiring verified for
  all 4 `evals.csv` sites plus `final_eval.csv`. Per-evaluation
  `invalid_episodes` sidecar persistence remains open (not required for the
  CSV-column portion of REQ-012's coverage-reporting invariant, which is now
  fully wired).
- Decision dependencies: none.

### Milestone 5 — PPO atomic-boundary stop condition (`REQ-002`/`DEC-015`)

- Status: `IMPLEMENTED`. `Sb3PpoPlannerBackend.atomic_boundary_remaining()`
  and `Agent.atomic_boundary_remaining()` added and unit-tested against a
  real (small) SB3 PPO model: remaining count decreases exactly by 1 per
  collected transition and returns to 0 exactly when the buffer resets after
  an update (`tests/test_ppo_atomic_boundary.py`, 2/2 passing). The training
  loop now completes the final rollout via one bounded extra `train_fn` call
  before the `FINALIZATION` block if `pending_atomic_steps > 0`, and records
  `budget.target_timesteps`/`actual_completed_timesteps`/`overshoot_steps` in
  `run_metadata.yaml`. **Live-verified for the zero-overshoot (TD3) path**
  (`agent.atomic_boundary_remaining()` returns 0 for a non-PPO planner with
  no crash; smoke run recorded `budget.overshoot_steps: 0`).
  **Live-verified 2026-07-25 for a real PPO run with `pending_atomic_steps > 0`**:
  a new PPO smoke preset (`conf/presets/test/smoke_train_ppo.yaml`, non-vectorized
  path, `num_envs=1`) was assembled with `agent.planner.algorithm.n_steps=8`,
  `batch_size=4`, `n_epochs=2` (overriding `ppo_sb3.yaml`'s
  `96`/`63` defaults, which do not divide evenly at smoke scale — the
  pre-existing `run_profile.planner.ppo.{n_steps,batch_size}` block in
  `smoke.yaml` was checked and confirmed to be dead configuration, not
  consumed by any code path in `src/thesis_rl/`, so it does not actually
  override the algorithm) and `experiment.total_timesteps=30`, deliberately
  not a multiple of the atomic unit (`n_steps * num_envs = 8`), forcing a
  real overshoot. The run completed end to end
  (`docker compose run --rm dev uv run --no-sync python -m thesis_rl.cli.train
  --config-name presets/test/smoke_train_ppo`): the training-monitor output
  showed `Run env steps` advance from `30/30` to `32/30` on the bounded extra
  `train_fn` call, and the resulting `run_metadata.yaml` recorded exactly
  `budget: {target_timesteps: 30, actual_completed_timesteps: 32,
  overshoot_steps: 2}` — the precise arithmetic the unit test already
  predicted, now confirmed against a real SB3 PPO model end to end, not just
  the small in-memory model in `tests/test_ppo_atomic_boundary.py`. The
  `Agent.train_vectorized()` interaction flagged `AWAITING_CONFIRMATION` in
  §4.1 remains open: this smoke run used the non-vectorized path
  (`num_envs=1`), so the overshoot-completion step's reuse of `train_fn`
  under `train_vectorized` (the vectorized-profile code path) is still not
  live-exercised.
- Expected files: `src/thesis_rl/runtime/loops/train_loop.py`,
  `src/thesis_rl/agent/agent.py`,
  `src/thesis_rl/agent/planners/algorithms/ppo_sb3.py`, new/extended tests
  under `tests/`.
- Tasks: resolve the `AWAITING_CONFIRMATION` item on `train_vectorized`'s
  interaction with the PPO buffer boundary (§4.1) before writing code;
  implement the bounded-overshoot extension per §7.1 and `DEC-EP-003`; record
  `target_timesteps`/`actual_completed_timesteps`/`atomic_collection_unit`/
  `overshoot_steps`.
- Tests/commands: `TEST-EP-001`, `TEST-EP-002`, `TEST-EP-003`; this is the
  highest-risk milestone (touches the core training loop) and requires a real
  `run_profile=smoke` PPO training smoke in addition to unit tests before
  being marked complete.
- Completion evidence: passing tests; a smoke run log showing the recorded
  overshoot; confirmation that TD3/SAC smoke runs are unaffected
  (`overshoot_steps == 0`).
- Decision dependencies: `DEC-EP-003` (resolved); the `train_vectorized`
  `AWAITING_CONFIRMATION` item must be closed first.

### Milestone 6 — Run disposition (`REQ-011`)

- Status: `IMPLEMENTED`; `disposition`/`disposition_rationale` fields
  (additive, default absent/`unset`) and `scripts/set_run_disposition.py`
  implemented and unit-tested (`tests/test_run_metadata.py`). Never set
  automatically, matching the REQ-011 invariant.
- Expected files: `src/thesis_rl/runtime/io/metadata.py`, new small CLI/script
  (path TBD during implementation), new test file.
- Tasks: add `disposition`/`disposition_rationale` fields per §7.6; add the
  manual-entry CLI/script.
- Tests/commands: `TEST-EP-014`, `TEST-EP-015`.
- Completion evidence: passing tests; a documented example of setting
  disposition for a fixture run.
- Decision dependencies: `DEC-EP-004` (resolved).

### Milestone 7 — Analysis pipeline: remove CI, reconcile R1--R3/R4 (`REQ-007`, `REQ-008`, `REQ-009`)

- Status: `PARTIAL`. **CI removal is complete** and covers a larger surface
  than originally scoped: the `1.96 * s / sqrt(n)` formula was found and
  removed not only in the 4 files identified during specification review
  (`common_stats.py`, `make_final_tables.py`, `make_plots.py`,
  `make_factor_effect_tables.py`) but also in 4 additional consumers of the
  shared `common_stats.mean_ci95` helper that the original grep for the
  literal `1.96` missed: `make_curriculum_tables.py`, `make_rulebook_tables.py`,
  `make_sample_efficiency_tables.py`, `make_generalization_tables.py`. The
  shared helper is renamed `mean_sd` (returns mean + sample SD only); every
  `_ci95`-suffixed CSV/report column across these 8 files is renamed to
  `_sd`; `make_final_tables.py` additionally now emits raw per-seed values
  (`{metric}_seed_values`, semicolon-joined) alongside mean/SD, satisfying
  REQ-009's "report all three raw seed values" invariant. Verified: all 8
  files import-check and syntax-check cleanly; a new regression test
  (`tests/test_analysis_no_confidence_interval.py`) exercises
  `build_final_tables` end-to-end on a synthetic 3-seed fixture and asserts
  no `ci95`/`ci_95`/"confidence" string appears in any output. A **separate,
  intentionally out-of-scope** per-chunk training-monitor statistic,
  `ep_rew_ci_95`/`ep_len_ci_95` in `train_chunks.csv` (computed in
  `agent.py:776-810,1527-1549` from a rolling window of recent training
  episodes within one run), was left untouched: it is a live-monitoring
  diagnostic over an episode population within a single run, not the
  cross-seed scientific claim `REQ-009`/`DEC-003` target ("the official
  condition summary" aggregating seed-level values); changing it would touch
  the training hot path for no REQ-009 benefit.
  **2026-07-25 pass (amended `DEC-003`, plus `REQ-007` R1--R3/R4 split)**:
  (a) added `common_stats.ci95(sd, n) = 1.96 * sd / sqrt(n)`, an optional,
  off-by-default confidence-interval helper. Wired an `include_ci: bool =
  False` parameter (default preserves byte-identical output) through
  `make_final_tables.build_final_tables`,
  `make_curriculum_tables.build_curriculum_tables`,
  `make_rulebook_tables.build_rulebook_tables`,
  `make_sample_efficiency_tables.build_sample_efficiency_tables`, and
  `make_generalization_tables.build_generalization_tables`; when enabled,
  each emits an additive `{metric}_ci95` column alongside the existing
  `_mean`/`_sd` columns, never replacing them. `run_analysis.py` gained a
  mirrored `--include-ci` CLI flag (same `action="store_true"`,
  default-`False` pattern as `--include-effects-tables`), threaded through
  `_build_tables_for_root`. **Not included in this CI-flag pass**:
  `make_plots.py` (a visualization, not a CSV table — adding a CI band to
  plots is a materially different mechanism than a column and was judged
  out of the additive-column scope this amendment describes) and
  `make_factor_effect_tables.py` (already gated behind the separate
  `--include-effects-tables` ablation flag; its own private `_mean_sd` was
  left untouched as lower-priority diagnostic-only surface). (b) **R1--R3/R4
  split implemented**: `Agent.evaluate()`'s `per_rule` rows use the four
  `MacroRule` names (`collision_impact`/`dynamic_interaction_safety`/
  `road_traffic_compliance`/`route_progress`, see
  `src/thesis_rl/rulebook/v2/types.py`); `route_progress` (R4) was previously
  reported in the same `rule_violation_by_rule.csv`/`.md` table as R1--R3
  using shared `violation_rate`/`violated` constraint-rule columns — a
  genuine REQ-007 conflation. `make_rulebook_tables.py` now splits
  `route_progress` rows into a new, separate
  `rulebook_r4_progress_margin.csv`/`.md` output with renamed,
  non-constraint columns (`negative_progress_rate_*`,
  `mean_progress_margin_*`, `min_progress_margin_*`,
  `max_progress_margin_*` -- the same underlying values, relabeled; no
  Rulebook aggregation/margin semantics changed), while
  `rule_violation_by_rule.csv`/`.md` now contains only R1--R3 rows.
- Expected files: `src/thesis_rl/analysis/common_stats.py`,
  `make_final_tables.py`, `make_plots.py`, `make_factor_effect_tables.py`,
  `make_rulebook_tables.py`, `make_curriculum_tables.py`,
  `make_sample_efficiency_tables.py`, `make_generalization_tables.py`,
  `run_analysis.py`, associated tests under `tests/`.
- Tasks: remove the four `1.96 * s / sqrt(n)` computations (done, prior
  pass); update every caller (done); implement the R1--R3/R4 metric split in
  the rulebook table builder per §7.7 (done); add the optional opt-in CI
  column per the 2026-07-25 `DEC-003` amendment (done for the 5 core CSV
  table builders listed above).
- Tests/commands: new `tests/test_analysis_optional_ci_and_r4_split.py` (6
  tests: `ci95` formula match, zero-`n` guard, default-off has no CI column,
  `include_ci=True` adds the column without removing mean/SD, R1--R3 and R4
  land in distinct tables, R4 table supports the optional CI column and
  never uses "violat*" terminology). `docker compose run --rm dev uv run
  --no-sync python -m pytest -q tests/test_analysis_no_confidence_interval.py
  tests/test_analysis_optional_ci_and_r4_split.py` -> `PASS` (8/8). `ruff
  check` on all touched analysis files -> `PASS`.
- Completion evidence: passing tests; a fixture 3-seed comparison run through
  the pipeline with no CI field anywhere in the output by default, and a
  verified `_ci95` column present only when `--include-ci`/`include_ci=True`
  is passed. R1--R3/R4 structural separation verified against a synthetic
  fixture with distinct R1 and R4 margins.
- Decision dependencies: none; depends on Milestone 4 (needs
  `data_abort_coverage` in CSV to correctly exclude aborted episodes from the
  fixture) only for full end-to-end validation, not for unit-level work.

### Milestone 8 — Canonical report and entry point (`REQ-016`, `REQ-018`, `DEC-010`)

- Status: `PARTIAL`. Discovered `src/thesis_rl/analysis/run_analysis.py`
  already gates factor-effect (ablation) tables behind an explicit
  `--include-effects-tables` flag (`action="store_true"`, default `False`) —
  `REQ-018`'s "core report never includes ablations by default" requirement
  was already satisfied at the code level. Discovered and fixed a genuine
  conformance bug: `run_analysis.py --seed-list` defaulted to the superseded
  10-seed list `"0,1,2,3,4,5,6,7,8,9"`; changed the default to the official
  `"0,1,2"` (`DEC-001`). Added a canonical, documented `make analyze
  RUN_PROFILE=<profile>` Makefile target (`ANALYSIS_ARGS=...` for extra
  flags) and recorded it in `AGENTS.md`'s Verified Commands.
  **2026-07-25 pass**: `conf/presets/td3/{td3_native,td3_monitor_only,
  td3_scalar_reward}_{curr,no_curr}.yaml` (the 6 files matching the
  ExecPlan §7.8 pattern; `td3_scalar_reward_scale_tuning_no_curr.yaml` is a
  separate diagnostics preset, not part of this 3x2 matrix, and was left
  untouched) each now carry an explicit `analysis.experiment_group`
  override, all distinct, all prefixed `ABLATION-CURRICULUM-` (e.g.
  `ABLATION-CURRICULUM-NATIVE-CURR-01`, `...-NATIVE-NOCURR-01`,
  `...-MONITOR-CURR-01`, `...-MONITOR-NOCURR-01`,
  `...-SCALARREWARD-CURR-01`, `...-SCALARREWARD-NOCURR-01`), so `REQ-018`'s
  "distinct comparison-block IDs" invariant is enforced by configuration
  (verified by composing each preset with Hydra and reading
  `cfg.analysis.experiment_group`), separable from `BASELINE-SCALAR-01`/
  `EXTENSION-ALGORITHM-01`. **Still not done**: an actual end-to-end `make
  analyze` run against real aggregated artifacts (only `build_final_tables`/
  `build_rulebook_tables` were exercised via synthetic fixtures, not the
  full `aggregate_runs` -> `run_analysis` chain against real run outputs).
- Expected files: `src/thesis_rl/analysis/run_analysis.py`, `Makefile`,
  `AGENTS.md`, `conf/presets/td3/td3_native_curr.yaml` and five siblings
  (all modified, 2026-07-25), `tests/test_td3_curriculum_ablation_tagging.py`
  (new).
- Tasks: confirm/add the canonical documented entry point (done, prior
  pass); gate factor-effect/generalization (ablation) outputs behind an
  explicit flag per §7.7 (done, prior pass); tag the six `td3`
  curriculum-ablation presets with a distinct `experiment_group` per §7.8
  (done, 2026-07-25).
- Tests/commands: new `tests/test_td3_curriculum_ablation_tagging.py` (3
  tests: each preset declares an `ABLATION-CURRICULUM-`-prefixed group, all
  6 groups are pairwise distinct, none collides with the two core
  comparison-block IDs). `docker compose run --rm dev uv run --no-sync
  python -m pytest -q tests/test_td3_curriculum_ablation_tagging.py
  tests/test_hydra_preset_run_configs.py tests/test_hydra_preset_test_configs.py
  tests/test_hydra_agent_presets.py` -> `PASS` (45/45, no regression in the
  42 pre-existing Hydra preset-composition tests). `ruff check` on the new
  test file -> `PASS`.
- Completion evidence: passing tests (done); two consecutive runs of the
  canonical command on the same fixture producing identical output
  (pre-existing, prior pass); documented command added to `AGENTS.md`
  (pre-existing, prior pass). End-to-end `make analyze` against real
  aggregated artifacts remains `NOT_RUN`.
- Decision dependencies: depends on Milestone 7 (report content must already
  be conformant before the entry point is declared canonical).

### Milestone 9 — Qualitative selection reconciliation (`REQ-014`)

- Status: `DONE` (2026-07-25, fourth pass). The one remaining gap from the
  third pass -- actual GIF pixel rendering for the periodic tracked subset,
  previously `AWAITING_CONFIRMATION` -- is closed. See the fourth-pass entry
  below the third-pass account (retained for history) for the resolution.
- Status (2026-07-25, third pass, historical). Both `AWAITING_CONFIRMATION`
  items from the previous pass were resolved by explicit user approval and
  implemented:
  1. **Category taxonomy** (approved, no compatibility shim): renamed
     `make_qualitative_manifest.py`'s categories to the four `REQ-014`
     names -- `best`->`representative_success`,
     `worst`->`representative_failure`,
     `rule_violation_case`->`severe_rule_violation`. `median` was **dropped**
     (no `REQ-014` equivalent -- the invariant requires "exactly one of four
     approved categories", and a median-route-completion pick does not map
     to success/failure/violation/disagreement). `curriculum_transition_case`
     was **dropped**, not reclassified as `algorithm_disagreement` -- reading
     `_pick_transition_case`, it picks the eval episode nearest a
     curriculum-promotion `global_step` within one run, a curriculum
     diagnostic, not a cross-condition shared-`scenario_uid` comparison, so
     relabeling it would misrepresent what it selects. `algorithm_disagreement`
     is a genuinely new selector (`_pick_disagreement`): for the current
     condition, it looks across every *other* condition in the same
     comparison for the identical `scenario_uid` with a diverging boolean
     `success` outcome, preferring the largest `route_completion` gap; ties
     with no diversity signal are `unavailable` with an explicit reason
     (`no_scenario_uid_recorded` / `no_other_conditions` /
     `no_shared_scenario_uid_with_diverging_outcome`). This uses only a
     differing `success` outcome as the divergence criterion (no continuous
     similarity/disagreement threshold), so no `AWAITING_CONFIRMATION`
     threshold question arose. `promotions_all_runs.csv` loading was removed
     (only consumer was the dropped transition-case selector).
  2. **Periodic tracked-subset render cadence** (approved: every 100,000
     timesteps): `train_loop.py` gained `_make_tracked_subset_render_gate(
     interval=100_000)` -- a stateful closure firing at the first periodic
     eval boundary at or after each crossed multiple of 100,000 (robust to
     any `eval_interval`, not just the default 25,000 that evenly divides
     it) -- and `_tracked_subset_uids_from_resolved_env_cfg()`, which reads
     `manifest.tracked_subset_uids` off whichever frozen panel manifest is
     configured for that eval's resolved env config
     (`provider.panel_manifest_path`). At the periodic-validation call site
     (right after each eval's `eval_episodes.csv` rows are written, before
     the curriculum-progression branch), `select_tracked_subset_episodes()`
     is called **unconditionally** every periodic eval (a cheap CSV scan,
     keeping the manifest complete per `REQ-014`'s "the manifest is always
     produced"), while a `tracked_subset_render_due(...)` check gates only a
     log line marking when the actual GIF render would fire. At the
     final-test call site, the same selection call fires unconditionally,
     with no cadence gate (final test always covers the complete panel).
  **Still `AWAITING_CONFIRMATION`**: actual GIF pixel rendering for the
  tracked subset. `render_selected_videos.py`/`render_qualitative_videos.py`
  operate by loading a Hydra config snapshot and checkpoint back off disk
  and replaying a fresh episode out-of-process -- they are post-hoc/replay
  scripts, not designed to capture frames from the *live* in-memory
  policy/env during an ongoing eval rollout. The infrastructure that *can*
  do that -- `eval_artifacts.LiveEvalEpisodeRecorder` /
  `maybe_build_live_final_eval_recorder_factory` -- already exists and is
  already wired for `eval_type="final"`, gated by
  `cfg.video.record_intermediate_evals` for non-final evals, but is not yet
  wired into the periodic-validation call site at all, and does not filter
  which episodes it records down to just the tracked-subset `scenario_uid`s
  (it currently records everything-or-nothing per eval). Threading
  `tracked_scenario_uids` through that factory to record only the tracked
  episodes, at the periodic call site, gated by the new cadence, is real
  additional surgery beyond a small reviewable change and was not attempted
  in this pass; the cadence gate, selection JSON, and an explicit log
  marker are implemented and tested so the remaining work is a clearly
  scoped, isolated follow-up.
  Separately, this pass also revised the tracked-subset *sizing and
  selection criterion itself* per a second approved user decision: from a
  flat 5-UID one-per-arm round-robin to **4 scenarios/arm (validation) / 10
  scenarios/arm (test)**, feature-diversity greedy-selected within each
  arm's already-drawn panel candidates (see §7.2 update above) -- this
  replaces the previous pass's `select_tracked_subset_uids()` runtime helper
  as the authoritative source; `train_loop.py` now reads
  `manifest.tracked_subset_uids` (persisted at build time) instead.
- Expected files: `src/thesis_rl/scenarios/panel_manifest.py`,
  `scripts/build_panel_manifest.py`, `src/thesis_rl/analysis/videos/
  make_qualitative_manifest.py`, `src/thesis_rl/runtime/loops/train_loop.py`
  (all modified, see §13). `render_selected_videos.py`/
  `render_qualitative_videos.py`: not modified -- the periodic in-process
  GIF-rendering wiring remains open, see above.
- Tasks: read both video-selection files in full (done, prior pass);
  reconcile category taxonomy with `REQ-014` (done, this pass); wire the
  tracked-subset selection + cadence gate into `train_loop.py` (done, this
  pass -- selection JSON only, GIF pixel rendering still open); revise
  tracked-subset sizing/diversity (done, this pass).
- Tests/commands: `tests/test_qualitative_manifest.py` (4 new tests: all
  four categories emitted, old category names never emitted,
  `algorithm_disagreement` selects the diverging shared-`scenario_uid`
  case, `algorithm_disagreement` reports `unavailable` for a single-condition
  comparison); `tests/test_panel_manifest.py` (+8 tests: diversity greedy
  selection prefers distinct feature-tuples, prefix-order fallback without a
  feature lookup, determinism, subset/duplicate invariants, save/load
  round-trip, backward-compatible load without the key);
  `tests/test_train_loop_eval_protocol_helpers.py` (+7 tests: cadence gate
  fires once per 100k window including with a non-divisor `eval_interval`,
  rejects non-positive interval, resolved-env-cfg tracked-uid resolution
  including the nested `config` shape and a real manifest round-trip).
  `docker compose run --rm dev uv run --no-sync python -m pytest -q
  tests/test_panel_manifest.py tests/test_tracked_subset_video_selection.py
  tests/test_video_selection_authoritative.py tests/test_qualitative_manifest.py
  tests/test_train_loop_eval_protocol_helpers.py tests/test_eval_artifacts.py`
  -> `PASS` (44/44). `ruff check` on every touched file -> `PASS`. Both
  committed panel manifests were regenerated against the real ScenarioNet
  catalog via `docker compose run --rm dataset-pipeline uv run --no-sync
  python scripts/build_panel_manifest.py --split {validation,test} --size
  {100,300} --seed 20260725` (see §7.2 update for hashes/counts).
- Completion evidence: passing tests (done); a fixture qualitative manifest
  exercising all four `REQ-014` categories including a shared-`scenario_uid`
  cross-condition `algorithm_disagreement` case -- done
  (`tests/test_qualitative_manifest.py`).
- Decision dependencies (third pass, historical): none remaining for category
  taxonomy or tracked-subset selection mechanism/sizing (both approved and
  implemented that pass); periodic in-process GIF pixel rendering was
  `AWAITING_CONFIRMATION`/open at that point -- resolved below.

**Fourth pass (2026-07-25): periodic in-process GIF pixel rendering
closed.** The blocker identified in the third pass -- `scenario_uid` is not
known at recorder-*creation* time for the parallel evaluation path
(`Agent._evaluate_parallel._install_episode` calls `artifact_recorder_factory`
before `env.reset_slots(...)`, so the tracked-subset allowlist can't gate
which episodes get a real recorder up front) -- was resolved without any
final-test-path change, by deferring the allowlist decision from recorder
*creation* to `finalize_episode`:

- `eval_artifacts.LiveEvalEpisodeRecorder.record_step` now latches
  `manifest_payload["scenario_uid"]` from the first `step_info` that carries
  one (`thesis_scenario_env.step`/`reset` always merge scenario metadata,
  including `scenario_uid`, into `info`), if not already set at
  construction.
- `LiveEvalEpisodeRecorder` gained two additive, default-preserving
  constructor kwargs: `output_dir_name: str = "final_eval"` (final-test
  keeps its exact pre-existing default) and
  `tracked_scenario_uids: frozenset[str] | None = None`. When the latter is
  set, `finalize_episode` persists to disk (GIF/manifest/trajectory) only if
  the now-known `scenario_uid` is in the allowlist; otherwise it discards
  the buffered frames and returns the same no-op-shaped result
  (`replay_warning="periodic_tracked_subset_skip:not_in_tracked_subset"`, all
  paths `None`) without ever writing to disk.
- New `build_periodic_tracked_subset_recorder_factory`/
  `maybe_build_periodic_tracked_subset_recorder_factory` in
  `eval_artifacts.py` (parallel to, but not sharing code path risk with,
  `build_live_final_eval_recorder_factory`/
  `maybe_build_live_final_eval_recorder_factory`, which are both completely
  untouched in signature and behavior). The `maybe_...` gate requires
  `cfg.video.enabled` + `mode=="live_final_eval"` (reused infra) and a
  non-empty tracked-subset, and is deliberately independent of
  `cfg.video.record_intermediate_evals` (that flag governs the separate,
  pre-existing full-panel intermediate recording, not this DEC-014
  mechanism).
- GIFs land under `videos_dir/periodic_eval/step_<global_step:07d>/
  eval_<eval_id:04d>/<scenario_uid>.gif` (plus sibling `.manifest.json`/
  `.trajectory.jsonl`), distinct from the final-test convention
  (`videos_dir/final_eval/eval_<eval_id:04d>/episode_<episode_id:04d>.gif`)
  and named by `scenario_uid` so a tracked case is trivially comparable
  across evals.
- `train_loop.py`'s periodic-validation call site now resolves
  `tracked_scenario_uids`/`periodic_tracked_subset_render_due` *before*
  `eval_agent.evaluate(...)` runs (the recorder factory must be passed into
  the live rollout, not built after it completes) and passes
  `artifact_recorder_factory=periodic_tracked_subset_artifact_factory` into
  that call. The stateful `tracked_subset_render_due(...)` gate closure is
  now called exactly once per periodic eval (previously it would have been
  called once for logging only, after the fact; calling it twice per eval
  would have silently skipped the next 100k-window). The post-eval block
  that writes `tracked_subset_selection.json` (unconditional, per `REQ-014`)
  now reuses these already-computed values instead of recomputing them.
- Trade-off recorded as an implementation decision (not a scientific-behavior
  deviation -- output content, selection, and manifests are unchanged; only
  the internal rendering mechanism was decided): when the cadence fires,
  every episode of that periodic evaluation gets a real
  `LiveEvalEpisodeRecorder` (frames buffered in memory), and only the
  tracked-subset episodes are persisted; untracked episodes' frames are
  discarded at `finalize_episode`. This bounds *on-disk* output to the small
  fixed tracked subset but not the *compute* cost of that one gated
  evaluation (bounded to once per 100,000 timesteps, not every periodic
  eval). This matches option (b) considered in the third-pass write-up
  ("render more broadly but bounded and safe") rather than a scenario_uid-uid
  aware pre-filter, because `scenario_uid` is unavailable before `env.reset`
  in the parallel evaluation path.
- Live end-to-end smoke (a real periodic eval producing on-disk GIFs during
  an actual training run) was not executed in this pass -- it requires a
  training run reaching the 100,000-timestep cadence boundary with
  `cfg.video.enabled=true`/`mode=live_final_eval`, which is not a
  few-minutes-cheap smoke path; deferred, see §11 and the completion
  evidence below for the exact follow-up command.
- Tests added: `tests/test_eval_artifacts.py` (+7: periodic recorder persists
  a tracked `scenario_uid` to the new path convention; discards an untracked
  `scenario_uid` with no disk writes; `maybe_build_...` returns `None` for
  empty tracked-uids / disabled video; `maybe_build_...` fires independently
  of `record_intermediate_evals`; final-test recorder factory regression --
  unaffected path/filename/unconditional-recording behavior).
  `tests/test_train_loop_eval_protocol_helpers.py` (+3: periodic
  `evaluate()` call site wires `artifact_recorder_factory=
  periodic_tracked_subset_artifact_factory`; the render gate is called
  exactly once per periodic eval; final-test `evaluate()` call site is
  unaffected/still uses `final_eval_artifact_factory`).
  `docker compose run --rm dev uv run --no-sync python -m pytest -q
  tests/test_eval_artifacts.py tests/test_train_loop_eval_protocol_helpers.py
  tests/test_tracked_subset_video_selection.py
  tests/test_video_selection_authoritative.py tests/test_panel_manifest.py`
  -> `PASS` (49/49). `ruff check` on
  `src/thesis_rl/runtime/io/eval_artifacts.py`,
  `src/thesis_rl/runtime/loops/train_loop.py`,
  `tests/test_eval_artifacts.py`,
  `tests/test_train_loop_eval_protocol_helpers.py` -> `PASS`.
- Decision dependencies: none remaining; `REQ-014`/`DEC-014` (as amended
  2026-07-25) is now fully implemented and tested.

### Milestone 10 — Applicability-aware R1--R3 aggregation (`REQ-008`)

- Status: `DONE` (2026-07-25). This was the sole remaining `NOT_IMPLEMENTED`
  requirement-level gap from §15's prior Final Reconciliation pass, added as
  a new milestone here since it was outside this ExecPlan's original
  Milestone 1--9 scope.
- Discovered: the per-step applicability signal spec §7.2 requires
  (`RuleComponentResult.applicable`) was already computed and already
  exposed on every step via `step_info["rule_components"][macro_rule_name]
  ["applicable"]` (`RulebookV2MonitorWrapper.step`,
  `src/thesis_rl/rulebook/v2/wrapper.py:220`) -- no Rulebook v2 change was
  needed or made. The actual gap was entirely in `Agent`'s episode/seed-level
  aggregation (`src/thesis_rl/agent/agent.py`): it read only
  `rule_reward_vector` (margins), never `rule_components`, so every step
  (applicable or not) fed episode-minimum-margin and violation counting, and
  `violation_rate` was computed as "episodes with >=1 violated step" over
  the *total* episode count rather than spec §7.2's `N'_s` (episodes with
  >=1 applicable step for that rule). See §7.9 for the full design account.
- Expected files: `src/thesis_rl/agent/agent.py`,
  `src/thesis_rl/runtime/io/csv_recorder.py`,
  `src/thesis_rl/runtime/loops/eval_loop.py`,
  `src/thesis_rl/runtime/loops/train_loop.py`,
  `src/thesis_rl/analysis/tables/make_rulebook_tables.py`,
  `tests/test_eval_protocol_req008_applicability_aggregation.py` (new).
- Tasks: add `Agent._extract_rule_applicability` (done); thread
  applicable-step counters through `_ParallelEvaluationEpisode`/
  `_aggregate_parallel_evaluation` and the duplicated serial `evaluate()`
  loop (done, both paths); redefine `violation_rate` to
  `mean(EpisodeViolationRate over N'_s episodes)` and add
  `applicable_episode_count`/`excluded_episode_count` to `per_rule` rows
  (done); add the two additive `rule_metrics.csv` columns and thread them
  through both `_append_rule_metrics_rows` call sites (done); surface the
  two counts as additive `_mean`/`_sd` columns in the R1--R3
  `rule_violation_by_rule.csv`/`.md` table only, not the R4 table (done).
- Tests/commands: new `tests/test_eval_protocol_req008_applicability_aggregation.py`
  (4 tests: violation rate excludes the zero-applicable-step episode from
  the denominator and numerator; episode-minimum-margin aggregation uses
  applicable steps only; the naive total-episode-count denominator would
  have produced a different, wrong value, confirming the fix is behavior-
  changing and not a no-op; the serial and parallel evaluation paths
  produce byte-identical `per_rule` rows for the same fixture). `docker
  compose run --rm dev uv run --no-sync python -m pytest -q
  tests/test_eval_protocol_req008_applicability_aggregation.py
  tests/test_parallel_evaluation.py tests/test_parallel_evaluation_data_abort.py
  tests/test_agent_pipeline.py tests/test_eval_artifacts.py
  tests/test_scenario_acl_usefulness.py tests/test_analysis_no_confidence_interval.py
  tests/test_analysis_optional_ci_and_r4_split.py
  tests/test_train_curriculum_helpers.py` -> `PASS` (51/51). Full repository
  suite: `docker compose run --rm dev uv run --no-sync python -m pytest -q`
  -> `970 passed, 4 failed` -- the same 4 pre-existing, unrelated
  `test_causal_semantic_batch.py` route-projection failures already recorded
  in this ExecPlan's Metadata section prior to this milestone (verified by
  inspecting the failing test names and tracebacks: `RoutePolyline.project`
  raising "no vertically compatible segment", a static-feature geometry
  path never touched by this change). `ruff check` on every touched
  production/test file -> `PASS`.
- Completion evidence: passing new regression tests demonstrating the fix
  is behavior-changing (not a no-op) and that both evaluation paths agree;
  no new failures in the full suite; `rule_metrics.csv`/
  `rule_metrics_all_runs.csv` schema changes are additive-only per this
  ExecPlan's Compatibility Constraints (existing columns/rows unchanged in
  shape, two new columns appended).
- Decision dependencies: none; no specification deviation was required --
  the fix implements spec §7.2's formulas as written using data already
  produced by the (untouched) Rulebook v2 module.

## 11. Progress And Findings Log

- **2026-07-24**: ExecPlan created following the approval of
  `EVAL-PROTOCOL` v1.0. Repository research for this plan confirmed the
  specification's own §11 gap list and additionally discovered: (a) a
  substantial pre-existing analysis pipeline under `src/thesis_rl/analysis/`
  (`run_analysis.py`, `aggregate/`, `comparisons/`, `tables/`, `plots/`,
  `videos/`) not previously inventoried during the specification review,
  meaning Milestones 7--9 are reconciliation work against existing code, not
  greenfield implementation; (b) the prohibited `1.96 * s / sqrt(n)` CI
  formula is implemented in exactly four places
  (`common_stats.py:21`, `make_final_tables.py:64`, `make_plots.py:65`,
  `make_factor_effect_tables.py:54`), giving Milestone 7 a precise, bounded
  scope; (c) `Agent.train_vectorized()`'s interaction with the PPO
  rollout-buffer boundary was not verified and is flagged
  `AWAITING_CONFIRMATION`, gating Milestone 5; (d) no existing test file
  matching the analysis pipeline (e.g. `tests/test_analysis*.py`) was found by
  name during this plan's research — this must be confirmed before Milestone
  7 begins, since new tests may need a new test module rather than extending
  an existing one. No implementation has started. Next step: user
  confirmation to begin Milestone 1 (lowest-risk, no cross-milestone
  dependency) or a preferred alternative starting point.
- **2026-07-24 (autonomous implementation pass)**: on explicit user
  instruction ("Procedi con la completa implementazione del piano in
  autonomia... evitando run lunghe... al più smoke"), implemented Milestones
  1, 3, 5, 6, 8 fully and Milestones 4 and 7 partially; Milestones 2 and 9 not
  started. Key findings beyond the original scope: (a) `common_stats.py`'s
  shared `mean_ci95` helper was consumed by four additional table builders
  (`make_curriculum_tables.py`, `make_rulebook_tables.py`,
  `make_sample_efficiency_tables.py`, `make_generalization_tables.py`) that
  the specification-review-time grep for the literal `1.96` missed —
  Milestone 7's CI removal therefore covers 8 files, not 4; (b)
  `run_analysis.py --seed-list` defaulted to the superseded 10-seed list —
  fixed to `0,1,2`; (c) `run_analysis.py` already gated ablation/factor-effect
  tables behind `--include-effects-tables` (default off), so `REQ-018`'s core
  invariant was already satisfied at the code level; (d) `torch.__version__`
  is a `TorchVersion` (str subclass) that PyYAML's `SafeDumper` cannot
  represent — caught by the new metadata tests and fixed with an explicit
  `str()` coercion before any real run was attempted. All changes were
  validated: focused unit tests for every implemented milestone (34 new/
  extended test cases across 5 new test files plus 2 extended files, all
  passing); `ruff check` clean on every touched file; a full `make smoke`
  (TD3, `run_profile=smoke`, real ScenarioNet data, real GPU) completed
  end-to-end and its `run_metadata.yaml`/`final_eval.csv`/`evals.csv`
  artifacts were inspected directly and confirmed to contain every new field
  with correct values; the full repository test suite (912 tests) was run and
  showed exactly 3 pre-existing failures, all in
  `tests/test_causal_semantic_batch.py` (Rulebook v2 route-projection
  geometry), unrelated to any file this plan touched and consistent with
  concurrent unrelated work visible in `git status` at session start
  (modified `src/thesis_rl/rulebook/v2/{controls,conflict_zones,transition}.py`).
  No PPO-specific end-to-end smoke was run (no ready-made PPO smoke preset
  exists; assembling one was judged not worth the risk/time given the
  `atomic_boundary_remaining()` unit test already validates the exact
  arithmetic against a real SB3 PPO model). Next step: Milestones 2 (panel
  manifest) and 9 (qualitative selection), the remaining CSV call sites for
  Milestone 4, the R1--R3/R4 split for Milestone 7, and the `conf/presets/td3`
  tagging for Milestone 8.
- **2026-07-25 (specification amendment round)**: before resuming Milestones
  2 and 9, the user proposed three refinements, all reviewed against already-
  `APPROVED` decisions and resolved directly in
  `docs/specifications/evaluation_protocol_v1.0_specification.md` (not a new
  spec version — recorded as amendments to `DEC-003`/`DEC-004`/`DEC-014`
  dated 2026-07-25): (a) a deterministic seed-driven balanced draw across the
  six scenario arms is now the approved mechanism for building the frozen
  panel under `REQ-004`/`DEC-005` (Milestone 2 scope), but the accompanying
  proposal to redraw a replacement scenario on a runtime data abort was
  **rejected** and `DEC-004`'s no-backfill policy reconfirmed: which scenario
  aborts depends on the policy under evaluation, not only the seed, so
  backfilling would make the effective evaluated scenario set
  policy-dependent and reintroduce the cross-condition panel-divergence risk
  `DEC-005` exists to prevent; (b) `REQ-014`/`DEC-014` amended to add a small
  fixed, pre-declared tracked subset of `scenario_uid`s (default 5 per split)
  rendered as GIFs unconditionally at every evaluation (periodic validation
  and final test alike, same UIDs, for training-progression visibility),
  while every other episode is recorded only in a per-episode manifest
  (scenario type/arm, characteristics, outcome, R1--R4 metrics) from which
  GIFs can be rendered later out-of-band by a separate manifest-driven
  script; the post-hoc four-category selection scheme for the qualitative
  appendix is otherwise unchanged; (c) `REQ-009`/`DEC-003` amended to allow
  an optional, off-by-default confidence-interval column
  (`1.96 * sd_a(x) / sqrt(n)`) alongside the mandatory raw-values/mean/SD
  core, gated by an analyst opt-in flag. Milestones 2 and 9 are unblocked and
  resume against this amended text.
- **2026-07-25 (implementation pass on Milestones 2/4/7/8/9)**: implemented,
  unit-tested, ruff-checked, and reconciled the requested milestones against
  the amended spec text above. Summary: Milestone 4 (data-abort coverage
  persistence) is now `IMPLEMENTED` -- all four `evals.csv` `append_row`
  call sites wire `_data_abort_coverage_fields`, guarded by a new
  static-scan regression test. Milestone 8's remaining gap (the 6 `td3`
  ablation presets' `experiment_group` tagging) is done, verified by
  composing each preset with Hydra and asserting 6 pairwise-distinct
  `ABLATION-CURRICULUM-`-prefixed values disjoint from the two core
  comparison-block IDs. Milestone 7 gained the amended opt-in
  `--include-ci`/`include_ci` CI95 column (5 core table builders +
  `run_analysis.py`, default-off byte-identical) and completed the
  previously-open R1--R3/R4 split in `make_rulebook_tables.py`
  (`route_progress` now reported in a separate
  `rulebook_r4_progress_margin.csv`/`.md` with non-constraint column names,
  never "violat*"). Milestone 2 (panel manifest) gained a new
  `src/thesis_rl/scenarios/panel_manifest.py` module (deterministic
  seed-driven balanced draw across the six arms, SHA-256 order-sensitive
  hash, fail-closed load verification) wired into both
  `envs/factory.py` `fixed_sequence`-provider construction sites via an
  opt-in `provider.panel_manifest_path` config key (no behavior change when
  unset); `save_run_metadata()` recording of the manifest identity per run,
  and generating/checking in a real validation/test manifest against the
  actual ScenarioNet catalog, remain open (no verified real-dataset access
  in this session; see Milestone 2 above). Milestone 9 (qualitative
  selection) gained the tracked-subset selection mechanism
  (`select_tracked_subset_uids` + `select_tracked_subset_episodes`,
  extending rather than duplicating `select_video_episodes.py`'s existing
  helpers) but **two sub-items are flagged `AWAITING_CONFIRMATION` and were
  deliberately not implemented, per this plan's own process rule against
  guessing observable/scientific-behavior-affecting choices**:
  1. **Live-rendering trigger policy inside `train_loop.py`'s periodic-eval
     hot path.** The amendment says the tracked subset is "rendered as GIFs
     unconditionally at every evaluation," but does not specify an
     acceptable wall-clock/storage overhead bound for rendering 5 extra
     GIFs at *every* periodic validation call across a full training run
     (potentially dozens of calls), nor whether this should block the
     training loop synchronously or be deferred/queued asynchronously like
     the existing checkpoint/eval machinery. Options: (a) synchronous
     inline rendering at each periodic eval (simplest, but adds unbounded
     wall-clock cost to the hot path scaling with eval frequency); (b)
     defer actual GIF rendering to a post-hoc pass over the now-complete
     per-episode manifest data (the tracked-subset *selection* records
     already support this, since every periodic eval's episodes are
     recorded with `scenario_uid`), rendering all tracked-subset GIFs for a
     run in one batch after training finishes or on demand -- weaker
     "training-progression visibility" if the goal was live monitoring
     during a run, but bounded and safe; (c) render only every Nth periodic
     evaluation (a sampling compromise) -- but the amendment's "at every
     evaluation" wording argues against silently sampling. Recommendation:
     option (b), since it reuses the exact selection mechanism already
     implemented, adds zero training-hot-path risk, and the
     `tracked_subset_selection.json` output already supports rendering
     on demand for any subset of eval_ids. Awaiting explicit confirmation
     before touching `train_loop.py`'s hot path.
  2. **`make_qualitative_manifest.py`'s category taxonomy** (`best`/
     `median`/`worst`/`rule_violation_case`/`curriculum_transition_case`)
     does not match `REQ-014`'s four fixed categories
     (`representative_success`/`representative_failure`/
     `severe_rule_violation`/`algorithm_disagreement`). Renaming the
     existing categories is an output-schema change for any downstream
     consumer of the current names (dashboards, prior qualitative-appendix
     drafts); `algorithm_disagreement` additionally requires a new
     cross-condition, shared-`scenario_uid` comparison capability this
     single-run builder does not have. Options: (a) rename in place
     (`best` -> `representative_success`, `worst` ->
     `representative_failure`, `rule_violation_case` ->
     `severe_rule_violation`) and additionally implement
     `algorithm_disagreement` as a new cross-run comparison pass; (b) add
     the four `REQ-014` category names as an additional, parallel output
     alongside the existing five (no removal, avoids breaking existing
     consumers, but leaves two overlapping taxonomies live); (c) keep the
     existing taxonomy as an internal implementation detail and only
     surface the four REQ-014 names at the final qualitative-appendix
     report layer (a thin remapping, not a rename). Awaiting explicit
     confirmation before choosing.
  All other work in this pass (Milestones 2's build/persist/load API and
  provider wiring, Milestone 4, Milestone 7, Milestone 8, and Milestone 9's
  tracked-subset selection mechanism) does not touch any of these two
  ambiguous points and was implemented, tested, and is safe to use today.
  Executed: `docker compose run --rm dev uv run --no-sync python -m pytest
  -q` against every new/extended test file listed in each milestone's
  "Tests/commands" above (all passing); `ruff check` on every touched file
  (all passing, per pyproject.toml's configured scope); no `make format`/
  global `make lint` was run (out of scope per `AGENTS.md`'s formatting
  baseline guidance -- only touched files were scoped). The full repository
  suite (`uv run --no-sync python -m pytest -q`) was **not** re-run in this
  pass; only the touched/related test files listed above were run, which is
  the narrower, targeted validation this pass's scope calls for. `docs/
  project_index.md`, `docs/engineering_workflow.md`, and
  `docs/templates/specification_template.md` were not modified.

- **2026-07-25 (Milestone 9, third pass -- category taxonomy + cadence
  wiring, then tracked-subset sizing revision).** Both `AWAITING_CONFIRMATION`
  items from the previous entry were approved by the user and implemented,
  and mid-pass the user approved a further revision to the tracked-subset
  mechanism itself. See §10 Milestone 9 for the full account; summary:
  1. **Category taxonomy** (approved: rename directly, no compatibility
     shim needed): `make_qualitative_manifest.py` now emits exactly
     `representative_success`/`representative_failure`/
     `severe_rule_violation`/`algorithm_disagreement`. `median` and
     `curriculum_transition_case` are dropped (neither maps to a `REQ-014`
     category -- verified by reading `_pick_median`/`_pick_transition_case`
     before removing them, not assumed). `algorithm_disagreement` is
     implemented as `_pick_disagreement()`: a genuinely new cross-condition
     selector comparing the identical `scenario_uid` across every other
     condition in the same comparison, picking a diverging boolean
     `success` outcome with the largest `route_completion` gap. No
     `AWAITING_CONFIRMATION` threshold question arose (`success` divergence
     is binary, not a tunable similarity cutoff).
  2. **Periodic render cadence** (approved: every 100,000 timesteps):
     `train_loop.py` gained a pure, unit-tested `_make_tracked_subset_render_gate`
     closure (fires at the first eval boundary at or after each crossed
     100k multiple, robust to any `eval_interval`) and
     `_tracked_subset_uids_from_resolved_env_cfg` (reads
     `manifest.tracked_subset_uids` off the resolved eval env config's
     `provider.panel_manifest_path`). Selection-JSON writing
     (`select_tracked_subset_episodes`) is unconditional at both the
     periodic-validation and final-test call sites, per `REQ-014`'s "the
     manifest is always produced" invariant; only a log marker (not yet
     actual GIF rendering) is cadence-gated. Actual periodic in-process GIF
     pixel rendering remains `AWAITING_CONFIRMATION`/open: it requires
     threading `tracked_scenario_uids` into
     `eval_artifacts.maybe_build_live_final_eval_recorder_factory` (which
     exists, is already wired for `eval_type="final"`, but is not wired for
     periodic evals and does not filter to specific `scenario_uid`s) --
     real additional surgery beyond a small reviewable change, deliberately
     not attempted rather than forced.
  3. **Tracked-subset sizing/diversity revision** (approved mid-pass,
     superseding the prior pass's flat 5-UID round-robin): `PanelManifest`
     gained a `tracked_subset_uids` field computed once at build time in
     `build_balanced_panel` via a deterministic greedy feature-diversity
     selection (`has_intersection`, `has_merge_or_roundabout`,
     `has_route_traffic_light`, `has_route_stop_sign`, `has_route_crosswalk`,
     `has_vehicle`, `has_pedestrian`, `has_cyclist`, `low_traffic`,
     `dense_traffic`, `vru_interaction`, `topology_tag`) within each arm's
     already-drawn panel candidates -- never a separate catalog draw.
     Approved sizing: 4 scenarios/arm for validation (24 total), 10
     scenarios/arm for test (60 total). Both committed manifest artifacts
     were regenerated against the real catalog (see §7.2 update for
     hashes). The prior pass's `select_tracked_subset_uids()` runtime
     helper is kept only for manifests predating this field and is not
     called by production code anymore.
  Executed: `docker compose run --rm dev uv run --no-sync python -m pytest
  -q tests/test_panel_manifest.py tests/test_tracked_subset_video_selection.py
  tests/test_video_selection_authoritative.py tests/test_qualitative_manifest.py
  tests/test_train_loop_eval_protocol_helpers.py tests/test_eval_artifacts.py`
  -> `PASS` (44/44, no regressions). `ruff check` on every touched file ->
  `PASS`. `docker compose run --rm dataset-pipeline uv run --no-sync python
  scripts/build_panel_manifest.py --split {validation,test} --size
  {100,300} --seed 20260725` -> both manifests regenerated successfully
  against the real ScenarioNet catalog (`/workspace/data/scenarionet/catalog/
  scenario_catalog.parquet` in the `dataset-pipeline` container's data
  volume). The full repository suite was **not** re-run in this pass; only
  the touched/related test files above were run (targeted validation,
  matching this ExecPlan's established pattern for incremental passes).
  `docs/project_index.md`, `docs/engineering_workflow.md`, and
  `docs/templates/specification_template.md` were not modified.

- **2026-07-25 (Milestone 9, fourth pass -- periodic in-process GIF pixel
  rendering).** Closed the one remaining gap from the third-pass entry
  above: actual GIF rendering for the tracked subset during a periodic
  validation eval, previously logged-only. See §10 Milestone 9's fourth-pass
  write-up for the full account; summary:
  1. Identified the real blocker: `Agent._evaluate_parallel._install_episode`
     calls `artifact_recorder_factory(episode_ctx)` *before*
     `env.reset_slots(...)`, so `scenario_uid` is unknown at recorder-
     creation time in the parallel evaluation path (used whenever
     `count_envs(env) > 1`, i.e. most periodic evals with >1 eval worker).
     Resolved by deferring the tracked-subset allowlist decision to
     `finalize_episode` instead of recorder creation: `record_step` latches
     `scenario_uid` from the first `step_info` that carries one (confirmed
     `thesis_scenario_env.step`/`reset` always merge it into `info`).
  2. `eval_artifacts.LiveEvalEpisodeRecorder` gained two additive kwargs
     (`output_dir_name`, `tracked_scenario_uids`, both default-preserving
     for the final-test path) and new
     `build_periodic_tracked_subset_recorder_factory`/
     `maybe_build_periodic_tracked_subset_recorder_factory` functions, kept
     fully separate from (not modifying) `build_live_final_eval_recorder_
     factory`/`maybe_build_live_final_eval_recorder_factory` -- the smaller,
     lower-risk of the two options the task considered, since it required no
     change to the final-test factory's signature or behavior at all.
  3. `train_loop.py`'s periodic-validation call site now resolves the
     tracked-subset UIDs and the cadence gate *before* calling
     `eval_agent.evaluate(...)` (previously both were computed only after,
     for logging) and passes the new factory in as
     `artifact_recorder_factory`. Fixed a latent risk in the same edit: the
     cadence gate closure is stateful (advances on each `due(...)` call), so
     the post-eval block was changed to reuse the already-computed
     `periodic_tracked_subset_render_due` value instead of calling
     `tracked_subset_render_due(...)` a second time (which would have
     silently skipped the next 100k-window).
  4. GIFs land under `videos_dir/periodic_eval/step_<global_step>/
     eval_<eval_id>/<scenario_uid>.gif`, distinct from
     `videos_dir/final_eval/eval_<eval_id>/episode_<episode_id>.gif`.
  5. Implementation decision (recorded, not requiring a scientific-behavior
     approval since REQ-014's observable output is unchanged): when the
     cadence fires, every episode of that one periodic eval gets a real
     recorder (frames buffered in memory) and only tracked-subset episodes
     are persisted to disk; untracked episodes' frames are computed then
     discarded at `finalize_episode`. On-disk output stays bounded to the
     small fixed tracked subset; compute cost for that one gated eval is not
     reduced, but the gate itself bounds this to once per 100,000 timesteps.
  Executed: `docker compose run --rm dev uv run --no-sync python -m pytest -q
  tests/test_eval_artifacts.py tests/test_train_loop_eval_protocol_helpers.py
  tests/test_tracked_subset_video_selection.py
  tests/test_video_selection_authoritative.py tests/test_panel_manifest.py`
  -> `PASS` (49/49, no regressions). `ruff check` on
  `src/thesis_rl/runtime/io/eval_artifacts.py`,
  `src/thesis_rl/runtime/loops/train_loop.py`,
  `tests/test_eval_artifacts.py`,
  `tests/test_train_loop_eval_protocol_helpers.py` -> `PASS`. Live smoke (a
  real training run reaching the 100,000-timestep cadence boundary with
  `cfg.video.enabled=true`/`mode=live_final_eval` and confirming actual GIF
  bytes on disk) was **not** executed -- it is not a few-minutes-cheap path
  (requires a real training run to a 100k-step boundary); this is a known,
  explicit gap, not a claimed-but-unexecuted check. Follow-up command, when a
  short real run is available: run a `td3_*` preset with
  `experiment.eval_interval` small enough to reach 100,000 steps quickly,
  `video.enabled=true`, `video.mode=live_final_eval`, and a configured
  `provider.panel_manifest_path`, then verify
  `videos_dir/periodic_eval/step_0100000/eval_*/<scenario_uid>.gif` files
  exist and are non-empty/openable. `docs/project_index.md`,
  `docs/engineering_workflow.md`, and `docs/templates/specification_template.md`
  were not modified.
- **2026-07-25 (Milestone 5 live PPO verification)**: closed the last
  `NOT_RUN`/unverified item flagged in the Milestone 5 entry above. Assembled
  `conf/presets/test/smoke_train_ppo.yaml` (non-vectorized, `num_envs=1`) and
  ran it live via `docker compose run --rm dev uv run --no-sync python -m
  thesis_rl.cli.train --config-name presets/test/smoke_train_ppo`. Before
  doing so, checked `run_profile.planner.ppo.{n_steps,batch_size}` in
  `smoke.yaml` and confirmed by grep across `src/thesis_rl/` that it is dead
  configuration -- no code path reads it -- so the preset overrides
  `agent.planner.algorithm.{n_steps,batch_size,n_epochs}` directly instead
  (`ppo_sb3.yaml`'s own defaults of `96`/`63` do not divide evenly at any
  small smoke `total_timesteps`). Set `total_timesteps=30`, deliberately not
  a multiple of the atomic unit (`n_steps * num_envs = 8`), so the run forces
  a real overshoot rather than landing exactly on a boundary. The run
  completed end to end: the training monitor showed `Run env steps` advance
  from `30/30` to `32/30` on the bounded extra `train_fn` call, and the
  resulting `run_metadata.yaml` recorded exactly `budget: {target_timesteps:
  30, actual_completed_timesteps: 32, overshoot_steps: 2}` -- the precise
  arithmetic `tests/test_ppo_atomic_boundary.py` already predicted against a
  small in-memory model, now confirmed against a real SB3 PPO model driven
  through the full training loop. The `Agent.train_vectorized()` interaction
  flagged `AWAITING_CONFIRMATION` in §4.1 remains open (this smoke run used
  the non-vectorized path); no other Milestone 5 gap remains.
  `docs/project_index.md`, `docs/engineering_workflow.md`, and
  `docs/templates/specification_template.md` were not modified.
- **2026-07-25 (Milestone 10, `REQ-008`)**: closed the last remaining
  `NOT_IMPLEMENTED` requirement-level gap. Verified the per-step
  applicability signal (`RuleComponentResult.applicable` per R1--R3 macro
  rule) was already computed and already exposed in `step_info
  ["rule_components"]` by the untouched Rulebook v2 wrapper; the gap was
  entirely in `Agent`'s episode/seed-level aggregation, which ignored it.
  Added `Agent._extract_rule_applicability`, threaded applicable-step
  counters through both the parallel (`_ParallelEvaluationEpisode`/
  `_aggregate_parallel_evaluation`) and serial (`evaluate()`, duplicated
  logic) evaluation paths, redefined `violation_rate` to spec §7.2's
  `RuleViolationRate_(i,s) = mean(EpisodeViolationRate_e over N'_s
  episodes)`, and added `applicable_episode_count`/`excluded_episode_count`
  to `per_rule` rows, `rule_metrics.csv`, and the R1--R3
  `rule_violation_by_rule.csv`/`.md` analysis table (R4's table untouched,
  since these counts have no meaning for a rule with no applicability
  concept). `aggregate_runs.py` needed no change (dynamic fieldname union).
  New `tests/test_eval_protocol_req008_applicability_aggregation.py` (4
  tests) demonstrates the fix is behavior-changing (the naive
  total-episode-count denominator would have produced a different, wrong
  value) and that the serial/parallel paths still agree byte-for-byte on
  `per_rule`. Full suite: `970 passed, 4 failed`, the same 4 pre-existing
  `test_causal_semantic_batch.py` route-projection failures already on
  record, unrelated to this change. See §7.9 and §10 Milestone 10 for the
  full design account. `docs/project_index.md`, `docs/engineering_workflow.md`,
  and `docs/templates/specification_template.md` were not modified.

## 12. Deviations

| ID | Original contract | Actual or proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-001` | ExecPlan §7.1/`DEC-EP-003` proposed a bounded "safety margin" loop to complete the final PPO rollout | Implemented as a single exact extra `train_fn` call sized to `agent.atomic_boundary_remaining()` (no iterative safety-bound loop needed) | The exact remaining-steps count is knowable in advance from the SB3 rollout buffer's `pos`/`buffer_size`, making a single precisely-sized call simpler and safer than an iterative bounded loop | Implementation detail, no observable behavior change from the approved `DEC-015` | `tests/test_ppo_atomic_boundary.py` |

No other deviations identified. Every implementation choice is a
private/internal detail that does not change observable scientific behavior,
per §6.

## 13. Files

| Path | Action | Purpose | Status |
|---|---|---|---|
| `src/thesis_rl/runtime/io/metadata.py` | Modified | M1 reproducibility fields; M6 disposition fields; M3/M5 field shapes used by `update_run_metadata` calls | Done |
| `src/thesis_rl/runtime/loops/train_loop.py` | Modified | M3 checkpoint hash/role; M4 data-abort coverage (all 4 `evals.csv` sites + `final_eval.csv`) | Done |
| `src/thesis_rl/agent/agent.py` | Modified | M5 `Agent.atomic_boundary_remaining()` passthrough | Done |
| `src/thesis_rl/agent/planners/algorithms/ppo_sb3.py` | Modified | M5 `Sb3PpoPlannerBackend.atomic_boundary_remaining()` | Done |
| `src/thesis_rl/runtime/io/csv_recorder.py` | Modified | M3/M4 additive CSV columns (`checkpoint_hash`, `checkpoint_role`, `data_abort_*`) | Done |
| `src/thesis_rl/analysis/common_stats.py` | Modified | M7 `mean_ci95` -> `mean_sd`; M7 (2026-07-25) `ci95()` opt-in helper | Done |
| `src/thesis_rl/analysis/tables/make_final_tables.py` | Modified | M7 CI removal, raw seed values; M7 (2026-07-25) `include_ci` param | Done |
| `src/thesis_rl/analysis/plots/make_plots.py` | Modified | M7 CI removal (mean±SD band); opt-in CI column NOT added (visualization, out of column-scope, see §11) | Done (CI-flag not applicable) |
| `src/thesis_rl/analysis/tables/make_factor_effect_tables.py` | Modified | M7 CI removal; opt-in CI column NOT added (already gated behind `--include-effects-tables`, lower priority) | Done (CI-flag deferred) |
| `src/thesis_rl/analysis/tables/make_curriculum_tables.py` | Modified | M7 CI removal (scope expansion); M7 (2026-07-25) `include_ci` param | Done |
| `src/thesis_rl/analysis/tables/make_rulebook_tables.py` | Modified | M7 CI removal; M7 (2026-07-25) R1--R3/R4 split into separate tables + `include_ci` param | Done |
| `src/thesis_rl/analysis/tables/make_sample_efficiency_tables.py` | Modified | M7 CI removal (scope expansion); M7 (2026-07-25) `include_ci` param | Done |
| `src/thesis_rl/analysis/tables/make_generalization_tables.py` | Modified | M7 CI removal (scope expansion); M7 (2026-07-25) `include_ci` param | Done |
| `src/thesis_rl/analysis/run_analysis.py` | Modified | M8 seed-list default fix; ablation gating confirmed pre-existing; M7 (2026-07-25) `--include-ci` flag wiring | Done |
| `Makefile` | Modified | M8 `analyze` target | Done |
| `AGENTS.md` | Modified | M8 Verified Commands entry | Done |
| `scripts/set_run_disposition.py` | New file | M6 manual disposition CLI | Done |
| `tests/test_run_metadata.py` | Modified | M1/M6 tests | Done |
| `tests/test_train_loop_eval_protocol_helpers.py` | Modified | M3/M4 helper unit tests; M4 (2026-07-25) static-scan regression test for all `evals.csv` sites | Done |
| `tests/test_ppo_atomic_boundary.py` | New file | M5 unit tests against a real SB3 PPO model | Done |
| `tests/test_analysis_no_confidence_interval.py` | New file | M7 regression test | Done |
| `src/thesis_rl/scenarios/panel_manifest.py` | Modified (2026-07-25, two passes) | M2 panel manifest build/persist/load; M9 `tracked_subset_uids` field + feature-diversity greedy selection (4/arm validation, 10/arm test) computed once at build time in `build_balanced_panel` | Done |
| `src/thesis_rl/envs/factory.py` | Modified (2026-07-25) | M2 `_resolve_frozen_panel_uids()` wired into both `fixed_sequence` provider construction sites | Done |
| `scripts/build_panel_manifest.py` | Modified (2026-07-25, two passes) | M2 operator script to build/persist a manifest from a real ScenarioNet catalog; M9 catalog-feature lookup + per-split `--tracked-subset-count-per-arm` default (4 validation / 10 test) | Executed against the real catalog for both splits (see §7.2 update) |
| `data/scenarionet/panels/validation_panel_manifest_v1.json` | Regenerated (2026-07-25, data volume, not git-tracked) | M9 tracked-subset feature-diversity revision; `sha256=24bc974a...760f88`, 24 tracked UIDs (4/arm) | Done |
| `data/scenarionet/panels/test_panel_manifest_v1.json` | Regenerated (2026-07-25, data volume, not git-tracked) | M9 tracked-subset feature-diversity revision; `sha256=35e0af63...bedd10`, 60 tracked UIDs (10/arm) | Done |
| `src/thesis_rl/analysis/videos/make_qualitative_manifest.py` | Modified (2026-07-25) | M9 category-taxonomy rename to the four approved REQ-014 categories (`representative_success`/`representative_failure`/`severe_rule_violation`/`algorithm_disagreement`); `median` and `curriculum_transition_case` dropped; new `_pick_disagreement()` cross-condition shared-`scenario_uid` selector | Done |
| `src/thesis_rl/analysis/videos/select_video_episodes.py` | Modified (2026-07-25, two passes) | M9 `select_tracked_subset_episodes()`, extending existing helpers | Done |
| `src/thesis_rl/runtime/loops/train_loop.py` | Modified (2026-07-25, two sub-passes) | M9 `_tracked_subset_uids_from_resolved_env_cfg()`, `_make_tracked_subset_render_gate()` (100,000-timestep cadence); periodic-validation call site now resolves the gate/tracked UIDs before `eval_agent.evaluate(...)` and passes `artifact_recorder_factory=periodic_tracked_subset_artifact_factory` into it (fourth pass) | Done |
| `src/thesis_rl/runtime/io/eval_artifacts.py` | Modified (2026-07-25, fourth Milestone 9 pass) | M9 `LiveEvalEpisodeRecorder` gained additive `output_dir_name`/`tracked_scenario_uids` kwargs (final-test defaults unchanged) and `scenario_uid` latching from `step_info` in `record_step`; new `build_periodic_tracked_subset_recorder_factory`/`maybe_build_periodic_tracked_subset_recorder_factory`, kept separate from the final-test factory functions | Done |
| `src/thesis_rl/runtime/io/metadata.py` | Modified (2026-07-25) | M2 `_snapshot_scenarionet_artifacts()` now also snapshots+hashes `env.provider.panel_manifest_path` (`panel_manifest_<stem>`/`panel_manifest_<stem>_sha256`) | Done |
| `tests/test_run_metadata.py` | Modified (2026-07-25) | M2 `test_run_metadata_records_panel_manifest_hash` | Done |
| `conf/presets/test/smoke_train_ppo.yaml` | New file (2026-07-25) | M5 live-verification PPO smoke preset (`n_steps=8`/`batch_size=4`/`n_epochs=2`, `total_timesteps=30` deliberately non-multiple of the atomic unit to force a real overshoot) | Done; executed live, see §7.1/§10 Milestone 5 |
| `conf/presets/td3/td3_native_curr.yaml` and five siblings | Modified (2026-07-25) | M8 distinct `experiment_group` tagging (`ABLATION-CURRICULUM-*`) | Done |
| `tests/test_panel_manifest.py` | Modified (2026-07-25, two passes) | M2/M9 panel manifest + tracked-subset-UID unit tests; M9 feature-diversity selection, persistence round-trip, and subset-invariant tests | Done |
| `tests/test_env_factory.py` | Modified (2026-07-25) | M2 `_resolve_frozen_panel_uids()` unit tests | Done |
| `tests/test_tracked_subset_video_selection.py` | New file (2026-07-25) | M9 tracked-subset episode selection unit tests | Done |
| `tests/test_qualitative_manifest.py` | New file (2026-07-25) | M9 four-category taxonomy + `algorithm_disagreement` cross-condition selector tests | Done |
| `tests/test_train_loop_eval_protocol_helpers.py` | Modified (2026-07-25, two sub-passes) | M9 render-cadence-gate and resolved-env-cfg tracked-uid unit tests; fourth pass adds periodic `evaluate()` call-site wiring regression tests | Done |
| `tests/test_eval_artifacts.py` | Modified (2026-07-25, fourth Milestone 9 pass) | M9 periodic tracked-subset recorder persistence/discard tests, `maybe_build_...` gating tests, final-test-path regression test | Done |
| `tests/test_td3_curriculum_ablation_tagging.py` | New file (2026-07-25) | M8 ablation `experiment_group` tagging unit tests | Done |
| `tests/test_analysis_optional_ci_and_r4_split.py` | New file (2026-07-25) | M7 opt-in CI column + R1--R3/R4 split unit tests | Done |
| `src/thesis_rl/agent/agent.py` | Modified (2026-07-25, Milestone 10) | M10 `_extract_rule_applicability`; applicability-aware `EpisodeViolationRate`/min-margin in `_ParallelEvaluationEpisode`/`_aggregate_parallel_evaluation` and the serial `evaluate()` loop; `applicable_episode_count`/`excluded_episode_count` in `per_rule` rows | Done |
| `src/thesis_rl/runtime/io/csv_recorder.py` | Modified (2026-07-25, Milestone 10) | M10 additive `applicable_episode_count`/`excluded_episode_count` columns in `rule_metrics.csv` | Done |
| `src/thesis_rl/runtime/loops/eval_loop.py` | Modified (2026-07-25, Milestone 10) | M10 `_append_rule_metrics_rows` passes through the two new columns | Done |
| `src/thesis_rl/runtime/loops/train_loop.py` | Modified (2026-07-25, Milestone 10) | M10 `_append_rule_metrics_rows` passes through the two new columns | Done |
| `src/thesis_rl/analysis/tables/make_rulebook_tables.py` | Modified (2026-07-25, Milestone 10) | M10 `applicable_episode_count_mean`/`_sd`, `excluded_episode_count_mean`/`_sd` additive columns in the R1--R3 table only | Done |
| `tests/test_eval_protocol_req008_applicability_aggregation.py` | New file (2026-07-25, Milestone 10) | M10 REQ-008 regression tests: applicability-filtered violation rate/min-margin, zero-applicable-step exclusion, serial/parallel path agreement | Done |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `uv run --no-sync python -m pytest -q tests/test_run_metadata.py` | `PASS` (3/3) | 2026-07-24 | M1/M6; caught and fixed a real `torch.__version__`/PyYAML bug before any live run |
| `uv run --no-sync python -m pytest -q tests/test_train_loop_eval_protocol_helpers.py tests/test_checkpointing.py` | `PASS` (13/13) | 2026-07-24 | M3/M4 helper units + no regression in existing checkpointing tests |
| `uv run --no-sync python -m pytest -q tests/test_ppo_atomic_boundary.py` | `PASS` (2/2) | 2026-07-24 | M5, against a real (small) SB3 PPO model, not a mock |
| `uv run --no-sync python -m pytest -q tests/test_analysis_no_confidence_interval.py` | `PASS` (2/2) | 2026-07-24 | M7 |
| `ruff check` on all files touched by this pass | `PASS` | 2026-07-24 | No lint findings |
| Import-check of all 8 modified `analysis/` modules | `PASS` | 2026-07-24 | `python -c "import ..."` for each, plus `run_analysis` |
| `make smoke` (TD3, `run_profile=smoke`, real dataset, real GPU) | `PASS` | 2026-07-24 | 2,000-step end-to-end run; `run_metadata.yaml`/`final_eval.csv`/`evals.csv` inspected directly and confirmed to contain correct M1/M3/M4/M5 fields (see §11 findings entry for values) |
| `uv run --no-sync python -m pytest -q` (full suite) | `PARTIAL PASS` (909 passed, 3 failed, 1 deselected during triage) | 2026-07-24 | The 3 failures are pre-existing, in `tests/test_causal_semantic_batch.py` (Rulebook v2 route geometry), unrelated to any file this plan touches; not caused by this work |
| PPO end-to-end smoke with `pending_atomic_steps > 0` | `NOT_RUN` | — | No ready-made PPO smoke preset exists; deferred, see §11 rationale. Residual risk: the loop-level wiring around the unit-tested `atomic_boundary_remaining()` call has not been exercised inside a real Hydra-driven training run for PPO specifically (only for TD3, where the code path is a no-op) |
| `make lint` (full repository scope) | `NOT_RUN` | — | Only the files touched by this pass were checked, per `docs/engineering_workflow.md`'s guidance to scope lint/format to modified files while the repository-wide baseline is not yet clean |
| `make analyze RUN_PROFILE=...` end-to-end against real aggregated artifacts | `NOT_RUN` | — | No comparison-block run registry with real aggregated CSVs was available in this session; only `build_final_tables` was exercised via a synthetic fixture |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_train_loop_eval_protocol_helpers.py` | `PASS` (6/6) | 2026-07-25 | M4: all 4 `evals.csv` sites wired, guarded by new static-scan regression test |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/runtime/loops/train_loop.py` | `PASS` | 2026-07-25 | M4 |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_td3_curriculum_ablation_tagging.py tests/test_hydra_preset_run_configs.py tests/test_hydra_preset_test_configs.py tests/test_hydra_agent_presets.py` | `PASS` (45/45) | 2026-07-25 | M8, no regression in 42 pre-existing Hydra preset-composition tests |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_analysis_no_confidence_interval.py tests/test_analysis_optional_ci_and_r4_split.py` | `PASS` (8/8) | 2026-07-25 | M7 |
| `docker compose run --rm dev uv run --no-sync ruff check` on all M7-touched analysis files (`common_stats.py`, `run_analysis.py`, and the 5 table builders) | `PASS` | 2026-07-25 | M7 |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_panel_manifest.py tests/test_env_factory.py` | `PASS` (18/18) | 2026-07-25 | M2 |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/scenarios/panel_manifest.py src/thesis_rl/envs/factory.py scripts/build_panel_manifest.py` | `PASS` | 2026-07-25 | M2 |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_env_factory.py tests/test_parallel_evaluation_config.py` | `PASS` (8/8) | 2026-07-25 | M2 regression check on `envs/factory.py` |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_tracked_subset_video_selection.py tests/test_video_selection_authoritative.py` | `PASS` (3/3) | 2026-07-25 | M9, no regression in the pre-existing video-selection test |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/analysis/videos/select_video_episodes.py tests/test_tracked_subset_video_selection.py` | `PASS` | 2026-07-25 | M9 |
| `scripts/build_panel_manifest.py` against a real ScenarioNet catalog | `NOT_RUN` | — | No verified real-dataset (`SCENARIONET_DATA_ROOT`) access confirmed in this session; script written and ruff-checked but not exercised end-to-end. Residual risk: only unit-tested against synthetic in-memory records, not a real catalog's actual arm distribution/scale |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_eval_artifacts.py tests/test_train_loop_eval_protocol_helpers.py tests/test_tracked_subset_video_selection.py tests/test_video_selection_authoritative.py tests/test_panel_manifest.py` | `PASS` (49/49) | 2026-07-25 | M9, fourth pass; periodic in-process GIF pixel-rendering implementation, no regression |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/runtime/io/eval_artifacts.py src/thesis_rl/runtime/loops/train_loop.py tests/test_eval_artifacts.py tests/test_train_loop_eval_protocol_helpers.py` | `PASS` | 2026-07-25 | M9, fourth pass |
| Live rendering of tracked-subset GIFs during a real periodic-validation training run | `NOT_RUN` | — | Implemented and unit-tested (mocked `save_gif`) in this pass; a real end-to-end live smoke reaching the 100,000-timestep cadence boundary was not executed because it is not a few-minutes-cheap path (requires a real training run), not because of an open implementation gap. See §10/§11 fourth-pass entries for the exact follow-up command |
| Full repository suite (`uv run --no-sync python -m pytest -q`) | `NOT_RUN` (this pass) | — | Only the touched/related test files listed above were run in this pass, per its narrower scope; the 2026-07-24 pass's full-suite baseline (909 passed / 3 pre-existing unrelated failures) is the last full-suite evidence on record |

## 15. Final Reconciliation

| Requirement | Status | Notes |
|---|---|---|
| `REQ-002`/`DEC-015` | `IMPLEMENTED` | Core logic unit-verified against a real SB3 PPO model; TD3 zero-overshoot path live-verified; PPO-overshoot path live-verified 2026-07-25 (`smoke_train_ppo.yaml`, `budget: {target: 30, actual: 32, overshoot: 2}`); vectorized `train_vectorized()` interaction still not live-exercised (§14) |
| `REQ-004` | `IMPLEMENTED` | Manifest build/persist/load + fail-closed provider wiring implemented and unit-tested; real validation (100, 4/arm tracked) and test (300, 10/arm tracked) manifests generated against the real ScenarioNet catalog and committed to the data volume; per-run metadata now records manifest path/hash (`test_run_metadata_records_panel_manifest_hash`); no Hydra preset yet points `provider.panel_manifest_path` at the committed manifests (config-wiring follow-up only, not a code/data/test gap) |
| `REQ-006` | `IMPLEMENTED` | Live-verified |
| `REQ-007` | `IMPLEMENTED` | R1--R3/R4 structurally split into distinct tables/columns in `make_rulebook_tables.py`, verified against a synthetic fixture with distinct R1/R4 margins |
| `REQ-008` | `IMPLEMENTED` (2026-07-25, Milestone 10) | Applicability-aware R1--R3 seed-level aggregation implemented and unit-tested: `RuleViolationRate_(i,s)`/`RuleMinimumMarginMean_(i,s)` now computed only over episodes with >=1 applicable step for that rule; `applicable_episode_count`/`excluded_episode_count` reported in `per_rule` rows, `rule_metrics.csv`, and the R1--R3 analysis table. R4 unaffected (no applicability concept, defaults to always-applicable). See §7.9/§10 Milestone 10 |
| `REQ-009` | `IMPLEMENTED` | CI removal verified across 8 files; mandatory-core regression test passing; optional opt-in `ci95` column (2026-07-25 amendment) implemented and unit-tested for 5 core table builders, default-off byte-identical |
| `REQ-011` | `IMPLEMENTED` | Live-verified via metadata tests; CLI script present |
| `REQ-012` | `IMPLEMENTED` | All 4 `evals.csv` call sites plus `final_eval.csv` wired and regression-tested; per-evaluation `invalid_episodes` JSON sidecar (§7.5) not added, a narrower residual gap than the CSV-column gap this requirement's core invariant targets |
| `REQ-014` | `DONE` | Tracked-subset selection mechanism (feature-diversity UIDs, 4/arm validation + 10/arm test, cross-eval-id episode matching), category-taxonomy reconciliation (four approved categories, `algorithm_disagreement` implemented), cadence-gated selection-JSON wiring, and the periodic in-process GIF pixel-rendering call itself are all implemented and unit-tested (§10 Milestone 9, §11). Only a real end-to-end live-smoke run reaching the 100,000-timestep cadence boundary remains `NOT_RUN` (not a correctness gap, see §14) |
| `REQ-015` | `IMPLEMENTED` | Live-verified; two fields degrade gracefully to `"unknown"`/`None` in environments without a resolvable value (documented limitation, not a defect) |
| `REQ-016` | `PARTIAL` | Canonical entry point confirmed/documented; ablation gating confirmed pre-existing; full end-to-end regeneration against real artifacts not run |
| `REQ-018` | `IMPLEMENTED` | Default gating confirmed; the 6 `td3` ablation presets now tagged with distinct `experiment_group` values, unit-tested |

**2026-07-25 reconciliation pass**: all `IMPLEMENTED`/`DONE`-status rows above
were re-verified against the current repository state (not re-taken on
trust from earlier progress-log entries): `REQ-002`/`DEC-015`,
`REQ-004`, `REQ-006`, `REQ-007`, `REQ-009`, `REQ-011`, `REQ-012`, `REQ-014`,
`REQ-015`, `REQ-018` all have real, currently-passing tests exercising the
described behavior (confirmed via a full repository test run: 966/970
passing; the 4 failures are pre-existing, in
`tests/test_causal_semantic_batch.py`, in a Rulebook v2 route-projection
module (`src/thesis_rl/rulebook/v2/geometry/route.py`) this plan never
touched, and correspond to files a concurrent, unrelated session has
modified per `git status` -- not a regression introduced by this ExecPlan).
**2026-07-25, Milestone 10 pass**: `REQ-008` closed (see §7.9/§10 Milestone
10). `REQ-016` remains `PARTIAL` (full end-to-end analysis regeneration
against real aggregated artifacts was never run against a real
multi-condition comparison block); this is now the sole remaining
requirement-level gap, an explicit, acknowledged residual, not silently
dropped work. Re-ran the full repository suite after the `REQ-008` change:
`970 passed, 4 failed` (the same 4 pre-existing, unrelated
`test_causal_semantic_batch.py` route-projection failures noted in the prior
pass above; no new failures).

None of the above reaches `VERIFIED`: `docs/engineering_workflow.md` reserves
`VERIFIED` for full reconciliation and required validation, which is not yet
complete for the one remaining residual gap above (`REQ-016`'s full
real-artifact regeneration not run) and for the two narrower,
explicitly-flagged residual items noted in their own rows above (`REQ-004`'s
preset-wiring follow-up; `REQ-014`'s live 100k-step GIF smoke). Every other
Milestone (1 through 6, 8, 9, 10) is complete and live- or unit-verified.
`docs/project_index.md`'s ExecPlan Registry row for this plan should be
updated from `IN_PROGRESS` to reflect this pass once reviewed.
