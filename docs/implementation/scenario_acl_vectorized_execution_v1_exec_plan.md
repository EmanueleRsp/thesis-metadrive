# Scenario ACL Vectorized Execution v1 — ExecPlan

## 1. Metadata

- Feature: deterministic process-based vectorized execution for ScenarioNet
  Scenario ACL.
- Plan ID: `ACL-SN-VEC-EXEC-V1`.
- Authoritative specifications:
  - `docs/specifications/automatic_curriculum_learning_v1_specification.md`,
    ACL v1, §12 and §28, `AUTHORITATIVE` (approved 2026-07-16; ScenarioNet
    amendment approved 2026-07-19).
  - `docs/specifications/scenarionet_integration_v1.1_specification.md`,
    `SCENARIONET-INTEGRATION` v1.1, §2.7, §23, §24, §27 and §28,
    `APPROVED`/authoritative (approved 2026-07-16).
- Related ADRs: ADR-001, ADR-014, and ADR-016.
- Status: `COMPLETED`.
- Created: 2026-07-20.
- Last updated: 2026-07-20.
- Branch: current working branch; no branch name was assumed.
- Owner: thesis repository maintainer.

## 2. Objective and scope

Enable `curriculum.kind=scenario_acl` to train on `N > 1` process-based
ScenarioNet environments while preserving the curriculum unit of attribution:

```text
one worker slot -> one ACL selection -> one frozen scenario episode
                -> one algorithm-specific learning potential
                -> one buffer/MAB attribution
```

For a fixed frozen software/data configuration, global seed, `num_envs`, and
checkpoint state, the feature must reproduce the logical sequence of ACL
selections, completed episode outcomes, buffer/MAB mutations, and worker reset
assignments. It does not promise bitwise equivalence across different hardware,
GPU drivers, Python/PyTorch versions, or simulator versions.

In scope:

- parent-controlled batched ACL selection for generated frozen catalog records
  and replay records;
- per-worker explicit setup and reset, without generic vector auto-reset racing
  ahead of ACL selection;
- per-scenario episode metrics and algorithm-specific LP attribution;
- deterministic buffer/MAB updates, persistence, checkpoint/resume and logs;
- configuration validation and an ACL-only vectorized execution path;
- focused unit, integration, resume, and real ScenarioNet smoke coverage.

Out of scope:

- mutation, generation of ScenarioDescriptions, writes to frozen datasets, or
  changes to arms A0--A5;
- changing ScenarioNet split/source policies, observations, reward semantics,
  termination/truncation contracts, planner algorithms, or generic non-ACL
  vectorized training behavior;
- changing `n_envs` during a run or resume;
- bitwise cross-platform determinism.

Compatibility constraints:

- retain the frozen catalog/runtime mapping and train split boundary;
- retain `spawn` process startup for ScenarioNet/MetaDrive workers;
- retain `ScenarioUsefulness.value = LP_alg` only; Rulebook values are
  diagnostic-only under ADR-014;
- retain stable planner/replay-buffer `n_envs` for an entire run and resume.

## 3. Authoritative requirements

| ID | Requirement | Specification section |
|---|---|---|
| REQ-VEC-001 | ACL selects exactly the six frozen ScenarioNet arms and never mutates or writes source scenarios. | ACL §28.1--§28.2; ADR-014 |
| REQ-VEC-002 | Each generated/replayed scenario has algorithm-specific learning potential, and curriculum decisions use only that value. | ACL §12, §28.3; ADR-014 |
| REQ-VEC-003 | Scenario buffer, MAB, staleness, warm-up, checkpoint and resume remain deterministic and persisted. | ACL §13--§16, §28.4 |
| REQ-VEC-004 | ACL remains external to ScenarioEnv and samples the source conditionally on the selected semantic arm. | ScenarioNet §2.7, §22, §31 |
| REQ-VEC-005 | Vectorization is process-based, uses one environment per process, uses `spawn`, and closes workers correctly. | ScenarioNet §23, §27.2, §28.28 |
| REQ-VEC-006 | Per-episode logging preserves scenario identity, source, arm, worker id, termination/truncation and outcomes; source-by-arm counters remain meaningful. | ScenarioNet §24, §30, §28.29 |
| REQ-VEC-007 | No curriculum metadata or future information reaches policy observations. | ScenarioNet §2.1, §27.3, §28.30 |
| REQ-VEC-008 | Native collision/destination termination and thesis truncation semantics are preserved. | ScenarioNet §20--§22, §27.4 |
| REQ-VEC-009 | The selected ScenarioNet scalar core rejects Rulebook-derived curriculum decisions and incompatible historical checkpoints. | ACL §28.3--§28.4; ADR-014 |

## 4. Current repository analysis

| Status | Evidence | Finding |
|---|---|---|
| VERIFIED | `src/thesis_rl/curriculum/scenario_acl/runtime.py::validate_scenario_acl_runtime_support` | ACL vector mode is accepted only for ScenarioNet, `num_envs > 1`, `spawn`, and six arms; single-env mode remains compatible. |
| VERIFIED | `src/thesis_rl/curriculum/scenario_acl/driver.py::run_scenario_acl_training` | The driver constructs one env, sets one selection before each reset, and updates buffer/MAB in a single-env episode callback. |
| VERIFIED | `src/thesis_rl/agent/agent.py::Agent.train_vectorized` | Generic vector collection is unchanged; ACL mode now requires a parent callback that receives completed slots and returns selective-reset observations. |
| VERIFIED | `src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py::_worker` | On `done`, workers auto-reset inside `step`; the parent cannot choose the next ACL scenario first. |
| VERIFIED | `src/thesis_rl/runtime/wiring/builders.py::build_train_env` | Non-ACL ScenarioNet vectorization already uses `DeterministicSubprocVecEnv`, per-worker providers and configurable process start method. |
| VERIFIED | `tests/test_scenarionet_vectorized_integration.py` | Two-worker ScenarioNet spawn/reset/auto-reset and mixed uniform provider smoke tests exist. |
| VERIFIED | `src/thesis_rl/envs/thesis_scenario_env.py::reset` | The environment publishes selected catalog metadata in reset/step info and can resolve a forced runtime index when catalog-backed. |
| VERIFIED | `src/thesis_rl/curriculum/scenario_acl/{buffer,mab,record,usefulness}.py` | Buffer and MAB are parent-process, single-record APIs; `ScenarioRecord` carries replay identity and staleness state. |
| VERIFIED | `src/thesis_rl/agent/planners/core/lifecycle.py` | LP values currently form a chunk-level update list; they do not retain episode/worker ownership. |
| VERIFIED | `src/thesis_rl/runtime/wiring/builders.py::set_planner_env_if_compatible` | Planner and transition replay buffer reject `n_envs` changes after initialization. |
| VERIFIED | `third_party/metadrive`, `third_party/scenarionet`, `third_party/stable-baselines3` | Current pinned working trees are MetaDrive `85e5dadc`, ScenarioNet `d4acdb5`, and the local SB3 fork `4e6c3db`. No dependency upgrade is authorized by this plan. |
| INFERRED | current generic vector protocol | The ACL route needs explicit selective reset rather than a post-hoc callback, otherwise the next worker scenario is selected too late. |

Call flow to preserve outside the ACL branch:

```text
train_loop -> build_train_env -> DeterministicSubprocVecEnv -> Agent.train_vectorized
```

Required ACL call flow:

```text
Scenario ACL parent state
  -> select one IterationSpec per slot
  -> configure worker slots
  -> reset selected slots
  -> vector step without auto-reset on completed ACL slots
  -> terminal transition + per-slot LP attribution
  -> stable parent-side buffer/MAB transaction
  -> select/configure/reset only completed slots
```

## 5. Assumptions and invariants

| Invariant | Basis | Enforcement |
|---|---|---|
| `num_envs` is positive and immutable after planner creation. | Existing planner/replay contract. | Fail early on checkpoint/env mismatch. |
| Worker process startup is `spawn`. | ScenarioNet §23; current config supports it. | Reject/warn unsupported ACL start methods; test spawned workers. |
| A reset selection is owned by exactly one slot and logical episode id. | REQ-VEC-002/003. | Immutable `ActiveAclSelection` keyed by slot and generation. |
| A terminal result is attributed to the selection active before that step, never to the auto-reset scenario. | REQ-VEC-002. | ACL worker mode has no done-time auto-reset. |
| Parent RNG is the sole RNG for MAB/replay choices; worker RNGs are derived from `SeedSequence([global_seed, worker_id])`. | ScenarioNet §23.2; deterministic requirement. | Serialize all RNG states and selection sequence state. |
| Parent commits simultaneous completions in ascending `(collection_tick, worker_id)` order. | Required to avoid timing-dependent process completion order. | Sort completed slot outcomes before normalization/MAB/buffer updates. |
| Fresh sampling exclusions are evaluated from the same parent-side buffer snapshot for a selection batch. | ACL plan fresh/replay disjointness. | Construct immutable batch exclusion set; define duplicate policy in DEC-VEC-002. |
| `terminated` and `truncated` remain distinct; terminal observation is used for learner transition storage. | ScenarioNet §20--§22. | Reuse `normalize_vector_transition_boundary`; regression-test both flags. |
| Policy observations never include selection IDs, arm names, source, LP or buffer data. | REQ-VEC-007. | Keep metadata in `info`/parent state only and test observation shape/content boundary. |

## 6. Decisions and approval gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| DEC-VEC-001 | implementation detail | Reset protocol after a worker terminates. | A: retain worker auto-reset and infer attribution; B: ACL-only parent-controlled selective reset. | B. | Correct scenario ownership; generic vector path remains unchanged. | Approved; ADR-016 |
| DEC-VEC-002 | specification clarification | Same frozen record can be selected twice in one fresh-selection batch. | A: permit as ScenarioNet §23.3 permits duplicates; B: prohibit only fresh duplicates within a parent selection batch, while allowing independently sampled non-ACL duplicates. | B. | Preserves ACL fresh/replay disjointness and avoids duplicate insert races; changes only ACL batch semantics. | Approved; ADR-016 |
| DEC-VEC-003 | specification clarification | Exact off-policy LP attribution point. | A: assign aggregate learner-update residuals to recent episodes; B: compute §12 residuals from each collected transition under a documented learner snapshot and aggregate by episode. | B. | Prevents replay-sampled old transitions from being attributed to the wrong current scenario. | Approved; ADR-016 |
| DEC-VEC-004 | implementation detail | Batch commit order for several completions. | A: arrival order; B: stable `(collection_tick, worker_id)` order. | B. | Seed reproducibility is insulated from OS scheduling. | Approved; ADR-016 |
| DEC-VEC-005 | blocking technical issue | PPO LP attribution across rollout buffers and partial episodes at chunk boundaries. | A: defer PPO vector ACL; B: add episode/slot provenance to rollout storage and carry partial state across chunks. | B. | Required for all currently supported scalar ACL planners; implementation is larger. | Approved; ADR-016 |
| DEC-VEC-006 | specification clarification | Deterministic scope. | A: promise bitwise cross-platform equality; B: same software/data/config/platform logical reproducibility only. | B. | Matches ScenarioNet §23.2 and realistic GPU behavior. | Approved; user confirmation 2026-07-20 |

DEC-VEC-001 through DEC-VEC-006 are approved by the user on 2026-07-20 and
recorded in ADR-016.

## 7. Proposed design

### 7.1 ACL vector execution boundary

Add an ACL-specific capability to the existing process vector environment rather
than changing its default protocol. In ACL mode, `step` returns terminal data
without resetting completed workers. It exposes batched/selective operations:

```text
configure_acl_selection(slot, serialized selection)
reset_slots(slots, reset_seeds/options)
```

The worker applies the selection to `ThesisScenarioEnv` immediately before the
corresponding reset. Generated records set arm/source/exclusions; replay records
force their catalog runtime index/reset seed. The command returns reset metadata
including a selection generation/token. Worker exceptions continue to produce
the existing crash logs.

The generic auto-reset vector behavior remains the default for non-ACL callers.

### 7.2 Parent state and deterministic batch selection

Create serializable dataclasses such as `AclSlotSelection` and
`AclVectorState` in the Scenario ACL package. A slot selection contains at
least: logical episode id, slot id, selection generation, mode, arm index/name,
arm probabilities, replay probabilities/record identity when present, forced
runtime index, reset seed, and source override.

At initial reset the parent samples `N` selections. At later steps it samples
only completed slots after committing their prior outcomes. The parent holds the
sole authoritative ScenarioBuffer, MAB, recent LP window and ACL RNG. Worker
processes never mutate those objects.

### 7.3 Per-episode metric and LP attribution

Extend the vector collector with per-slot episode accumulators and a structured
terminal callback/result. It passes the active selection plus terminal metrics
to the parent transaction. The collector must preserve final observations and
termination/truncation boundaries before selective reset.

Implement a backend-facing LP attribution interface that returns values keyed
by `(slot_id, logical_episode_id)`. TD3/SAC must retain per-transition episode
provenance and evaluate the specified residual on the selected timing defined
by DEC-VEC-003. PPO must retain slot/episode provenance through rollout and
emit GAE-derived LP for completed episodes, including a defined treatment of
episodes spanning chunks. Existing aggregate LP remains a diagnostic metric,
not ACL feedback.

### 7.4 Parent-side transaction

For each vector collection tick, sort completed slots deterministically. For
each outcome: compute normalized LP against the deterministic recent window;
update MAB only for fresh generated selections; insert/update exactly one
ScenarioRecord; refresh ranks; update staleness/`num_seen`; append per-episode
event artifacts. Persist buffer and state only after the complete tick
transaction has succeeded. A failed transaction must leave no partially
published buffer checkpoint; use temporary file plus atomic replace.

### 7.5 Checkpoint and resume

Persist a versioned ACL vector state beside existing ACL state: active slot
selections, per-slot logical episode/metric accumulator state, parent RNG,
worker RNG derivation identity, collection tick, and expected `n_envs`.
Planner checkpoint and transition replay buffer persistence remain paired with
this state. Resume rejects missing/incompatible vector state or mismatched
`n_envs`; it never silently falls back to single-env behavior.

### 7.6 Configuration, observability and errors

Add an explicit ACL vector execution mode/config validation, enabled only when
`curriculum.kind=scenario_acl` and `env.vectorized.num_envs > 1`. Maintain
single-env ACL as the default compatible mode. Logs and CSV/JSONL payloads gain
slot id, logical episode id, selection token, per-episode LP, and deterministic
commit order. Preserve existing source-by-arm aggregate statistics.

Reject unsupported combinations explicitly: non-`spawn` process start if
chosen by the approved gate, dynamic `n_envs`, unavailable per-backend LP
attribution, malformed/foreign replay record, stale selection token, or an
attempt to configure a worker while it has an unfinished episode.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| REQ-VEC-001 | AC-VEC-001, AC-VEC-002 | ACL selection batch/state and worker configuration | TEST-VEC-001--003 | VERIFIED for the implemented parent/vector path |
| REQ-VEC-002 | AC-VEC-003, AC-VEC-004 | collector provenance and planner LP attribution | TEST-VEC-004--007 | VERIFIED for TD3/SAC and PPO provenance; real TD3/PPO vector smoke passed |
| REQ-VEC-003 | AC-VEC-005, AC-VEC-006 | transaction, persistence, resume | TEST-VEC-008--010 | VERIFIED with deterministic active-slot restart on real checkpoint resume |
| REQ-VEC-004 | AC-VEC-001, AC-VEC-002 | `ThesisScenarioEnv` selection command | TEST-VEC-002--003 | VERIFIED for the implemented parent/vector path |
| REQ-VEC-005 | AC-VEC-007 | ACL VecEnv protocol/close | TEST-VEC-011--012 | VERIFIED |
| REQ-VEC-006 | AC-VEC-008 | artifacts/CSV/event logging | TEST-VEC-013 | VERIFIED by real TD3/PPO artifacts |
| REQ-VEC-007 | AC-VEC-009 | metadata boundary | TEST-VEC-014 | VERIFIED |
| REQ-VEC-008 | AC-VEC-010 | transition boundary path | TEST-VEC-015 | VERIFIED |
| REQ-VEC-009 | AC-VEC-005 | ACL config/resume compatibility | TEST-VEC-008, TEST-VEC-016 | VERIFIED; real checkpoint compatibility smoke passed |

## 9. Test strategy defined before implementation

Acceptance criteria:

- AC-VEC-001: initial `N=2` selection creates two independently traceable
  worker assignments; each uses one of frozen A0--A5 and no dataset write.
- AC-VEC-002: one batch can mix fresh generation and replay; each slot loads
  precisely its assigned catalog identity.
- AC-VEC-003: modifying LP for one completed slot changes only that slot's
  record and, if fresh, that arm's MAB update.
- AC-VEC-004: LP is the specified per-episode algorithm-specific value, never
  a chunk mean assigned to several records.
- AC-VEC-005: equal seed/config/software with `N=2` reproduces selection,
  commit, buffer/MAB and resume traces; incompatible historical checkpoint is
  rejected.
- AC-VEC-006: resume at a non-empty active-slot boundary reproduces the
  uninterrupted logical trace and rejects `n_envs` mismatch.
- AC-VEC-007: spawned ACL workers reset repeatedly, terminate, and close
  without leaked processes; generic non-ACL auto-reset still works.
- AC-VEC-008: ACL artifacts contain per-slot selection and LP provenance while
  aggregate `source × arm` counters remain consistent.
- AC-VEC-009: selection/curriculum metadata cannot appear in policy
  observations.
- AC-VEC-010: vectorized terminal transition preserves final observation and
  separate termination/truncation flags before a slot is reset.

Mandatory test matrix:

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| TEST-VEC-001 | Unit | deterministic batch selector | seed, buffer and two slots | stable selections/tokens/RNG state | REQ-VEC-001,003 |
| TEST-VEC-002 | Unit | fresh/replay worker configuration | fake catalog records | slot-local arm/source/forced seed only | REQ-VEC-001,004 |
| TEST-VEC-003 | Unit | duplicate policy | two fresh selections targeting same UID | approved DEC-VEC-002 behavior | REQ-VEC-001 |
| TEST-VEC-004 | Unit | TD3 LP ownership | interleaved two-slot transitions | LP is segmented by episode/slot | REQ-VEC-002 |
| TEST-VEC-005 | Unit | SAC LP ownership | interleaved two-slot transitions | entropy-aware residuals stay slot-local | REQ-VEC-002 |
| TEST-VEC-006 | Unit | PPO LP ownership | two slot rollout with boundaries | positive GAE LP per episode | REQ-VEC-002 |
| TEST-VEC-007 | Unit | no aggregate feedback fallback | differing worker LP values | no chunk mean reaches buffer/MAB | REQ-VEC-002 |
| TEST-VEC-008 | Unit | vector ACL state serialization | active slots + RNG + buffer/MAB | exact round trip; old/incompatible state rejected | REQ-VEC-003,009 |
| TEST-VEC-009 | Unit | transaction order | same outcomes permuted by arrival | identical resulting state | REQ-VEC-003 |
| TEST-VEC-010 | Integration | deterministic resume | fake two-worker controlled env | uninterrupted and resumed traces equal | REQ-VEC-003 |
| TEST-VEC-011 | Integration | selective reset protocol | two controllable subprocess envs | done slot is not reset before parent configuration | REQ-VEC-005 |
| TEST-VEC-012 | Integration | process lifecycle | spawned envs/repeated reset/close | clean close and no live workers | REQ-VEC-005 |
| TEST-VEC-013 | Integration | artifact attribution | controlled mixed batch | one event/record per slot with provenance | REQ-VEC-006 |
| TEST-VEC-014 | Regression | observation isolation | selection-rich info payload | observation contract unaffected | REQ-VEC-007 |
| TEST-VEC-015 | Regression | boundary handling | terminated and truncated slots | final observation and flags preserved | REQ-VEC-008 |
| TEST-VEC-016 | Regression | existing single-env ACL guard behavior | `num_envs=1` and invalid old state | existing path retained; explicit errors | REQ-VEC-003,009 |
| TEST-VEC-017 | Smoke/integration | real frozen ScenarioNet `N=2` | prepared matching runtime/train fixture | generate[2] -> buffer[2] -> replay[2], mixed batch, clean close | REQ-VEC-001--006 |

Commands to run from repository root after implementation:

```bash
uv run --no-sync python -m pytest -q tests/test_scenario_acl_config.py tests/test_scenario_acl_mab.py tests/test_scenario_acl_buffer.py tests/test_scenario_acl_scenario_env.py
uv run --no-sync python -m pytest -q tests/test_scenarionet_vectorized_integration.py tests/test_transition_boundary.py
make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/curriculum/scenario_acl src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py src/thesis_rl/agent tests/test_scenario_acl_config.py tests/test_scenarionet_vectorized_integration.py"
make lint PYTHON_QUALITY_PATHS="src/thesis_rl/curriculum/scenario_acl src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py src/thesis_rl/agent tests/test_scenario_acl_config.py tests/test_scenarionet_vectorized_integration.py"
git diff --check
```

The exact names of new focused tests will be added before production code.
`make smoke` is a representative end-to-end command but must be run only after
confirming its preset selects the intended ACL vector feature; it is not yet a
mandatory acceptance command. No global static-type command is configured.

## 10. Milestones

### M1 — Approve semantics and freeze tests

- Status: completed.
- Files: this plan; an ADR only if an approved decision changes scientific or
  observable behavior.
- Tasks:
  - [x] resolve DEC-VEC-001 through DEC-VEC-005;
  - [x] update requirements/test matrix and add ADR-016;
  - [x] confirm no mutation/dataset-write path enters the feature.
- Evidence: approved plan and decisions.

### M2 — Parent selection state and selective worker protocol

- Status: completed, including provisioned real-fixture integration.
- Expected files: `src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py`,
  `src/thesis_rl/curriculum/scenario_acl/`, `src/thesis_rl/envs/thesis_scenario_env.py`,
  focused tests.
- Tasks:
  - [x] add serialized slot selection and worker commands;
  - [x] add ACL-only no-auto-reset/selective-reset protocol;
  - [x] validate token/slot lifecycle and process cleanup;
  - [x] retain generic vector protocol unchanged.
- Tests: TEST-VEC-001--003, TEST-VEC-011--012.

### M3 — Collector and backend LP provenance

- Status: completed for the supported scalar backends and real PPO vector smoke.
- Expected files: `src/thesis_rl/agent/agent.py`, lifecycle/backend algorithms,
  transition replay components, focused tests.
- Tasks:
  - [x] add per-slot episode accumulators and LP ownership primitive;
  - [x] propagate transition provenance into TD3/SAC/PPO LP computation;
  - [x] retain aggregate metrics only as diagnostics;
  - [x] enforce final-observation and boundary correctness in ACL worker mode.
- Tests: TEST-VEC-004--007, TEST-VEC-015.

### M4 — ACL transaction, artifacts and resume

- Status: completed, including the approved active-slot restart policy and real checkpoint/resume smoke.
- Expected files: Scenario ACL driver/buffer/state artifacts, checkpoint tests,
  recorder/event schemas as needed.
- Tasks:
  - [x] deterministic completion ordering and atomic state persistence;
  - [x] vector ACL state versioning and mismatch errors;
  - [x] add provenance fields to artifacts without leaking into observations;
  - [x] enable vector ACL configuration only after required capabilities exist.
- Tests: TEST-VEC-008--010, TEST-VEC-013--014, TEST-VEC-016.

### M5 — Frozen ScenarioNet integration and reconciliation

- Status: completed against the provisioned matching frozen catalog/runtime fixture.
- Expected files: integration/smoke tests, configs and this plan; update index
  only if implementation status changes.
- Tasks:
  - [x] execute real matching-runtime `N=2` generate/replay smoke;
  - [x] run focused regressions and quality checks;
  - [x] inspect full diff and reconcile each requirement/criterion.
- Tests: TEST-VEC-017 and commands in §9.

## 11. Progress and findings log

- 2026-07-20 — Created after repository analysis. Generic ScenarioNet
  process-based vectorization is present and integration-tested, while ACL is
  deliberately single-env. The hard blocker is attribution/control-plane
  semantics, not ScenarioNet process capability. No production files changed
  and no tests were run for this planning task.
- 2026-07-20 — User approved DEC-VEC-001--DEC-VEC-005 and DEC-VEC-006; ADR-016
  records the material decisions. Added parent-side serializable ACL vector
  state, per-episode LP accumulators, deterministic completion ordering,
  atomic persistence, and ACL-only selective-reset subprocess commands. System
  Python lacks project dependencies and the local uv environment lacks pytest;
  provisioned-environment validation remains pending.
- 2026-07-20 — Compose focused ACL suite passed (`24 passed`), and focused Ruff
  check/format passed for the new vector state and tests. The existing
  ScenarioNet vector regression ran 7 tests with 1 failure: its fixed-sequence
  fixture selected a `test` record while the environment split was `train`;
  the worker then exited with EOFError. This remains an integration blocker.
- 2026-07-20 — Fixed ScenarioNet provider construction to filter catalog
  records by the requested split before worker partitioning. The vector
  integration and transition-boundary regression suite now passes (`8 passed`).
  Added the ACL parent reset callback to `Agent.train_vectorized`; backend LP
  values and driver transaction wiring still require integration.
- 2026-07-20 — Added `AclVectorSelectionCoordinator` for deterministic parent
  selection batches, fresh exclusion enforcement, stale-generation rejection,
  worker configuration, and selective reset. The combined ACL/vector suite and
  Ruff check pass (`33 passed`).
- 2026-07-20 — Added collection-time TD3/SAC residual interfaces for custom and
  SB3 backends. The lifecycle attributes residuals by `(acl_slot_id,
  acl_episode_id)` and never uses replay-batch residuals for that value. PPO
  rollout buffers now retain the same provenance and aggregate positive GAE
  advantages per episode, including fragments retained across rollout updates.
  Focused planner/ACL validation passes (`51 passed` after the driver and state
  persistence integration).
- 2026-07-20 — Wired the main Scenario ACL driver to the ACL-mode process vector
  environment. Parent selection now resolves frozen train-catalog records,
  configures and selectively resets completed slots, commits completions in
  `(collection_tick, worker_id)` order, persists pending PPO completions and
  last observations, and writes per-slot event provenance. A legacy fixture
  regression in `_select_provider_seed` was fixed by treating the new ACL
  attribute as optional. The complete suite reported `693 passed, 1 skipped,
  3 failed` before that fix; the ACL-related failure is resolved, while the two
  remaining failures are unrelated pre-existing failures in the forced-rule
  debug tool and golden recorder fixture.
- 2026-07-20 — Real frozen ScenarioNet ACL smoke completed with two `spawn`
  workers and 600 transitions. It produced deterministic parent events with
  two fresh buffer insertions and one replay update, per-episode TD3-SB3 LP,
  vector state, active selections and persisted observations. The first resume
  attempt exposed a blocking boundary: newly spawned workers do not restore
  MetaDrive/Rulebook simulator state from the learner checkpoint, so reusing
  persisted observations without a worker reset causes
  `RulebookV2MonitorWrapper.step called before reset`. Exact mid-episode resume
  therefore required an explicit policy decision. The approved policy is to
  reset each persisted active slot with its persisted selection seed before
  the first resumed step, retain logical episode provenance and accumulators,
  and emit a restart event. Exact simulator trajectory-prefix reconstruction
  remains out of scope because the learner checkpoint does not serialize
  MetaDrive/Rulebook state.
- 2026-07-20 — The approved resume policy was implemented and the real
  checkpoint resume completed 10 additional transitions with two newly spawned
  workers. Each active slot emitted an explicit restart event and no worker
  attempted a step before reset.
- 2026-07-20 — A compact real ScenarioNet smoke with `horizon=10` and 60
  transitions verified two fresh insertions followed by replay updates on both
  slots, with distinct per-episode TD3 LP values and deterministic commit
  ordering. A matching PPO smoke completed six episodes and persisted two
  unresolved rollout completions exactly once while LP attribution was pending.
- 2026-07-20 — Fixed duplicate persistence of PPO completions whose LP becomes
  available after the terminal callback; same-slot/same-tick completion ties
  now use `episode_id` as an explicit deterministic tiebreaker. The regression
  suite passed (`56 passed`).
- 2026-07-20 — Added validation evaluation to the ACL vectorized driver after
  every training chunk, using the validation split and the existing
  checkpoint/evaluation artifact conventions. The final test evaluation remains
  separate. A real 20-transition smoke with `eval_interval=10` produced two
  intermediate terminal evaluations, `evaluation_started/finished` events,
  `evals.csv`, and the final test evaluation.
- 2026-07-20 — Aligned the vectorized final test path with the sequential path:
  it now prints the `Final Evaluation` summary, records `final_eval.csv`, and
  passes the configured live final-evaluation artifact recorder. A compact real
  smoke produced the final-test GIF, manifest, and trajectory log under
  `videos/final_eval/eval_0002/`.
- 2026-07-20 — Made the heavy final-evaluation trajectory log profile-aware:
  it is enabled for `smoke` and disabled for all other standard profiles,
  while explicit `video.save_trajectory_log` overrides remain supported.

## 12. Deviations

No deviations identified.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/implementation/scenario_acl_vectorized_execution_v1_exec_plan.md` | Added | This implementation plan. |
| `src/thesis_rl/curriculum/scenario_acl/driver.py` | Modified | Parent-side batch selection, transaction, artifacts and resume. |
| `src/thesis_rl/curriculum/scenario_acl/vectorized.py` | Added | Parent-owned slot state, deterministic batch selection, per-episode provenance, LP attribution, ordering, and atomic checkpoint state. |
| `src/thesis_rl/curriculum/scenario_acl/runtime.py` | Modified | Validated ACL vector capability gate. |
| `src/thesis_rl/runtime/execution/deterministic_subproc_vec_env.py` | Modified | ACL-only selective reset worker protocol. |
| `src/thesis_rl/runtime/wiring/builders.py` | Modified | Build the approved ACL vector env path without generic seed partitioning. |
| `src/thesis_rl/envs/thesis_scenario_env.py` | Modified | Safe application/reporting of slot-local ACL selections. |
| `src/thesis_rl/envs/factory.py` | Modified | Preserve train/validation/test split boundaries when partitioning ScenarioNet provider records. |
| `src/thesis_rl/agent/agent.py` | Modified | Parent callback boundary for ACL selective reset after terminal transition storage. |
| `src/thesis_rl/agent/planners/core/backend_base.py` | Modified | Collection-time LP backend capability with no replay fallback. |
| `src/thesis_rl/agent/planners/core/lifecycle.py` | Modified | Per-slot/episode residual attribution and lifecycle LP access. |
| `src/thesis_rl/agent/planners/core/buffers.py` | Modified | PPO rollout slot/episode provenance storage. |
| `src/thesis_rl/agent/planners/algorithms/{td3,td3_sb3,sac,sac_sb3,ppo,ppo_sb3}.py` | Modified | Collection-time TD3/SAC residuals and PPO episode-level GAE provenance. |
| `src/thesis_rl/agent/planners/core/lifecycle.py` | Modified | LP provenance interface. |
| `src/thesis_rl/agent/planners/algorithms/{ppo,ppo_sb3,td3,td3_sb3,sac,sac_sb3}.py` | Modified | Backend-specific per-episode LP implementation. |
| `tests/test_scenario_acl_*.py` | Modified/added | ACL state, transaction, LP and resume coverage. |
| `tests/test_scenario_acl_vectorized_state.py` | Added | Vector state, duplicate, ordering, LP, partial PPO, and RNG regression coverage. |
| `tests/test_scenarionet_vectorized_integration.py` | Existing regression | Generic spawned ScenarioNet vector smoke; ACL-specific frozen-catalog smoke executed through the real CLI fixture. |

## 14. Validation results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| Repository/specification inspection | PASS | 2026-07-20 | Complete instruction, authoritative specification, ADR-014, and worktree inspection. |
| System Python focused pytest | NOT_RUN | 2026-07-20 | Collection blocked by missing `omegaconf`. |
| Compose focused ACL pytest | PASS | 2026-07-20 | `24 passed`: vector state/config/usefulness tests. |
| Compose ScenarioNet vector regression | PASS | 2026-07-20 | `8 passed` after filtering provider records by requested split before worker partitioning. |
| Compose ACL/vector combined suite | PASS | 2026-07-20 | `33 passed` across ACL state/config/usefulness, ScenarioNet vector integration, and transition-boundary tests. |
| Compose collection-LP focused suite | PASS | 2026-07-20 | `50 passed` before final collector boundary extension; latest focused rerun `26 passed`. |
| Compose ACL/vector/driver focused suite | PASS | 2026-07-20 | Latest ACL/ScenarioNet/boundary suite `56 passed`, including state observation, resume restart, PPO pending-completion deduplication, and the legacy ScenarioEnv regression. |
| Compose full pytest suite | PARTIAL | 2026-07-20 | `699 passed, 1 skipped, 2 failed`; both remaining failures are unrelated pre-existing tests: forced-rule debug scenario configuration and golden recorder fixture missing `paths.run_dir`. No ACL/ScenarioNet test failed. |
| Real ScenarioNet ACL vector smoke | PASS | 2026-07-20 | Frozen catalog/runtime, `spawn`, `num_envs=2`, 600 transitions; three completion events, two fresh inserts and one replay update; vector state/checkpoints persisted. |
| Real ScenarioNet ACL vector resume | PASS | 2026-07-20 | After the approved active-slot restart fix, `latest` resume completed 10 additional transitions with `num_envs=2`, `spawn`, and no pre-reset worker step. Restart events were persisted for both active slots. |
| Real ScenarioNet ACL vector generate/replay smoke | PASS | 2026-07-20 | Compact `horizon=10`, 60-transition TD3 run: two fresh insertions at tick 0 and replay updates on both slots at ticks 1--2; per-episode LP values were distinct. |
| Real ScenarioNet ACL vector PPO smoke | PASS | 2026-07-20 | Compact `horizon=10`, 60-transition PPO-SB3 run: six episodes completed; pending terminal completions were persisted once each while rollout LP attribution remained incomplete at chunk end. |
| Real ScenarioNet ACL vector periodic evaluation smoke | PASS | 2026-07-20 | Compact `horizon=10`, 20-transition TD3 run with `eval_interval=10`: two validation evaluations appeared in the terminal and `evals.csv`; final test evaluation also completed. |
| Real ScenarioNet ACL vector final-evaluation artifact smoke | PASS | 2026-07-20 | Compact `horizon=10`, 2-transition TD3 run: `Final Evaluation` appeared in the terminal, `final_eval.csv` was written, and `videos/final_eval/eval_0002/` contained a GIF, manifest, and trajectory log. |
| Ruff check for all modified feature files | PASS | 2026-07-20 | Ruff check passed for collector, ACL package, factory, env, subprocess vector, builders, and focused tests. |
| Python compile check | PASS | 2026-07-20 | `PYTHONPYCACHEPREFIX=/tmp/thesis-metadrive-pycache .venv/bin/python -m compileall -q ...`. |
| `git diff --check` | PASS | 2026-07-20 | No whitespace errors. |
| Ruff check for new files | PASS | 2026-07-20 | New vector state and test files pass. |
| Ruff format check for new files | PASS | 2026-07-20 | New vector state and test files are formatted. |
| Full focused Ruff directory check | PASS | 2026-07-20 | Ruff check passed; format-check is not a gate because it includes pre-existing baseline formatting debt. |

## 15. Final reconciliation

Reconciliation is complete for the parent/vector control plane, worker
protocol, collection-time TD3/SAC/PPO attribution, observation isolation,
termination boundary, artifacts, and checkpoint resume. AC-VEC-001--010 and
TEST-VEC-001--017 are covered by focused tests and/or the provisioned real
fixture smokes. The two unrelated full-suite failures remain separately
documented and are not part of this feature change.
