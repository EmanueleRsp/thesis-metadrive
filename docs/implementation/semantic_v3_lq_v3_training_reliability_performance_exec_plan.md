# semantic_v3 / latent_query_v3 training reliability and performance

## 1. Metadata

- Plan ID: `SEMANTIC-V3-LQ-V3-RUNTIME-PERF`
- Status: `IMPLEMENTED`
- Created / updated: 2026-07-22
- Authoritative specifications: OBS-V1.2 (`docs/specifications/observation_v1.2_specification.md`), ENC-V1.1 (`docs/specifications/encoder_v1.1_specification.md`), RULEBOOK-V4.7 (`docs/specifications/rulebook_v4.7_specification.md`), RL-BASELINES v1 (`docs/specifications/rl_baselines_v1_specification.md`), ACL v1, and ScenarioNet v1.1; all `APPROVED`.
- Applicable decisions: ADR-016, ADR-022, ADR-023.

## 2. Objective and scope

Restore robust progress for the canonical `semantic_v3` + `latent_query_v3`
training path, determine measured bottlenecks, and apply only behavior-preserving
optimizations. The target benchmark is a finite, deterministic diagnostic run
using the same provider, seed, profile, worker count, and learner overrides
before and after each change. FPS means committed global environment steps
divided by elapsed monotonic wall-clock seconds, including collection and
learner updates; its denominator is also reported by phase.

In scope: the attached `CanonicalGeometryError`, worker progress/cleanup,
profiling instrumentation emitted only in internal training artifacts, redundant runtime work,
and TD3/SAC/PPO integration validation. Out of scope: a scientific-contract,
dataset, reward, Rulebook, termination/truncation, observation-token, or
learner-hyperparameter change; mixed precision, compilation, and dependencies
require a separate approval gate.

## 3. Authoritative requirements

| ID | Requirement | Source |
|---|---|---|
| REQ-RP-01 | `semantic_v3` has the exact causal 3,064-value contract and `lq_v3` has 143 tokens. | OBS-V1.2 §§4--13; ENC-V1.1 §§2--6 |
| REQ-RP-02 | Conflict geometry uses the frozen 1 mm canonicalization; if no valid component remains, no pertinent zone exists. | RULEBOOK-V4.7 §§2.6, 2.8, 15.11 |
| REQ-RP-03 | Vector worker failures are explicit and owned workers close cleanly; global steps retain their meaning. | ScenarioNet v1.1 §§23--24; RL-BASELINES §8 |
| REQ-RP-04 | TD3, SAC, and PPO collect real transitions and update their learners without NaN/Inf. | RL-BASELINES §§8--11 |
| REQ-RP-05 | Optimizations preserve physical perception, logging, step count, and frozen experiment configuration. | OBS-V1.2 §§6--11; user objective |

## 4. Repository analysis and invariants

- **VERIFIED:** `canonicalize_geometry()` is fail-fast after `set_precision(..., mode="valid_output")`; vehicle and crosswalk candidate builders currently canonicalize each post-intersection polygon.
- **VERIFIED:** the attached error occurs in `build_vehicle_conflict_zone_candidates()` while canonicalizing a post-intersection component.
- **VERIFIED:** `semantic_v3` and `lq_v3` are explicit configuration selectors; no legacy fallback is allowed.
- **INVARIANT:** a valid pre-snap overlap with area below or unstable at the frozen grid may represent no valid conflict-zone component, not a fatal scenario-wide error, provided this is the semantics prescribed by RULEBOOK-V4.7 §2.8.
- **AWAITING_MEASUREMENT:** present GPU availability, dataset mount, baseline phase timings, source of a possible fast-profile hang, and contention.

## 5. Acceptance criteria and frozen mandatory matrix

| ID | Acceptance criterion | Test / command |
|---|---|---|
| AC-01 | A valid corridor overlap which collapses at the frozen grid yields no candidate; invalid source geometry still fails explicitly. | New deterministic Rulebook geometry regression |
| AC-02 | The attached canonical run advances beyond the failing transition with finite values and clean worker shutdown. | Reproduction command, fixed-seed diagnostic smoke |
| AC-03 | The profiler reports reset, step, observation, encoder, transition collection, learner, logging/callback, and total time in internal training artifacts. | Focused vector/environment integration matrix |
| AC-04 | Under identical benchmark parameters, before/after global-step FPS and phase medians are recorded; no prohibited feature is disabled. | Reproducible benchmark command |
| AC-05 | TD3, SAC, and PPO each reset, collect, update, and close with finite data. | Three bounded provider-backed smokes |

| Test ID | Level | Expected behavior | Requirement |
|---|---|---|---|
| TEST-RP-01 | Unit | snapped-away valid overlap is filtered; malformed geometry raises `CanonicalGeometryError` | REQ-RP-02 |
| TEST-RP-02 | Integration | phase counters/timers are finite and recorded without altering public observation/reward contracts | REQ-RP-05 |
| TEST-RP-03 | Integration | fixed-seed vector training reports progress without worker crash/hang | REQ-RP-03 |
| TEST-RP-04 | Smoke | TD3/SAC/PPO each make at least one actual update with finite transitions | REQ-RP-04 |

## 6. Design, decisions, and milestones

`DEC-RP-01` (implementation detail, pending evidence): filter only the
post-boolean component that legitimately disappears at the already-frozen grid;
do not repair, broaden, or alter invalid source corridors. This is expected to
implement the “no component remains” Rulebook outcome and requires a focused
regression before use.

- [x] M1: reproduce the attached crash, establish baseline and phase profile.
- [x] M2: add failing regression(s), apply the smallest contract-preserving fix.
- [x] M3: profile, remove measured redundant overhead, and benchmark again.
- [x] M4: validate all three algorithms, quality checks, smoke, and reconcile. The 20 FPS target is not reached under the frozen benchmark and is recorded as a limitation, not a deviation.

## 7. Traceability and files

| Requirement | Implementation | Tests | Status |
|---|---|---|---|
| REQ-RP-02 | `rulebook/v2/geometry/conflict_zones.py`, complete pair cache in `transition.py` | `tests/test_rulebook_v2_geometry.py`, `tests/test_rulebook_v2_transition.py` | Implemented; focused tests passed |
| REQ-RP-03--05 | agent/vector-worker/environment timing, sparse PER persistence, lane/occupancy geometry caches | replay/vector/environment integration matrix, full Rulebook matrix, and three smokes | Implemented; bounded validation passed |

| Path | Planned action | Purpose |
|---|---|---|
| `src/thesis_rl/rulebook/v2/geometry/conflict_zones.py` | Possible modification | Filter only snapped-away derived overlaps. |
| `tests/test_rulebook_v2_*.py` | Add regression | Preserve canonical geometry semantics. |
| `src/thesis_rl/**` | Possible measured-only modification | Remove demonstrated overhead. |
| `scripts/profile_rulebook_scenarionet_case.py` | Add diagnostic | Reproduce and phase-profile a fixed real ScenarioNet case. |
| this plan | Update | Evidence, benchmark, reconciliation. |

## 8. Validation and findings log

| Date | Evidence / result |
|---|---|
| 2026-07-22 | Attached production traceback read: worker slot 2 fails while a post-intersection component collapses after 1 mm snapping. Existing code confirms the exception is propagated rather than hidden. |
| 2026-07-22 | Corrected candidate construction to canonicalize source corridors before the boolean operation, as required by RULEBOOK-V4.7 §2.8. Derived components that alone collapse after that operation are discarded as no valid zone; invalid source corridors remain fail-fast. `tests/test_rulebook_v2_geometry.py`: `27 passed in 0.19s`; focused Ruff and format checks passed. |
| 2026-07-22 | Correction to the initial artifact lookup: the fixed-seed five-worker GPU TD3 diagnostic did complete 64 global steps and 65 update calls (including setup accounting) by 08:08:51. The vector-driver CSV currently drops the `fps`/elapsed fields returned by `Agent.train_vectorized`, so it cannot support the requested benchmark yet. The smoke replay persistence serialised `latest_replay_buffer.pkl` at 7,370,421,217 bytes after this 64-step run; this measurable checkpoint I/O is a major smoke-profile overhead candidate, but its enabled policy is approved and may not simply be disabled. |
| 2026-07-22 | Restored vector CSV FPS/elapsed metrics and added parent, worker, environment, and Rulebook phase timing. Baseline TD3 was 2.45 FPS (26.53 s/64 global steps); wrapper Rulebook evaluation was dominant. Sparse PER persistence reduced a 64-step replay artifact from 7.37 GB to 1.6 MB while preserving replay capacity, active rows, priorities, RNG, and sampling. |
| 2026-07-22 | Rulebook pair-cache regression proves complete canonical candidates are reused with unchanged zone ID. Caching the invariant 1 cm lane coverage envelope reduced TD3 diagnostic wall time to 15.75 s / 64 steps (4.13 FPS); Rulebook evaluator reduced from 22.3 to 8.88 worker-seconds. No policy-visible value, logging stream, reward, geometry tolerance, or step count changed. |
| 2026-07-22 | A second measured redundancy was removed: crosswalk, vehicle-yield, and RSS used to recompute lane association for the same live ego/vehicles during one transition. Associations are now materialized once per pre/post snapshot and reused; a bounds rejection avoids Shapely `covers` calls which cannot match under the frozen 1 cm lane-association tolerance. Focused Rulebook regression and formatting/lint checks passed; this incremental change awaits an uncontended end-to-end FPS measurement. |
| 2026-07-22 | The identical TD3 five-worker diagnostic with the association reuse completed the 64-step training chunk, 65 real gradient steps, and finite losses at **5.2558 FPS** / 12.3672 s. The valid one-episode final evaluation wrote `final_eval.csv`, `final.zip`, `final_replay_buffer.pkl`, `latest_rng_state.pkl`, and final-evaluation video/trajectory artifacts, proving cleanup. Timings are learner update 5.9195 s, parent `env.step` 5.5467 s, transition collection 0.7301 s, and encoder action 0.1629 s. The zero-final-evaluation variant instead fails fast by design (`n_eval_episodes must be positive`) after the completed training chunk. |
| 2026-07-22 | The previously non-terminating real ScenarioNet integration case at index 1358 was traced to vehicle-yield occupancy prediction for a 699-vertex conflict zone. The frozen deterministic ear clipping was O(n³) from per-ear point allocation/full scans. Precomputed vertices, a bounds filter, prepared coverage, a bounded exact decomposition cache, prepared CTRV intersection, and within-transition pair reuse preserve the same decomposition rules and tie-break. The first real step now completes in 10.33 s and the next nine in 0.60--1.09 s; `tests/test_rulebook_v2_scenarionet_integration.py` passes both source scenarios in `22.03 s`. |
| 2026-07-22 | Full Rulebook v2 matrix now completes: `197 passed in 26.36s`. The focused integrated matrix passes again after the occupancy change: `85 passed in 18.08s`. Ruff, formatting, and `git diff --check` pass over the modified scope. |
| 2026-07-22 | Contention audit found two obsolete Rulebook test containers started by this task at ~100% of one CPU each; GPU utilization was 0%. After stopping only those owned diagnostics, the identical TD3 benchmark completed with final evaluation/checkpoint artifacts at **5.3462 FPS** (65 steps / 12.1581 s), versus 5.2558 FPS under that CPU contention. Contention is therefore recorded but is not the primary throughput cause. |
| 2026-07-22 | The bounded `fast` TD3 diagnostic after all changes also completed cleanly: 65 steps, 65 gradient steps, finite losses, 5.3590 FPS / 12.1291 s, `final_eval.csv`, `final.zip`, and RNG state present. This closes the reported fast-profile progress/cleanup concern. |
| 2026-07-22 | Fast-profile TD3 diagnostic completed without hang at 2.45 FPS before lane-envelope caching. SAC bounded smoke completed 64 steps/65 updates, finite losses and replay/checkpoint cleanup. PPO bounded smoke completed 80 steps/10 gradient steps and final-test cleanup with `final_eval_episodes=1`; a deliberately invalid zero final-evaluation override correctly failed fast and was not treated as a runtime regression. |
| 2026-07-22 | Focused integrated regression matrix passed: `85 passed in 17.88s` across replay persistence, agent/vector execution, ScenarioNet environment, semantic-v3, ENC-V1.1, and checkpoint contracts. Ruff and formatting checks passed on all owned modified files; `git diff --check` passed. The full Rulebook glob matrix displayed no failures but the terminal did not return its final summary, so it remains non-conclusive evidence rather than a claimed pass. |

Planned commands (only successful results will be recorded): focused geometry
pytest, focused profiler pytest, `make rulebook-v2-check`, bounded container
smokes for TD3/SAC/PPO, focused Ruff/format check, and `git diff --check`.

## 9. Deviations and final reconciliation

No deviations identified. The measured 5.3462 FPS diagnostic maximum remains
below the requested 20 FPS target. The residual cost is physical MetaDrive
stepping plus the deliberately forced TD3 learner updates (together roughly
11.5 s of a 12.16 s chunk); reaching the target under this frozen benchmark would
require a further measured equivalent optimization or an approved change to
the runtime/benchmark configuration. Candidate approval-gated alternatives are
different vector-worker/learner scheduling, Torch compilation, and mixed
precision; none was introduced. All other requirements reconcile to code and
successful validation, so the implementation status is `IMPLEMENTED` rather
than `VERIFIED` solely because the indicative 20 FPS target is unmet.
