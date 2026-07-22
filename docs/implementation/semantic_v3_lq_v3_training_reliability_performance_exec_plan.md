# semantic_v3 / latent_query_v3 training reliability and performance

## 1. Metadata

- Plan ID: `SEMANTIC-V3-LQ-V3-RUNTIME-PERF`
- Status: `IN_PROGRESS`
- Created / updated: 2026-07-22
- Authoritative specifications: OBS-V1.2 (`docs/specifications/observation_v1.2_specification.md`), ENC-V1.1 (`docs/specifications/encoder_v1.1_specification.md`), RULEBOOK-V4.7 (`docs/specifications/rulebook_v4.7_specification.md`), RL-BASELINES v1 (`docs/specifications/rl_baselines_v1_specification.md`), ACL v1, and ScenarioNet v1.1; all `APPROVED`.
- Applicable decisions: ADR-016, ADR-022, ADR-023.
- Additional approval: on 2026-07-22, the user approved a checked constrained
  triangulation for every valid polygon for which the prescribed ear clipping
  cannot select an ear. The geometric occupancy domain must remain identical;
  the internal triangle partition need not be identical.

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
and TD3/SAC/PPO integration validation. The throughput expansion approved on
2026-07-22 additionally covers isolated diagnostic scaling experiments,
policy-action batching, deterministic parallel evaluation of independent
Rulebook work, replay/IPC/copy profiling, and equivalent scheduling/caching
optimizations. Out of scope: a scientific-contract,
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
| REQ-RP-06 | Aggregate transition throughput is measured consistently while scaling worker count and resource use. | User objective 2026-07-22 |
| REQ-RP-07 | Parallel work has deterministic reduction and cannot reorder normative transitions, ACL decisions, or learner updates. | User objective; ScenarioNet v1.1 §23; ACL v1 §28 |

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
- [ ] M5: reproduce the 2026-07-22 fast TD3 worker failure, repair every valid
  ear-exhaustion case while preserving the occupancy geometry, and add
  deterministic regressions before verifying a real fast run past the failure.
- [x] M6: establish an isolated TD3 scaling curve (5/10/16/20 workers; the ACL
  vector contract rejects one worker), with
  matching global transition/update accounting, CPU/GPU samples, and no change
  to any scientific profile.
- [x] M7: profile and remove equivalent pipeline overhead (IPC, copies, replay,
  learner scheduling, and independent Rulebook subphases); validate exact
  transition outputs and all three algorithm paths.
- [ ] M8: reconcile the measured throughput ceiling against the 20/30 FPS
  target, documenting residual approval-gated alternatives.
- [x] M9: with explicit user approval, isolate TD3 replay/PER versus learner
  compute time, then benchmark a non-default batch size and mixed precision
  only in labelled diagnostic runs. Async collection/policy lag is excluded.
- [x] M10: with explicit user approval, screen diagnostic-only LQ-lite (eight
  latents, two blocks, unchanged 143-token input and 256-output interface) for
  finite training, memory, and throughput. No learning-quality claim is made.
- [x] M11: with explicit user approval, screen diagnostic-only LQ-micro (four
  latents, one block) plus a `[128, 128]` decoder. The `make run` launcher now
  exposes `ENCODER=<Hydra encoder config>` and defaults to the approved
  20-worker diagnostic setting; neither architecture is a scientific default.
- [x] M12: recheck worker-thread scaling after the reduced learner screen and
  probe 24 workers with the same explicit one-thread override. Retain the
  largest configuration that completes the benchmark rather than treating a
  stalled bootstrap as throughput.
- [x] M13: test an explicit, diagnostic-only pre-spawn numeric-library thread
  limit. It must be inherited before child imports, leave the default unchanged,
  and demonstrate clean worker startup plus finite real transitions before any
  throughput claim. Final evaluation remains a separate non-training teardown.
- [x] M14: propagate the same explicit pre-spawn cap to the asynchronous
  evaluator process and its vector children, with a regression that confirms
  scoped inheritance and parent-environment restoration.
- [x] M15: screen an explicit parent-process PyTorch thread cap after seeding
  but before planner construction. Keep the default null and compare only
  matched diagnostic training chunks.
- [x] M16: vectorize independent proportional-PER prefix-tree lookups while
  preserving the same float64 tree, stratified masses, and selected indices.
  Verify scalar/batched equivalence before a matched benchmark.
- [x] M17: batch PER priority-tree propagation after the prescribed duplicate
  maximum reduction, preserving raw priorities, tree masses, and exact
  current-maximum bookkeeping.

## 7. Traceability and files

| Requirement | Implementation | Tests | Status |
|---|---|---|---|
| REQ-RP-02 | `rulebook/v2/geometry/conflict_zones.py`, complete pair cache in `transition.py` | `tests/test_rulebook_v2_geometry.py`, `tests/test_rulebook_v2_transition.py` | Implemented; focused tests passed |
| REQ-RP-03--05 | agent/vector-worker/environment timing, sparse PER persistence, lane/occupancy geometry caches | replay/vector/environment integration matrix, full Rulebook matrix, and three smokes | Implemented; bounded validation passed |
| REQ-RP-03, REQ-RP-06 | pre-spawn numeric-library thread-limit inheritance for an explicit diagnostic override | spawned vector-worker inheritance regression; bounded 24-worker benchmark | Pending |

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
| 2026-07-22 | The first repair was insufficient: the next identical fixed-seed run failed at the same collection point, and its traceback reached the ordinary zero/one-hole `ValueError` branch. A minimal valid concave shell with one hole reproduces ear exhaustion. The root cause is therefore the incomplete local ear predicate, not the number of holes or invalid ScenarioNet geometry. |
| 2026-07-22 | With explicit user approval, every valid ear-exhaustion case now uses GEOS constrained triangulation of the original polygon, filters invalid/degenerate components, proves coverage by symmetric difference, and applies the existing deterministic output ordering. Normal ear-clipping and its frozen tie-break remain the primary path. Regressions cover multi-hole and single-hole exhaustion, exact area, and repeated deterministic WKB order. |
| 2026-07-22 | Generalized regressions plus the live ScenarioNet integration cases passed: `34 passed in 22.28s`. The complete Rulebook v2 matrix then passed: `200 passed in 26.69s`. A full fast run through the former collection point remains required to close M5; it was not started automatically because it requires roughly twelve minutes of GPU execution. |
| 2026-07-22 | Matched 320-transition/320-gradient-step TD3 diagnostics scale from 5.9231 FPS (5 workers) to 6.5841 (10) and 6.9489 (16). Limiting each of 16 workers to one PyTorch thread gives 7.0028 FPS. A 32-worker attempt remained in worker bootstrap, consumed about 21 GiB and 3,830 processes/threads, and made no transition; it was an owned diagnostic and was stopped. The practical saturation is therefore below 32 workers, but the 1-worker point remains pending. |
| 2026-07-22 | At 16 workers and one worker thread, aggregate worker time is 58.009 s for 320 transitions while the parent waits 14.183 s: physical ScenarioNet/MetaDrive stepping is 32.903 worker-s and Rulebook evaluation 24.697 worker-s. Within the Rulebook evaluator, vehicle yield is 6.546 s, registry/aggregation 4.038 s, drivable-area 3.627 s, RSS candidates 0.652 s, and crosswalk 0.358 s. These totals are parallel-worker sums, not wall-clock components. Learner update is 30.216 wall-s of the 45.696-s training chunk; collection-time TD residuals account for a separate 1.029-s transition-collection phase. |
| 2026-07-22 | An equivalent learner-scheduling diagnostic changed TD3 `train_freq` from 1 to 4 with automatic `gradient_steps` from 16 to 64, preserving exactly 320 gradients for 320 transitions while reducing update calls from 20 to 5. It achieved 6.9711 FPS versus 7.0028 FPS for the matched baseline, so scheduling-call overhead is not a worthwhile optimization. The learner cost is predominantly necessary replay/PER sampling, CPU-to-device batch conversion, actor/critic forward-backward work, delayed-policy updates, and an additional replay-sample TD-residual pass used for ACL learning potential. |
| 2026-07-22 | The non-vector one-environment diagnostic completed at 3.8557 FPS, but it uses the distinct episode-sampling ACL path and is not comparable with vector execution. Forcing vector execution correctly fails fast because the approved ACL runtime contract requires `env.vectorized.num_envs > 1`; therefore the comparable scaling curve starts at five workers. |
| 2026-07-22 | The 16-worker learner split reports 31.6524 s inside `TD3.train()` and 0.9816 s for the additional ACL replay sample/residual pass, of 32.6352 s learner-update wall time. The ACL pass is only 3.0% of learner time, so changing its implementation alone cannot materially affect throughput. The same run reaches 6.6374 FPS; small run-to-run variability is expected from real scenarios. |
| 2026-07-22 | A matched 20-worker TD3 diagnostic completes 320 real transitions and 320 gradient steps in 44.3519 s, or **7.2150 FPS** (16 update calls). This is the best measured correct configuration so far, but only 3.0% above 16 workers and 21.8% above five workers. Together with the 32-worker bootstrap failure, it establishes a practical worker-scaling ceiling far below the 20 FPS target. |
| 2026-07-22 | Vehicle-yield now deduplicates candidate-zone expansion for actors with the same `(ego movement, other movement)` pair while preserving the independent actor priority/occupancy pass. Regression verifies one candidate for two same-movement actors before and after the episode cache. The matched 20-worker benchmark is 7.1861 FPS versus 7.2150 FPS before the change (within normal real-scenario variance); its measured benefit is therefore negligible for this workload, although it removes work in multi-actor same-movement scenes. |
| 2026-07-22 | Full post-change Rulebook validation passes: `make rulebook-v2-check` reports `200 passed in 26.78s`, scoped Ruff passes, and `git diff --check` passes. |
| 2026-07-22 | Read-only runtime audit confirms the reproducible container uses PyTorch `2.9.1+cu128`, CUDA is available, and `torch.compile` is exposed. There is no existing compile/AMP path in the training code. This establishes technical availability only; it does not establish numerical equivalence or authorize enabling it. |
| 2026-07-22 | With explicit user authorization, an isolated `torch.compile` diagnostic was attempted and then fully removed from the working tree because neither available backend is valid for the real TD3 path in the reproducible image. Inductor fails before the first committed transition with `PermissionError: nvcc` (the runtime exposes the CUDA driver but no CUDA compiler); full CUDAGraph compilation fails on overwritten graph outputs, and actor-only CUDAGraph compilation fails when collection batch size 20 changes to replay batch size 256. No scientific or diagnostic profile was changed. A usable compilation experiment requires either a CUDA compiler toolchain or a fixed-shape compile-aware design, both material approval/dependency gates. |
| 2026-07-22 | ACL audit: `num_arms=6` is a fixed semantic A0--A5 contract, while `target_sync_interval=5` (`N_MAB`) counts generated episode completions, not worker slots or environment steps. Parent-side completion commits are deterministically ordered by `(collection_tick, worker_id, episode_id)`. The matched 320-transition worker diagnostics completed no episodes, so ACL/MAB updates did not contribute to their measured FPS. Increasing workers does not require changing `N_MAB`; changing it would alter the curriculum feedback dynamics and is an ablation, not a throughput optimization. |
| 2026-07-22 | With explicit user authorization, a labelled diagnostic-only switch skipped `vehicle_yield` **and its complete input path** (movement-pair candidate generation/cache, conflict-zone selection, priority/occupancy checks, and CTRV occupancy prediction). The matched 20-worker TD3 run completed 320 transitions and 320 gradient steps at **7.6867 FPS** / 41.6305 s, versus 7.2150 FPS / 44.3519 s canonically: **+0.4717 FPS, +6.5%**. It is not a scientific result and must remain false by default. Learner update is still 31.3569 s, so the change does not alter the conclusion that removing vehicle-yield cannot approach the 20 FPS target. |
| 2026-07-22 | Fine learner profiling of a completed matched 20-worker TD3 run: `TD3.train()` is 30.9229 s; replay sampling including conversion/device transfer is 1.5835 s (5.1%), PER priority update 0.2367 s (0.8%), and the residual actor/critic forward-backward/optimizer compute is about 29.1026 s (94.1%). A user-authorized batch-512 diagnostic is slower: 6.7960 FPS / 47.0863 s versus 7.0469 FPS / 45.4103 s with batch 256 under the same instrumentation; `TD3.train()` increases to 32.6411 s. Batch 512 is excluded. |
| 2026-07-22 | The user-authorized, diagnostic-only CUDA FP16 autocast TD3 run completes 320 transitions and 320 gradient steps with finite losses, but is slower: **6.6239 FPS** / 48.3102 s and `TD3.train()` 33.9825 s. It remains false by default and is excluded as a throughput option for this architecture/runtime. |
| 2026-07-22 | The user-authorized LQ-lite technical screen (8 latents and 2 blocks, retaining the exact 143 semantic tokens and output width 256) completes 320 transitions and 320 gradient steps with finite losses at **9.2786 FPS** / 34.4881 s. Learner update is 21.6241 s and `TD3.train()` 21.1041 s, versus 31.6848 s and 30.9229 s for the profiled canonical LQ run: about 31.8% faster learner wall time. The live post-training GPU reading is 38,760 MiB / 97,871 MiB and 42% utilization; it is not a peak-memory comparison. This is a speed/finite-data screen only, not evidence that the reduced capacity learns safely or equivalently. |
| 2026-07-22 | The user-authorized LQ-micro + decoder-lite technical screen (4 latents, 1 block; decoder `[128,128]`; unchanged 143-token input and encoder output 256) completes 320 transitions and 320 gradient steps with finite losses at **10.5423 FPS** / 30.3539 s. Learner update is 17.0556 s and `TD3.train()` 16.6427 s. This is +13.6% over the LQ-lite screen and about +49.6% over the profiled canonical LQ screen, but remains a throughput/finite-data result only. |
| 2026-07-22 | Matched LQ-micro + decoder-lite worker-thread screens complete at 10.5423 FPS (1 thread), 10.4888 FPS (2), and 10.5145 FPS (4). The 0.5% spread is within run variability and gives no evidence that a higher per-worker PyTorch thread count is beneficial; retain the explicit one-thread diagnostic override rather than changing the global `null` default. |
| 2026-07-22 | A 24-worker LQ-micro + decoder-lite probe with `worker_num_threads=1` did not commit a first transition after 3m44s of bootstrap. It reached about 3,000 processes/threads and 18.43 GiB, then the owned diagnostic was stopped. A separate pre-spawn `worker_library_num_threads=1` probe reduced this to 294 processes/threads, proving that NumPy/BLAS/OpenMP pools were created before the worker function. It still failed to commit a transition after more than four minutes because one real ScenarioNet reset remained CPU-bound; it too was stopped. Thus the thread-pool explosion is fixed diagnostically, but 24 workers are not yet a usable benchmark configuration. |
| 2026-07-22 | With both explicit worker limits, 21 LQ-micro + decoder-lite workers completed 336 real transitions and 336 gradient steps with finite losses in 30.9078 s: **10.8710 FPS**. The parent phase split is learner 17.5352 s, synchronous `env.step` 12.5245 s, action/encoder 0.3367 s; worker summed wrapped stepping is 53.7689 s and Rulebook evaluator 23.1481 s. This is a +3.1% matched training-chunk gain over the 20-worker LQ-micro screen (10.5423 FPS), while retaining the fixed transition/gradient ratio. The separate final-evaluation teardown was stopped after the completed chunk because it is outside chunk FPS and remains slow for this real scenario; it is not a training failure. |
| 2026-07-22 | Architecture audit: TD3 with `share_features_extractor=false` owns four independent feature-extractor instances (actor, actor target, twin critic, critic target). This is intentional baseline behavior; enabling sharing changes critic-gradient routing and the effective target-update behavior, so it is not an equivalent performance optimization. Encoder parameter counts are canonical LQ-v3 1,121,344, LQ-lite 589,888, and LQ-micro 324,160 per instance. Further material learner reduction therefore requires an explicitly approved diagnostic architecture ablation rather than an internal sharing shortcut. |
| 2026-07-22 | The numeric-library cap is now applied not only to training/evaluation vector children but also to the asynchronous evaluator process itself, which otherwise imports NumPy before it can build capped children. A deterministic fake-process regression verifies `OPENBLAS_NUM_THREADS=2` is visible at evaluator process start and the parent value is restored. Focused worker/vector/async-evaluation matrix: `20 passed`; scoped Ruff, format, and `git diff --check` pass. |
| 2026-07-22 | An explicit parent-only `runtime.parent_torch_num_threads=1` screen at 21 LQ-micro workers reduced the live process count to 212 but slowed the matched training chunk to **10.5399 FPS** / 31.8790 s. Learner update rose from 17.5352 to 18.5020 s while synchronous `env.step` was unchanged (12.4956 s). The default parent PyTorch thread setting is therefore retained; further small parent caps are not justified. New helper regressions plus worker/vector/async-evaluation tests: `23 passed`; scoped Ruff, format, and `git diff --check` pass. |
| 2026-07-22 | PER sampling now resolves the same stratified float64 prefix masses in batch, and PER priority updates reduce duplicate indices with the prescribed maximum before batch tree-delta propagation. Scalar/batched boundary and duplicate-update regressions prove identical selected indices, raw priorities, tree values, and maximum bookkeeping; replay/TD3 matrix: `29 passed`. At 21 LQ-micro workers, sampling falls from 1.5619 s to 1.1247 s and priority update from 0.2501 s to 0.1491 s. End-to-end chunk throughput rises from 10.8710 to **10.9154 FPS** / 30.7822 s (+0.4%); the remaining learner compute dominates. |
| 2026-07-22 | A single 22-worker probe with both numeric caps and fully vectorized PER remained before its first committed transition after about 1m44s. It held 282 processes/threads, so this is not the former oversubscription failure: one physical ScenarioNet reset stayed CPU-bound while the synchronous parent waited. The owned probe was stopped; 21 workers remains the largest completed configuration and further worker counts are not a credible throughput path. |

Planned commands (only successful results will be recorded): focused geometry
pytest, focused profiler pytest, `make rulebook-v2-check`, bounded container
smokes for TD3/SAC/PPO, focused Ruff/format check, and `git diff --check`.

## 9. Deviations and final reconciliation

The approved ear-exhaustion robustness correction is a bounded implementation
deviation from the literal ear-clipping-only wording: it preserves the
geometric occupancy domain and deterministic output order but not necessarily
the same internal triangle partition. It is recorded above with explicit user
approval.

The fastest completed **canonical** matched diagnostic is 20 vector workers
with one PyTorch thread per worker: **7.2150 aggregate transitions/s** (320
real transitions, 320 gradient steps, 44.3519 s). The fastest completed
**diagnostic architecture screen** is LQ-micro plus decoder-lite at 21 workers
with explicit PyTorch and numeric-library worker caps: **10.9154 FPS** (336
real transitions, 336 gradient steps, 30.7822 s). The latter preserves the
observation stream, Rulebook, ACL, and transition/gradient ratio, but not the
approved model capacity; it is not a scientific-result configuration. The 20
and 30 FPS targets are **not reached**. The canonical scaling curve is 5:
5.9231, 10: 6.5841, 16: 7.0028, and 20: 7.2150 FPS; 22--24 worker probes with
numeric pools capped still did not commit a first transition because of a
single CPU-bound physical ScenarioNet reset. The ceiling is not CPU-core
saturation: it is the sequential learner plus the parent wait for the slowest
fully physical environment step. At canonical 20 workers, `TD3.train()` alone
consumes 30.4202 s, while parent `env.step` consumes 11.9455 s. Thus, even
eliminating every environment cost without changing the learner would only
reach about 10.5 FPS for that benchmark.

The tested equivalent alternatives have negligible residual headroom:
worker-thread limiting gains 0.8% at 16 workers; grouping four learner updates
is 0.5% slower while preserving the gradient ratio; ACL's extra replay pass is
only 3.0% of learner time; and same-movement candidate deduplication is within
benchmark variance. `torch.compile` was explicitly tested but is unavailable
for the real run with this image: Inductor requires a missing `nvcc`, and the
available CUDAGraph path cannot safely handle the live collection/replay batch
shape change. The remaining credible ways to exceed this ceiling are therefore
mixed precision, a materially different learner batch/update configuration,
asynchronous producer/consumer collection, or provisioning a compiler/toolchain
followed by a new compile-aware diagnostic. They can change numerical execution,
optimizer timing, policy lag, dependencies, or experimental behavior and were
not introduced. The implementation status remains `IN_PROGRESS` until an
approval decision or a separately validated diagnostic for one of those
alternatives resolves M8.

The user-authorized upper-bound ablation that removes vehicle-yield and all of
its prerequisite transition work raises the matched 20-worker diagnostic only
from 7.2150 to 7.6867 FPS (+6.5%). Since it changes the Rulebook reward and
removes precedence/occupied-conflict-zone compliance, it is retained solely as
an explicitly labelled diagnostic switch, disabled by default, and is not a
candidate scientific profile.

Fine TD3 profiling and the approved learner diagnostics close the remaining
non-async alternatives tested here: replay/PER work is only 5.9% of
`TD3.train()`; batch 512 and FP16 autocast are respectively 3.6% and 6.0%
slower than the profiled batch-256 diagnostic. The learner is therefore
compute-bound in actor/critic forward-backward and optimizer work. Async
collection/policy lag remains explicitly excluded by the user; no further
equivalent learner optimization has demonstrated material headroom.
