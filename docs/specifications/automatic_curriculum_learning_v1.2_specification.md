# ScenarioNet ACL with Semantic-Arm EMA Selection and Prioritized Replay

## Metadata

- Feature: ScenarioNet automatic curriculum learning with EMA arm selection
- Specification ID: `ACL-SN-EMA-001`
- Version: `v1.2`
- Status: `APPROVED`
- Date: `2026-07-27`
- Supersedes: `docs/specifications/automatic_curriculum_learning_v1.1_specification.md`
- Related specifications: `docs/specifications/scenarionet_integration_v1.1_specification.md`, `docs/specifications/rl_baselines_v1_specification.md`
- Related ADRs: `docs/decisions/ADR-014-scenarionet-acl-learning-potential-only.md`, `docs/decisions/ADR-016-scenario-acl-vectorized-execution.md`, `docs/decisions/ADR-030-acl-recorded-rationale-record.md`, `docs/decisions/ADR-029-acl-reward-scale-normalization.md`
- Authoritative: `YES`
- Source ExecPlan: `docs/implementation/automatic_curriculum_learning_v1.2_exec_plan.md` (`ACL-SN-INIT-002`)

## 1. Purpose and context

This version makes three changes relative to v1.1 and records the rationale for design choices that v1.1 left undocumented.

**Changed behavior (`DEC-006`/`DEC-007`, approved 2026-07-27):**

1. The per-episode learning potential is normalized by a per-arm reward-scale estimate before it enters the shared rank window, removing a cross-arm confound in which arms with structurally larger reward magnitude (driven by rulebook violation propensity, not learnability) received systematically higher scores (`FIND-004`).
2. The off-policy learning-potential formulas (TD3, SAC) use the positive part of the TD residual, `max(delta, 0)`, instead of the absolute value `|delta|`, restoring for those algorithms the zone-of-proximal-development (ZPD) hopelessness filter that the PPO formula `max(GAE, 0)` already had structurally.
3. The same reward-scale-normalized value from (1), not the raw learning potential, is stored on `ScenarioRecord` and therefore governs replay-buffer retention/eviction priority, not only the MAB feedback used for Generate sampling (`DEC-007`). Without this, the confound in (1) would persist through the 60% of training time spent on Replay even after the MAB itself was fixed.

**Unchanged, now documented (`RAT-001..RAT-009`, §15):** every other mechanism — rank normalization, the shared rank window, the bounded EMA update, the absence of an inverse-probability correction, the disabled target-MAB path, the temperature term, the semantic-tier initialization question, and the uniform initial score — is retained from v1.1 exactly as implemented. `DEC-001` (per-arm initialization) was reconsidered and closed as "keep `q_i = 0.50` uniform for all arms," now supported by measured evidence (`MEAS-002`) rather than left as an unexplained default.

This is not a direct reproduction of Peng or Abouelazm: Peng motivates adaptive arm selection, while learning potential, scenario storage, and staleness are inspired by PLR-family work. EMA, temperature, frozen semantic arms A0–A5, immutable ScenarioNet records, absence of mutation, the 40/60 mode ratio, and the reward-scale normalizer introduced in this version are project choices.

## 2. Scope

### In scope

- six frozen ScenarioNet semantic arms A0–A5;
- deterministic Generate/Replay selection;
- EMA score update for the selected generated arm;
- temperature-scaled softmax with explicit exploration floor;
- 40% Generate and 60% Replay after warm-up;
- existing 70/30 usefulness/staleness replay sampling;
- per-arm reward-scale normalization of the LP fed to the rank window (new in v1.2);
- positive-part (`max(., 0)`) TD3/SAC learning-potential formulas (new in v1.2);
- reward-scale-normalized LP as the value stored for replay-buffer retention/eviction priority, not only for the MAB feedback path (extended in v1.2, `REQ-007`);
- checkpoint/resume persistence of scores, reward-scale estimates, buffer, counters, and RNG;
- default use by `make run` when no curriculum override is supplied.

### Out of scope

- mutation, generated children, or writes to ScenarioDescription/source data;
- Rulebook criticality, safety rank, or margins in ACL usefulness;
- the PPO learning-potential formula (already positive-part; unchanged);
- the chunk-level LP fallback used only when no per-episode LP is available (`arm_index = -1`; cannot be arm-scale-normalized by construction);
- variance inflation of prediction-error-based LP at critic convergence (§15, `LIM-002`);
- changes to transition replay, reward scalarization, observation schema, or dataset splits.

## 3. Terminology and invariants

`K=6` is the number of arms. `q_i` is the EMA score of arm `i`, `p_i` its Generate selection probability, `\widetilde{LP}` normalized learning potential, `LP_i` the raw per-episode learning potential attributed to arm `i`, `s_i` the per-arm reward-scale EMA, and `alpha` the EMA coefficient (shared by `q_i` and `s_i`). Scores and normalized LP are finite values in `[0,1]`. `s_i` is a finite non-negative reward-magnitude estimate, clamped away from zero at read time by a fixed floor `s_min = 1e-3`. `eta` is the exploration mixture coefficient and `tau` the positive softmax temperature. Generate means selecting a fresh frozen catalog record; Replay means selecting a record already in the scenario buffer. During warm-up, all selections are Generate.

## 4. Inputs and prohibited information

| Input | Meaning | Range/source | Missing behavior | Policy-visible |
|---|---|---|---|---|
| arm index | selected A0–A5 | integer `[0,5]`, frozen catalog | fatal configuration error | No |
| raw LP | algorithm-specific episode learning potential | finite `>=0`, ACL §12 | fatal invalid metric | No |
| normalized LP | rank of the reward-scale-normalized LP within the shared window | finite `[0,1]` | fatal invalid metric | No |
| RNG state | deterministic parent selector state | NumPy generator state | fatal checkpoint error | No |
| buffer record | frozen ScenarioNet identity and metrics | immutable dataset identity | replay selection error | No |

Future tracks, privileged Rulebook criticality, safety rank, evaluator metrics, mutation output, and transition-replay priorities are prohibited from arm scores, replay ranking, or mode selection.

## 5. Outputs

The selector emits a mode, arm (for Generate), scenario identity, and selection probability. Diagnostics expose current scores, reward-scale estimates, probabilities, update count, and mode counts, including both the raw LP (`U`) and the reward-scale-normalized LP (`U_scaled`) per committed episode. These are curriculum diagnostics/reproducibility metadata, not policy observations or training rewards.

## 6. Functional requirements

### REQ-001: EMA arm score

For a generated episode assigned to arm `i`, update only that arm:

`q_i <- (1-alpha) q_i + alpha * normalized_LP`.

No inverse-probability correction is applied (`RAT-004`). Scores remain finite and in `[0,1]`.

### REQ-002: Temperature and exploration

Before Generate sampling, compute `p_i=(1-eta) softmax(q_i/tau)+eta/K`. Require `0<eta<=1`, `tau>0`, `0<alpha<=1`; probabilities must be finite, sum to one within numerical tolerance, and satisfy `p_i >= eta/K` (`RAT-006`).

### REQ-003: Generate/Replay schedule

While `len(buffer)<warmup_buffer_size`, select Generate. Afterwards select Generate with probability `0.40` and Replay with probability `0.60`. Replay never creates or mutates a ScenarioDescription.

### REQ-004: Replay ranking preservation

Retain usefulness/staleness mixture `0.70/0.30`, rank exponent `1.0`, and staleness offset `1`. No Rulebook-derived value may affect rank or replay probability.

### REQ-005: Persistence and compatibility

Persist EMA scores, reward-scale estimates, update count, buffer, counters, and RNG under checkpoint schema `acl_ema_v2`. Checkpoints created under `acl_ema_v1` (pre-`v1.2`) or the cumulative-weight schema are incompatible and require explicit restart; silent interpretation under a different schema is forbidden.

### REQ-006: Default curriculum

The root `make run` configuration selects this approved ACL profile by default. Explicit `curriculum=disabled` or another approved override remains authoritative.

### REQ-007: Per-arm reward-scale normalization (new in v1.2, `DEC-006`)

Maintain a per-arm reward-scale EMA `s_i`, initialized to `1.0` identically for every arm and updated only on committed Generate episodes using the same `alpha` as `q_i`:

`s_i <- (1-alpha) s_i + alpha * |episode_reward|`.

For every committed episode (Generate or Replay) assigned to arm `i`, before it enters the shared rank window, compute:

`LP_scaled = LP_i / max(s_i, s_min)`,

using the value of `s_i` as it stands *before* that same episode's `update_reward_scale` call, so an episode cannot normalize itself. `_normalize_learning_potential` and the shared rank window operate on `LP_scaled`, not on raw `LP_i`. `LP_scaled` is also the value stored in `ScenarioRecord.learning_potential`/`usefulness` and therefore governs replay-buffer retention/eviction priority (`buffer.insert`'s worst-of-buffer replacement): this closes the residual confound a first version of `DEC-006` would otherwise have left open in the 60% of training time spent on Replay (`DEC-007`, ExecPlan §11). The raw `LP_i` remains unchanged only in diagnostic logs (`RAT-001`; `live_event_context["U"]`, `buffer_events`).

### REQ-008: Positive-part off-policy learning potential (new in v1.2, `DEC-006`)

The production TD3 and SAC learning-potential computation (`agent/planners/core/lifecycle.py: acl_learning_potential`, `acl_ready_learning_potentials`) and its documented fallback (`curriculum/scenario_acl/usefulness.py: compute_td3_learning_potential`, `compute_sac_learning_potential`) use

`LP = mean(max(delta_t, 0))`

over the episode's collected TD residuals `delta_t`, matching the existing PPO formula `mean(max(A_t, 0))`. This does not resolve the irreducible variance-inflation limitation shared by all three algorithms (`LIM-002`).

## 7. Algorithmic contract

Initialization: `q_i=0.50` and `s_i=1.0` for every arm (`RAT-008`; `DEC-001` closed as "keep uniform", `MEAS-002`). For every Generate selection, sample from the formula in REQ-002, run the frozen scenario, compute the raw LP per REQ-008, normalize it by reward scale per REQ-007, normalize the result by the existing rank-normalization contract, then apply REQ-001 and update `s_i` immediately in the parent process. Replay does not update the MAB score `q_i` or the reward-scale estimate `s_i`, but its LP is still reward-scale-normalized before entering the shared rank window (REQ-007), preserving the softmax shift-invariance property that makes the shared window safe across Generate and Replay episodes (`RAT-002`, `MEAS-001`). Use numerically stable log-sum-exp softmax.

## 8. State, timing, reset, serialization

State is parent-owned and updated after an episode's LP is available. Vectorized completions are committed in deterministic `(collection_tick, worker_id, episode_id)` order. Reset clears no EMA state; a new run initializes scores to `0.50` and reward-scale estimates to `1.0`. Resume restores all selector state and rejects incompatible schema/version (`acl_ema_v1` checkpoints cannot resume under `v1.2`).

## 9. Configuration

| Field | Type | Default | Range | Frozen |
|---|---|---:|---|---|
| `buffer_capacity` | int | 1000 | `>0` | YES |
| `warmup_buffer_size` | int | 100 | `>=0` | YES |
| `generate_probability` | float | 0.40 | `[0,1]` | YES |
| `exploit_probability` | float | 0.60 | `[0,1]`, complementary | YES |
| `mab.num_arms` | int | 6 | exactly 6 | YES |
| `mab.update_method` | string | `ema` | exactly `ema` | YES |
| `mab.alpha` | float | 0.10 | `(0,1]` | YES |
| `mab.initial_score` | float | 0.50 | `[0,1]` | YES |
| `mab.eta` | float | 0.20 | `(0,1]` | YES |
| `mab.temperature` | float | 0.50 | `>0` | YES |
| `mab.use_importance_correction` | bool | false | false only | YES |
| `mab.use_target_mab` | bool | false | bool; optional delayed target scores | YES |
| `mab.target_sync_interval` | int | 1 | `>0`; used when target MAB is enabled | YES |
| replay weights | floats | 0.70/0.30 | nonnegative, sum 1 | YES |

The per-arm reward-scale estimator (`s_i`, initial value `1.0`, floor `1e-3`) is **not** independently configurable: it is an implementation constant reusing `mab.alpha`, per the parsimony argument in `RAT-004`/`RAT-007` against unjustified new hyperparameters (`ADR-029`). Invalid values fail before learner construction. The target-MAB path is retained but disabled by default. Legacy cumulative fields (`weight_clip_*`, `initial_weight_decay`) are rejected or migrated explicitly, never silently ignored.

## 10. Errors and diagnostics

Non-finite LP, invalid probability, wrong arm count, mutation configuration, incompatible checkpoint, or missing frozen record is fatal. Each committed episode records arm, pre/post score, raw LP (`U`), reward-scale-normalized LP (`U_scaled`), rank-normalized LP (`U_norm`), probability, and update count. Rulebook diagnostics remain diagnostic-only.

## 11. Reproducibility and compatibility

The specification ID/version, resolved YAML, dataset/catalog hashes, seed, and selector state are persisted. Checkpoints from `v1.1` (`acl_ema_v1`) cannot resume under `v1.2` (`acl_ema_v2`) without an explicit migration artifact; the default migration policy is restart. Previous experiments remain reproducible under their recorded `v1.1` configuration and are unaffected retroactively — this version changes only the signal computed for future episodes.

## 12. Acceptance criteria

### AC-001: EMA bounded update

Given `q=[0.5]*6`, arm A0 receives normalized LP `1` repeatedly and A1–A5 receive `0`; scores move toward those values, remain in `[0,1]`, and only the selected arm changes per update. (REQ-001)

### AC-002: Probability contract

The resulting probabilities are finite, sum to one, satisfy the exploration floor, and with A0 higher than the others have `p(A0)>1/6`. (REQ-002)

### AC-003: Warm-up and 40/60 schedule

Before 100 records every selection is Generate; after warm-up a seeded long sequence uses both modes with deterministic replay and empirical proportions consistent with 0.40/0.60 within the test tolerance. (REQ-003)

### AC-004: Replay and Rulebook separation

Changing Rulebook diagnostics while holding LP fixed leaves score, rank, replacement, replay probability, and MAB feedback unchanged. (REQ-004)

### AC-005: Checkpoint compatibility

`acl_ema_v2` state round-trips exactly (scores, target scores, reward-scale estimates, update count); `acl_ema_v1` and cumulative-schema checkpoints are rejected with an explicit incompatibility error. (REQ-005)

### AC-006: Default composition

Composing the root config without a curriculum override resolves to the `v1.2` ACL profile; explicit disabled/alternative profiles still resolve as requested. (REQ-006)

### AC-007: Reward-scale normalization removes the cross-arm scale confound

Given two arms with identical raw LP distributions but different reward-magnitude distributions, after each arm's reward-scale estimate has converged, the reward-scale-normalized LP distributions are approximately equal, and the shared rank window assigns the two arms statistically indistinguishable ranks. Before any reward-scale update (`t=0`), normalization is a no-op for every arm (`s_i=1.0` uniformly). (REQ-007)

### AC-008: Positive-part TD3/SAC learning potential

Given a residual sequence with mixed signs, `compute_td3_learning_potential` and `compute_sac_learning_potential`, and the production path `acl_learning_potential`, equal `mean(max(delta,0))`, not `mean(|delta|)`; an all-negative residual sequence yields `LP=0`. (REQ-008)

## 13. Required validation categories

Required: nominal/boundary, invalid configuration, numerical stability, update order, reset, deterministic seed, checkpoint/resume, no privileged information, replay separation, regression, and end-to-end smoke. Mutation validation remains required as a prohibited feature. The reward-scale estimator requires: uniform-initialization no-op behavior, EMA update correctness, floor clamping, and checkpoint round-trip/incompatibility tests. Distributional and lexicographic LP variants are covered by existing algorithm-specific tests; no new formula for those is introduced here.

## 14. Traceability

| Requirement | Acceptance | Source/decision |
|---|---|---|
| REQ-001/002 | AC-001/002 | Proposal §2–6; project adaptation |
| REQ-003 | AC-003 | Proposal §7/9; ADR-014 |
| REQ-004 | AC-004 | ACL v1.1 §28.3; ADR-014 |
| REQ-005 | AC-005 | ACL v1.1 §28.4; ADR-016; `DEC-005` |
| REQ-006 | AC-006 | User request 2026-07-23 |
| REQ-007 | AC-007 | `FIND-004`; `DEC-006`; `DEC-007`; `ADR-029` |
| REQ-008 | AC-008 | `FIND-002`; `DEC-006`; `ADR-029` |

## 15. Recorded rationales, open decisions, and known limitations

### 15.1 Recorded rationales (`RAT-001..RAT-009`)

The following mechanisms were already implemented and approved under `v1.1` but never had their justification written down. They are reconstructed here from the code and the reference paper, and are the authoritative rationale as of `v1.2`. Full derivations are in `docs/implementation/automatic_curriculum_learning_v1.2_exec_plan.md` §6.2; summarized:

- **`RAT-001`** — Raw LP cannot feed the EMA directly: it drifts down as the value function converges (independent of true across-arm rank), has an algorithm-dependent unbounded scale, and is heavy-tailed. Rank normalization removes drift and scale by construction and is outlier-robust.
- **`RAT-002`** — The rank window is shared across arms, not per-arm, because a per-arm window would converge every arm to `~0.5` by construction, leaving the bandit no cross-arm signal. This is necessary for the mechanism to function, and creates the residual non-stationarity that `DEC-006`/`REQ-007` addresses for reward-scale, but not for LP magnitude in general.
- **`RAT-003`** — A bounded EMA replaces the paper's cumulative accumulation because, with non-negative rank feedback, cumulative accumulation degenerates (every arm's weight increases monotonically to the clip, then the softmax returns to uniform). This is a structural incompatibility, not a preference.
- **`RAT-004`** — No inverse-probability (`1/p_i`) correction: `update()` is called only for the sampled arm, so there are no zero-contribution rounds to balance; the correction would move the EMA's fixed point to `E[U_i]/p_i`, a bias, not a fix. Quantitatively it would saturate scores at `1.0` on floor-probability arms in the vast majority of updates.
- **`RAT-005`** — The target-MAB path (periodic synchronization) stays disabled: in the paper it is the sole update mechanism (cumulative accumulator); layered on an already-smoothing EMA it adds a delay and a new hyperparameter with no identified problem to solve. Retained as a diagnostic option.
- **`RAT-006`** — The temperature term exists because bounded `q in [0,1]` scores would otherwise cap the maximum arm-preference ratio at `e^1`; `tau=0.5` restores an expressive ratio (`e^2`).
- **`RAT-007`** — Full fidelity to the reference paper (cumulative accumulation, `1/p` correction, `N_MAB` resync, re-centered `2U-1` feedback, no temperature) is rejected on cost/benefit: it requires rewriting the core update/selection contract, breaks checkpoint compatibility, and introduces an untuned hyperparameter with no ablation budget to justify it.
- **`RAT-008`** — Initialization uses a semantic tier, not the positional arm index, because the taxonomy does not guarantee an ordinal difficulty ordering across `A1..A4` the way the paper's vehicle-count index does; only `A0` (upper-bounded complexity) is structurally guaranteed to be the easiest arm.
- **`RAT-009`** — Replay episodes entering the shared rank window is measured (`MEAS-001`) to be a level shift only (dispersion unaffected: `0.3031` mixed vs `0.2972` Generate-only), which the softmax is invariant to. No code change from this finding alone.

### 15.2 Closed decisions

| ID | Question | Resolution | Status |
|---|---|---|---|
| `DEC-001` | Per-arm initial score `q_i`? | Uniform `q_i=0.50` for all arms, supported by `MEAS-002` (A0 is empirically the lowest- or near-lowest-LP arm from the earliest measurement onward, across three algorithms and four runs) | CLOSED 2026-07-27 |
| `DEC-004` | Does Replay contamination of the rank window need a fix? | No code change; measured to be a benign level shift (`MEAS-001`) | CLOSED 2026-07-27 |
| `DEC-005` | Bump the checkpoint schema for `v1.1`→`v1.2` documentation-only changes? | N/A at the time (no state change); superseded by `DEC-006`, which does add persisted state and does bump the schema (`acl_ema_v1`→`acl_ema_v2`) | CLOSED 2026-07-27 |
| `DEC-006` | How to address `FIND-004` (cross-arm reward-scale confound) and align TD3/SAC LP sign handling with PPO? | Per-arm reward-scale EMA normalization (REQ-007) combined with positive-part TD residuals (REQ-008) | APPROVED 2026-07-27 |
| `DEC-007` | Does `DEC-006`'s reward-scale normalization need to extend to replay-buffer retention/eviction priority (`ScenarioRecord.learning_potential`), or only to the MAB feedback path? | Extend it: `LP_scaled` (not raw `LP_i`) is stored in `ScenarioRecord.learning_potential`/`usefulness` and therefore governs `buffer.insert` eviction, closing the residual confound in the 60% of training time spent on Replay. Decided while the three experiments were already scheduled for a restart, avoiding a second future restart. | APPROVED 2026-07-27 |

### 15.3 Known limitations (not resolved by this version)

- **`LIM-002`** — Variance inflation: at critic convergence `E[delta]=0`, so both `mean(|delta|)` (pre-`v1.2`) and `mean(max(delta,0))` (`v1.2`) are proportional to `sigma(delta)`. A noisy-but-unlearnable scenario still yields nonzero LP. This predates `v1.2`, is not introduced by `DEC-006`, and affects PPO's existing formula identically. No fix within the prediction-error LP family is known; the principled fix is a learning-progress signal (slope of return per arm over repeated visits), explicitly out of scope: no ablation budget, requires an architectural change.
- **`LIM-003`** — `MEAS-003` (measured on 15 runs before `DEC-006`) shows the bandit's arm-probability ordering is close to static (median Spearman rank correlation `+0.66` between the first and last recorded chunk) rather than tracking an easy-to-hard progression; the direction of drift is correct (mass moves toward `A3/A4/A5`) but the ordering is largely set by `t~25000` steps. Whether `DEC-006`/`DEC-007` change this materially is an empirical question for the re-run experiments, not established by this specification.
- **`LIM-004`** — The `commit_event`/`collect_catalog_episode` call-site wiring of `LP_scaled` into the buffer-record constructors (`REQ-007`/`DEC-007`) is verified by direct code reading and by the full passing focused test suite (which exercises `ScenarioArmBandit.normalize_learning_potential_by_reward_scale` and `ScenarioRecord`/`compute_scenario_usefulness` individually), but has no dedicated closure-level regression test, because `commit_event` is a private closure inside the ~800-line vectorized training loop with no existing harness for isolated testing. Residual risk: a future edit to that closure could silently revert to the raw value without a test failing. Follow-up command, if a harness is built: a test asserting that, given two arms with equal raw LP but different reward-scale estimates, `buffer.insert` prefers the arm with the higher `LP_scaled`, not the higher raw LP.

## 16. References

`automatic_curriculum_learning_v1.1_specification.md`; `automatic_curriculum_learning_v1_specification.md` §§12–13/28; ADR-014; ADR-016; ADR-030; ADR-029; `docs/implementation/automatic_curriculum_learning_v1.2_exec_plan.md`; Peng et al., 2024, IROS (`docs/papers/curriculum learning/`); stable-baselines3 and repository deterministic vector execution contracts.

## 17. Implementation handoff checklist

- [x] Scope, exclusions, formulas, state, configuration, diagnostics, compatibility, and acceptance criteria defined.
- [x] Material decisions (`DEC-001`, `DEC-004`, `DEC-005`, `DEC-006`, `DEC-007`) approved by the user during the 2026-07-27 session.
- [x] Canonical filename and `Authoritative: YES` set after approval.
- [x] Implementation complete: `mab.py`, `driver.py`, `lifecycle.py`, `usefulness.py`; focused tests, lint, and format-check pass (ExecPlan §14).

## 18. Approval record

- Approved by: user
- Approval date: 2026-07-27
- Approval evidence: explicit request "procedi con l'implementazione" following review of `DEC-006`'s consolidated design (reward-scale normalization + positive-part TD3/SAC formula) and its cost (redoing the three in-progress experiments)
- Repository path: `docs/specifications/automatic_curriculum_learning_v1.2_specification.md`
- Project index updated: YES
