# ScenarioNet ACL with Semantic-Arm EMA Selection and Prioritized Replay

## Metadata

- Feature: ScenarioNet automatic curriculum learning with EMA arm selection
- Specification ID: `ACL-SN-EMA-001`
- Version: `v1.1`
- Status: `APPROVED`
- Date: `2026-07-23`
- Supersedes: `docs/specifications/automatic_curriculum_learning_v1_specification.md` only for the selected ScenarioNet scalar ACL core
- Related specifications: `docs/specifications/scenarionet_integration_v1.1_specification.md`, `docs/specifications/rl_baselines_v1_specification.md`
- Related ADRs: `docs/decisions/ADR-014-scenarionet-acl-learning-potential-only.md`, `docs/decisions/ADR-016-scenario-acl-vectorized-execution.md`, `docs/decisions/ADR-028-scenario-acl-generate-eligibility-renormalization.md` (REQ-002/REQ-003 amendment, approved 2026-07-27)
- Authoritative: `YES`

## 1. Purpose and context

This specification defines a project adaptation named **ScenarioNet ACL with semantic-arm EMA selection and learning-potential/staleness-prioritized scenario replay**. It replaces the cumulative positive-feedback arm update used by the selected ScenarioNet scalar ACL core with a bounded exponential moving average (EMA) of recent normalized algorithmic learning potential. It is not a direct reproduction of Peng or Abouelazm: Peng motivates adaptive arm selection, while learning potential, scenario storage, and staleness are inspired by PLR-family work. EMA, temperature, frozen semantic arms A0--A5, immutable ScenarioNet records, absence of mutation, and the 40/60 mode ratio are project choices.

## 2. Scope

### In scope

- six frozen ScenarioNet semantic arms A0--A5;
- deterministic Generate/Replay selection;
- EMA score update for the selected generated arm;
- temperature-scaled softmax with explicit exploration floor;
- 40% Generate and 60% Replay after warm-up;
- existing 70/30 usefulness/staleness replay sampling;
- checkpoint/resume persistence of scores, buffer, counters, and RNG;
- default use by `make run` when no curriculum override is supplied.

### Out of scope

- mutation, generated children, or writes to ScenarioDescription/source data;
- Rulebook criticality, safety rank, or margins in ACL usefulness;
- changes to algorithm-specific LP formulas;
- changes to transition replay, reward scalarization, observation schema, or dataset splits.

## 3. Terminology and invariants

`K=6` is the number of arms. `q_i` is the EMA score of arm `i`, `p_i` its Generate selection probability, `\widetilde{LP}` normalized learning potential, and `alpha` the EMA coefficient. Scores and normalized LP are finite values in `[0,1]`. `eta` is the exploration mixture coefficient and `tau` the positive softmax temperature. Generate means selecting a fresh frozen catalog record; Replay means selecting a record already in the scenario buffer. During warm-up, all selections are Generate.

## 4. Inputs and prohibited information

| Input | Meaning | Range/source | Missing behavior | Policy-visible |
|---|---|---|---|---|
| arm index | selected A0--A5 | integer `[0,5]`, frozen catalog | fatal configuration error | No |
| normalized LP | algorithm-specific episode learning potential rank | finite `[0,1]`, ACL §12 | fatal invalid metric | No |
| RNG state | deterministic parent selector state | NumPy generator state | fatal checkpoint error | No |
| buffer record | frozen ScenarioNet identity and metrics | immutable dataset identity | replay selection error | No |

Future tracks, privileged Rulebook criticality, safety rank, evaluator metrics, mutation output, and transition-replay priorities are prohibited from arm scores, replay ranking, or mode selection.

## 5. Outputs

The selector emits a mode, arm (for Generate), scenario identity, and selection probability. Diagnostics expose current scores, probabilities, update count, and mode counts. These are curriculum diagnostics/reproducibility metadata, not policy observations or training rewards.

## 6. Functional requirements

### REQ-001: EMA arm score

For a generated episode assigned to arm `i`, update only that arm:

`q_i <- (1-alpha) q_i + alpha * normalized_LP`.

No inverse-probability correction is applied. Scores remain finite and in `[0,1]`.

### REQ-002: Temperature and exploration

Before Generate sampling, compute `p_i=(1-eta) softmax(q_i/tau)+eta/K` over the arms currently eligible for Generate (`K_eligible <= K`; an arm is eligible when at least one of its frozen catalog records is not already held by the scenario buffer or claimed elsewhere in the same vectorized batch). An ineligible arm receives `p_i=0`. Require `0<eta<=1`, `tau>0`, `0<alpha<=1`; probabilities must be finite, sum to one within numerical tolerance, and satisfy `p_i >= eta/K_eligible` for every eligible arm. **Amended 2026-07-27 by ADR-028** (`DEC-EXH-001`, `DEC-EXH-002`): the original wording used `K` unconditionally, which let an arm whose entire pool had been absorbed into the buffer keep being drawn, silently degrade to Replay, and freeze its EMA score (FIND-001). When every arm is simultaneously ineligible, sampling falls back to the existing missing-frozen-record Replay/fatal path unchanged.

### REQ-003: Generate/Replay schedule

While `len(buffer)<warmup_buffer_size`, select Generate. Afterwards select Generate with probability `0.40` and Replay with probability `0.60`. Replay never creates or mutates a ScenarioDescription. **Amended 2026-07-27 by ADR-028**: the 0.40/0.60 split is measured over Generate draws that produce a committed Generate episode; an arm made ineligible by REQ-002 is excluded from the Generate arm draw rather than counted as a degraded Replay, so the split holds even while one or more arms are temporarily ineligible.

### REQ-004: Replay ranking preservation

Retain usefulness/staleness mixture `0.70/0.30`, rank exponent `1.0`, and staleness offset `1`. No Rulebook-derived value may affect rank or replay probability.

### REQ-005: Persistence and compatibility

Persist EMA scores, update count, buffer, counters, and RNG. Checkpoints created under the cumulative-weight schema are incompatible and require explicit restart/migration; silent interpretation as EMA is forbidden.

### REQ-006: Default curriculum

The root `make run` configuration selects this approved ACL profile by default. Explicit `curriculum=disabled` or another approved override remains authoritative.

## 7. Algorithmic contract

Initialization: `q_i=0.50` for every arm. For every Generate selection, sample from the formula in REQ-002, run the frozen scenario, normalize LP by the existing rank-normalization contract, then apply REQ-001 immediately in the parent process. Replay does not update the MAB. Use numerically stable log-sum-exp softmax.

## 8. State, timing, reset, serialization

State is parent-owned and updated after an episode's LP is available. Vectorized completions are committed in deterministic `(collection_tick, worker_id, episode_id)` order. Reset clears no EMA state; a new run initializes scores to `0.50`. Resume restores all selector state and rejects incompatible schema/version.

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

Invalid values fail before learner construction. The target-MAB path is retained but disabled by default; when enabled, sampling uses the delayed target EMA scores and synchronizes them every `target_sync_interval` updates. Legacy cumulative fields (`weight_clip_*`, `initial_weight_decay`) are rejected or migrated explicitly, never silently ignored.

## 10. Errors and diagnostics

Non-finite LP, invalid probability, wrong arm count, mutation configuration, incompatible checkpoint, or missing frozen record is fatal. Each committed Generate records arm, pre/post score, normalized LP, probability, and update count. Each episode records mode, scenario identity, and source. Rulebook diagnostics remain diagnostic-only.

## 11. Reproducibility and compatibility

The specification ID/version, resolved YAML, dataset/catalog hashes, seed, and selector state are persisted. Existing cumulative-MAB checkpoints cannot resume under v1.1 without an explicit migration artifact; the default migration policy is restart. Previous experiments remain reproducible under their recorded v1 configuration.

## 12. Acceptance criteria

### AC-001: EMA bounded update

Given `q=[0.5]*6`, arm A0 receives normalized LP `1` repeatedly and A1--A5 receive `0`; scores move toward those values, remain in `[0,1]`, and only the selected arm changes per update. (REQ-001)

### AC-002: Probability contract

The resulting probabilities are finite, sum to one, satisfy the exploration floor, and with A0 higher than the others have `p(A0)>1/6`. (REQ-002)

### AC-003: Warm-up and 40/60 schedule

Before 100 records every selection is Generate; after warm-up a seeded long sequence uses both modes with deterministic replay and empirical proportions consistent with 0.40/0.60 within the test tolerance. (REQ-003)

### AC-004: Replay and Rulebook separation

Changing Rulebook diagnostics while holding LP fixed leaves score, rank, replacement, replay probability, and MAB feedback unchanged. (REQ-004)

### AC-005: Checkpoint compatibility

EMA state round-trips exactly; cumulative-schema checkpoints are rejected with an explicit incompatibility error. (REQ-005)

### AC-006: Default composition

Composing the root config without a curriculum override resolves to the v1.1 ACL profile; explicit disabled/alternative profiles still resolve as requested. (REQ-006)

## 13. Required validation categories

Required: nominal/boundary, invalid configuration, numerical stability, update order, reset, deterministic seed, checkpoint/resume, no privileged information, replay separation, regression, and end-to-end smoke. Mutation validation remains required as a prohibited feature. Distributional and lexicographic LP variants are covered by existing algorithm-specific tests; no new formula is introduced here.

## 14. Traceability

| Requirement | Acceptance | Source/decision |
|---|---|---|
| REQ-001/002 | AC-001/002 | Proposal §2--6; project adaptation |
| REQ-003 | AC-003 | Proposal §7/9; ADR-014 |
| REQ-004 | AC-004 | ACL v1.1 §28.3; ADR-014 |
| REQ-005 | AC-005 | ACL v1.1 §28.4; ADR-016 |
| REQ-006 | AC-006 | User request 2026-07-23 |

## 15. Open decisions and limitations

| ID | Question | Recommendation | Status |
|---|---|---|---|
| DEC-001 | Approve EMA/temperature/40-60 as frozen scientific defaults? | Approve this v1.1 specification | APPROVED 2026-07-23 |
| DEC-002 | Migrate old cumulative checkpoints? | Reject and restart unless a separately reviewed converter is required | APPROVED 2026-07-23 |
| DEC-003 | An arm whose entire frozen pool is absorbed by the buffer keeps being drawn, silently degrades to Replay, and freezes its EMA score forever (FIND-001, ADR-028) — how should REQ-002/REQ-003 handle it? | Exclude ineligible arms before sampling and renormalize the softmax and `eta/K` floor over the eligible subset | APPROVED 2026-07-27 |

## 16. References

`automatic_curriculum_learning_v1_specification.md` §§12--13/28; ADR-014; ADR-016; attached proposal `pasted-text-1.txt`; stable-baselines3 and repository deterministic vector execution contracts.

## 17. Implementation handoff checklist

- [x] Scope, exclusions, formulas, state, configuration, diagnostics, compatibility, and acceptance criteria defined.
- [x] Material decisions approved by the user's explicit implementation request on 2026-07-23.
- [x] Canonical filename and `Authoritative: YES` set after approval.

## 18. Approval record

- Approved by: user
- Approval date: 2026-07-23
- Approval evidence: explicit request to implement the attached ACL changes
- Repository path: `docs/specifications/automatic_curriculum_learning_v1.1_specification.md`
- Project index updated: YES
