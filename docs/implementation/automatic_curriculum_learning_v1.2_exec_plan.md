# Automatic Curriculum Learning v1.2 ExecPlan — Arm Score Initialization and Recorded Rationales

## 1. Metadata

- Feature/plan ID: `ACL-SN-INIT-002`
- Authoritative specification (current): `docs/specifications/automatic_curriculum_learning_v1.2_specification.md`, `APPROVED`, `Authoritative: YES`
- Target specification (deliverable of this plan): `automatic_curriculum_learning_v1.2_specification.md`, written and approved 2026-07-27
- Status: `IMPLEMENTED_PENDING_RERUN`
- Created: 2026-07-27
- Last updated: 2026-07-27
- Branch: `scenarionet-implementation`
- Related ADRs: ADR-014, ADR-016 (existing); ADR-028, ADR-029 (new, this plan)
- Reference paper: Peng et al., 2024, *Reward-Driven Automated Curriculum Learning for Interaction-Aware Self-Driving at Unsignalized Intersections*, IROS. Local copy: `docs/papers/curriculum learning/`

## 2. Objective and scope

### 2.1 Observable capability

Two independent deliverables, one behavioral and one documentary:

1. **Behavioral (REQ-101..REQ-103).** Replace the uniform scalar initialization of the semantic-arm bandit scores with an explicit per-arm initialization that boosts only the structurally-easiest arm (`A0_simple_low_traffic`), reproducing the *effect* of the reference paper's Eq. 17 under this repository's selection function.
2. **Documentary (REQ-104).** Record, in the specification and in ADRs, the rationale for four algorithmic choices that are currently implemented and approved but carry **no written justification anywhere in the repository**. These rationales are reconstructed and verified in §6.2 of this plan.

A third, investigation-only item (REQ-105) records a defect candidate found during analysis (Replay contamination of the rank window) and defines how to measure it from already-produced run logs, without ablation.

### 2.2 Why it is needed

- The current uniform initialization (`q_i = 0.50` for all arms) is a project choice that diverges from the reference paper's difficulty-informed initialization without a recorded reason.
- The four undocumented rationales are the principal scientific-defensibility gap in the ACL component: the repository currently states *what* it does but never *why* it departs from the cited paper. In a thesis defence this is attackable; with the rationales recorded, each departure becomes a defensible engineering decision.

### 2.3 Success criteria

- A fresh run initializes arm scores per REQ-101 and the sampling distribution at `t=0` matches the values recorded in §7.2 within numerical tolerance.
- Setting the configuration to a uniform mapping reproduces current v1.1 behavior bit-for-bit (regression guard).
- Specification v1.2 and the new ADRs contain every rationale listed in §6.2, each traceable to the evidence recorded here.

### 2.4 In scope

- `ScenarioArmBandit` initialization; ACL configuration schema and validation; default ACL YAML profiles.
- Specification v1.2, ADRs, `docs/project_index.md`.
- Focused and regression tests for initialization and configuration.
- Measurement procedure for REQ-105.

### 2.5 Out of scope

- The EMA update rule, importance correction, target-MAB path, temperature, `eta`, `alpha`, window size, replay ranking (0.70/0.30), Generate/Replay ratio (40/60): all confirmed unchanged (§6.2).
- The uncalibrated arm-classification thresholds in `src/thesis_rl/scenarios/arms.py:19-26`. Separate, already-tracked work item; this plan does not touch scenario classification.
- Any change to `arms.py` classification logic, catalog, or dataset policy.

### 2.6 Compatibility constraints

- Checkpoint schema `acl_ema_v1` is **unchanged**: initialization affects fresh runs only; `from_state_dict` restores persisted scores (`mab.py:92-120`). See DEC-005.
- The three training runs in progress as of 2026-07-27 were produced under v1.1 defaults. Runs produced after this change are **not directly comparable** to them; the user has stated the intention to re-run.

## 3. Authoritative requirements

Requirements REQ-001..REQ-006 of ACL v1.1 remain in force and unmodified. This plan proposes the following additional requirements for specification v1.2.

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-101` | Arm scores are initialized from an explicit per-arm mapping; unlisted arms use a documented default. | v1.2 §7 (Algorithmic contract), supersedes v1.1 §7 sentence "Initialization: `q_i=0.50` for every arm" |
| `REQ-102` | A configuration mapping that assigns the same value to all six arms reproduces v1.1 behavior exactly. | v1.2 §9 (Configuration) |
| `REQ-103` | Initialization applies to fresh runs only; resume restores persisted scores and does not re-initialize. | v1.2 §8 (State, timing, reset, serialization) |
| `REQ-104` | The specification records the rationale for: rank normalization, shared rank window, EMA update, absence of inverse-probability correction, disabled target MAB, and temperature. | v1.2 §1 and new "Design rationale" section |
| `REQ-105` | The Replay contamination of the rank window is measured and its outcome recorded before any corrective change. | v1.2 §15 (Open decisions and limitations) |

## 4. Current repository analysis

All statements below are labelled per `.agent/PLANS.md`.

### 4.1 Initialization — `VERIFIED`

`src/thesis_rl/curriculum/scenario_acl/mab.py:29-32`:

```python
self.scores = np.full(
    int(self.config.num_arms), float(self.config.initial_score), dtype=np.float64
)
self.target_scores = self.scores.copy()
```

A single scalar (`initial_score`, default `0.50`) is broadcast to all six arms. Validation at `mab.py:25-26` requires `initial_score ∈ [0,1]`. Configuration parsing and range check at `src/thesis_rl/curriculum/config.py:316,322`. YAML default at `conf/curriculum/scenario_acl.yaml:22`. The frozen contract is stated in `docs/specifications/automatic_curriculum_learning_v1.1_specification.md:90` ("Initialization: `q_i=0.50` for every arm") and `:107` (table row, `Frozen: YES`).

### 4.2 Arm identity and ordering — `VERIFIED`

`src/thesis_rl/scenarios/arms.py:10-17` defines `ARMS` as a tuple of six names in the order `A0..A5`. The bandit addresses arms by **positional index only** (`mab.py`, `numpy` vectors). There is **no** explicit difficulty field (`difficulty_class`, `tier`, `rank`) anywhere in the codebase. The A0..A5 ordering is a declaration order, not a validated ordinal difficulty axis (see §6.2 RAT-008).

### 4.3 Selection function — `VERIFIED`

`mab.py:39-57`: `p_i = (1-eta)·softmax(q_i/tau) + eta/K`, with `eta=0.20`, `tau=0.50`, `K=6`. Exploration floor is `eta/K = 0.0333...`. Structurally identical to reference paper Eq. 8, except that the paper has no temperature term.

### 4.4 Update rule — `VERIFIED`

`mab.py:63-82`: `scores[i] = clip((1-alpha)·scores[i] + alpha·value, 0, 1)`, `alpha=0.10`, applied to the sampled arm only. `selection_probability` is accepted by the signature and explicitly discarded at `mab.py:70` (`del selection_probability  # EMA deliberately has no importance correction.`). `use_importance_correction=true` is hard-rejected at `config.py:324-325`. `use_target_mab` default `false`; when true, `target_scores = scores.copy()` every `target_sync_interval` updates (`mab.py:81-82`).

### 4.5 Feedback signal — `VERIFIED`

Raw learning potential: `src/thesis_rl/curriculum/scenario_acl/usefulness.py:48-97` — PPO `mean(max(GAE,0))`, TD3 `mean(|TD residual|)`, SAC entropy-aware TD residual. Always `>= 0`, unbounded above, scale differs by algorithm.

Rank normalization: `src/thesis_rl/curriculum/scenario_acl/driver.py:316-328` — `U = 1 - (rank-1)/(M-1)` with averaged tied ranks, computed against `recent_usefulness`, a sliding window of the last `recent_window_size = 100` raw LP values.

### 4.6 Rank window is shared across arms — `VERIFIED`

`recent_usefulness` is a single flat `list[float]` (`driver.py:531`), never keyed by arm. Same object used at `driver.py:779`, `driver.py:1762`, `driver.py:2005`; appended at `driver.py:794` and `driver.py:1765`. No `dict[arm_index, list[float]]` exists.

### 4.7 Replay episodes enter the rank window — `VERIFIED`

In `commit_event` (`driver.py:775-805`) the append at `driver.py:794` is **unconditional**, while the bandit update at `driver.py:798` is guarded by `completion.selection.mode == "generate"`. Consequently Replay episodes (60% of episodes after warm-up, REQ-003) contribute to the rank set although only Generate episodes update the bandit. Replay selection prefers high stored usefulness (0.70/0.30 usefulness/staleness, REQ-004). Consequence analysis in §6.2 RAT-009.

### 4.8 Diagnostics already emitted — `VERIFIED`

`driver.py:785-787` logs per episode: `origin` (`"replay"` or `"new"`), `U` (raw LP), `U_norm` (rank-normalized). This is sufficient to measure REQ-105 from existing run logs with no code change.

### 4.9 Reference paper — `VERIFIED` (read from local PDF)

- Eq. 8 (p. 5090): `p_i(t) = (1-η)·e^{w_i(t)}/Σ_j e^{w_j(t)} + η/(N_sv^max+1)`.
- Eq. 10 (p. 5090): `r̂_i(t) = r_norm_i(t)/p_i(t)`, with `r_norm_i(t) = 2(r_i - k0·R_min)/(k1·R_max - k0·R_min) - 1`, i.e. a **signed** value in `[-1,1]` derived from **episode reward**, not from a learning potential.
- Eq. 11 (p. 5091): `w_i(t+1) = w_i(t) + α·r̂_i(t)` — cumulative additive.
- Eq. 12 (p. 5091): target-MAB synchronization every `N_MAB` rounds; `N_MAB = 1000` in their experiments.
- Eq. 17 (p. 5092): `w_i(0) = e^{-2i}`, `i = 0..6`, where **`i` is the number of surrounding vehicles**, i.e. a physically ordinal difficulty axis. This is the paper's main method.
- Eq. 18 (p. 5092): `w_i(0) = 1` — uniform initialization, used by the paper as the *ablation*, not as the method.

### 4.10 Documentation gap — `VERIFIED`

No rationale exists anywhere in the repository for the EMA choice, the removal of importance correction, the disabled target MAB, or the rank normalization beyond a single sentence at `docs/specifications/automatic_curriculum_learning_v1_specification.md:1292`. Searched: both ACL specifications, all ADRs in `docs/decisions/`, `docs/implementation/automatic_curriculum_learning_v1.1_exec_plan.md`, and commit `db80eeb`. All sources state only "project choice" / "deliberately" / "APPROVED 2026-07-23". The proposal document cited as reference in v1.1 §16 (`pasted-text-1.txt`) is not present in the repository.

## 5. Assumptions and invariants

| Item | Value | How established | Violation handling |
|---|---|---|---|
| Score domain | `q_i ∈ [0,1]` | `mab.py:79` clip; `mab.py:111-118` checkpoint validation | `ValueError` before learner construction |
| Feedback domain | `U ∈ [0,1]` | `mab.py:75-76` validation; rank construction | `ValueError` |
| Feedback distribution | `U` approximately uniform on `[0,1]` | Consequence of rank normalization over a window | Not enforced; used only in rationale arithmetic |
| Exploration floor | `p_i >= eta/K = 0.0333...` | `mab.py:54`; v1.1 REQ-002 | `ValueError` if probabilities invalid |
| Arm count | exactly 6 | `config.py:318-319` | `ValueError` |
| Structurally easiest arm | `A0_simple_low_traffic` | `arms.py:70-78`: the only arm requiring *absence* of topology **and** an upper bound `relevant_vehicles_q90 <= 8.0` | Assumption is documented, not enforced in code |
| Arms A1..A4 relative order | **unknown / not established** | `arms.py:29-79`: A1 doubles as the non-topological fallback; A4 is an orthogonal VRU axis | No ordinal claim may be made about them (see RAT-008) |
| Checkpoint schema | `acl_ema_v1`, unchanged | `mab.py:18,98-100` | Explicit rejection of foreign schema |
| Seeds/determinism | parent-owned, deterministic commit order | v1.1 §8; ADR-016 | Unchanged by this plan |

## 6. Decisions and approval gates

### 6.1 Open gates (require explicit user approval before implementation)

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-001` | Specification deviation | v1.1 §7 freezes `q_i = 0.50` for every arm. REQ-101 replaces it with a per-arm mapping. | A: keep uniform. B: per-arm mapping boosting A0 only. C: exponential over the positional index, as in paper Eq. 17 applied literally. | **Revised to A** — see MEAS-002 | Changes the sampling distribution at `t=0` | **Approved 2026-07-27 as B, then SUPERSEDED BY EVIDENCE the same day (MEAS-002). Awaiting re-decision.** |
| `DEC-002` | Implementation detail | Exact value of `q_{A0}`. | 0.65 / 0.85 / 1.0 | **Moot if DEC-001 = A** | Strength of the initial prior | Blocked on DEC-001 re-decision |
| `DEC-003` | Implementation detail | Configuration shape for per-arm initialization. | A: explicit mapping `arm name -> score`. B: `base` + `spread` + tier table. C: positional list. | **A** (unchanged; applies only if DEC-001 = B) | Configuration schema and validation | Approved 2026-07-27; blocked on DEC-001 re-decision |
| `DEC-004` | Blocking technical issue (investigation) | Replay episodes enter the rank window (§4.7). | A: measure from existing logs, then decide. B: change window to Generate-only immediately. | **A executed; outcome = no change** | Measured: level shift only, softmax-invariant (MEAS-001) | **RESOLVED 2026-07-27 — no code change** |
| `DEC-005` | Implementation detail | Whether the checkpoint schema must be bumped. | A: keep `acl_ema_v1`. B: bump. | **A** — the schema tracks the *meaning* of persisted data, which is unchanged; bumping would conflate state compatibility with configuration provenance and degrade the guard's diagnostic value | Checkpoint compatibility | Approved 2026-07-27 |

### 6.2 Recorded rationales (REQ-104)

The following are **not new decisions**. They are already-implemented, already-approved behaviors whose justification was never written down (§4.10). Each entry is reconstructed from code and from the reference paper, with the evidence that supports it. These are the texts to be carried into specification v1.2 and the ADRs.

#### `RAT-001` — Why rank normalization of the learning potential

Raw LP cannot be used directly and cannot be min-max normalized as in the paper, for three independent reasons:

1. **Systematic drift.** Both `mean(max(GAE,0))` and `mean(|TD residual|)` are prediction-error quantities that shrink as the value function converges. Feeding raw LP to a bounded EMA would drive every arm's score down over training regardless of relative merit. A rank measures "how high is this relative to what is being observed now", which removes the global drift by construction and is precisely the quantity a bandit needs in order to allocate attention.
2. **Unknown, algorithm-dependent scale.** The repository supports PPO, TD3 and SAC with different LP formulas (`usefulness.py:143-202`) whose magnitudes are not comparable. A fixed temperature and a `[0,1]`-bounded EMA cannot operate on an input of unknown scale. Some normalization is therefore mandatory, not optional.
3. **Heavy tails.** TD-residual and advantage spikes are common. Min-max normalization is outlier-dominated (one extreme episode compresses all others toward zero); rank normalization is outlier-robust by construction.

The reference paper's Eq. 10 normalizes **episode reward**, which has externally-defined bounds from the reward function design, not a learning potential, which has none. Its `[-1,1]` centering additionally serves its cumulative accumulator (RAT-003). Its normalization is therefore not transferable to this signal.

`INFERRED, VERIFY BEFORE CITING`: rank-based prioritization is the characteristic mechanism of the PLR family, which v1.1 §1 cites as the lineage for learning potential and staleness. The PLR paper is not present in `docs/papers/`; this claim must be checked against the primary source before it appears in the thesis.

Known cost, to be recorded as a limitation: rank normalization discards magnitude information (an arm with ten times the LP of another is only reported as "above the median"), and pins the mean of all arm scores near 0.5 by construction, so scores can express relative ordering but never "the whole curriculum is exhausted".

#### `RAT-002` — Why the rank window is shared across arms rather than per-arm

A per-arm window would rank each arm's episodes against that arm's own history. By construction every arm would then converge to approximately 0.5 (any sample sits near the middle of its own distribution) and the bandit would have **no signal with which to distinguish arms**. Cross-arm comparability is the property a MAB requires. The shared window is therefore necessary for the mechanism to function, not an oversight.

Known limitation: the composition of the shared window depends on the current sampling distribution, so it is non-stationary. This is intrinsic to any relative measure and is accepted.

#### `RAT-003` — Why a bounded EMA instead of the paper's cumulative accumulation (Eq. 11)

The paper's per-round feedback is **signed**, in `[-1,1]` (Eq. 10). The feedback used here is a rank, hence **always non-negative**, in `[0,1]`.

With a cumulative update `w += α·U` and non-negative feedback, every arm's weight increases monotonically and never decreases; arms differ only in growth rate, all eventually reach the clip (`±5` in ACL v1), and once there the softmax returns to uniform — the bandit stops discriminating. This was the structural condition of the ACL v1 configuration.

A bounded EMA instead converges to the recent mean rank of that arm: an interpretable, bounded, approximately stationary quantity. The change from cumulative to EMA therefore corrected a real degeneracy; cumulative accumulation and rank-based feedback are **structurally incompatible**, and this is not a matter of preference.

#### `RAT-004` — Why no inverse-probability correction (paper Eq. 10)

The `1/p_i` factor is the EXP3 device that makes the estimator **unbiased for a sum**: with `r̂_i = r_i·1{i chosen}/p_i` one has `E[r̂_i] = p_i·(r_i/p_i) = r_i`, and accumulating over rounds estimates `Σ_t r_i(t)` correctly, precisely because unchosen rounds contribute exactly zero.

That construction does not carry over to this implementation. `update()` is invoked **only for the sampled arm** (`driver.py:798`), so conditioning on selection is already implicit and there are no zero-contribution rounds to balance the inflation. Dividing by `p_i` would move the EMA's fixed point from `E[U_i]` to `E[U_i]/p_i`, i.e. produce a score systematically biased by how often the arm happens to be sampled — the opposite of a correction. Restoring unbiasedness would require updating *all* arms every round (zero for unchosen ones), which is a different algorithm.

Quantitatively, the correction would also be unusable here: because `U` is a rank it is approximately uniform on `[0,1]`, so `U >= eta/K = 0.0333` holds about 97% of the time; on an arm near the exploration floor the product `U/p_i` would exceed 1 in almost every update and the score would saturate at 1.0 essentially every time. This is the typical case, not an edge case.

Conclusion: removing the correction when moving to EMA was mathematically correct. Only the justification was missing.

#### `RAT-005` — Why the target-MAB path stays disabled

In the reference paper the periodic synchronization *is* the mechanism by which the sampling weights are updated at all (Eq. 12), because the underlying update is cumulative. Layered on top of an EMA — which already smooths — it would add a delay and one more hyperparameter (`N_MAB`) without addressing any identified problem. The path is retained as a diagnostic option should thrashing be observed in the sampling distribution; enabling it without such evidence would be an unmotivated change.

#### `RAT-006` — Why a temperature term exists although the paper has none

The paper's `w` is unbounded, so `e^{w}` can express arbitrarily large preference ratios. Here `q ∈ [0,1]`, so without a temperature the maximum ratio between the best and worst arm would be `e^1 ≈ 2.72`, too flat to express a preference. With `tau = 0.5` the maximum ratio is `e^2 ≈ 7.39`. The temperature compensates for the bounded score domain introduced by RAT-003.

#### `RAT-007` — Why full fidelity to the paper is rejected

A fully faithful variant requires *all* of: cumulative accumulation, `1/p` correction, `N_MAB` resynchronization, **re-centered feedback** (`2U-1`, otherwise it degenerates per RAT-003), and removal of the temperature. It is internally coherent, but it requires rewriting `probabilities()`, `update()` and configuration validation, breaks the `acl_ema_v1` checkpoint schema, and introduces `N_MAB` as a quantity to tune — with no evidence of benefit and no ablation budget available. Rejected on cost/benefit, not on preference.

#### `RAT-008` — Why initialization uses a semantic tier and not the positional index

Applying `e^{-2i}` to the positional index of `ARMS` would silently assert an ordinal difficulty ordering `A1 < A2 < A3 < A4` that the classifier does not guarantee: `A1_traffic` doubles as the fallback for any non-topological scenario that fails the A0 bound (`arms.py:79`), and `A4_vru` is an orthogonal axis (VRU presence) rather than a step above `A3` (`arms.py:58-59`). In the paper, by contrast, `i` is the number of surrounding vehicles — a genuine physical difficulty axis.

The arithmetic in §7.2 shows that the paper's initialization, once passed through its own selection function, is functionally "boost the easiest arm, leave everything else flat". That shape requires identifying **only the easiest arm** — which is exactly the one ordinal fact this taxonomy does establish structurally (`A0` is the only arm defined by an upper bound on complexity, `arms.py:70-78`). The hardest arm is deliberately *not* suppressed, matching the paper, which does not suppress its hard end either.

#### `RAT-009` — Replay contamination of the rank window (defect candidate, unquantified)

Per §4.7, Replay episodes enter the rank window while only Generate episodes update the bandit, and Replay selection prefers high stored usefulness. Generated episodes are therefore ranked against a window biased upward.

Ordering is provably unaffected: the window is common to all arms, so the induced shift is common, and the softmax is invariant to a constant offset. What is lost is **resolution** — if generated episodes concentrate in the lower ranks, `U` values compress into a narrower band, the differences between arm EMAs shrink proportionally, and the bandit discriminates less. The effect is partially self-limiting, because a replayed scenario's LP decreases as the agent learns it, so the magnitude is unknown.

Measurement procedure requiring no code change and no ablation: compare the distribution of `U` for `origin="replay"` against `origin="new"` in existing run logs (`driver.py:785-787`). If the distributions overlap, the effect is negligible and nothing is done. If Replay sits systematically higher, the targeted fix is to restrict the append at `driver.py:794` to Generate episodes.

## 7. Proposed design

### 7.1 Affected modules

| Module | Change |
|---|---|
| `src/thesis_rl/scenarios/arms.py` | Add an explicit, documented mapping of the structurally-easiest arm (no change to classification logic) |
| `src/thesis_rl/curriculum/config.py` | New field `initial_scores` (mapping) alongside the existing `initial_score`; validation |
| `src/thesis_rl/curriculum/scenario_acl/mab.py` | Build the initial score vector from the mapping instead of `np.full` |
| `conf/curriculum/scenario_acl.yaml` | Declare the per-arm mapping |

### 7.2 Initialization arithmetic (evidence for DEC-002)

Reference paper, Eq. 17 fed into Eq. 8 (`p_i ∝ e^{w_i}`, seven arms, uniform share = 14.3%):

| arm `i` | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| `w_i(0) = e^{-2i}` | 1.0 | 0.135 | 0.018 | 0.0025 | 3.4e-4 | 4.5e-5 | 6.1e-6 |
| share `∝ e^{w_i}` | **30.6%** | 12.9% | 11.5% | 11.3% | 11.3% | 11.3% | 11.3% |

The exponential collapses to `≈1` from `i=2` onward: the paper's initialization is **not** a gradient across arms but "the easiest arm receives about twice the uniform share, all others are essentially flat" (`30.6% / 14.3% = 2.14x` uniform).

Translation into this repository's selection function, where `p_i ∝ e^{q_i/tau}` with `tau = 0.5`, i.e. `∝ e^{2q}`:

- Target: `A0` at roughly twice the share of the others, i.e. `e^{2(q_0 - 0.5)} = 2`, giving `q_0 = 0.5 + ln(2)/2 = 0.8466`, rounded to **0.85**.
- Resulting shares before the exploration mix: `A0 = 28.7%`, each other arm `14.26%` (uniform = 16.7%).
- After mixing with `eta = 0.20`: `A0 = 26.3%`, each other arm `14.74%`.

`VERIFIED LIMITATION`: exact numerical equivalence with the paper is **not achievable**. Matching its `2.14x`-uniform boost would require `q_0 ≈ 1.01`, outside the `[0,1]` score domain; `q_0 = 1.0` would match it (`2.11x` uniform) but sits exactly on the domain boundary, leaving the EMA no headroom above the initial value. `0.85` is the closest usable point that preserves the shape while leaving headroom. Lowering `tau` to sharpen the softmax was considered and rejected: it would alter the sampling dynamics for the entire run, not only the initial transient.

Decay of the prior: with `alpha = 0.10`, after `n` updates on an arm the residual weight of the initial value is `0.9^n` (≈35% after 10 updates). The initialization is therefore an early-training prior, consistent with the paper's intent, not a persistent bias.

### 7.3 Configuration shape (DEC-003)

```yaml
mab:
  initial_score: 0.50          # default for arms not listed below
  initial_scores:              # explicit per-arm overrides
    A0_simple_low_traffic: 0.85
```

Rationale: an explicit name-keyed mapping is auditable, does not depend on tuple ordering, and does not encode an ordinal axis that §4.2/RAT-008 show does not exist. Validation: every key must be a member of `ARMS`; every value must lie in `[0,1]`; the resulting vector must be finite with exactly `num_arms` entries.

### 7.4 Backward compatibility

Omitting `initial_scores` entirely, or listing all six arms at `0.50`, must reproduce v1.1 behavior exactly (REQ-102, TEST-004). Resume path is untouched (REQ-103): `from_state_dict` (`mab.py:92-120`) restores persisted vectors and never calls the initializer.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-101` | `AC-101` | `mab.py` `__post_init__`, `config.py`, `conf/curriculum/scenario_acl.yaml` | `TEST-001`, `TEST-002` | Planned |
| `REQ-102` | `AC-102` | `mab.py`, `config.py` | `TEST-004` | Planned |
| `REQ-103` | `AC-103` | `mab.py:92-120` (unchanged) | `TEST-005` | Planned |
| `REQ-104` | `AC-104` | `docs/specifications/automatic_curriculum_learning_v1.2_specification.md`, new ADRs | Documentation review (no automated test) | Planned |
| `REQ-105` | `AC-105` | Analysis of existing run logs | `TEST-008` (procedure recorded, not automated) | Planned |

### Acceptance criteria

- `AC-101`: a fresh bandit built from the default v1.2 profile yields `scores = [0.85, 0.50, 0.50, 0.50, 0.50, 0.50]` in `ARMS` order, and `probabilities()` returns `A0 = 0.2630 ± 1e-3` with every other arm at `0.1474 ± 1e-3`.
- `AC-102`: with a uniform mapping, `scores`, `probabilities()` and the post-update state are bit-identical to the v1.1 implementation for the same seed and input sequence.
- `AC-103`: a checkpoint written before the change restores unchanged scores and update count under the new code.
- `AC-104`: specification v1.2 contains RAT-001..RAT-009 verbatim in substance, each with its supporting evidence.
- `AC-105`: the `U` distribution comparison between `origin="replay"` and `origin="new"` is recorded in §11 with the resulting decision.

## 9. Test strategy (to be frozen at approval)

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-001` | Unit | Per-arm initialization vector | Default v1.2 mapping | `[0.85, 0.50, 0.50, 0.50, 0.50, 0.50]` | `REQ-101` |
| `TEST-002` | Unit | Initial sampling distribution | Same, `eta=0.20`, `tau=0.50` | `A0 = 0.2630`, others `0.1474` (tol `1e-3`), sum `= 1` | `REQ-101` |
| `TEST-003` | Unit | Invalid configuration | Unknown arm name; value `>1`; value `<0`; non-mapping type | `ValueError` before learner construction | `REQ-101` |
| `TEST-004` | Regression | v1.1 equivalence | Uniform mapping / omitted field, fixed seed | State and probabilities identical to current behavior | `REQ-102` |
| `TEST-005` | Compatibility | Resume does not re-initialize | `acl_ema_v1` checkpoint with non-default scores | Restored scores preserved exactly | `REQ-103` |
| `TEST-006` | Unit | Exploration floor preserved | Boosted initialization | `p_i >= eta/K` for every arm | `REQ-101`, v1.1 REQ-002 |
| `TEST-007` | Integration | Hydra composition | `conf/curriculum/scenario_acl_scenarionet.yaml` | Resolves with the per-arm mapping; no legacy field accepted | `REQ-101` |
| `TEST-008` | Analysis | Replay contamination measurement | Existing run logs | Recorded distribution comparison and decision | `REQ-105` |

Mandatory-test policy: after approval, `TEST-001..TEST-007` may be strengthened but not removed, weakened, skipped, or aligned to the implementation without a new approval.

### Available validation commands (verified in `AGENTS.md`)

| Purpose | Command |
|---|---|
| Focused tests | `uv run --no-sync python -m pytest tests/test_scenario_acl_mab.py tests/test_scenario_acl_config.py -q` |
| Full suite (primary environment) | `make test` |
| Full suite (provisioned container) | `uv run --no-sync python -m pytest -q` |
| Lint | `make lint` |
| Format check (focused) | `make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/curriculum/scenario_acl/mab.py src/thesis_rl/curriculum/config.py tests/test_scenario_acl_mab.py"` |
| Compose validation | `make config` and `make config-gpu` |
| Smoke | `make smoke` |
| Whitespace | `git diff --check` |

No type-checking target is configured repository-wide; static checking is not claimed for this change.

## 10. Milestones

- [x] **M0** — Analysis: read reference paper (Eq. 8, 10, 11, 12, 17, 18), current implementation, both ACL specifications, ADR-014/016, and the v1.1 ExecPlan. Establish the documentation gap (§4.10) and the Replay-window finding (§4.7).
- [x] **M1** — Write this ExecPlan with the recorded rationales (§6.2).
- [x] **M2** — Obtain approval for `DEC-001`, `DEC-004`, `DEC-005`, `DEC-006`. `DEC-002`/`DEC-003` moot (superseded by `DEC-001` closing as "keep uniform").
- [x] **M3** — Author specification v1.2 (`docs/specifications/automatic_curriculum_learning_v1.2_specification.md`, `APPROVED`) and two ADRs: `ADR-028` (RAT-001..RAT-009 retrospective rationale), `ADR-029` (DEC-006 reward-scale normalization + positive-part TD3/SAC formula). Update `docs/project_index.md`.
- [x] **M4** — Implement `mab.py` (reward-scale EMA, checkpoint schema `acl_ema_v2`), `driver.py` (both the vectorized `commit_event` path and the sequential `collect_catalog_episode` path), `lifecycle.py` (`acl_learning_potential`/`acl_ready_learning_potentials` positive-part), `usefulness.py` (fallback formulas kept consistent with production).
- [x] **M5** — Implement/extend tests in `test_scenario_acl_mab.py`, `test_scenario_acl_usefulness.py`, `test_scenario_acl_vectorized_state.py`; run focused tests, lint, focused format check. All pass (§14).
- [x] **M6** — `TEST-008`/`REQ-105` (`DEC-004`) executed against existing run logs in the analysis session; resolved as no-code-change (`MEAS-001`).
- [ ] **M7** — Run `make test`, `make config`, `make smoke`, `git diff --check` against the three re-run experiments once the user restarts them. *Blocked on the user stopping/restarting TD3/SAC/PPO; not run yet.*
- [x] **M8** — Final reconciliation (§15); `docs/project_index.md` updated; ChatGPT source synchronization status reported to the user.

## 11. Progress and findings log

### 2026-07-27 — Analysis session

Completed: full review of the ACL arm initialization, update, and feedback path against the reference paper.

Findings:

1. `VERIFIED` — Reference paper Eq. 17 is the paper's **main method**, not an ablation; Eq. 18 (uniform) is the ablation. The repository currently implements the paper's ablation configuration without recording that it does so.
2. `VERIFIED` — Eq. 17, once passed through Eq. 8, produces "easiest arm boosted, remainder flat", not a gradient. This removes the obstacle that had blocked a difficulty-informed initialization, because only the easiest arm must be identified (§7.2, RAT-008).
3. `VERIFIED` — Exact numerical transfer of Eq. 17 is impossible in a `[0,1]` score domain (§7.2 limitation).
4. `VERIFIED` — The absence of the inverse-probability correction is mathematically correct for an EMA estimator, not a regression against the paper (RAT-004). An earlier recommendation within this analysis to reintroduce it with a capped `1/p_i` factor was **withdrawn**: the cap would have been an unvalidated invented parameter compensating for a correction that does not apply to this estimator class.
5. `VERIFIED` — Cumulative accumulation with non-negative rank feedback degenerates (RAT-003). The v1.1 move to EMA fixed a real defect that was never documented as such.
6. `VERIFIED` — Replay episodes enter the rank window (§4.7). Ordering is protected by softmax shift invariance; resolution is not. Unquantified; measurable from existing logs.
7. `VERIFIED` — No rationale for any of the above exists in the repository (§4.10).

Decisions needed: `DEC-001..DEC-005`.

Next step: user approval of the gates in §6.1.

### 2026-07-27 — Measurement session (REQ-105 / `TEST-008`)

`DEC-001`, `DEC-002`, `DEC-003`, `DEC-005` approved by the user. `DEC-004` approved as "measure first". The measurement was executed before any production change and produced two results, the second of which supersedes `DEC-001`.

Data source: `scenario_buffer_events.jsonl` and `mab_history.jsonl` from four in-progress/completed runs under `EXP_thesis_RP_thesis_CUR_scenario_acl_scenarionet_REW_scalar_reward` (TD3 seed_0 x2, PPO seed_0, SAC seed_0). Generate episodes are identified by `buffer_action ∈ {inserted, rejected}`, Replay by `{updated, skipped_evicted_before_commit}`. The event order in the JSONL matches the append order into `recent_usefulness` (`driver.py:794` precedes `driver.py:848`). The analysis script replays the stream and reproduces `_normalize_learning_potential` exactly, maintaining both the current mixed window and a counterfactual Generate-only window.

#### `MEAS-001` — Replay contamination is real but functionally benign (`DEC-004` resolved)

Aggregate over 10 430 Generate and 17 539 Replay episodes:

| Quantity | Value |
|---|---|
| Median raw LP, Replay / Generate | `4.07 / 1.54 = 2.64x` |
| Fraction of Replay episodes above the Generate median | `0.896` (0.500 would mean no contamination) |
| Mean `U` fed to the MAB — mixed window (current) | `0.3792` |
| Mean `U` fed to the MAB — Generate-only window | `0.5056` |
| Dispersion of `U` — mixed window | `0.3031` |
| Dispersion of `U` — Generate-only window | `0.2972` |

Contamination is confirmed and large in *level*: Replay episodes have 2.6x the median raw LP of Generate episodes, and the mean `U` reaching the bandit is depressed by `-0.13`.

However, the hypothesis recorded in `RAT-009` — that this compresses discriminative resolution — is **refuted**. Dispersion is statistically unchanged (`0.3031` vs `0.2972`; the mixed window is in fact 2.0% *wider*), because the two LP distributions overlap substantially rather than separating. The effect is therefore a common downward shift applied to every arm, and the softmax is invariant to a constant offset, so sampling probabilities are unaffected.

**Resolution: no code change.** `driver.py:794` stays unconditional. `RAT-009` is corrected accordingly: the phenomenon exists, is quantified, and is benign. Retaining the mixed window additionally preserves the larger, fresher reference sample (a Generate-only window would refill at 40% of the rate and therefore span ~2.5x more training time).

#### `MEAS-002` — `A0` is empirically the *lowest*-learning-potential arm, contradicting `DEC-001`

Per-arm EMA score trajectories from `mab_history.jsonl`, all four runs:

| Run | First recorded chunk (25 011 steps) | Last recorded chunk |
|---|---|---|
| TD3 seed_0 (225902) | A0 `0.402` — lowest of six | A0 `0.163` — lowest band |
| TD3 seed_0 (055617) | A0 `0.379` — second lowest | A0 `0.156` |
| PPO seed_0 (024112) | A0 `0.620` — lowest of six | A0 `0.326` |
| SAC seed_0 (225902) | A0 `0.652` — near lowest | A0 `0.169` — lowest of six |

`A0_simple_low_traffic` is the lowest or near-lowest scoring arm from the earliest recorded measurement onward, consistently across three algorithms. Corroborating evidence from the replay buffer of TD3 seed_0 (055617), which retains the 1 000 most useful scenarios: `A4_vru` 333, `A3_complex_junction` 264, `A5_critical_mixed` 223, `A2_junction` 82, `A1_traffic` 80, **`A0_simple_low_traffic` 18**.

The premise behind `DEC-001`/`DEC-002` was that an untrained policy extracts more learning potential from the structurally simplest arm, justifying an initial boost. The measured behavior is the opposite: simple low-traffic scenarios yield the least learning signal, while VRU and complex-junction scenarios yield the most. Raising `q_{A0}` to `0.85` would direct roughly ten additional percentage points of the early Generate budget to the empirically least informative arm.

`LIMITATION`: the earliest available chunk is at 25 011 steps with ~200 MAB updates (~33 per arm), by which point the initial value has largely decayed (`0.9^33 ≈ 0.03`). There is therefore **no direct observation of steps 0–25 000**, which is precisely the interval the initialization governs. The evidence is strongly directional and consistent across algorithms and seeds, but it does not conclusively describe the initial transient.

**Recommendation: revise `DEC-001` to option A (retain uniform initialization).** This is no longer an unexplained default: it becomes a choice supported by measured per-arm learning potential, which closes the documentation gap that motivated this plan without changing behavior. `REQ-101`, `REQ-102`, `REQ-103` and their tests would then be withdrawn; `REQ-104` (recorded rationales) and `REQ-105` (this measurement) remain and become the whole deliverable.

#### `FIND-001` — Separate defect candidate: arm `A4_vru` can freeze and self-lock

In TD3 seed_0 (055617) the `A4_vru` score is **bit-identical** (`0.715461969165923`) from MAB update 1 420 through update 3 118 — 1 698 consecutive updates during which `A4` received none, while holding the *highest* score of all six arms and therefore the highest Generate selection probability (~34% at `tau = 0.5`).

This is self-locking: a high score attracts selections, the selections never commit an update, and the score can only be corrected by an update that never arrives. The remaining three runs analyzed do not show the freeze, so it is run-specific rather than systematic. A plausible mechanism, not yet verified, is the interaction between the forced `A4` Waymo-only source (`curriculum/scenario_acl/arms.py:40-43`) and the runtime data-abort/quarantine path (ADR-024), combined with the known unresolved catalog gap recorded for `live_training_reliability_v1` in `docs/project_index.md`.

Out of scope for this plan. Requires its own bugfix ExecPlan and a regression test. Reported to the user 2026-07-27.

Next step: user re-decision on `DEC-001` in light of `MEAS-002`.

### 2026-07-27 — Learning-potential attribution audit

Triggered by a discussion of whether the ZPD (zone-of-proximal-development) filtering property of the PPO formula `mean(max(A, 0))` can be extended to the off-policy formulas, which use `mean(|delta|)`. A prerequisite question was raised first: *on which transitions* is the off-policy LP actually computed. An earlier reading of this session suspected it came from a replay-buffer batch and was therefore scenario-independent. **That suspicion is refuted.** Full trace below.

#### `FIND-002` — The production LP path is per-episode and on-policy for every backend (`VERIFIED`)

Traced end to end:

| Stage | Location | Behavior |
|---|---|---|
| Per-transition residual | `agent/planners/core/lifecycle.py:124-146` | Calls `backend.collection_learning_potential_batch(...)` on the transitions just collected, then accumulates each residual under `_acl_collection_residuals[(acl_slot_id, acl_episode_id)]` taken from `infos`. |
| TD3 residual | `agent/planners/algorithms/td3_sb3.py:444-480` | `r + (1-done)*gamma*min Q_target(s', pi_target(s')+clipped noise) - min Q(s, a)` on the **collected** batch. Docstring: *"Compute TD3 residuals on collected transitions, not replay samples."* |
| SAC residual | `agent/planners/algorithms/sac_sb3.py:429-468` | Same, with the entropy-aware target `Q_target - alpha * log pi(a'|s')`. |
| PPO | `agent/planners/core/backend_base.py:43-60` returns `None`; PPO instead stores per-episode values via `pop_acl_episode_learning_potential` (`ppo_sb3.py:567`, `ppo.py:695`), filled from `_acl_episode_advantages` (`ppo_sb3.py:461`, `ppo.py:561`). | Per-episode GAE, positive part. |
| Per-episode aggregation | `lifecycle.py:148-157` (`acl_learning_potential`) and `lifecycle.py:159-172` (`acl_ready_learning_potentials`) | `float(np.abs(np.asarray(values)).mean())` — **this is where the absolute value is applied in production for TD3/SAC**. |
| Transport | `agent/agent.py:1385-1420` | Written into the worker payload as `learning_potential` / `ready_learning_potentials`. |
| Consumption | `curriculum/scenario_acl/driver.py:759-762` | `learning_potentials[(slot, episode_id)] = payload["learning_potential"]`, then rank-normalized at `driver.py:778-779`. |

Consequences:

1. The per-arm differentiation observed in `MEAS-002` is genuine: LP is attributed to the scenario that produced the transitions, using the actions the current policy actually took. There is no scenario-independence defect.
2. `backend_base.py:53-56` documents the design intent explicitly — a backend that cannot produce a collection-time snapshot returns `None` and the driver *must not* substitute a replay aggregate.
3. The replay-batch LP at `td3_sb3.py:556-590` and `sac_sb3.py:520-540` is **not** the per-scenario signal. It reaches `lifecycle.py:187-189` (`learning_potential_values`), is averaged into the chunk summary (`agent.py:883`, `agent.py:1636`), and is used by `driver.py:1988-2004` only as a chunk-level fallback when *no* episode produced per-episode feedback. Monitoring and last-resort, not attribution.
4. **Correction with design impact:** the `np.abs` that governs production TD3/SAC LP is at `lifecycle.py:153` and `lifecycle.py:163`, *not* at `usefulness.py:92,97`. A change of `abs` to positive-part restricted to `usefulness.py` would have no effect on any ACL run. Any future ZPD alignment must target `lifecycle.py:153,163` (or move the sign decision into the backend residual functions).
5. Because `acl_learning_potential` reduces the episode to a mean of absolute values before the driver sees it, the sign distribution of the residuals is destroyed at the aggregation site and is **not recoverable from any existing log**. A retrospective estimate of the impact of `max(delta,0)` versus `|delta|` on the completed runs is impossible; only a new instrumented run can provide it.

#### `FIND-003` — `AclEpisodeAccumulator` is vestigial (cleanup candidate, not executed)

`curriculum/scenario_acl/vectorized.py:57-105` defines a per-episode accumulator with `rewards`, `td_residuals`, `sac_residuals` and `ppo_*` fields, an `add_transition(**values)` ingest, and a `learning_potential(algorithm)` dispatcher. The driver instantiates one per slot (`driver.py:676`), persists it in `AclVectorState` (`vectorized.py:256`, `276`, `314`), and discards it in `clear_completed`. **`add_transition` is never called from production code** — a repository-wide search finds it only in its own definition and in `tests/test_scenario_acl_vectorized_state.py:220-230`. Every accumulator therefore serializes as empty, and `learning_potential(algorithm)` is unreachable in production. The live path is `FIND-002` above.

Not removed in this plan, for three reasons:

- `AclEpisodeAccumulator.from_dict` is `cls(**dict(payload))` (`vectorized.py:105`) and `AclVectorState.from_dict` rejects any version mismatch (`vectorized.py:295-297`). Dropping fields would raise `TypeError` on resume of a checkpoint written before the change, which would break crash-recovery of the three runs currently in progress.
- Removal requires deleting `tests/test_scenario_acl_vectorized_state.py:218-231`, which `AGENTS.md` classifies as mandatory tests that may not be removed without explicit approval.
- It is unrelated to the objective of this plan.

Proposed as a separate cleanup, to be scheduled after the in-progress runs complete: delete the class and its `AclVectorState` field, make `AclVectorState.from_dict` ignore a legacy `accumulators` key so pre-existing checkpoints still load, and remove the three tests. `usefulness.compute_td3_td_residuals` / `compute_sac_td_residuals` and the `compute_learning_potential` branches at `usefulness.py:174-201` are *not* part of this: they are unreached in production only because `train_summary["learning_potential"]` short-circuits at `usefulness.py:159-163`, and they remain a documented fallback with test coverage.

#### `MEAS-003` — The bandit does drift toward the harder arms, but weakly and without reordering them

Question: does the curriculum progressively move the agent toward harder scenarios as it learns? Measured by recomputing the production selection function (`softmax(q/0.50)` mixed with `eta = 0.20` over `K = 6`) from the logged EMA scores in `mab_history.jsonl`, then comparing the first recorded chunk (25 011 steps) with the last. 15 runs with at least 6 logged chunks, spanning 150 066 to 475 209 steps (script: scratchpad `arm_drift.py`).

| Quantity | Result |
|---|---|
| Runs where `p(A3)+p(A4)+p(A5)` increases | 14 of 15 |
| Typical drift of that mass | `+0.06` (range `-0.058` to `+0.103`) |
| Mass on `A3+A4+A5` at the *first* recorded chunk | `0.50`–`0.61` (uniform would be `0.50`) |
| Spearman rank correlation, first vs last arm ordering | median `+0.66`, range `-0.31` to `+0.94` |
| Observed probability range across all arms and runs | `0.106`–`0.336` (uniform `0.167`; the `eta/K` floor is `0.033`) |

Three conclusions:

1. **The direction is correct.** `A0_simple_low_traffic` loses mass in 11 of 15 runs and `A4_vru` gains it in 13 of 15, consistent with easy scenarios being progressively exhausted of learning signal.
2. **It is not an easy-to-hard progression.** The bandit is already tilted toward the hard arms at the earliest measurement available; the later drift extends a pre-existing tilt rather than traversing a curriculum. This is consistent with `MEAS-002`.
3. **The ordering is close to static.** With a median rank correlation of `+0.66` and a probability range of roughly `0.11`–`0.34` against an available floor of `0.033`, the mechanism is behaving as a mildly non-uniform, nearly time-invariant sampler rather than as a competence-tracking curriculum.

The measurement cannot by itself distinguish "easy arms were mastered, so their LP fell" from "residual magnitude grows with scenario complexity and the bandit is tracking complexity, not learnability" (the variance-inflation failure mode). One partial discriminator: PPO uses `mean(max(A, 0))`, which does filter hopeless outcomes, and PPO shows the same drift (`+0.022` to `+0.095`), so the drift is not purely an artifact of the off-policy `|delta|`. Variance inflation still applies to both.

#### `FIND-004` — Root cause of the immediate-then-frozen arm ranking: LP is not scale-invariant

Question raised by the user: the bandit differentiates the arms almost immediately and then stops moving. Where is the defect — in the LP formulas, or elsewhere?

Measured from `logs/events.jsonl` (`scenario_acl_episode_ended`, whose `live_event_context` carries `arm`, raw LP as `U`, rank-normalized value as `U_norm`), Generate episodes only, split into quintiles of episode order. Scripts: scratchpad `lp_diagnosis.py`, `lp_scale_test.py`.

**Observation 1 — raw LP collapses globally while competence does not improve.** TD3 seed_0 (225902), median raw LP per arm from first to last quintile: `A0 1.434 -> 0.622` (-57%), `A1 -63%`, `A2 -65%`, `A3 5.046 -> 0.684` (-86%), `A4 6.056 -> 1.251` (-79%), `A5 -81%`. Over the same interval mean `route_completion` moves `A0 +6.8pp`, `A1 +5.7pp`, `A2 +8.8pp`, `A3 -8.7pp`, `A4 -4.4pp`, `A5 -0.7pp`. An 80% drop in the learning signal accompanied by flat-to-negative competence means the LP is tracking critic-residual decay, which is a global property of the optimizer, not per-arm learnability.

**Observation 2 — the cross-arm ordering of LP is the cross-arm ordering of reward magnitude.** Per-arm medians, four runs across three algorithms:

| Run | Spearman(median LP per arm, median \|reward\| per arm) | Relative dispersion of LP across arms | Same, after dividing LP by \|reward\| |
|---|---|---|---|
| td3 seed_0 (225902) | `+0.771` | `0.994` | `0.298` |
| sac seed_0 (225902) | `+0.886` | `0.533` | `0.361` |
| ppo seed_0 (024112) | `+0.829` | `0.296` | `0.216` |
| td3 seed_0 (055617) | `+0.943` | `1.268` | `0.352` |

Dividing the LP by the episode reward magnitude removes between 27% and 72% of the between-arm dispersion. Example (td3 055617): raw LP `A0 0.589` vs `A4 5.807` — a 9.9x gap — becomes `0.0428` vs `0.0985`, a 2.3x gap; median `|reward|` for the same arms is `13.65` vs `57.92`. The episode reward is dominated by the scalarized rulebook term (`scalar_rule_reward` is two orders of magnitude larger than `env_reward` in the logged metrics), so arms that trigger heavier rule penalties mechanically produce larger residuals.

**Causal chain:**

1. `delta = r + gamma*Q' - Q` carries the units of the return, so `mean(|delta|)` is proportional to the reward scale of the arm.
2. Arms differ in reward scale by 2x-4x, driven by rulebook violation propensity, which is a **static property of the scenario class**.
3. `_normalize_learning_potential` (`driver.py:316-328`) ranks each episode against a window shared by **all** arms, converting that static scale gap into a stable per-arm percentile `E[U_i]`.
4. The EMA with `alpha = 0.10` has an effective horizon of ~10 updates per arm and therefore reaches `E[U_i]` after ~60 total updates, i.e. roughly 7 000 steps of a 350 000-step run. Hence "immediate".
5. `E[U_i]` does not move afterwards because the reward scale does not move. Hence "then frozen".
6. Rank normalization is purely relative, so the one genuinely temporal signal present — the 80% global LP decay — is cancelled exactly and never reaches the bandit.

**Answer to the question as posed: the defect is not in the sign of the formulas.** Replacing `|delta|` with `max(delta, 0)` changes neither the units nor the scale proportionality, so it would leave both symptoms intact. It is also not an implementation bug: `_normalize_learning_potential`, `ScenarioArmBandit.update` and the attribution path of `FIND-002` all behave as specified. It is a **design defect in the feedback signal**: a scale-carrying quantity is being compared across classes with different scales.

**Consequence worth recording:** in this run the bandit increases the probability mass on `A3`, `A4`, `A5` (`MEAS-003`) precisely while `route_completion` on those arms declines. The curriculum is directing sampling toward the arms on which the agent is regressing.

Candidate remedies, none yet approved, listed with their honest cost:

| Option | Change | Effect | Limitation |
|---|---|---|---|
| `R-A` | Per-arm rank window: rank each episode against the recent window **of its own arm** | Removes the scale confound by construction; `U` becomes a within-arm trend detector | At steady state every arm returns `E[U] ~ 0.5`, so the bandit degenerates to near-uniform whenever no arm has a moving LP — defensible as a neutral default, but a much weaker curriculum |
| `R-B` | Scale-normalize the residual, e.g. divide by a running per-arm estimate of `\|r\|` or of `\|delta\|` | Keeps the shared window and the cross-arm comparison | Requires choosing the normalizer; measurement above shows it removes only 27-72% of the dispersion, so the confound is reduced, not eliminated |
| `R-C` | Replace LP with learning progress: slope of return (or `route_completion`) per arm over repeated visits | Directly measures the intended quantity; immune to both scale and variance inflation | Lagged, needs several visits per arm, and is an architectural change rather than a formula change |
| `R-D` | Reduce the reward-scale spread upstream (rulebook scalarization) | Attacks the root cause | Out of ACL scope, changes the learning problem itself, invalidates comparability with completed runs |

This finding supersedes the framing of the LP discussion recorded earlier in this session: the `max` versus `|.|` question remains valid on its own merits (hopelessness filtering) but is **not** the cause of the observed behavior and must not be presented as its fix.

### 2026-07-27 — `DEC-006`: consolidated remedy design (R-B + C3)

User selected `R-B` (per-arm reward-scale normalization) over `R-A`/`R-C`, combined with `C3` (`max(delta, 0)` instead of `|delta|` for TD3/SAC), on the grounds that both require re-running the in-progress experiments anyway and address two distinct, complementary failure modes: `R-B` removes the cross-arm scale confound (`FIND-004`), `max(delta,0)` restores the within-arm ZPD hopelessness filter that PPO already has structurally (matching `RAT-008`'s framing, not paper fidelity — Peng's Eq. 10 normalizes signed episode reward with a cumulative accumulator, a different construction already rejected as non-transferable in `RAT-001`).

**Approved design:**

1. **Scale estimator**: one scalar per arm, EMA of `|episode reward|` (the same field already used in `FIND-004`'s diagnostic, `metrics.get("reward", 0.0)`), updated only on Generate episodes (mirrors the bandit's own update gating at `driver.py:798`).
2. **Decay rate**: reuse `alpha = 0.10` (the existing bandit EMA rate). No new hyperparameter — matches the parsimony argument already used in `RAT-004`/`RAT-007` to reject unjustified new knobs.
3. **Initialization**: `1.0` for every arm, identical across arms. The literal value is not load-bearing; what matters is that it is *shared*, so normalization is a no-op during warm-up (~10-60 updates per arm) and the correction phases in only as per-arm estimates diverge from the common prior — degrading gracefully to `R-0` exactly when there is insufficient data to trust a per-arm estimate. A numerical safety clamp (`max(estimate, 1e-3)`) is a guard, not a tunable choice.
4. **Normalization point**: after per-episode aggregation (after the mean over timesteps and after the `max(.,0)`/`|.|` gate), not per-transition. `LP_scaled = LP_raw / scale_estimate[arm]`, computed with the **pre-update** estimate (the estimate is updated with this episode's reward magnitude after normalizing, to avoid the episode leaking into its own denominator). Feeds into the existing `_normalize_learning_potential` unchanged.
5. **Formula change**: `compute_td3_learning_potential` and `compute_sac_learning_potential` semantics move from `mean(|delta|)` to `mean(max(delta, 0))`; production code path is `lifecycle.py:153` and `lifecycle.py:163` (`acl_learning_potential`, `acl_ready_learning_potentials`), per `FIND-002` — not `usefulness.py`, which is unreached in production.

**Declared limitation (irreducible, to be written into spec v1.2 and the thesis):** variance inflation. At critic convergence `E[delta] = 0`, so `mean(max(delta,0)) ~ sigma(delta)`: a noisy-but-unlearnable scenario still scores nonzero LP. Neither `R-B` nor the sign change resolves this; it affects PPO too and is shared across the whole prediction-error family of LP measures. The principled fix is a learning-progress signal (slope of return per arm over repeated visits, same family as `R-C`), explicitly out of scope here: no ablation budget, requires an architectural change, and would need its own ExecPlan.

**Status:** approved design, **not yet implemented**. Implementation requires stopping and re-running the three in-progress experiments (TD3, SAC, PPO), which is a resource-affecting decision the user must confirm explicitly before any run is touched.

### 2026-07-27 — `DEC-006` implementation

User confirmed ("D'accordo comunque, procedi con l'implementazione") and implementation proceeded per the approved design in the previous entry. Delivered:

- `mab.py`: `reward_scale` array, `reward_scale_estimate`, `normalize_learning_potential_by_reward_scale`, `update_reward_scale`; checkpoint schema `acl_ema_v1` -> `acl_ema_v2` with explicit rejection of the old schema.
- `driver.py`: both `commit_event` (vectorized) and `collect_catalog_episode` (sequential) normalize LP by the arm's reward-scale estimate before it enters the shared rank window, for both Generate and Replay episodes; `update_reward_scale` is called only on committed Generate episodes, mirroring the existing `bandit.update` gate. The sequential path needed one implementation detail not covered by the original design write-up: Replay iterations there carry `spec.arm_index = -1` (unlike the vectorized selector, which always resolves a real index), so the arm is resolved by name (`SCENARIO_ARM_NAMES.index(spec.arm_name)`) for scale lookup only, matching what the vectorized selector already does structurally.
- `lifecycle.py`: `acl_learning_potential`/`acl_ready_learning_potentials` use `np.maximum(residuals, 0.0).mean()` instead of `np.abs(residuals).mean()`, at the actual production site identified by `FIND-002`.
- `usefulness.py`: `compute_td3_learning_potential`/`compute_sac_learning_potential` updated identically, keeping the documented fallback consistent with production even though `FIND-002` established it is not reached in the current pipeline.
- Tests: 4 new tests in `test_scenario_acl_mab.py` (reward-scale init/EMA/floor, checkpoint round-trip, legacy-schema rejection); `test_scenario_acl_usefulness.py` updated for positive-part semantics including an all-negative case; one new test in `test_scenario_acl_vectorized_state.py` locking in positive-part behavior at the real production call site, not only the fallback.
- Validation: 102 focused tests pass (`acl`/`lifecycle`/`curriculum`, run in an isolated `docker compose run --rm dev` container separate from the three in-progress training containers); `ruff check`/`ruff format --check` clean on all touched files after one auto-format pass.
- Documentation: `docs/specifications/automatic_curriculum_learning_v1.2_specification.md` (APPROVED), `ADR-028` (RAT-001..009), `ADR-029` (DEC-006 design), `docs/project_index.md`.

### 2026-07-27 — `DEC-007`: extend `REQ-007` to replay-buffer retention priority

Same-day follow-up. User asked directly whether the buffer-retention confound flagged as `LIM-001` in the completed `DEC-006` implementation should also be fixed. Investigation: `buffer.py:51`, `worst = min(self._records, key=lambda current: current.usefulness)`, and `record.usefulness` is populated from the raw `learning_potential` argument passed to `_build_record_from_catalog_entry`/`_update_replay_record` in `driver.py` -- which, before this entry, was still the raw `lp`/`episode_usefulness`, not the reward-scale-normalized value already computed for the rank window in `DEC-006`.

Reasoning for extending rather than deferring: `exploit_probability=0.60`, so 60% of post-warm-up training time is Replay sampled from this buffer. `MEAS-002`'s corroborating evidence (buffer composition skewed `A4_vru=333` vs `A0_simple_low_traffic=18` of 1000 slots) shows the same `FIND-004` confound already biases buffer composition. Leaving it unfixed while fixing only the Generate-side MAB feedback would mean the confound continues to bias 60% of training time through a second channel, undermining much of `DEC-006`'s intended effect. Since the three experiments already require a restart for `DEC-006`, fixing this now avoids a second future restart.

**Approved and implemented:** `driver.py`'s `commit_event` (vectorized) and `collect_catalog_episode` (sequential) now pass `lp_scaled`/`scaled_episode_usefulness` — the same value already computed for `REQ-007`'s rank-window normalization — as the `learning_potential` argument to `_build_record_from_catalog_entry` and `_update_replay_record`, instead of the raw value. Diagnostic fields (`live_event_context["U"]`, `buffer_events`, `log_event`) are unaffected and continue to report the raw LP. The sequential path required declaring `scaled_episode_usefulness: float | None = None` at the top of `collect_catalog_episode`, since the buffer-record block runs outside the `if episode_usefulness is not None:` scope where the variable was otherwise only defined.

Validation: full focused suite re-run after the change (102 passed, no regressions), `ruff check`/`ruff format --check` clean on `driver.py`. No dedicated closure-level regression test was added for the call-site substitution itself (`commit_event`/`collect_catalog_episode` are private closures inside the large training-loop functions with no existing isolated-testing harness); recorded as `LIM-004` with an explicit follow-up test description, per the requirement to record the reason and remaining risk for checks not run.

`docs/specifications/automatic_curriculum_learning_v1.2_specification.md`, `ADR-029` (amended in place, same ADR since it is a same-day extension of the same decision), and `docs/project_index.md` updated accordingly. No new ADR was created for `DEC-007`: it extends `ADR-029`'s `REQ-007` rather than introducing a new mechanism.

## 12. Deviations

| ID | Original contract | Proposed change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-001` | ACL v1.1 §7: "Initialization: `q_i=0.50` for every arm"; §9 table row `mab.initial_score`, `Frozen: YES` | Per-arm initialization mapping, `A0 = 0.85`, others `0.50` | Reproduce the effect of reference paper Eq. 17 (the paper's method) under this repository's selection function, using the only ordinal fact the taxonomy establishes (RAT-008, §7.2) | **Pending — DEC-001** | Spec v1.2, new ADR, `TEST-001..TEST-004`, `tests/test_scenario_acl_mab.py`, `tests/test_scenario_acl_config.py` |

No other deviation is proposed. All items in §6.2 document existing approved behavior and introduce no change.

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `docs/implementation/automatic_curriculum_learning_v1.2_exec_plan.md` | Created | This plan |
| `docs/specifications/automatic_curriculum_learning_v1.2_specification.md` | Planned creation | REQ-101..REQ-105 and the recorded rationales |
| `docs/decisions/ADR-0XX-acl-arm-score-initialization.md` | Planned creation | DEC-001/002/003 |
| `docs/decisions/ADR-0XX-acl-ema-arm-update-recorded-rationale.md` | Planned creation | RAT-001..RAT-007 (retrospective) |
| `docs/project_index.md` | Planned modification | Register spec v1.2 and the new ADRs |
| `src/thesis_rl/curriculum/config.py` | Planned modification | `initial_scores` field and validation |
| `src/thesis_rl/curriculum/scenario_acl/mab.py` | Planned modification | Build the initial vector from the mapping |
| `src/thesis_rl/scenarios/arms.py` | Planned modification | Documented constant for the structurally-easiest arm |
| `conf/curriculum/scenario_acl.yaml` | Planned modification | Default per-arm mapping |
| `tests/test_scenario_acl_mab.py` | Planned modification | `TEST-001`, `TEST-002`, `TEST-004`, `TEST-005`, `TEST-006` |
| `tests/test_scenario_acl_config.py` | Planned modification | `TEST-003`, `TEST-007` |

## 14. Validation results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `TEST-008` measurement script over four run logs | `PASS` | 2026-07-27 | Executed against 27 969 recorded episodes (10 430 Generate, 17 539 Replay) from TD3 x2 / PPO / SAC. Results in §11 `MEAS-001` and `MEAS-002`. Script is analysis-only and reproduces `_normalize_learning_potential` exactly; it modifies nothing in the repository. |
| `arm_drift.py` measurement script over 15 run logs (`MEAS-003`) | `PASS` | 2026-07-27 | Analysis-only, no repository changes. |
| `lp_diagnosis.py` / `lp_scale_test.py` measurement scripts (`FIND-004`) | `PASS` | 2026-07-27 | Analysis-only, no repository changes. |
| `git diff --check` | `PASS` | 2026-07-27 | No whitespace errors. |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/ -k "acl or lifecycle or curriculum"` | `PASS` | 2026-07-27 | 102 passed, 923 deselected. Includes 6 new/modified tests: reward-scale EMA init/update/floor, checkpoint round-trip, legacy-schema rejection (`test_scenario_acl_mab.py`); positive-part TD3/SAC formula incl. all-negative case (`test_scenario_acl_usefulness.py`); production-site positive-part regression at `_DelegatingLifecycle.acl_learning_potential` (`test_scenario_acl_vectorized_state.py`). Run in an isolated `docker compose run --rm dev` container, separate from the three in-progress training containers. |
| `ruff check` on `mab.py`, `driver.py`, `usefulness.py`, `lifecycle.py`, and the three modified test files | `PASS` | 2026-07-27 | Focused scope per `AGENTS.md` (`PYTHON_QUALITY_PATHS` narrowed to touched files). |
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/ -k "acl or lifecycle or curriculum"` (re-run after `DEC-007`) | `PASS` | 2026-07-27 | 102 passed, 923 deselected, no regressions from routing `LP_scaled` into `_build_record_from_catalog_entry`/`_update_replay_record`. |
| `ruff check` / `ruff format --check` on `driver.py` (re-run after `DEC-007`) | `PASS` | 2026-07-27 | No changes needed beyond the earlier formatting pass. |
| `ruff format --check` then `ruff format` on `mab.py`, `driver.py` (focused scope) | `PASS` after formatting | 2026-07-27 | Two files needed reformatting (line length in new methods); applied and re-verified; tests re-run and still pass. Repository-wide format baseline is not clean and was not touched, per `AGENTS.md`. |
| Repository-wide `make test`, `make config`, `make smoke` | `NOT_RUN` | 2026-07-27 | Deferred until the user stops and restarts the three in-progress experiments (TD3, SAC, PPO seed_0), since `acl_ema_v1` checkpoints from those runs are rejected by the new `acl_ema_v2` schema by design (`REQ-005`). Exact follow-up command: `make test`, `make config`, `make config-gpu`, `make smoke` after restart. |

## 15. Final reconciliation

**Status: implementation complete; three experiments pending user restart.**

Requirements and acceptance criteria (`automatic_curriculum_learning_v1.2_specification.md` §12) reconciled against code:

| Requirement | Implementation location | Test |
|---|---|---|
| REQ-001..REQ-006 (unchanged from v1.1) | `mab.py`, `driver.py` (pre-existing) | pre-existing suite, all passing |
| REQ-007 (reward-scale normalization, MAB feedback + buffer priority) | `mab.py: reward_scale_estimate/normalize_learning_potential_by_reward_scale/update_reward_scale`; `driver.py: commit_event` (vectorized) and `collect_catalog_episode` (sequential), both the rank-window append and the `_build_record_from_catalog_entry`/`_update_replay_record` calls | `test_scenario_acl_mab.py` (4 new tests); buffer call-site wiring verified by code reading only (`LIM-004`, no dedicated closure-level test) |
| REQ-008 (positive-part TD3/SAC LP) | `lifecycle.py: acl_learning_potential/acl_ready_learning_potentials`; `usefulness.py: compute_td3_learning_potential/compute_sac_learning_potential` | `test_scenario_acl_vectorized_state.py`, `test_scenario_acl_usefulness.py` |

Approved decisions: `DEC-001` (closed, uniform init), `DEC-004` (closed, no change), `DEC-005` (superseded by `DEC-006`'s actual schema bump), `DEC-006` (approved and implemented), `DEC-007` (same-day follow-up: extend `REQ-007` to replay-buffer retention priority, approved and implemented). No deviation from the approved design occurred during implementation; the sequential-path arm-index resolution for Replay episodes (`spec.arm_index == -1` requiring a name lookup, absent from the original `DEC-006` write-up) was a necessary implementation detail to keep the two collection paths consistent, not a change of design intent.

**Unresolved / deferred:**

- **M7 blocked**: repository-wide `make test`/`make config`/`make smoke` deferred until the user restarts the three experiments, since old checkpoints cannot resume under the new schema by design.
- **`LIM-002`**: variance inflation of prediction-error-based LP is irreducible within this measure family; a learning-progress signal is the principled fix, out of scope (no ablation budget, architectural change).
- **`LIM-003`**: whether `DEC-006`/`DEC-007` change the near-static arm-ordering behavior observed in `MEAS-003` is an open empirical question for the re-run experiments, not established by this plan.
- **`LIM-004`**: the `DEC-007` buffer call-site wiring (`lp_scaled`/`scaled_episode_usefulness` substituted for the raw value at the two record-construction sites) has no dedicated closure-level regression test, only code-reading verification plus the still-passing existing suite. `commit_event`/`collect_catalog_episode` are private closures inside the large vectorized/sequential training-loop functions with no isolated-testing harness. Follow-up if a harness is built: assert that, given two arms with equal raw LP but different reward-scale estimates, `buffer.insert` prefers the higher `LP_scaled`, not the higher raw LP.
- **`FIND-001`** (A4 self-lock) and the uncalibrated arm-classification thresholds (`src/thesis_rl/scenarios/arms.py:19-26`): delegated to background tasks (`task_0d02fdc0`, `task_4d34cd32`), independent of this plan.
- The PLR-family attribution supporting `RAT-001` is `INFERRED` and must be verified against the primary source before it is used in the thesis.
- `FIND-003` (`AclEpisodeAccumulator` dead code): cleanup deferred until after the re-run experiments complete, to avoid breaking crash-recovery resume during the transition.
