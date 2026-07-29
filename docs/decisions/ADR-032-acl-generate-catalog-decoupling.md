# ADR-032: Decoupling Generate Eligibility From Scenario-Buffer Membership, Arm Coverage Cycles, And Buffer Selectivity

- Status: APPROVED
- Date: 2026-07-29
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-29
- Supersedes: none; narrows the operative scope of `ADR-028` (see "Consequences")
- Affected specifications: `docs/specifications/automatic_curriculum_learning_v1.3_specification.md` (`ACL-SN-EMA-001`, REQ-002, REQ-003, REQ-004, REQ-005, REQ-009, §9)
- Affected ExecPlan: `docs/implementation/automatic_curriculum_learning_v1.3_exec_plan.md` (`ACL-SN-CAT-003`, `DEC-008`, `DEC-009`, `DEC-010`, `DEC-011`)
- Related ADRs: `ADR-014` (LP-only usefulness, mutation prohibited), `ADR-016` (deterministic vectorized execution), `ADR-024` (runtime scenario data-abort), `ADR-028` (Generate-arm eligibility renormalization), `ADR-029` (reward-scale normalization)

## Context

### The rule under review

`ACL v1.1`/`v1.2` §3 define Generate as "selecting a **fresh** frozen catalog record". The implementation
(`src/thesis_rl/curriculum/scenario_acl/selection.py`) realizes "fresh" as **"not currently held by the scenario
buffer"**:

```python
effective_excluded = set(excluded_scenario_uids)
effective_excluded.update(str(record.scenario_id) for record in buffer.records())
```

`excluded_scenario_uids` carries the in-flight UIDs required for vectorized batch uniqueness (`ADR-016`,
`DEC-VEC`) plus run-local quarantine (`ADR-024`). The buffer-membership term is an addition on top of those.

### Where the rule comes from

Neither reference method has such a rule. Abouelazm et al. (2025) §III-E.1: *"the random generator produces new
scenarios using the autoregressive process... Each sampled scenario is evaluated based on its learning potential
and is only added to the buffer if its learning potential exceeds the current minimum value in Λ."* There is no
exclusion of buffer members from the exploration draw, in that paper or in the PLR/ACCEL family it builds on,
because the level space `Θ` is combinatorial/continuous: the probability of redrawing a level already in `Λ` is
effectively zero. The paper's phrase *"randomly sampling unseen scenarios"* describes a **property of the space**,
not an enforced constraint.

This project finitized that space into a frozen 2,000-record training catalog. In doing so the descriptive
adjective was translated into an executed constraint. That translation, not the frozen dataset, is the defect.

### Verified magnitudes

Measured from `data/scenarionet/frozen/scenario_selection_index.json` on 2026-07-29 (`VERIFIED`):

| Split | Arm | Source | Records |
|---|---|---|---:|
| train | `A0_simple_low_traffic` | pg | 333 |
| train | `A1_traffic` | pg | 333 |
| train | `A2_junction` | pg | 334 |
| train | `A3_complex_junction` | waymo | 334 |
| train | `A4_vru` | waymo | 333 |
| train | `A5_critical_mixed` | waymo | 333 |

Training catalog: **2,000 records**, six single-source arms of ~333. Configured `buffer_capacity`: **1,000**
(`conf/curriculum/scenario_acl.yaml`), i.e. 50% of the catalog.

### Two independent consequences of the rule

**(1) Arm absorption — `FIND-001`, already observed.** Because `333 < 1000`, an entire arm's pool can be absorbed
into the buffer, after which the arm has no Generate-eligible record. This was found live in
`td3_sb3/seed_0/20260726_055617`, where `A4_vru`'s EMA score stayed bit-identical from MAB update 1420 to 3118
while holding the highest Generate probability (~0.336); three other completed runs sat at 332, 224 and 207 of the
same 333 records. `ADR-028` corrected the acute symptom by removing ineligible arms from the softmax, so a starved
arm can no longer freeze while holding probability mass. It did not remove the underlying condition: an absorbed
arm still leaves the curriculum and stops receiving EMA updates, and the feedback is inverted — the higher an
arm's learning potential, the less likely its records are evicted, so **the most informative arms are the first to
be removed from exploration**.

**(2) Systematic downward bias of the MAB feedback — `FIND-005`, not previously recorded.** The buffer retains an
arm's high-LP records by construction (`buffer.insert` admits a candidate only if it beats the buffer minimum).
Excluding buffer members from the Generate draw therefore makes the eligible residual pool of an arm converge
toward *records already tried and rejected as low-LP*. The bandit keeps receiving feedback, but drawn from the
lower tail of the arm's LP distribution: `q_i` estimates `E[LP | arm i, LP below the eviction threshold]` rather
than `E[LP | arm i]`. An arm is penalized precisely in proportion to how productive it is. Unlike (1), this
degrades the signal continuously and from early training, and at `|Λ|/|C| = 0.5` it applies to all six arms, not
only to fully absorbed ones. `REQ-001`'s EMA is defined over an unbiased per-arm sample; this rule violates that
premise.

### Buffer selectivity

A second, independent fidelity gap: in Abouelazm `N = 1000` selects from an unbounded space, so `|Λ|/|C| → 0` and
the admission test "LP exceeds the current minimum in `Λ`" is a strong filter. At `|Λ|/|C| = 0.5` the same test
degrades to "above the median", the claim that the buffer stores high-learning-potential scenarios becomes nearly
vacuous, and the Explore and Exploit distributions substantially overlap. The value `1000` was inherited from the
paper, where it denoted something structurally different; it has no independent provenance in this project.

## Decision

Four decisions, approved together on 2026-07-29. `DEC-008` is a correction; `DEC-009` and `DEC-010` are project
design choices that this correction makes coherent to state.

### `DEC-008` — Generate eligibility is independent of scenario-buffer membership

Buffer membership is removed from the Generate candidate filter. The only exclusions are run-local quarantine
(`ADR-024`) and in-flight UIDs (`ADR-016` batch uniqueness). A Generate draw may therefore select a record the
buffer already holds; when it does, the committed episode **updates** the existing buffer entry instead of
attempting a rejected insert, and the MAB is updated as for any Generate episode.

This restores the exploration semantics of PLR/ACCEL/Abouelazm rather than departing from them. It resolves
`FIND-005` and removes the structural condition behind `FIND-001`.

### `DEC-009` — Per-arm coverage cycles (sampling without replacement within a cycle)

Within the drawn arm, Generate samples uniformly among the arm's records **not yet visited in the arm's current
coverage cycle**. When an arm's cycle is exhausted, the cycle counter increments, the visited set clears, and the
full arm pool becomes eligible again. Cycles are per-arm, not global, because the MAB draws arms at very different
rates and a global cycle would make fast arms wait for slow ones.

This is an addition to the reference methods, not a restoration, and it is justified specifically by the finite
catalog. With `m = 333` records per arm, `n` i.i.d. draws with replacement yield `m(1 - e^{-n/m})` distinct
records:

| Generate draws on one arm | distinct records, i.i.d. | distinct records, coverage cycles |
|---:|---:|---:|
| 333 | 210 (63%) | 333 (100%) |
| 600 | 278 (83%) | 333 (100%) |
| ~1000 | 316 (95%) | 333 (100%) |

In a representative completed run (~9,500 episodes, 40% Generate → ~3,800 Generate draws distributed very
unevenly across arms by the bandit), arms with low selection probability would receive on the order of 300 draws
and leave ~40% of their curated records never explored. On an unbounded level space this concern is meaningless —
which is exactly why the reference methods have no such mechanism — but on a frozen, curated, expensive catalog it
wastes the asset and weakens the claim that training covers the curated distribution. Coverage cycles do not
change how many draws an arm receives; they make those draws cover strictly more distinct records.

### `DEC-010` — `buffer_capacity` reduced from 1000 to 250

`|Λ| = 250` is 12.5% of the 2,000-record training catalog. With the frozen `warmup_buffer_size = 100`, the
warm-up fill ratio becomes `ρ = 0.40`, close to Abouelazm's `ρ = 0.5`. The buffer holds ~42 records per arm on
average, enough for replay diversity while keeping the admission test genuinely selective and keeping the Explore
and Exploit distributions distinct.

After `DEC-008` this value no longer affects MAB liveness or feedback bias — it governs only replay selectivity.
It is an explicit project choice and is **not** validated by ablation; that limitation is recorded in the
specification (`LIM-005`).

### `DEC-011` — Legacy sequential ACL path

The non-vectorized `collect_catalog_episode`/`choose_acl_episode` path in `driver.py` carries the same
buffer-exclusion semantics and is unreachable for production runs (`train_loop.py:791` returns early into
`run_scenario_acl_training`). It is aligned with the new semantics rather than deleted, so the two paths cannot
diverge silently; its removal is tracked separately and is out of scope here.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Keep the exclusion; handle exhaustion with a fallback tier (draw the arm's least-recently-generated buffered record) | No new persisted state, no schema bump, resume-compatible | Fixes `FIND-001` only; leaves `FIND-005` (the stronger defect) fully in place | Rejected once redoing the runs was accepted: its only advantage was avoiding a restart |
| Keep the exclusion; only reduce `buffer_capacity` below the smallest arm pool (333) | Smallest possible diff; makes total absorption impossible | Buys liveness by numerical accident, not by design; breaks if the arm partition changes; leaves `FIND-005` intact | Rejected as a non-fix of the actual defect |
| `DEC-008` alone (i.i.d. uniform draw, no coverage cycles) | Maximal fidelity to PLR/ACCEL/Abouelazm; no new persisted state; no schema bump | Leaves ~17–40% of each arm's curated records unexplored in a typical run (table above) | Rejected on the finite-catalog argument, but recorded as the strictly paper-faithful baseline; `DEC-009` is declared as an addition on top of it, not as part of the fix |
| Intra-arm prioritization by visit count / staleness instead of uniform-within-cycle | Finer control of coverage order | Within a cycle each record is visited at most once, so visit counts differ by at most 1 and the rule degenerates to deterministic round-robin; adds a second prioritization mechanism competing with the MAB for attribution | Rejected; uniform-within-cycle recorded as `RAT-010` |
| Buffer coincident with the catalog, replacing Generate/Replay with a single prioritized draw over all records | Simpler; close to PLR | Removes the arm-level MAB's only feedback channel; discards the Peng-derived half of the method | Rejected: it is a different method, not a repair of this one |

## Consequences

- **`ADR-028` narrows in scope, and stays.** After `DEC-008` an arm becomes Generate-ineligible only if its whole
  pool is quarantined or in flight — rare but possible. `ScenarioArmBandit.probabilities(eligible_mask=...)`
  remains the correct handling for that residual case and is retained as a safety net. A side effect is a gain in
  fidelity to Peng et al. §III-B, whose `eta/K` floor exists to guarantee that *"all arms have a probability of
  being selected throughout the entire training process"*: with no arms being excluded in normal operation, the
  floor again covers all six arms as intended.
- **Checkpoint schema bumps `acl_ema_v2` → `acl_ema_v3`.** `DEC-009` adds persisted per-record coverage state.
  Existing checkpoints cannot resume; the migration policy is restart, consistent with `REQ-005`.
- **The observable RNG call order changes.** `selection.py` declares that order as part of the reproducibility
  contract; seeds from `v1.1`/`v1.2` runs no longer reproduce those runs. Accepted: the experiments are being
  redone.
- **Runs completed under `v1.1`/`v1.2` keep the caveat already recorded for `FIND-001`** and gain the `FIND-005`
  caveat: curriculum-arm-selection claims derived from them are not reliable. Their policy-performance results are
  unaffected by this ADR.
- **The buffer becomes a genuine active subset** (12.5% of the catalog), so `buffer_events` eviction traffic will
  increase substantially relative to `v1.2` runs. This is intended, not a regression.
- Evicting a record from the buffer has never deleted it from the catalog or from its arm, and still does not; the
  catalog partition into arms was already immutable and remains so.

## Validation And Traceability

Mandatory tests are frozen in `docs/implementation/automatic_curriculum_learning_v1.3_exec_plan.md` §9. The
decision-critical ones:

- Generate candidate sets are invariant to buffer contents (direct test of `FIND-005` removal).
- An arm whose entire pool is held by the buffer still receives Generate draws and EMA updates (regression for
  `FIND-001` at its root, complementing the existing `ADR-028` renormalization tests).
- A coverage cycle visits every eligible record of an arm exactly once, then restarts with the counter
  incremented.
- A Generate draw that lands on a buffered record updates the entry and updates the MAB; it is not counted as a
  rejected insert.
- Coverage state round-trips through persistence; `acl_ema_v2` checkpoints are rejected with an explicit error.

## Approval Record

- Approved by: user
- Approval date: 2026-07-29
- Approval evidence: session of 2026-07-29. The user established that specifications are a record of decisions and
  must be corrected when wrong ("le specifiche le uso più per tenere traccia di ciò che faccio e con quali
  motivazioni, ma non devono essere prese per verità assolute se sono presenti errori"), accepted the cost of
  redoing the in-progress runs ("Non mi interessa se poi devo rifare le run"), reviewed the paper-fidelity analysis
  that splits the change into `DEC-008` (restoration) and `DEC-009` (original finite-catalog addition) together
  with the revised `DEC-010` capacity recommendation, and instructed "procedi".
