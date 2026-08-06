# Specification: Priority-Weighted Rank Scalarization Amendment

> Review candidate. Amends `SCAL-V1.0` only in the sections listed below.
> Every section of `SCAL-V1.0` not listed here is unchanged and remains in
> force. Do not begin production implementation while `Status: UNDER_REVIEW`.

## Metadata

- Feature: `Priority-weighted rank scalarization mode and optional post-hoc reward compression`
- Specification ID: `SCAL-V1.1`
- Version: `1.1`
- Status: `APPROVED`
- Date: `2026-08-06`
- Supersedes: `NONE` — amends `docs/specifications/rulebook_scalarization_v1.0_specification.md`, version `1.0`, without replacing it
- Related specifications:
  - `docs/specifications/rulebook_scalarization_v1.0_specification.md`, version `1.0` (base contract; §1–§5, §8, §10, §11, §13, §15, §16 unchanged)
  - `docs/specifications/rulebook_v4.7_specification.md` through `rulebook_v4.12_specification.md` (upstream margin contract; unchanged by this document)
- Related ADRs: `docs/decisions/ADR-011-rulebook-scalarization-v1.md`; `docs/decisions/ADR-056-wrongway-cost-physics-noise-deadband.md` (companion Rulebook fix, `DEC-SCAL11-003`, closed); `docs/decisions/ADR-057-scalarization-v1.1-defaults.md` (`DEC-SCAL11-001`, `002`, `004`)
- Authoritative: `YES`

## 1. Purpose And Context

This amendment responds to a finding from the 2026-08-01/02 scalarization
audit (Codex session, cross-referenced in this conversation): the approved
default mode `bounded_satisfaction_rank` (`SCAL-V1.0` §7.5) provably
preserves satisfaction-*pattern* dominance, but dilutes every margin's
continuous severity into one shared tie-breaker
\(T(\mathbf m)=\tfrac14\sum_k m_k\), weighted identically regardless of a
rule's priority. A change in \(m_1\) (collision severity) and a change in
\(m_4\) (progress) reach the learner with the same weight. This was
identified, independently of this amendment, as a plausible cause of the
measured gap between a native-reward baseline (route completion 0.521,
success 0.20) and the scalar-rulebook baseline (0.327, 0.05) on matched
seed/encoder/budget conditions — not proof of causation, but a concrete,
addressable asymmetry in the current formula.

This document keeps the vector rulebook, the four-margin macro contract, and
the three existing modes exactly as approved. It adds a fourth mode that
embeds each rule's continuous margin inside its own priority-weighted term
instead of a shared tie-breaker, and an optional, separately configured
post-hoc reward-compression stage to manage the resulting wider dynamic
range without touching the hierarchy proof.

### 1.1 Scientific source and project adaptation boundary

The new mode's structure was proposed directly by the user, not derived from
a literature formula. Its priority base is a further project-specific
departure from Veer et al.'s construction: their theorem's continuous
robustness term is a single, unweighted, shared tie-breaker external to the
priority digits (exactly what `bounded_satisfaction_rank` already
implements); embedding severity inside each weighted digit is a project
adaptation that requires its own dominance proof, because it doubles the
adverse excursion each higher digit must absorb, invalidating the literature
value \(a>2\) and the currently frozen `a=2.01`. §7.6 below re-derives the
required bound analytically.

The optional reward-compression stage is inspired by symlog/twohot reward
and value prediction in DreamerV3 (Hafner et al., 2023) and by earlier
return-rescaling techniques (van Hasselt et al., 2016, PopArt; Pohlen et
al., 2018). DreamerV3's empirical validation is in a model-based setting
with a discrete twohot regression head across highly heterogeneous reward
scales across domains; this project uses standard scalar-head PPO/TD3/SAC
critics on one fixed reward scale. The mechanism (reducing MSE sensitivity
to rare large-magnitude targets) is architecture-agnostic and expected to
transfer qualitatively, but no direct empirical evidence in this exact
setting (rulebook-hierarchy reward, SB3 baselines) currently exists; this is
recorded as an open ablation, not a proven result.

## 2. Scope

### 2.1 In Scope

- a fourth scalarization mode, `bounded_priority_weighted_rank`;
- its exact formula, domain, range, and dominance proof;
- the required frozen `priority_base` for this mode;
- an optional, independently configured, deterministic post-hoc reward
  compression stage (`none` or `symlog`) applied after scalarization;
- diagnostics distinguishing the raw (pre-compression) and delivered
  (post-compression) scalar reward;
- configuration, resume-compatibility, and logging amendments needed for the
  new mode and the compression stage;
- an explicit, separate dependency on a rulebook-side tolerance-ownership
  correction (`DEC-SCAL11-003`), scoped to a companion Rulebook amendment,
  not implemented by this document.

### 2.2 Out Of Scope

- everything already out of scope per `SCAL-V1.0` §2.2;
- changing the rule hierarchy, the four-margin macro contract, or any
  rulebook metric, cost, or margin formula;
- implementing the rulebook-side tolerance-ownership correction itself
  (component code in `components/road.py`, `components/controls.py`,
  `geometry/conflict_zones.py`) — that is a Rulebook specification amendment
  with its own ADR, out of this document's authority;
- amending or retracting `ADR-050` — this document only records that
  `bounded_priority_weighted_rank`'s correctness depends on retracting its
  "cost stays untouched" decision, and defers that retraction to the
  companion Rulebook amendment;
- selecting a specific compression scale parameter beyond the parameter-free
  `symlog` form (see `DEC-SCAL11-002`);
- making `bounded_priority_weighted_rank` the configuration default (see
  `DEC-SCAL11-001`).

### 2.3 Optional Or Deferred

- full training ablation comparing `bounded_satisfaction_rank` against
  `bounded_priority_weighted_rank` with and without compression;
- a compression scale parameter other than `τ=1`;
- migrating the reward-compression concept into a general SB3-level
  normalization wrapper (currently disabled by `SCAL-V1.0` §8.4 policy).

## 3. Terminology, Assumptions, And Preconditions

Adds to `SCAL-V1.0` §3.1:

| Symbol | Meaning | Domain |
|---|---|---|
| \(I_k'\) | priority-weighted-rank satisfaction indicator, \(k=1,2,3\) | \(-\mathbf 1[m_k<0]\in\{-1,0\}\), after the same numerical canonicalization as §3.3 |
| \(a'\) | priority base for `bounded_priority_weighted_rank` | frozen to `3.0` in conformant core experiments using this mode |
| \(h(r)\) | optional post-hoc reward compression | \(\operatorname{sign}(r)\log(1+|r|)\), applied only when `reward_compression.mode=symlog` |

\(I_k'\) is numerically identical to \(I_k-1\) from `SCAL-V1.0` §3.1
(\(I_k=\mathbf 1[m_k=0]\Rightarrow I_k-1=-\mathbf 1[m_k<0]\)); the new
symbol is introduced only because this mode adds \(m_k\) rather than
\(T(\mathbf m)\) to the same term, so the two must be visibly distinguished
in the formula.

Preconditions 1–8 of `SCAL-V1.0` §3.2 apply unchanged. §3.3 (numerical
canonicalization) applies unchanged to margins consumed by this mode.

### 3.1 Pre-existing tolerance-ownership defect, verified already active in `SCAL-V1.0`

This is not a risk specific to the new mode. It is a verified, already-active
defect in the currently approved `bounded_satisfaction_rank` default, found
by re-examining `ADR-050` on its merits rather than treating it as settled
because it is approved (per the user's explicit instruction in this
conversation: approved decisions are re-evaluated against new evidence, not
treated as immutable).

The categorical indicator \(I_k=\mathbf 1[m_k=0]\) — used by
`bounded_satisfaction_rank` today, unchanged by this amendment — depends on
`canonical_margin[:3]==0.0` exactly, with only the inherited numerical
canonicalization tolerance \(\varepsilon_0=10^{-8}\) (`SCAL-V1.0` §3.3).
`ADR-050`'s own stated noise floor is `0.1 m/s`, many orders of magnitude
above \(\varepsilon_0\). In `components/road.py:174-207`, `wrongway`'s
`cost` has no deadband at that scale; only the diagnostic `status` field
does (`WRONGWAY_STATUS_SPEED_EPSILON_MPS=0.1`, added by `ADR-050`). This
means a stationary ego's residual physics-solver velocity noise already
flips the canonicalized \(m_3\) hairline-negative **today**, which already
flips \(I_3\) categorically — and the categorical term
`priority_base^1 * (pattern-1)` is applied at full weight in the currently
approved formula (only the *continuous* margin, not the categorical jump,
is diluted by `bounded_satisfaction_rank`'s shared \(T(\mathbf m)/4\)
tie-breaker). `ADR-050`'s record — *"cost stays the same continuous
function... so the scalarizer's input... [is] unaffected"* — is accurate
about what that ADR itself changed, but does not establish that the reward
was already correct; it explicitly scoped itself to the diagnostic status
marker only, leaving this reward-affecting exposure untouched and
unaudited.

For comparison, `ADR-048` (`RSS_STANDSTILL_SPEED_MPS`,
`components/rss.py:52,104-105`) does not share this defect: verified in this
repository snapshot, that tolerance gates *applicability* itself (standstill
candidates are dropped before cost is computed, `cost=0.0`), not only a
diagnostic field. `wrongway` is the exception, not the general pattern.

**Update, 2026-08-06: fixed independently.** The `wrongway` instance of this
defect is corrected by `docs/specifications/rulebook_v4.12_specification.md`
(`ADR-056`, explicit user approval in this conversation), which gives `cost`
the same `0.1 m/s` deadband already used by `status` (`ADR-050`), as a
continuous reparameterization, and simplifies `status` to derive from
`cost > 0.0`. This was pursued as its own priority item, independent of this
document's approval, per `DEC-SCAL11-003`'s recommendation `A`.

The other candidate subrules were re-examined the same day, on their code
(not only on the original audit's table), with materially different
findings — `wrongway` had independent empirical evidence (a video artifact,
`ADR-050`) that its noise source (`0.1 m/s`-scale physics-solver contact
jitter) is real and many orders of magnitude above the `1e-8` numerical
tolerance; the other three do not have comparable evidence:

- **Stop dwell** (`components/controls.py::evaluate_stop`): the dwell timer
  accumulates by repeated float addition (`carried_continuous_s +
  delta_t_s`). Back-of-envelope floating-point summation error over a
  plausible number of steps is of order `1e-14`, already several orders of
  magnitude *below* the existing `1e-8` scalarizer tolerance — the opposite
  relationship from `wrongway`. Absent concrete evidence of an observed
  phantom violation, this is assessed as **not an active problem**, and is
  not queued as a fix.
- **TTC** (`components/ttc.py::evaluate_ttc`) and **crosswalk/yield temporal
  gap** (`components/controls.py::evaluate_vehicle_yield`,
  `evaluate_crosswalk_yield`, backed by
  `geometry/conflict_zones.py::worst_case_temporal_gap_violation`): both
  have the same `max(0, 1 - x/threshold)` shape, fed by a continuous
  geometric root-finding calculation (`predict_occupancy_interval`).
  **Update, 2026-08-06 (code audit completed):** re-read all three call
  sites end-to-end. In every one, `status` is derived from the *same*
  `cost` value that feeds the scalarizer
  (`ComponentStatus.VIOLATED if cost > 0.0 else ...`) — there is no second,
  separately-toleranced field the way `wrongway`'s `status` had
  `WRONGWAY_STATUS_SPEED_EPSILON_MPS` while its `cost` had none. That
  specific defect class (a deadband applied to the diagnostic field but
  silently omitted from the reward-affecting field) is therefore verified
  **absent** here — there is nothing of that shape to fix. `continuous_sat.py`
  additionally carries its own internal epsilons throughout
  (`AREA_EPSILON_M2`, `INTERVAL_EPSILON_S`, a `1e-12` coefficient guard),
  applied uniformly to the one value both `cost` and `status` are computed
  from, not asymmetrically. A separate, strictly weaker question remains
  open — whether the SAT solver's *numerical* error near the threshold
  ever exceeds the shared `1e-8` tolerance at all — but that is a
  measurement question about solver precision, not a known code defect,
  and no artifact or report motivates investigating it now.

`DEC-SCAL11-003` is updated accordingly: `wrongway` closed (defect
confirmed and fixed); stop dwell downgraded to "no action absent evidence"
(defect ruled out by magnitude); TTC and crosswalk/yield gap closed as
"no defect of this class found" (structurally verified absent by code
audit) — none of the three remaining subrules require a fix.

**Consequence for this document:** `bounded_priority_weighted_rank` does not
introduce this problem; it makes an already-active problem's magnitude
harder to ignore, because the same categorical jump that today contributes
`-2.01` will contribute `-3` under this mode with no change to its
relationship to the diluted tie-breaker (there no longer is one). Fixing the
`wrongway` cost deadband has value independent of whether this amendment is
approved at all, and should be pursued on its own merits as a correction to
the currently deployed reward, not deferred as a mode-specific
precondition. It remains, nonetheless, a genuine precondition for trusting
`bounded_priority_weighted_rank`'s output once enabled, and is retained as
`DEC-SCAL11-003` for that reason. A non-exhaustive candidate list for other
subrules (proposed by the referenced Codex audit, only `wrongway` and `RSS`
independently re-verified by this document; the rest require re-verification
during the companion ExecPlan, since `components/road.py`,
`components/controls.py`, and `rulebook/v2/wrapper.py` are modified on the
current branch):

| Subrule | Candidate finding |
|---|---|
| `wrongway` (`components/road.py`) | `cost` has no deadband; `status` has `WRONGWAY_STATUS_SPEED_EPSILON_MPS=0.1 m/s` (`ADR-050`). Verified present as described in this repository snapshot. |
| TTC (R2) (`components/ttc.py::evaluate_ttc`) | Re-verified 2026-08-06: `status` derives from `cost > 0.0`, no separate epsilon. No wrongway-class defect. No fix. |
| Stop dwell (`components/controls.py::evaluate_stop`) | Re-verified 2026-08-06: float-accumulation error `~1e-14`, below the `1e-8` tolerance. No fix. |
| Crosswalk/yield temporal gap (`components/controls.py::evaluate_vehicle_yield`/`evaluate_crosswalk_yield`, `geometry/conflict_zones.py::worst_case_temporal_gap_violation`) | Re-verified 2026-08-06: `status` derives from `cost > 0.0`, no separate epsilon. No wrongway-class defect. No fix. |
| RSS longitudinal/lateral, VRU clearance | Candidate: no new tolerance — changing these would alter safety semantics and needs separate justification, not a numerical-noise argument. |
| Off-road, wrong-carriageway | Already have `OFFROAD_AREA_EPSILON_M2=1e-4 m^2`; no change proposed. |
| Solid/dashed line markings | Already have a `1e-2 m` geometric buffer; no change proposed. |

This table is a starting point for the companion Rulebook ExecPlan, not a
normative requirement of this scalarization amendment.

## 4. Inputs And Prohibited Information

Unchanged from `SCAL-V1.0` §4, with one addition: the optional compression
stage (§7.7) consumes only the already-computed scalar reward and the
`reward_compression` configuration; it has the same prohibited-access list
as the scalarizer (no observations, actions, simulator state, or future
information) because it is a fixed deterministic function of one already-
computed scalar.

## 5. Outputs

Adds to `SCAL-V1.0` §5:

| Output | Meaning/type | Shape/unit/range | Ordering/mask | Consumer | Guarantees/edge cases |
|---|---|---|---|---|---|
| `raw_scalar_reward` | scalar reward before compression | scalar finite float | not applicable | diagnostics | equals `scalar_reward` when `reward_compression.mode=none` |
| `scalar_reward` (amended meaning) | scalar reward delivered to PPO/TD3/SAC | scalar finite float | not applicable | PPO/TD3/SAC | equals `raw_scalar_reward` transformed by `h` when compression is enabled |
| `reward_compression_mode` | selected compression identifier | scalar enum | not applicable | logs/manifest | exact configured value |

The raw, pre-compression reward must remain recoverable from diagnostics,
mirroring the existing requirement that raw margins remain recoverable
separately from the scalar reward (`SCAL-V1.0` §5).

## 6. Functional Requirements

### REQ-SCAL11-001: Add A Fourth Named Mode

The implementation must accept a fourth mode value,
`bounded_priority_weighted_rank`, alongside the three modes of `SCAL-V1.0`
REQ-SCAL-001. Per `DEC-SCAL11-001` (approved `A`, `ADR-057`), this mode
becomes the default for new conformant configuration; `bounded_satisfaction_rank`
remains available and unchanged for any run that explicitly selects it.

### REQ-SCAL11-002: Define The Priority-Weighted Rank Formula

`bounded_priority_weighted_rank` must:

- consume the same four bounded macro margins as the other bounded modes,
  after the same numerical canonicalization (`SCAL-V1.0` §3.3);
- compute \(I_k'=-\mathbf 1[m_k<0]\) for \(k=1,2,3\) from the canonicalized
  margins directly, with no scalarizer-owned tolerance beyond §3.3's
  existing numerical canonicalization;
- add each margin \(m_k\), \(k=1,2,3\), to its own priority-weighted term,
  not to a shared tie-breaker;
- retain \(m_4\) at unit weight, not divided by 4;
- use `priority_base = 3.0` exactly (see REQ-SCAL11-003);
- never gate \(R_4\) categorically, matching `SCAL-V1.0` REQ-SCAL-003/004.

### REQ-SCAL11-003: Freeze The Priority Base For This Mode

Conformant core experiments using `bounded_priority_weighted_rank` require
`priority_base = 3.0` exactly. This is a different frozen value from the
`priority_base = 2.01` required by `bounded_centered_sigmoid` and
`bounded_satisfaction_rank` (`SCAL-V1.0` §9); the configuration validator
must select the required frozen value from the selected mode, not accept one
global frozen constant across all bounded modes. Selecting
`bounded_priority_weighted_rank` with any `priority_base != 3.0` fails
configuration loading for conformant runs. Changing `3.0` requires a new
dominance proof and an approved specification/ADR amendment, exactly as
`SCAL-V1.0` §9 requires for `2.01`.

### REQ-SCAL11-004: Optional Post-Hoc Reward Compression

The implementation must accept a `reward_compression.mode` field with
values `none` (default) and `symlog`, applied identically to the output of
any of the four scalarization modes, as a separate deterministic, stateless
transform:

\[
h(r)=\operatorname{sign}(r)\,\log(1+|r|).
\]

- `none` leaves `scalar_reward` unchanged and equal to `raw_scalar_reward`.
- `symlog` sets `scalar_reward = h(raw_scalar_reward)` and preserves
  `raw_scalar_reward` unchanged in diagnostics.
- The transform must not depend on running statistics, episode history, or
  any state; two calls with the same `raw_scalar_reward` and configuration
  must return the same `scalar_reward`, independent of seed, algorithm, or
  call order.
- `reward_compression.mode` is frozen for the run and is part of the
  checkpoint/resume compatibility identity (§11).

### REQ-SCAL11-005: Preserve Per-Transition Dominance Under Compression

For any two canonical margin vectors \(\mathbf m^A,\mathbf m^B\) such that
\(r^{\rm rank'}(\mathbf m^A) > r^{\rm rank'}(\mathbf m^B)\) under
`bounded_priority_weighted_rank`, `h(r^{\rm rank'}(\mathbf m^A)) >
h(r^{\rm rank'}(\mathbf m^B))` must also hold whenever compression is
enabled, because `h` is strictly increasing. This requirement is a
consequence of `h`'s monotonicity (§7.7) and must be covered by a
regression test rather than assumed.

### REQ-SCAL11-006: Disclose The Compression Trade-Off, Do Not Hide It

Compression is not a substitute for §1's return-level limitation. It must
not be described, in logs, diagnostics, or documentation, as improving
return-level lexicographic behavior; it only affects the numerical scale
and variance of the training signal. Enabling `symlog` must be logged with
an explicit flag distinguishing conformant hierarchy-only runs from runs
that also apply compression, so results are never silently pooled across
the two conditions.

## 7. Mathematical And Algorithmic Contract

### 7.6 Mode D — `bounded_priority_weighted_rank`

For \(k=1,2,3\), using the canonicalized margins of `SCAL-V1.0` §3.3:

\[
I_k'(\mathbf m)=-\mathbf 1[m_k<0]\in\{-1,0\}.
\]

The reward is:

\[
\boxed{
r^{\mathrm{rank}'}(\mathbf m)
=
a'^{\,3}\bigl(I_1'+m_1\bigr)
+
a'^{\,2}\bigl(I_2'+m_2\bigr)
+
a'\bigl(I_3'+m_3\bigr)
+
m_4
}
\]

with:

\[
\boxed{a'=3}.
\]

#### Domain and range

\(m_{1:3}\in[-1,0]\), \(m_4\in[-1,1]\). Define \(q_k=I_k'+m_k\):

- if \(m_k=0\) (satisfied), \(I_k'=0\) and \(q_k=0\);
- if \(m_k<0\) (violated), \(I_k'=-1\) and \(q_k\in[-2,-1)\), so
  \(q_k\in[-2,0]\) over the full domain.

Maximum: \((m_1,m_2,m_3,m_4)=(0,0,0,1)\Rightarrow r_{\max}=1\).

Minimum: \((m_1,m_2,m_3,m_4)=(-1,-1,-1,-1)\Rightarrow
r_{\min}=27(-2)+9(-2)+3(-2)+(-1)=-79\).

\[
r^{\mathrm{rank}'}\in[-79,1].
\]

Neutral vector: \(r^{\mathrm{rank}'}(\mathbf 0)=0\).

#### Satisfaction-pattern dominance proof

The worst-case adverse excursion of any single \(q_k\) is its full range,
\(2\) (from `0`, satisfied, down to the infimum `-2` as \(m_k\to-1\)
violated); \(m_4\)'s worst-case adverse excursion is its full range, `2`
(`+1` to `-1`). Unlike `SCAL-V1.0` §7.5, there is no separate,
independently smaller shared tie-breaker term: every lower-priority
opposition is now itself weighted by its own priority base.

For \(R_3\) (weight \(a'\)) to dominate the worst-case opposition from
\(m_4\) alone:

\[
a'\cdot 1 > 2
\quad\Longleftrightarrow\quad
a'>2.
\]

(The `1` coefficient is \(q_3\)'s indicator jump `-1`→`0` times the base
`a'`; the worst \(m_4\) swing contributes up to `2` directly since it has
unit weight.)

For \(R_2\) (weight \(a'^2\)) to dominate the worst-case combined opposition
from \(R_3\) and \(R_4\):

\[
a'^2 > 2a' + 2.
\]

For \(R_1\) (weight \(a'^3\)) to dominate the worst-case combined opposition
from \(R_2\), \(R_3\), and \(R_4\):

\[
a'^3 > 2a'^2 + 2a' + 2.
\]

Checking \(a'=2.01\) (the value frozen for the other three modes): fails
the second and third conditions (\(4.0401 < 6.02\); \(8.120601 < 14.1002\)).
Only R3-over-R4 dominance would hold.

Checking \(a'=3\): all three hold —
\(3>2\); \(9>2(3)+2=8\); \(27>2(9)+2(3)+2=26\). The numeric root of
\(a^3=2a^2+2a+2\) is at \(a\approx2.91964\); `3` is the smallest integer
above it and yields the closed-form digit weights `27, 9, 3, 1`.

Therefore, for every pair of satisfaction patterns and every valid margin
combination, the vector satisfying the first differing rule (in priority
order R1 > R2 > R3) has strictly greater \(r^{\mathrm{rank}'}\), independent
of all lower indicators and margin values.

#### Same-pattern monotonicity

Within a fixed satisfaction pattern, \(r^{\mathrm{rank}'}\) is exactly
linear and strictly increasing in every \(m_k\) with a fixed positive
coefficient (\(a'^3\), \(a'^2\), \(a'\), or `1`), with no saturation over
the entire margin range. This is the property `bounded_satisfaction_rank`
lacked (its tie-breaker gave every margin the same 1/4 weight, diluting
higher-priority severity) and `bounded_centered_sigmoid` only approximated
poorly (saturating to near-constant well inside the domain at the frozen
`c=30`).

#### Numerical margin of the proof

`27 > 26` and `9 > 8` are margins of `1`, i.e. roughly `3.7%` and `12.5%` of
the dominant term — several orders of magnitude larger than the inherited
numerical tolerance `1e-8`, so floating-point noise cannot violate the
proof. The margin is nonetheless thin relative to future changes: if a
future rulebook amendment widens any bounded margin's range beyond
`[-1,0]`/`[-1,1]`, or adds a fifth satisfaction-gated priority level, the
inequalities above must be re-derived and re-verified before reuse; §12
requires a direct regression test of the raw inequalities (not only of
their numeric instantiation at `a'=3`) to catch this automatically.

#### What the proof does not establish

Identical to `SCAL-V1.0` §7.5's disclaimer: no continuous lexicographic
severity ordering within a fixed pattern beyond monotonicity, and no
lexicographic optimality of the expected discounted return
\(\sum_t\gamma^t r^{\mathrm{rank}'}(\mathbf m_t)\) relative to the separate
objective returns \(J_1,\ldots,J_4\). This limitation is inherent to any
per-step scalarization, not specific to this formula (§1.1).

### 7.7 Post-hoc reward compression

\[
h(r)=\operatorname{sign}(r)\log(1+|r|).
\]

Properties:

- odd, strictly increasing, \(h(0)=0\);
- for `bounded_priority_weighted_rank`'s range \([-79,1]\):
  \(h(-79)=-\log(80)\approx-4.382\), \(h(1)=\log(2)\approx0.693\), an
  approximately `17.8`x range compression (from `80` to `~5.08`);
- near zero, \(h(r)\approx r\) for \(|r|\ll1\) but already compresses
  moderate values noticeably (\(h(1)\approx0.693\) vs raw `1`, a `~31%`
  reduction), so compression is not tail-only: it also reduces the
  magnitude of the everyday progress-only signal, not just rare violation
  spikes;
- compresses the ratio between severities: e.g. a mild-vs-severe R1
  violation at raw `-27` vs `-54` (ratio `2.0`) compress to
  \(h(-27)\approx-3.332\), \(h(-54)\approx-4.007\) (ratio `1.20`),
  reducing the gradient signal that separates degrees of severity within
  the same violated category — an explicit trade of hierarchy-separation
  strength for critic-target variance reduction, not a free improvement
  (§1.1, REQ-SCAL11-006).

`h` has no free parameter in this specification. A scaled variant
\(h_\tau(r)=\tau\cdot\operatorname{sign}(r)\log(1+|r|/\tau)\) is explicitly
deferred (§2.3): DreamerV3 uses `τ=1` uniformly to avoid per-domain tuning
across highly heterogeneous reward scales, a justification that does not
directly transfer to this project's single, fixed reward scale; picking a
different `τ` here would need its own justification and is not addressed by
this document.

## 8. Applicability, State, And Timing

Unchanged from `SCAL-V1.0` §8, with one addition: the compression stage
(§7.7) is applied after scalarization, before the transition is inserted
into the rollout/replay buffer, with the same purity/statelessness
requirements as §8.2 (no history, no running statistics, a no-op `reset()`
if present).

## 9. Configuration

```yaml
scalarization:
  specification_id: SCAL-V1.1
  version: 1.1
  mode: bounded_priority_weighted_rank   # new default; DEC-SCAL11-001=A
  vector_schema_id: rulebook_v2_macro_v4
  priority_base: 3.0                     # required 2.01 for the three SCAL-V1.0 modes

  reward_compression:
    mode: none                      # none | symlog; DEC-SCAL11-002=B, off by default
```

| Field | Type | Default | Valid range | Meaning | Required | Frozen for experiments |
|---|---:|---:|---:|---|---|---|
| `scalarization.mode` | enum | `bounded_priority_weighted_rank` | four named modes | formula selection | `YES` | `YES` |
| `scalarization.priority_base` | float | mode-dependent | `2.01` for legacy/centered-sigmoid/rank; `3.0` for priority-weighted-rank | priority separation | `YES` | `YES` |
| `scalarization.reward_compression.mode` | enum | `none` | `none`, `symlog` | post-hoc scalar compression | `YES` | `YES` |

Validation rules (in addition to `SCAL-V1.0` §9):

- selecting `bounded_priority_weighted_rank` with `priority_base != 3.0`
  fails configuration loading;
- selecting any other mode with `priority_base != 2.01` still fails
  configuration loading, exactly as today;
- `reward_compression.mode` may be combined with any of the four
  scalarization modes;
- no field may be overridden per algorithm, scenario, arm, source, or seed
  (inherited from `SCAL-V1.0` §9).

## 10. Errors, Logging, And Diagnostics

Adds to `SCAL-V1.0` §10.1 (fatal errors): unknown `reward_compression.mode`;
`priority_base` mismatched with the selected mode's required frozen value;
non-finite `scalar_reward` after compression; mode/compression mismatch on
checkpoint resume.

Adds to `SCAL-V1.0` §10.3 (required run-level logging):
`reward_compression_mode`.

Adds to `SCAL-V1.0` §10.4 (required transition/episode diagnostics):
`raw_scalar_reward` alongside `scalar_reward`, whenever compression is
enabled, so the raw hierarchy-only reward remains reconstructable.

## 11. Reproducibility And Compatibility

Unchanged determinism/legacy-compatibility requirements from `SCAL-V1.0`
§11.1–§11.2. Amends §11.3 (checkpoint/replay compatibility identity): the
required field list gains `reward_compression_mode`. A change of mode
between `bounded_satisfaction_rank` and `bounded_priority_weighted_rank`, or
a change of `reward_compression.mode`, is a reward-semantics change under
the same rules as any other mode change in `SCAL-V1.0` §11.3 (new run
identity, empty rollout/replay state, separate result labeling).

## 12. Acceptance Criteria

### AC-SCAL11-001: Fourth Mode Registered, Default Unchanged Unless Approved

- Given: a valid configuration without an explicit mode;
- When: scalarization configuration is loaded;
- Then: the selected mode is `bounded_priority_weighted_rank` with
  `priority_base=3.0` (`DEC-SCAL11-001`, approved `A`).
- Given: `mode=bounded_priority_weighted_rank`;
- Then: configuration loading succeeds only with `priority_base=3.0`.
- Related requirements: `REQ-SCAL11-001`, `REQ-SCAL11-003`.

### AC-SCAL11-002: Priority-Weighted Rank Reference Values

For `bounded_priority_weighted_rank` with `a'=3`:

\[
r(0,0,0,0)=0,\quad
r(0,0,0,1)=1,\quad
r(0,0,0,-1)=-1,
\]
\[
r(0,0,-1,1)=-5,\quad
r(0,-1,0,1)=-17,\quad
r(-1,0,0,1)=-53,\quad
r(-1,-1,-1,-1)=-79.
\]

All values must agree within \(10^{-6}\). Related requirements:
`REQ-SCAL11-002`.

### AC-SCAL11-003: Exhaustive Satisfaction-Pattern Dominance

- Given: every pair of the \(2^3\) satisfaction patterns and boundary/extreme
  continuous margin combinations;
- When: the first differing rule (priority order R1 > R2 > R3) is
  identified;
- Then: the vector satisfying that rule has strictly greater
  `bounded_priority_weighted_rank` scalar reward, independent of all lower
  indicators and valid margin values.
- Related requirements: `REQ-SCAL11-002`.

### AC-SCAL11-004: Algebraic Guard On The Dominance Inequalities

- Given: `a'=3` and the general symbolic bounds
  \(a'>2\), \(a'^2>2a'+2\), \(a'^3>2a'^2+2a'+2\);
- When: a regression test evaluates the three inequalities directly (not
  only the specific reference values of `AC-SCAL11-002`);
- Then: all three hold strictly.
- Purpose: catch, at test time, any future change to a bounded margin's
  range or to the number of satisfaction-gated priority levels that would
  silently invalidate the proof.
- Related requirements: `REQ-SCAL11-002`.

### AC-SCAL11-005: Same-Pattern Monotonicity, No Saturation

- Given: two valid vectors with identical satisfaction indicators and all
  margins equal except one margin larger in vector A;
- When: `bounded_priority_weighted_rank` reward is evaluated;
- Then: \(r(A)>r(B)\) by exactly the fixed linear coefficient times the
  margin difference, for every value of the differing margin across its
  entire domain (no saturation).
- Related requirements: `REQ-SCAL11-002`.

### AC-SCAL11-006: Compression Preserves Pairwise Order

- Given: any two canonical margin vectors with
  \(r^{\mathrm{rank}'}(\mathbf m^A)>r^{\mathrm{rank}'}(\mathbf m^B)\);
- When: `reward_compression.mode=symlog`;
- Then: \(h(r^{\mathrm{rank}'}(\mathbf m^A))>h(r^{\mathrm{rank}'}(\mathbf m^B))\).
- Related requirements: `REQ-SCAL11-005`.

### AC-SCAL11-007: Compression Reference Values And Reversibility Of Diagnostics

- Given: `reward_compression.mode=symlog` and a known `raw_scalar_reward`;
- When: the transition is scalarized and compressed;
- Then: `scalar_reward = sign(raw_scalar_reward) * log(1 + |raw_scalar_reward|)`
  within \(10^{-6}\), and `raw_scalar_reward` remains separately logged and
  recoverable.
- Given: `reward_compression.mode=none`;
- Then: `scalar_reward == raw_scalar_reward` exactly.
- Related requirements: `REQ-SCAL11-004`, `REQ-SCAL11-006`.

### AC-SCAL11-008: Resume Compatibility Includes Compression Mode

- Given: a checkpoint/replay state created with one `reward_compression.mode`;
- When: resume is requested with a different value;
- Then: same-run resume fails before learning continues.
- Related requirements: `REQ-SCAL11-004`; amends `SCAL-V1.0` AC-SCAL-015.

## 13. Required Validation Categories

Same categories as `SCAL-V1.0` §13, all `REQUIRED` for the new mode and the
compression stage, plus:

- algebraic-guard regression (`AC-SCAL11-004`) — `REQUIRED`, new category
  specific to this amendment.

## 14. Traceability

| Requirement | Acceptance criteria | Scientific source or approved decision |
|---|---|---|
| `REQ-SCAL11-001` | `AC-SCAL11-001` | user proposal, this conversation, 2026-08-06 |
| `REQ-SCAL11-002` | `AC-SCAL11-002`, `003`, `005` | user proposal + Veer et al. adaptation with re-derived bound |
| `REQ-SCAL11-003` | `AC-SCAL11-001` | dominance proof §7.6 |
| `REQ-SCAL11-004` | `AC-SCAL11-007` | DreamerV3 symlog (Hafner et al., 2023), PopArt (van Hasselt et al., 2016) |
| `REQ-SCAL11-005` | `AC-SCAL11-006` | monotonicity of `h` |
| `REQ-SCAL11-006` | `AC-SCAL11-007` | this conversation, 2026-08-06 |

## 15. Open Decisions And Limitations

| ID | Question | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|
| `DEC-SCAL11-001` | Should `bounded_priority_weighted_rank` become the new default mode, or remain opt-in alongside the existing default? | `A`: new default; `B`: opt-in only, `bounded_satisfaction_rank` stays default | `A`, once `DEC-SCAL11-003`'s companion Rulebook amendment lands, on the strength of §1's diluted-signal finding | Which formula every new conformant run uses by default | `RESOLVED: A` (`ADR-057`, 2026-08-06). `DEC-SCAL11-003`'s precondition is satisfied (closed above). |
| `DEC-SCAL11-002` | Should `reward_compression.mode=symlog` be enabled by default once implemented? | `A`: default on; `B`: default off, explicit opt-in ablation | `B` — the hierarchy-separation trade-off (§7.7) is unquantified in this exact setting and should be measured, not assumed | Training stability vs. hierarchy-separation strength | `RESOLVED: B` (`ADR-057`, 2026-08-06). `symlog` remains available but is not the default. |
| `DEC-SCAL11-003` | The `wrongway` cost/status tolerance mismatch (§3.1) is verified already active in the currently approved `SCAL-V1.0` default, independent of this amendment. Should its correction be pursued as its own, separately prioritized Rulebook ADR/ExecPlan regardless of `SCAL-V1.1`'s outcome, and treated as a precondition only for enabling `bounded_priority_weighted_rank` specifically? | `A`: yes, both — independent priority item and a precondition for this mode; `B`: bundle it only with this amendment, no independent priority | `A` — it already affects deployed reward computation; waiting on this document's approval to fix it is not justified | Correctness of the currently running default reward, and whether this mode can be enabled before that separate fix lands | `CLOSED`, all four candidates re-examined. `wrongway`: defect confirmed and fixed (`rulebook_v4.12_specification.md`/`ADR-056`, 2026-08-06, `A` applied). Stop dwell: re-examined and found **not** to share this defect (accumulation error `~1e-14`, already below the `1e-8` tolerance) — no fix. TTC and crosswalk/yield temporal gap: full code audit of `ttc.py::evaluate_ttc`, `controls.py::evaluate_vehicle_yield`/`evaluate_crosswalk_yield`, and `conflict_zones.py::worst_case_temporal_gap_violation` (2026-08-06) found `status` is derived from the same `cost` value in every case (`VIOLATED if cost > 0.0`) — the wrongway-class asymmetric-tolerance defect does not exist there. No fix needed; the precondition for enabling `bounded_priority_weighted_rank` (§3.1's "Consequence for this document") is satisfied. |
| `DEC-SCAL11-004` | Should the algebraic-guard test (`AC-SCAL11-004`) also assert the analogous inequalities for the three existing modes retroactively, closing a gap `SCAL-V1.0` did not require? | `A`: yes, add retroactively; `B`: no, out of scope for this amendment | `B` — out of scope; note as a candidate follow-up, not a blocking gap of this document | Test coverage of `SCAL-V1.0`'s existing proof | `RESOLVED: B` (`ADR-057`, 2026-08-06). Recorded as a candidate follow-up for a future, separately scoped `SCAL-V1.0` test-coverage amendment; not part of this document's implementation. |

### Intentional limitations (unchanged in kind from `SCAL-V1.0` §16.1, restated for this mode)

1. No continuous lexicographic severity ordering across satisfaction
   patterns; no lexicographic optimality of the expected discounted return.
2. The categorical jump remains discrete; function approximation is
   expected to smooth it near the boundary, more so in under-sampled
   regions (rare, severe violations), independent of any transform applied
   after scalarization.
3. Compression, if enabled, trades hierarchy-separation strength for
   reduced target variance; it is not a solution to limitation 1.
4. The dominance proof's numerical margin (`27` vs `26`, `9` vs `8`) is thin
   relative to future rulebook range changes; `AC-SCAL11-004` guards against
   silent invalidation but does not prevent the underlying fragility.

## 16. References

- [R2] S. Veer, K. Leung, R. Cosner, Y. Chen, P. Karkus, and M. Pavone,
  “Receding Horizon Planning with Rule Hierarchies for Autonomous Vehicles,”
  ICRA 2023. https://arxiv.org/abs/2212.03323 — source of the exponential
  priority-separation principle; §7.6 explicitly diverges from its shared,
  unweighted tie-breaker construction and re-derives the required bound for
  this project's embedded-severity variant.
- D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, “Mastering Diverse Domains
  through World Models,” 2023. https://arxiv.org/abs/2301.04104 — source of
  the symlog reward/value transform (§7.7); validated in a model-based,
  cross-domain, twohot-regression setting, not this project's setting.
- H. van Hasselt, A. Guez, M. Hessel, V. Mnih, D. Silver, “Learning values
  across many orders of magnitude,” NeurIPS 2016 (PopArt).
  https://arxiv.org/abs/1602.07714 — related return-rescaling precedent,
  cited for context in §1.1; not adopted (it is adaptive/stateful, which
  conflicts with `SCAL-V1.0` §8.2's statelessness requirement for the
  scalarizer).
- `docs/decisions/ADR-050-wrongway-status-deadband.md` — the existing,
  approved decision that §3.1/`DEC-SCAL11-003` identifies as needing
  amendment for this mode's correctness.
- `docs/specifications/rulebook_scalarization_v1.0_specification.md` — base
  contract this document amends.

## 17. Implementation Handoff Checklist

Before setting `Status: APPROVED`:

- [x] `DEC-SCAL11-001` through `DEC-SCAL11-004` resolved (`A`, `B`, `CLOSED`, `B`; `ADR-057`).
- [x] Companion Rulebook tolerance-ownership amendment approved (`DEC-SCAL11-003`
      resolved `A`; `rulebook_v4.12_specification.md`/`ADR-056`).
- [x] New mode's dominance proof (§7.6) and algebraic-guard test
      (`AC-SCAL11-004`) reviewed.
- [x] Compression stage's statelessness and resume-compatibility fields
      reviewed against `SCAL-V1.0` §8.2/§11.3.
- [x] The user explicitly approves this complete amendment.
- [x] Status becomes `APPROVED`; file moves to `docs/specifications/` without
      `_UNDER_REVIEW`; `SCAL-V1.0`'s `project_index.md` row is updated to
      reference `SCAL-V1.1` as the amended authoritative version, following
      the same pattern already used for `rulebook_v4.8`–`v4.11` amending
      `rulebook_v4.7`.

## 18. Approval Record

- Approved by: `user`
- Approval date: `2026-08-06`
- Approval evidence: "Top. Avevo letto le altre decisioni e approvo i suggerimenti, quindi procedi pure con l'implementazione automatica del piano" (this conversation), approving the recommended resolution of `DEC-SCAL11-001` (`A`), `DEC-SCAL11-002` (`B`), and `DEC-SCAL11-004` (`B`); `DEC-SCAL11-003` was already closed independently on 2026-08-06.
- Approval notes: implementation proceeds per the accompanying ExecPlan (`docs/implementation/rulebook_scalarization_v1.1_exec_plan.md`) and `ADR-057`.
- Repository path: `docs/specifications/rulebook_scalarization_v1.1_specification.md` (post-approval)
- Project index updated: `YES`
