# Specification: Configurable Rulebook Scalarization for Scalar RL Baselines

> This document was explicitly approved by the user on 2026-07-17 after the
> complete-text review. It is authoritative for Rulebook scalarization v1.0;
> production implementation remains subject to the linked ExecPlan and its
> validation gates.

## Metadata

- Feature: `Rulebook scalarization for PPO, TD3, and SAC baselines`
- Specification ID: `SCAL-V1.0`
- Version: `1.0`
- Status: `APPROVED`
- Date: `2026-07-17`
- Supersedes: `NONE — no authoritative scalarization specification is registered`
- Related specifications:
  - `docs/specifications/rulebook_v4.6_specification.md`, version `4.6-final-implementation-complete`
  - `docs/specifications/rulebook_v4.7_specification.md`, version `4.7-final-implementation-complete`
  - `docs/specifications/automatic_curriculum_learning_v1_specification.md`
- Related ADRs: `docs/decisions/ADR-011-rulebook-scalarization-v1.md`
- Authoritative: `YES`

## 1. Purpose And Context

The selected rulebook emits an ordered per-transition margin vector

\[
\mathbf m_t =
\left(
m_1(t),m_2(t),m_3(t),m_4(t)
\right),
\]

ordered as

\[
R_1^{\mathrm{collision}}
\succ
R_2^{\mathrm{interaction}}
\succ
R_3^{\mathrm{compliance}}
\succ
R_4^{\mathrm{progress}}.
\]

The four-component notation above is the bounded macro schema. The legacy mode
uses the explicitly declared ordered schema length `N` described in §3.1 and
does not require the selected rulebook to expose exactly four margins.

The authoritative Rulebook v4.6 contract gives

\[
m_1,m_2,m_3\in[-1,0],
\qquad
m_4\in[-1,1].
\]

For the first three margins, zero means that no evaluable violation is
present and a negative value means violation. The fourth margin is a continuous
route-progress objective: positive means forward progress, zero means stopping,
and negative means regression.

PPO, TD3, and SAC as used by the scalar baselines require one scalar reward per
transition. Every mode receives a vector of ordered margins, but the input
contract differs by mode:

| Mode | Input contract |
|---|---|
| `legacy_scaled_sigmoid` | An explicitly adapted ordered vector of `N` finite margins and exactly `N` compatible positive scales. |
| `bounded_centered_sigmoid` | Exactly the four bounded macro-margins of the Rulebook v2 implementation family, when the selected specification satisfies the v4.6/v4.7 macro contract. |
| `bounded_satisfaction_rank` | Exactly the four bounded macro-margins of the Rulebook v2 implementation family, when the selected specification satisfies the v4.6/v4.7 macro contract. |

This specification defines three selectable scalarization modes:

1. `legacy_scaled_sigmoid`, preserving the historical implemented formula;
2. `bounded_centered_sigmoid`, a bounded-input smooth revision;
3. `bounded_satisfaction_rank`, the default centered satisfaction-rank formula.

The implementation must support all three modes through configuration. The
scientific default configuration is Rulebook implementation family v2,
specification version 4.7, with `bounded_satisfaction_rank`. Bounded modes are
mathematically compatible with v4.6 and v4.7 because both satisfy the same
four-margin contract, although results must remain labeled by the exact
rulebook specification version. Implementing the three modes does not require
running all three in the main experimental matrix. The legacy and smooth modes
are optional experimental ablations unless a separate approved protocol makes
them mandatory.

The scalarizer is always downstream of the selected rulebook:

```text
RulebookResult.margins -> configured scalarizer -> scalar_reward -> PPO/TD3/SAC
```

The existing `scalar_reward` field/interface remains the learner-facing scalar
reward. Exactly one configured scalarization mode is active for each run; two
alternative rewards are not delivered concurrently to the learner. The native
environment reward may remain in diagnostics but has zero weight and is not
mixed into conformant core scalar rewards.

This specification governs scalar baselines only. A true lexicographic learner
continues to consume and optimize separate objective returns and does not use
these formulas to define its policy objective.

### 1.1 Scientific source and project adaptation boundary

The design is inspired primarily by the rank-preserving reward construction of
Veer et al. Their construction:

- maps rule satisfaction patterns to exponentially separated reward levels;
- uses average rule robustness as a same-rank tie-breaker;
- provides a non-differentiable step-function construction with a rank
  guarantee;
- replaces the step with a sigmoid when differentiability is required for
  continuous trajectory optimization;
- scales raw STL robustness because their raw values are not guaranteed to lie
  in the theorem's bounded domain.

The formulas in this specification are project adaptations, not direct copies:

- the inputs are per-step rulebook margins rather than whole-trajectory STL
  robustness values;
- the four-margin macro inputs for the bounded modes are already bounded;
- only \(R_1,R_2,R_3\) possess satisfaction/violation semantics;
- \(R_4\) is retained exclusively as a continuous objective;
- rewards are consumed by model-free RL and accumulated into expected
  discounted returns;
- the default guarantee concerns per-transition satisfaction patterns, not
  continuous lexicographic ordering of severities and not lexicographic
  optimality of expected discounted return vectors.

## 2. Scope

### 2.1 In Scope

- deterministic scalarization of an ordered rulebook margin vector;
- configuration-driven selection among three scalarization modes;
- exact mathematical behavior of every mode;
- numerical canonicalization at the inherited rulebook comparison tolerance;
- scalar reward delivery to PPO, TD3, and SAC;
- storage and preservation of the original margin vector for diagnostics,
  evaluation, and rule-aware curriculum functions;
- interactions with one-step, GAE, and current replay storage;
- conditional compatibility rules for future N-step and PER extensions;
- logging, reproducibility, checkpoint compatibility, and mandatory tests;
- a theoretical record of why the formulas differ and what each one does and
  does not guarantee.

### 2.2 Out Of Scope

- changing the rule hierarchy or any rulebook metric;
- changing the maximum aggregation inside \(R_2\) or \(R_3\);
- selecting rulebook physical, semantic, geometric, or temporal thresholds;
- defining the true lexicographic PPO, TD3, or SAC learners;
- defining distributional critics;
- adding a safety shield or guaranteeing safety during learning;
- changing environment termination or truncation;
- mixing the native MetaDrive/ScenarioEnv reward into core thesis rewards;
- tuning scalarization parameters separately per algorithm, dataset source,
  scenario arm, seed, or experimental result.

### 2.3 Optional Or Deferred

- full training ablations for all three modes;
- learned scalarization;
- thresholded lexicographic action selection;
- Chebyshev, \(p\)-mean, hypervolume, or reference-point scalarizations;
- constrained-RL replacements such as CPO;
- non-Archimedean numerical scalarization;
- changing the continuous same-rank tie-breaker;
- introducing a minimum semantic violation threshold distinct from the
  rulebook's numerical tolerance.

## 3. Terminology, Assumptions, And Preconditions

### 3.1 Symbols

| Symbol | Meaning | Domain |
|---|---|---|
| \(m_k(t)\) | ordered rulebook margin for component \(k\) | finite scalar; bounded-mode domain is \(m_{1:3}\in[-1,0]\), \(m_4\in[-1,1]\) |
| \(\mathbf m_t\) | ordered rulebook margin vector | shape `(N,)`; bounded-mode macro schema is `(4,)` |
| \(N\) | number of margins in the selected legacy schema | positive integer, explicit in the adapter/configuration |
| \(a\) | exponential priority base | frozen to `2.01` in conformant core experiments |
| \(c\) | sigmoid sharpness | frozen to `30.0` when a sigmoid mode is selected |
| \(s_k\) | legacy component-specific scale | positive finite scalar, explicitly configured for every `k=1,...,N` |
| \(\rho_k\) | legacy normalized robustness | \((-1,1)\) for finite \(m_k/s_k\) |
| \(\sigma(x)\) | logistic sigmoid | \(1/(1+\exp(-x))\) |
| \(I_k\) | satisfaction indicator for \(R_k\) | \(\mathbf 1[m_k=0]\), \(k=1,2,3\) |
| \(T(\mathbf m)\) | continuous tie-breaker | \(\frac14\sum_{k=1}^4m_k\) |
| \(\varepsilon_0\) | inherited numerical zero/range tolerance | \(10^{-8}\) |

### 3.2 Preconditions

1. **Complete rulebook evaluation — upstream guarantee.** Scalarization is
   called only after the selected rulebook has completed evaluation. For the
   v2 implementation family this is represented by
   `RulebookResult.complete_evaluation == True`; other rulebooks require an
   explicit adapter-level completion assertion.
2. **Explicit ordering — upstream/adaptor guarantee.** Every input vector has
   a declared schema and order. The bounded macro order is collision,
   interaction, compliance, progress.
3. **Bounded margins — bounded-mode runtime validation.** Bounded modes reject
   values outside their specified ranges beyond \(\varepsilon_0\). Legacy mode
   validates finite values and relies on its explicit schema/scales rather than
   applying the bounded macro ranges.
4. **Finite values — runtime validated.** NaN and infinity are rejected in all
   modes.
5. **No applicability mask required — upstream guarantee for bounded macro
   schema.** A non-applicable macro-rule produces margin zero according to the
   rulebook contract.
6. **Per-transition operation — normative.** Scalarization is applied once to
   each emitted transition, including terminal and time-limit-truncated
   transitions.
7. **No state — normative.** The scalarizer has no episodic memory and no reset
   behavior beyond immutable configuration.
8. **No policy dependence — normative.** The same input vector and
   configuration produce the same scalar reward for every algorithm and seed.

### 3.3 Numerical canonicalization

Before evaluating a bounded mode:

1. reject non-finite values;
2. reject \(m_{1:3}< -1-\varepsilon_0\) or \(m_{1:3}>\varepsilon_0\);
3. reject \(m_4<-1-\varepsilon_0\) or \(m_4>1+\varepsilon_0\);
4. canonicalize values within \(\varepsilon_0\) of `0.0` to exact `0.0`;
5. canonicalize numerical overshoot within \(\varepsilon_0\) of `-1.0` or
   `1.0` to the corresponding boundary;
6. perform no other clipping.

For `legacy_scaled_sigmoid`, the adapter validates finite values and explicit
schema/scales. It does not impose the bounded macro ranges or silently clip raw
rulebook margins.

This is a numerical convention inherited from the rulebook comparison tolerance,
not a semantic near-satisfaction threshold. A later semantic tolerance would
change the scientific meaning and requires a new approved decision.

## 4. Inputs And Prohibited Information

| Input | Meaning/type | Shape/unit/frame | Range/time | Source/validity | Missing-data behavior | Policy-visible |
|---|---|---|---|---|---|---|
| `margins` | ordered rulebook margins | `(N,)`, dimensionless or rulebook-defined units | current transition | validated rulebook adapter | fail fast | `NO` |
| `vector_schema_id` | explicit margin schema/order identifier | scalar string | run-constant | validated configuration/adapter | fail fast | `NO` |
| `mode` | scalarization enum | scalar string | run-constant | validated configuration | fail fast | `NO` |
| `priority_base` | exponential base | scalar | run-constant | configuration/manifest | fail fast | `NO` |
| `sigmoid_sharpness` | sigmoid \(c\) | scalar | run-constant | required by sigmoid modes | fail fast | `NO` |
| `legacy_rule_scales` | legacy \(s_1,\ldots,s_N\) | `(N,)`, positive finite | run-constant | required only by legacy mode | fail fast | `NO` |

The scalarizer must not access:

- observations or actions;
- current or future simulator state;
- future tracks or outcomes;
- termination cause except for ordinary downstream logging;
- curriculum arm, usefulness, scenario difficulty, or source;
- actor IDs, rule diagnostics, raw physical quantities, or policy features;
- critic values, TD errors, advantages, return estimates, or gradients;
- native environment reward;
- algorithm identity for formula selection;
- seed-dependent or performance-dependent parameter overrides.

## 5. Outputs

| Output | Meaning/type | Shape/unit/range | Ordering/mask | Consumer | Guarantees/edge cases |
|---|---|---|---|---|---|
| `scalar_reward` | scalar transition reward | scalar finite float | not applicable | PPO/TD3/SAC | deterministic |
| `mode` | selected formula identifier | scalar enum | not applicable | logs/manifest | exact configured value |
| `canonical_margins` | validated margins used in calculation | `(N,)` | schema order | diagnostics/evaluation | not policy-visible |
| `priority_contributions` | per-rule gated/shaped terms | mode-dependent tuple | rulebook order | diagnostics | finite |
| `continuous_tie_breaker` | \(T(\mathbf m)\) or legacy equivalent | scalar | not applicable | diagnostics | finite |
| `satisfaction_pattern` | \(I_1,I_2,I_3\) where applicable | `(3,)` booleans or `None` | priority order | diagnostics | not used by policy |

The original vector margins must remain available separately from the scalar
reward. Scalarization must not overwrite, discard, or reinterpret the raw
rulebook diagnostics.

## 6. Functional Requirements

### REQ-SCAL-001: Support Three Named Modes

The implementation must accept exactly:

```text
legacy_scaled_sigmoid
bounded_centered_sigmoid
bounded_satisfaction_rank
```

The default must be:

```text
bounded_satisfaction_rank
```

Unknown aliases or spelling variants fail configuration loading. Mode selection
must remain fixed for the complete scientific run.

### REQ-SCAL-002: Preserve Historical Legacy Formula

`legacy_scaled_sigmoid` must reproduce the historical formula defined in
Section 7.2, including:

- one explicit positive scale per input margin;
- an explicit vector schema identifier and ordering;
- exactly `N` scales for an ordered vector of length `N`;
- `tanh` normalization;
- uncentered sigmoid;
- exponential terms for all `N` components;
- the legacy average normalized-robustness term.

Selecting this mode without exactly `N` explicit valid scales must fail. The
legacy formula may be used with any rulebook that exposes a finite, ordered
margin vector and an explicit compatible scale set; it is not automatically
compatible with every rulebook. Existing repository values must be labeled as
repository legacy defaults and recorded with their source path, digest, schema
identifier, and repository commit.

### REQ-SCAL-003: Provide Smooth Bounded Revision

`bounded_centered_sigmoid` must:

- consume the bounded margins directly;
- use no rule-specific scales and no preliminary `tanh`;
- center each sigmoid contribution at zero;
- apply sigmoid shaping only to \(R_1,R_2,R_3\);
- retain all four margins in the continuous tie-breaker;
- use \(a=2.01\) and \(c=30.0\) in conformant core experiments.

This mode is a smooth approximation and must not be described as satisfying the
strict satisfaction-rank guarantee.

### REQ-SCAL-004: Provide Default Satisfaction-Rank Formula

`bounded_satisfaction_rank` must:

- use exact satisfaction indicators for \(R_1,R_2,R_3\);
- never gate \(R_4\);
- retain all four margins in the continuous tie-breaker;
- use \(a=2.01\);
- produce zero for the all-zero vector;
- satisfy the per-transition satisfaction-pattern dominance conditions proven
  in Section 7.5.

### REQ-SCAL-005: Disable Native Reward Mixing

The conformant scalar reward is entirely determined by the selected formula.
Native MetaDrive/ScenarioEnv reward weight must be exactly zero. A non-zero
mixing coefficient fails configuration validation for thesis-core runs.

Historical runs that used native reward mixing may be reported only as
non-conformant historical results with their exact old configuration. They are
not reproduced by silently enabling mixing in this specification.

### REQ-SCAL-006: Preserve Algorithm-Independent Semantics

Given the same canonical margin vector and scalarization configuration, PPO,
TD3, and SAC must receive numerically equal scalar rewards.

No algorithm-specific rescaling, offset, clipping, normalization, or formula
variant is permitted inside the scalarizer.

### REQ-SCAL-007: Define Downstream Return Semantics

- PPO stores the scalar reward in its rollout buffer and computes returns/GAE
  using its ordinary scalar-reward procedure.
- TD3 and SAC store the scalar reward in the replay transition.
- If a future N-step extension is enabled, its targets must sum the already
  scalarized per-step rewards:
  \[
  R_t^{(n)}=\sum_{j=0}^{n-1}\gamma^j r_{t+j}^{\mathrm{scalar}}.
  \]
- If a future PER extension is enabled, priorities for scalar baselines must
  derive from the scalar critic target/error, not directly from the margin
  vector.
- Rule-aware curriculum criticality and evaluation metrics continue to use the
  ordered vector and diagnostics rather than reconstructing safety information
  from the scalar reward.

Scalarizing an N-step accumulated margin vector instead of accumulating
per-step scalar rewards is prohibited because nonlinear scalarization and
temporal accumulation do not commute. N-step replay extensions and PER are not
implemented by this scalarization feature. `PPO.n_steps` is the rollout length,
not an N-step return definition.

### REQ-SCAL-008: Handle Terminal And Truncated Transitions Uniformly

The formula is evaluated on every valid emitted transition before storage.
Termination and time-limit truncation do not change the formula and do not add:

- collision terminal bonuses;
- success bonuses;
- time-limit corrections inside the scalarizer;
- survival rewards;
- failure penalties beyond the margins already emitted by the rulebook.

Bootstrapping masks remain the responsibility of PPO/TD3/SAC and the replay
extension, not the scalarizer.

### REQ-SCAL-009: Emit Decomposed Diagnostics

Every scalarization result must expose enough information to reconstruct the
reward from logged fields:

- selected mode;
- canonical margins;
- priority contribution per component;
- continuous contribution;
- \(a\), and \(c\) or legacy scales where applicable;
- final scalar reward;
- specification version.

These values are diagnostic/reproducibility metadata and never policy inputs.

### REQ-SCAL-010: Prevent Reward-Semantics Mixing Across Runs

A scientific run may resume only when the scalarization mode and all relevant
parameters exactly match the checkpoint/run manifest.

Changing mode, \(a\), \(c\), or legacy scales requires:

- a new run identity;
- an empty on-policy rollout state;
- an empty replay buffer unless a separate approved migration explicitly
  recomputes every stored reward from preserved margin vectors;
- separate result labeling.

Loading neural-network weights as transfer initialization may be allowed by a
separate experiment, but it must not be represented as continuation of the same
run.

## 7. Mathematical And Algorithmic Contract

### 7.1 Common notation

Define the continuous bounded-margin tie-breaker:

\[
T(\mathbf m)
=
\frac14
\sum_{k=1}^{4}m_k.
\]

Its range is:

\[
T(\mathbf m)\in[-1,0.25].
\]

The full excursion is:

\[
\Delta T_{\max}=0.25-(-1)=1.25.
\]

The asymmetric range follows from \(m_{1:3}\in[-1,0]\) and
\(m_4\in[-1,1]\).

### 7.2 Mode A — `legacy_scaled_sigmoid`

For an explicitly adapted ordered vector of length `N` and positive scales
\(s_1,\ldots,s_N\):

\[
\rho_k
=
\tanh\left(\frac{m_k}{s_k}\right),
\qquad k=1,\ldots,N.
\]

For ordered priority index \(k\in\{1,\ldots,N\}\):

\[
e_k=N-k+1,
\]

so the exponents are:

\[
(e_1,\ldots,e_N)=(N,N-1,\ldots,1).
\]

Define:

\[
\sigma(x)=\frac{1}{1+\exp(-x)}.
\]

The historical scalar reward is:

\[
\boxed{
r^{\mathrm{legacy}}(\mathbf m)
=
\sum_{k=1}^{N}
\left[
a^{e_k}\sigma(c\rho_k)
+
\frac{\rho_k}{N}
\right]
}
\]

with historical defaults:

\[
a=2.01,
\qquad
c=30.0.
\]

#### Verified and intended historical properties

- It is smooth in every margin for valid positive scales.
- Rule-specific scales control where `tanh` saturates.
- The sigmoid approximates a rule-satisfaction step.
- Every input component, including any progress component present in the
  selected legacy schema, receives an exponential sigmoid term.
- The formula is retained for repository legacy reproduction, not selected as
  the new scientific default.

#### Limitations

1. **Calibration dependence.** The scales \(s_k\) change the shape and effective
   sensitivity of every rule.
2. **Redundant normalization under bounded macro schemas.** Bounded macro
   inputs are already bounded, so the preliminary scale and `tanh` are not
   required for boundedness.
3. **Positive offset at neutral state.** For the four-component historical
   configuration, at \(\mathbf m=\mathbf 0\):
   \[
   r^{\mathrm{legacy}}(\mathbf 0)
   =
   \frac12
   \left(a^4+a^3+a^2+a\right)
   =
   15.246554505.
   \]
   With variable episode lengths, this offset can act as an implicit survival
   reward.
4. **Progress semantic mismatch.** \(R_4\) is treated by the same sigmoid
   hierarchy machinery even though the current rulebook defines it as a
   continuous subordinate objective rather than a satisfied/violated rule.
5. **No exact finite-\(c\) rank guarantee.** A finite sigmoid only approximates
   a step.
6. **Historical formula, new input contract.** Reusing the formula with the new
   bounded rulebook does not recreate the exact old experimental condition if
   the old rule definitions or ranges were different.

A global enclosure for the four-component historical configuration is:

\[
r^{\mathrm{legacy}}
\in
\left(
W\sigma(-c)-1,
W\sigma(c)+1
\right),
\]

where

\[
W=a^4+a^3+a^2+a=30.49310901.
\]

The actual reachable range is narrower and depends on the configured scales and
input bounds.

### 7.3 Transition from legacy to bounded smooth mode

The transition is justified by four independent observations.

#### Step 1: remove \(s_k\) and the preliminary `tanh`

Veer et al. scale robustness because their STL robustness values do not
necessarily lie in the theorem's required bounded interval. The current
rulebook already returns bounded margins. Therefore:

\[
\rho_k=\tanh(m_k/s_k)
\quad\longrightarrow\quad
m_k.
\]

This removes empirical rule-specific scale calibration from the new modes.

#### Step 2: center the sigmoid

The uncentered sigmoid has \(\sigma(0)=0.5\). Define instead:

\[
g_c(m)
=
2\sigma(cm)-1.
\]

Then:

\[
g_c(0)=0,
\]

and, equivalently:

\[
g_c(m)=\tanh\left(\frac{cm}{2}\right).
\]

This algebraic identity does not reintroduce the legacy rule-specific
normalization. It is only the centered logistic shape.

#### Step 3: do not sigmoid-gate progress

The new rulebook states that progress is continuous. Therefore the exponential
shaping applies only to \(R_1,R_2,R_3\), while \(m_4\) remains in the continuous
tie-breaker.

#### Step 4: reduce the gated exponent sequence

There are three satisfaction-gated rules, so the priority exponents become:

\[
(3,2,1),
\]

not the historical four-rule sequence \((4,3,2,1)\).

### 7.4 Mode B — `bounded_centered_sigmoid`

Define:

\[
g_c(m)=2\sigma(cm)-1.
\]

The smooth bounded reward is:

\[
\boxed{
r^{\mathrm{smooth}}(\mathbf m)
=
a^3g_c(m_1)
+
a^2g_c(m_2)
+
ag_c(m_3)
+
T(\mathbf m)
}
\]

with:

\[
a=2.01,
\qquad
c=30.0.
\]

#### Properties

- no rule-specific scales;
- no preliminary `tanh`;
- neutral vector reward:
  \[
  r^{\mathrm{smooth}}(\mathbf 0)=0;
  \]
- safe positive progress:
  \[
  r^{\mathrm{smooth}}(0,0,0,m_4)=m_4/4;
  \]
- differentiable with respect to all margins;
- monotone increasing in each margin while other margins are fixed;
- easier to approximate smoothly than a discontinuous gate.

Its range is:

\[
r^{\mathrm{smooth}}
\in
\left[
-\left(a^3+a^2+a\right)\tanh(c/2)-1,
\;0.25
\right].
\]

For \(a=2.01,c=30\), the lower endpoint is numerically extremely close to
\(-15.170701\).

#### Critical limitation: no satisfaction-rank guarantee

For any finite \(c\):

\[
\lim_{\epsilon\to0^+}g_c(-\epsilon)=0.
\]

Therefore a sufficiently small high-priority violation can be compensated by
lower-priority contributions.

A concrete counterexample with \(c=30\) is:

\[
\mathbf m^A=(0,0,0,-1),
\]

\[
\mathbf m^B=(0,0,-10^{-6},1).
\]

Trajectory/transition \(A\) satisfies \(R_3\), while \(B\) violates it. However,
the smooth \(R_3\) penalty for \(B\) is arbitrarily small and its progress term
is larger, so:

\[
r^{\mathrm{smooth}}(\mathbf m^B)
>
r^{\mathrm{smooth}}(\mathbf m^A).
\]

This is expected behavior for the smooth approximation and must not be reported
as a bug or as strict rank preservation.

### 7.5 Mode C — `bounded_satisfaction_rank` — default

For \(k=1,2,3\), define:

\[
I_k(\mathbf m)
=
\mathbf 1[m_k=0],
\]

after numerical canonicalization.

The default reward is:

\[
\boxed{
r^{\mathrm{rank}}(\mathbf m)
=
a^3(I_1-1)
+
a^2(I_2-1)
+
a(I_3-1)
+
T(\mathbf m)
}
\]

with:

\[
a=2.01.
\]

Equivalently, each satisfied rule contributes zero categorical penalty and each
violated rule contributes the negative priority jump:

\[
I_k-1
=
\begin{cases}
0,&m_k=0,\\
-1,&m_k<0.
\end{cases}
\]

#### Reward range

The maximum occurs for:

\[
(m_1,m_2,m_3,m_4)=(0,0,0,1),
\]

giving:

\[
r_{\max}=0.25.
\]

The minimum occurs for:

\[
(m_1,m_2,m_3,m_4)=(-1,-1,-1,-1),
\]

giving:

\[
r_{\min}
=
-\left(a^3+a^2+a\right)-1
=
-15.170701.
\]

The all-zero vector gives:

\[
r^{\mathrm{rank}}(\mathbf 0)=0.
\]

#### Satisfaction-pattern dominance proof

The tie-breaker can change by at most:

\[
\Delta T_{\max}=1.25.
\]

For \(R_3\), the categorical advantage of satisfaction over violation is \(a\).
The worst possible lower/continuous opposition is \(1.25\). Since:

\[
a=2.01>1.25,
\]

satisfying \(R_3\) dominates violating \(R_3\) when higher satisfaction
indicators are equal.

For \(R_2\), the categorical advantage is \(a^2\). The lower \(R_3\) categorical
term and the entire continuous tie-breaker can oppose it by at most:

\[
a+1.25=3.26.
\]

Since:

\[
a^2=4.0401>3.26,
\]

satisfying \(R_2\) dominates violating \(R_2\) when \(I_1\) is equal.

For \(R_1\), all lower categorical terms and the tie-breaker can oppose it by at
most:

\[
a^2+a+1.25=7.3001.
\]

Since:

\[
a^3=8.120601>7.3001,
\]

satisfying \(R_1\) dominates violating \(R_1\).

Therefore the formula preserves the total order of the three-rule satisfaction
patterns, with \(R_1\) most important, followed by \(R_2\), then \(R_3\).

#### What the proof does not establish

The formula does not guarantee continuous lexicographic severity ordering.

If two vectors have the same satisfaction pattern, the categorical terms are
equal and the decision is made by:

\[
T(\mathbf m)=\frac14\sum_km_k.
\]

Consequently, continuous improvements can trade off within a pattern. For
example, among two transitions that both violate \(R_1\), a small improvement
in \(m_1\) can be outweighed by larger changes in lower margins.

The formula also does not prove equivalence between:

\[
\max_\pi
\mathbb E_\pi
\left[
\sum_t\gamma^t
r^{\mathrm{rank}}(\mathbf m_t)
\right]
\]

and lexicographic optimization of:

\[
\left(
J_1(\pi),J_2(\pi),J_3(\pi),J_4(\pi)
\right),
\]

where:

\[
J_k(\pi)
=
\mathbb E_\pi
\left[
\sum_t\gamma^t m_k(t)
\right].
\]

The first is ordinary scalar RL with a satisfaction-rank-shaped per-step reward.
The second is the objective of a true lexicographic learner.

### 7.6 Reference pseudocode

```python
def scalarize_rulebook_margins(
    margins: tuple[float, ...],
    cfg: ScalarizationConfig,
) -> ScalarizationResult:
    m = canonicalize_and_validate_margins(margins, cfg=cfg, tolerance=1.0e-8)

    if cfg.mode == "legacy_scaled_sigmoid":
        scales = require_positive_finite_legacy_scales(cfg.legacy_rule_scales)
        if len(scales) != len(m):
            raise ScalarizationConfigurationError("Legacy scales must match vector length")
        rho = tuple(math.tanh(value / scale) for value, scale in zip(m, scales))
        exponents = tuple(range(len(m), 0, -1))
        priority_terms = tuple(
            (cfg.priority_base ** exponent)
            * stable_sigmoid(cfg.sigmoid_sharpness * value)
            for exponent, value in zip(exponents, rho)
        )
        continuous = sum(rho) / len(rho)
        reward = sum(priority_terms) + continuous
        pattern = None

    elif cfg.mode == "bounded_centered_sigmoid":
        priority_terms = (
            cfg.priority_base**3
            * centered_sigmoid(cfg.sigmoid_sharpness * m[0]),
            cfg.priority_base**2
            * centered_sigmoid(cfg.sigmoid_sharpness * m[1]),
            cfg.priority_base
            * centered_sigmoid(cfg.sigmoid_sharpness * m[2]),
        )
        continuous = sum(m) / 4.0
        reward = sum(priority_terms) + continuous
        pattern = tuple(value == 0.0 for value in m[:3])

    elif cfg.mode == "bounded_satisfaction_rank":
        pattern = tuple(value == 0.0 for value in m[:3])
        priority_terms = (
            cfg.priority_base**3 * (float(pattern[0]) - 1.0),
            cfg.priority_base**2 * (float(pattern[1]) - 1.0),
            cfg.priority_base * (float(pattern[2]) - 1.0),
        )
        continuous = sum(m) / 4.0
        reward = sum(priority_terms) + continuous

    else:
        raise ScalarizationConfigurationError(cfg.mode)

    if not math.isfinite(reward):
        raise ScalarizationEvaluationError("Non-finite scalar reward")

    return ScalarizationResult(
        reward=float(reward),
        mode=cfg.mode,
        canonical_margins=m,
        priority_contributions=priority_terms,
        continuous_tie_breaker=continuous,
        satisfaction_pattern=pattern,
    )
```

`stable_sigmoid` must use a numerically stable implementation that avoids
overflow for large-magnitude arguments while preserving the mathematical
logistic result.

## 8. Applicability, State, And Timing

### 8.1 Transition timing

The scalarizer consumes the rulebook result generated from the same explicit:

```text
pre_state -> post_state
```

transition. It runs after complete rulebook evaluation and before the transition
is inserted into the rollout or replay buffer.

### 8.2 State and reset

The scalarizer is pure and stateless:

- no history;
- no timers;
- no reset mutation;
- no episode-dependent scaling;
- no running reward normalization;
- no percentile estimation;
- no online calibration.

A framework-required `reset()` method, if present, must be a no-op.

### 8.3 Termination and truncation

The terminal or truncated transition is scalarized normally. The next episode
starts with the same immutable scalarization configuration. No state crosses the
episode boundary because the scalarizer owns no state.

### 8.4 Reward normalization wrappers

Automatic reward normalization, reward clipping, or return-based rescaling
outside the scalarizer is disabled in the core comparison unless separately
approved and applied identically to all relevant scalar baselines.

Observation normalization does not modify this requirement.

## 9. Configuration

```yaml
scalarization:
  specification_id: SCAL-V1.0
  version: 1.0
  mode: bounded_satisfaction_rank
  vector_schema_id: rulebook_v2_macro_v4

rulebook:
  implementation_family: v2
  specification_id: RULEBOOK-V4.7
  version: 4.7-final-implementation-complete

  priority_base: 2.01
  numerical_tolerance: 1.0e-8
  native_environment_reward_weight: 0.0

  sigmoid:
    sharpness: 30.0

  legacy:
    vector_schema_id: null
    rule_scales: null
```

| Field | Type | Default | Valid range | Meaning | Required | Frozen for experiments |
|---|---:|---:|---:|---|---|---|
| `scalarization.specification_id` | string | `SCAL-V1.0` | exact value | contract identity | `YES` | `YES` |
| `scalarization.version` | string | `1.0` | exact value | contract version | `YES` | `YES` |
| `scalarization.mode` | enum | `bounded_satisfaction_rank` | three named modes | formula selection | `YES` | `YES` |
| `scalarization.vector_schema_id` | string | `rulebook_v2_macro_v4` | explicit adapter schema | margin order and shape | `YES` | `YES` |
| `rulebook.implementation_family` | string | `v2` | selected implementation family | upstream rulebook adapter | `YES` | `YES` |
| `rulebook.specification_id` | string | `RULEBOOK-V4.7` | exact approved identifier | upstream rulebook contract | `YES` | `YES` |
| `rulebook.version` | string | `4.7-final-implementation-complete` | exact selected version | upstream rulebook contract | `YES` | `YES` |
| `scalarization.priority_base` | float | `2.01` | finite \(>1\) | priority separation | `YES` | `YES` |
| `scalarization.numerical_tolerance` | float | `1e-8` | exact inherited value for conformant runs | numerical canonicalization | `YES` | `YES` |
| `scalarization.native_environment_reward_weight` | float | `0.0` | exact zero for conformant runs | native reward mixing | `YES` | `YES` |
| `scalarization.sigmoid.sharpness` | float | `30.0` | finite \(>0\) | sigmoid steepness | sigmoid modes only | `YES` |
| `scalarization.legacy.vector_schema_id` | string or null | `null` | explicit legacy schema | legacy margin order | legacy mode only | `YES` |
| `scalarization.legacy.rule_scales` | list[float] or null | `null` | exactly `N` positive finite values | repository legacy normalization | legacy mode only | `YES` |

Validation rules:

- all supplied numeric values must be finite;
- bounded modes require the four-margin macro schema and reject non-null legacy
  scales or a legacy schema identifier;
- legacy mode requires an explicit schema identifier and exactly `N` positive
  scales;
- conformant core experiments require \(a=2.01\);
- changing \(a\) requires a new dominance proof and approved specification or
  ADR;
- conformant sigmoid experiments require \(c=30.0\);
- changing \(c\) is an explicit ablation and must have a separate run identity;
- no field may be overridden per algorithm, scenario, arm, source, or seed.
- the default preset is implementation family `v2`, specification version
  `4.7-final-implementation-complete`, and mode `bounded_satisfaction_rank`;
- selecting v4.6 with a bounded mode is valid only when its exact four-margin
  contract is active and the run is labeled v4.6.

## 10. Errors, Logging, And Diagnostics

### 10.1 Fatal errors

The following fail before storing the transition:

- wrong margin shape or order metadata;
- NaN or infinity;
- range violation beyond \(\varepsilon_0\);
- unknown mode;
- missing/invalid legacy scales;
- non-zero native reward weight in a conformant run;
- non-finite final reward;
- mode/config mismatch on checkpoint resume;
- attempt to switch mode with non-empty rollout or replay data;
- duplicate scalarization with different modes for the same transition.

### 10.2 No fallback policy

There is no fallback from one scalarization mode to another. A legacy
configuration error must not silently select the default. A numerical problem in
the smooth mode must not silently select the step mode.

### 10.3 Required run-level logging

The run manifest must include:

```text
scalarization_specification_id
scalarization_version
scalarization_mode
scalarization_vector_schema_id
priority_base
sigmoid_sharpness_if_applicable
legacy_rule_scales_if_applicable
legacy_scale_source_path_if_applicable
legacy_scale_source_digest_if_applicable
numerical_tolerance
native_environment_reward_weight
rulebook_implementation_family
rulebook_specification_id
rulebook_specification_version
algorithm
discount_factor
n_step_configuration_if_enabled
per_configuration_if_enabled
reward_normalization_configuration
```

### 10.4 Required transition/episode diagnostics

At minimum, evaluation artifacts must permit reporting:

- scalar reward;
- the complete ordered margin vector and its schema identifier;
- three satisfaction indicators for bounded modes;
- categorical/priority contribution;
- continuous tie-breaker;
- per-rule peak and sum metrics from the rulebook;
- scalar return and vector returns separately.

Logging frequency and storage format belong to the ExecPlan, but the information
must be recoverable for representative evaluation runs.

## 11. Reproducibility And Compatibility

### 11.1 Determinism

For identical canonical margins and serialized configuration, output must be
deterministic and independent of seed, algorithm, device, and scenario source,
within an absolute reference tolerance of \(10^{-6}\).

### 11.2 Legacy compatibility

The legacy mode preserves the historical mathematical formula. Codex must inspect
the repository to identify:

- current implementation path;
- existing field names;
- exact stored scale values;
- whether native reward mixing is currently present;
- stable sigmoid implementation;
- existing tests.

Repository facts must be mapped to this public contract without silently
changing the formulas. Any discrepancy is reported before implementation.

The current repository configuration may establish a `repository legacy
reproduction`, but it does not by itself establish an exact historical paper
reproduction. A claim of exact historical reproduction requires a verifiable
artifact containing the original scale values, formula, rulebook version,
configuration, and code/commit.

### 11.3 Checkpoint and replay compatibility

A change of scalarization mode changes reward semantics.

The checkpoint/run compatibility identity must include:

```text
rulebook_specification_id
rulebook_version
scalarization_specification_id
scalarization_version
scalarization_mode
priority_base
sigmoid_sharpness
numerical_tolerance
legacy_vector_schema_id
legacy_rule_scales
legacy_scale_source_path
legacy_scale_source_digest
native_environment_reward_weight
```

These fields must be recorded in both the checkpoint manifest and run metadata.
Resume must compare them before loading the learner, optimizer, replay buffer,
or rollout state. A mismatch fails same-run resume.

- Existing replay buffers containing only scalar rewards are not portable across
  modes.
- Replay buffers that also retain the complete canonical margin vector could be
  migrated only through a separately approved deterministic re-scalarization
  procedure; no such migration is implemented by this specification.
- PPO rollout buffers are never migrated.
- A model checkpoint may be loaded as transfer initialization in a new run, but
  optimizer, replay, rollout, and run identity from the old scientific run must
  not be continued under incompatible reward semantics.
- Results from different modes must not be pooled as one experimental
  condition.

### 11.4 Rulebook v4.7 compatibility

The CTRV amendment changes conflict-zone occupancy prediction but explicitly
preserves:

- the four macro-rule order;
- the margin ranges;
- the learner output interface;
- the maximum macro aggregation;
- the definition of progress.

Therefore this scalarization contract is mathematically compatible with v4.6 and
v4.7 when the exact four-margin macro contract is active, while results remain
labeled by the exact rulebook version because the underlying margins may differ
on curved scenarios.

## 12. Acceptance Criteria

### AC-SCAL-001: Mode Registry And Default

- Given: a valid configuration without an explicit mode;
- When: scalarization configuration is loaded;
- Then: the selected mode is `bounded_satisfaction_rank`.
- Given: the default scientific configuration;
- Then: the selected upstream contract is Rulebook implementation family v2,
  specification version `4.7-final-implementation-complete`.
- Given: an unknown mode;
- Then: configuration loading fails.
- Related requirements: `REQ-SCAL-001`.

### AC-SCAL-002: Input Validation And Canonicalization

- Given: valid boundary vectors and values within \(10^{-8}\) numerical
  tolerance;
- When: canonicalization runs;
- Then: near-zero and boundary overshoots are canonicalized exactly.
- Given: NaN, infinity, wrong shape, or an out-of-range value beyond tolerance;
- Then: scalarization fails before transition storage.
- Given: legacy mode;
- Then: the adapter requires a declared vector schema, finite ordered margins,
  and exactly one positive finite scale per margin, without applying bounded
  macro clipping.
- Related requirements: `REQ-SCAL-004`, `REQ-SCAL-008`.

### AC-SCAL-003: Legacy Neutral Reference

- Given: legacy mode with the four-component historical schema, four valid
  scales, \(a=2.01,c=30\), and
  \(\mathbf m=(0,0,0,0)\);
- When: reward is computed;
- Then:
  \[
  r=15.246554505
  \]
  within \(10^{-6}\).
- Related requirements: `REQ-SCAL-002`.

### AC-SCAL-004: Legacy Scale Requirement

- Given: legacy mode with missing, non-finite, non-positive, or length-mismatched
  scales, or without an explicit vector schema identifier;
- When: configuration is loaded;
- Then: loading fails without selecting another mode.
- Related requirements: `REQ-SCAL-002`.

### AC-SCAL-005: Smooth Neutral And Progress Behavior

- Given: smooth mode;
- Then:
  \[
  r(0,0,0,0)=0,
  \]
  \[
  r(0,0,0,1)=0.25,
  \]
  \[
  r(0,0,0,-1)=-0.25.
  \]
- Related requirements: `REQ-SCAL-003`.

### AC-SCAL-006: Smooth Mode Is Not Misclassified As Rank-Preserving

- Given:
  \[
  \mathbf m^A=(0,0,0,-1),
  \qquad
  \mathbf m^B=(0,0,-10^{-6},1);
  \]
- When: smooth rewards are computed with \(a=2.01,c=30\);
- Then:
  \[
  r^{\mathrm{smooth}}(\mathbf m^B)
  >
  r^{\mathrm{smooth}}(\mathbf m^A),
  \]
  and documentation/diagnostics identify the mode as approximate.
- Related requirements: `REQ-SCAL-003`.

### AC-SCAL-007: Default Reference Values

For default mode:

\[
r(0,0,0,0)=0,
\]

\[
r(0,0,0,1)=0.25,
\]

\[
r(0,0,0,-1)=-0.25,
\]

\[
r(0,0,-1,1)=-2.01,
\]

\[
r(0,-1,0,1)=-4.0401,
\]

\[
r(-1,0,0,1)=-8.120601,
\]

\[
r(-1,-1,-1,-1)=-15.170701.
\]

All values must agree within \(10^{-6}\).

- Related requirements: `REQ-SCAL-004`.

### AC-SCAL-008: Exhaustive Satisfaction-Pattern Dominance

- Given: every pair of the \(2^3\) satisfaction patterns and boundary/extreme
  continuous margin combinations;
- When: the first differing rule is identified;
- Then: the vector satisfying that rule has strictly greater default scalar
  reward, independent of all lower indicators and valid margin values.
- Related requirements: `REQ-SCAL-004`.

### AC-SCAL-009: Same-Pattern Monotonicity

- Given: two valid vectors with identical satisfaction indicators and all
  margins equal except one margin that is larger in vector \(A\);
- When: default or smooth reward is evaluated;
- Then: \(r(A)>r(B)\).
- For legacy mode the same property is tested under fixed positive scales.
- Related requirements: `REQ-SCAL-002`, `REQ-SCAL-003`, `REQ-SCAL-004`.

### AC-SCAL-010: Progress Is Never Categorically Gated In New Modes

- Given: bounded smooth or default mode and fixed \(m_1,m_2,m_3\);
- When: \(m_4\) moves continuously through negative, zero, and positive values;
- Then: its reward contribution is exactly \(m_4/4\), with no sigmoid or
  indicator jump.
- Related requirements: `REQ-SCAL-003`, `REQ-SCAL-004`.

### AC-SCAL-011: Algorithm Parity

- Given: one validated margin vector;
- When: the PPO, TD3, and SAC reward adapters process it under the same mode;
- Then: stored scalar rewards are equal within \(10^{-6}\).
- Related requirements: `REQ-SCAL-006`.

### AC-SCAL-012: N-Step Order Of Operations

- If a future N-step extension is enabled, given a sequence of margin vectors
  and \(n>1\);
- When: an N-step target is built;
- Then: each vector is scalarized first and the discounted scalar rewards are
  summed.
- A test that scalarizes an accumulated vector must produce a distinguishable
  result for at least one nonlinear-mode fixture and be rejected as the wrong
  path.
- This criterion is conditional and is not an implementation acceptance gate
  for the current PPO/TD3/SAC baseline paths.
- Related requirements: `REQ-SCAL-007`.

### AC-SCAL-013: Terminal And Truncated Transitions

- Given: identical margins on non-terminal, terminal, and truncated transitions;
- When: scalarization runs;
- Then: the scalar reward is identical, with bootstrapping handled downstream.
- Related requirements: `REQ-SCAL-008`.

### AC-SCAL-014: No Native Reward Mixing

- Given: non-zero native reward weight;
- When: a conformant configuration is loaded;
- Then: loading fails.
- Given: different native environment rewards and identical margin vectors under
  a valid configuration;
- Then: scalar rewards are identical.
- Related requirements: `REQ-SCAL-005`.

### AC-SCAL-015: Resume Compatibility

- Given: a checkpoint or replay state created with one mode;
- When: resume is requested with a different mode or parameter set;
- Then: same-run resume fails before learning continues.
- Related requirements: `REQ-SCAL-010`.

### AC-SCAL-016: Diagnostics Reconstruct Reward

- Given: any valid scalarization result;
- When: the reward is recomputed from logged canonical margins, priority
  contributions, continuous term, and configuration;
- Then: it matches the stored scalar reward within \(10^{-6}\).
- Related requirements: `REQ-SCAL-009`.

## 13. Required Validation Categories

| Category | Requirement |
|---|---|
| nominal and boundary behavior | `REQUIRED` |
| invalid and incomplete inputs | `REQUIRED` |
| masks, padding, state, reset, and update order | state/reset `REQUIRED`; masks/padding `NOT APPLICABLE` |
| termination and truncation | `REQUIRED` |
| deterministic seeds and reproducibility | `REQUIRED` |
| numerical stability, NaN, and infinity | `REQUIRED` |
| compatibility and migration | `REQUIRED` |
| absence of future and privileged information | `REQUIRED` |
| upstream, downstream, and end-to-end integration | `REQUIRED` |
| regressions for known bugs | `REQUIRED` when discovered |
| N-step and PER implementation | `CONDITIONAL`; required only if the corresponding future extension is enabled |
| full three-mode training ablation | `OPTIONAL`, not an implementation acceptance criterion |

Exact test modules, repository paths, commands, and fixtures belong to the
ExecPlan created by Codex after approval.

## 14. Traceability

| Requirement | Acceptance criteria | Scientific source or approved decision |
|---|---|---|
| `REQ-SCAL-001` | `AC-SCAL-001` | user decision 2026-07-17 |
| `REQ-SCAL-002` | `AC-SCAL-003`, `004`, `009` | historical implementation and thesis slides; repository legacy provenance boundary in §11.2 |
| `REQ-SCAL-003` | `AC-SCAL-005`, `006`, `009`, `010` | Veer sigmoid approximation plus project bounded-input adaptation |
| `REQ-SCAL-004` | `AC-SCAL-007`, `008`, `009`, `010` | Veer step construction plus project proof for current ranges |
| `REQ-SCAL-005` | `AC-SCAL-014` | Rulebook v4.6 native reward disabled |
| `REQ-SCAL-006` | `AC-SCAL-011` | approved algorithm-comparison fairness decision |
| `REQ-SCAL-007` | `AC-SCAL-012` | scalar critic/replay semantics; conditional future N-step/PER direction |
| `REQ-SCAL-008` | `AC-SCAL-013` | rulebook transition contract |
| `REQ-SCAL-009` | `AC-SCAL-016` | reproducibility and observability decision |
| `REQ-SCAL-010` | `AC-SCAL-015` | scientific-condition compatibility decision |

## 15. Scientific Comparison And Alternatives Reviewed

### 15.1 Censi et al. rulebooks

Censi et al. define a rulebook as a preordered collection of violation metrics
that induces a preorder over realizations. The hierarchy is part of the
behavioral specification and does not inherently require a scalar reward.

Implication for this project:

- the vector rulebook is the primary specification;
- scalarization is an external adapter for scalar RL baselines;
- a scalar reward must not be confused with the rulebook itself;
- retaining the vector and raw diagnostics is mandatory.

### 15.2 Veer et al. rank-preserving reward

Veer et al. analytically separate satisfaction ranks with exponentially
increasing terms and use average robustness to distinguish equal-rank
trajectories. Their theorem concerns a step-function construction over bounded
whole-trajectory robustness. Their sigmoid version is introduced to permit
gradient-based continuous optimization.

Benefits transferred to this project:

- analytical priority separation rather than arbitrary weighted-sum tuning;
- explicit satisfaction pattern;
- average continuous robustness as a tie-breaker;
- clear distinction between exact step and smooth sigmoid forms.

Limits on transfer:

- their object is trajectory robustness, not immediate model-free RL reward;
- their planner uses a two-stage receding-horizon optimization method;
- their sigmoid is an approximation of the theorem's step;
- their same hyperparameters across scenarios demonstrate robustness in their
  evaluated setting, not universal absence of tuning;
- their scale constants are selected according to robustness ranges;
- their guarantees do not automatically establish lexicographic expected-return
  optimality for PPO, TD3, or SAC.

### 15.3 Skalse et al. lexicographic RL

Skalse et al. define lexicographic RL directly over separate expected discounted
reward objectives and provide value-based and policy-gradient algorithm
families with convergence results under their assumptions.

Implication:

- a true lexicographic learner is scientifically distinct from every scalar mode
  in this specification;
- the default scalarizer is a strong satisfaction-rank baseline, not a
  replacement for the lexicographic algorithm;
- the thesis comparison remains meaningful even when the scalar baseline needs
  no empirical weight sweep.

### 15.4 Thresholded lexicographic ordering

Thresholded lexicographic methods retain lower-priority actions or policies once
a higher-priority objective is within a specified threshold. They can improve
practical behavior when critic estimates are noisy or strict equality prevents
progress, but they introduce thresholds with semantic and algorithmic impact.

Decision:

- thresholding belongs to the future lexicographic learner specification;
- it is not implemented by changing scalarizer gates;
- the numerical \(\varepsilon_0\) in this document is not a thresholded
  lexicographic slack.

### 15.5 Weighted sums

A conventional weighted sum:

\[
r=\sum_kw_km_k
\]

is simple and smooth but requires weights to express trade-offs. With continuous
objectives and no minimum improvement resolution, finite weights cannot in
general guarantee that every arbitrarily small improvement of a higher
objective dominates all lower objectives.

Decision:

- no ordinary weighted-sum mode is added;
- the legacy and new exponential modes are retained because they represent
  satisfaction-rank structure, not arbitrary continuous exchange rates.

### 15.6 Chebyshev and other Pareto scalarizations

Weighted Chebyshev, augmented Chebyshev, \(p\)-means, utility functions, and
reference-point methods are useful when the goal is to explore or select points
on a Pareto front. They require weights, a reference point, an exponent, an
augmentation coefficient, or another preference parameter.

Decision:

- they do not directly encode the fixed total priority hierarchy required here;
- they are outside the scalar-baseline core;
- they may be studied only under a separate multi-objective experimental
  question.

### 15.7 Constrained RL

CPO and related constrained-RL methods optimize a performance objective subject
to expected-cost constraints and may provide near-constraint-satisfaction
properties during policy updates under their assumptions.

Decision:

- constraints require approved thresholds;
- one constrained performance problem is not equivalent to the four-level
  total hierarchy;
- CPO is not a drop-in scalarization mode and is outside this specification.

### 15.8 Non-Archimedean scalarization

Infinite and infinitesimal weights can express exact lexicographic relations in
special numerical frameworks.

Decision:

- ordinary PyTorch/SB3 floating-point arithmetic does not provide such numbers;
- introducing a non-standard numerical system would materially change the
  optimization stack;
- it is not selected for this thesis implementation.

### 15.9 Internal maximum aggregation is a separate issue

The rulebook currently defines macro-costs using maximum aggregation among
subcomponents. That choice affects the meaning of \(m_2\) and \(m_3\) before
scalarization.

This specification:

- neither endorses the maximum as the unique mathematical aggregation nor
  replaces it;
- bounded modes receive only the four finalized macro margins; the legacy mode
  may receive another explicitly adapted ordered schema;
- records that boundedness makes the maximum numerically stable, while semantic
  comparability depends on the rulebook's thresholds and normalization;
- leaves any change to internal aggregation to a new approved rulebook version.

## 16. Limitations And Approved Decisions

### 16.1 Intentional limitations

1. The default formula preserves satisfaction-pattern hierarchy only.
2. It does not preserve continuous lexicographic severity ordering.
3. It does not guarantee lexicographic ordering of expected discounted returns.
4. The discontinuous gate gives the same categorical jump to every negative
   violation, while severity enters only through the continuous tie-breaker.
5. The smooth mode retains a sharpness parameter and no strict rank guarantee.
6. The legacy mode retains scales, an offset, and progress gating.
7. All scalar modes can hide vector information if only scalar returns are
   reported; therefore vector evaluation remains mandatory.
8. None of the modes guarantees safe exploration or safe rollouts.
9. Discounting can make temporally distant violations contribute less.
10. Implementing all modes increases code/test surface but does not mandate
    full training ablations.
11. N-step replay extensions and PER are conditional future compatibility
    constraints, not implementation scope for this scalarization feature.
12. The repository can establish a legacy configuration reproduction, but an
    exact historical paper reproduction requires a separately verifiable
    artifact.

### 16.2 Approved design decisions

| ID | Decision | Status | Evidence |
|---|---|---|---|
| `DEC-SCAL-001` | implement all three modes | `APPROVED` | user decision 2026-07-17 |
| `DEC-SCAL-002` | default to `bounded_satisfaction_rank` | `APPROVED` | user decision 2026-07-17 |
| `DEC-SCAL-003` | retain legacy mode for compatibility | `APPROVED` | user decision 2026-07-17 |
| `DEC-SCAL-004` | retain smooth bounded mode as configurable alternative | `APPROVED` | user decision 2026-07-17 |
| `DEC-SCAL-005` | no rule-specific scales in new bounded modes | `APPROVED` | user decision 2026-07-17 |
| `DEC-SCAL-006` | no categorical gate on progress in new modes | `APPROVED` | Rulebook v4.6 semantics and user discussion |
| `DEC-SCAL-007` | no native reward mixing in core | `APPROVED` | Rulebook v4.6 contract |
| `DEC-SCAL-008` | optional, not mandatory, full mode ablations | `APPROVED` | scope recommendation accepted through this review candidate |
| `DEC-SCAL-009` | scalarization is downstream of every selected rulebook; bounded modes require the v4.6/v4.7 four-margin macro contract | `APPROVED` | user clarification in review on 2026-07-17 |
| `DEC-SCAL-010` | the existing `scalar_reward` interface remains, with exactly one configured mode active per run | `APPROVED` | user clarification in review on 2026-07-17 |
| `DEC-SCAL-011` | legacy mode accepts an explicit ordered vector of length `N` and exactly `N` scales, with exponents `N,...,1` | `APPROVED` | user clarification in review on 2026-07-17 |
| `DEC-SCAL-012` | N-step/PER rules are conditional future constraints and are not implemented here | `APPROVED` | user clarification in review on 2026-07-17 |
| `DEC-SCAL-013` | checkpoint/run metadata must contain exact rulebook and scalarization identity and reject incompatible resume before state load | `APPROVED` | user clarification in review on 2026-07-17 |
| `DEC-SCAL-014` | distinguish repository legacy reproduction from exact historical experiment reproduction | `APPROVED` | user clarification in review on 2026-07-17 |

No material scientific decision is intentionally left open. Repository mappings
remain implementation facts for Codex to verify.

## 17. References

### Project documents

- [P1] `docs/specifications/rulebook_v4.6_specification.md`,
  `4.6-final-implementation-complete`.
- [P2] `docs/specifications/rulebook_v4.7_specification.md`,
  `4.7-final-implementation-complete`, integrating the approved causal CTRV
  amendment.
- [P3] `docs/specifications/automatic_curriculum_learning_v1_specification.md`.
- [P4] thesis meeting slides dated 2026-07-03, historical scalarization section.
- [P5] `docs/engineering_workflow.md`.
- [P6] `docs/templates/specification_template.md`.
- [P7] `docs/project_index.md`.

### Scientific literature

- [R1] A. Censi, K. Slutsky, T. Wongpiromsarn, D. Yershov, S. Pendleton,
  J. Fu, and E. Frazzoli, “Liability, Ethics, and Culture-Aware Behavior
  Specification using Rulebooks,” ICRA 2019.
  https://arxiv.org/abs/1902.09355
- [R2] S. Veer, K. Leung, R. Cosner, Y. Chen, P. Karkus, and M. Pavone,
  “Receding Horizon Planning with Rule Hierarchies for Autonomous Vehicles,”
  ICRA 2023.
  https://arxiv.org/abs/2212.03323
- [R3] J. Skalse, L. Hammond, C. Griffin, and A. Abate,
  “Lexicographic Multi-Objective Reinforcement Learning,” IJCAI 2022.
  https://www.ijcai.org/proceedings/2022/0476.pdf
- [R4] A. Tercan and V. S. Prabhu,
  “Thresholded Lexicographic Ordered Multiobjective Reinforcement Learning,”
  ECAI 2024.
  https://arxiv.org/abs/2408.13493
- [R5] J. Achiam, D. Held, A. Tamar, and P. Abbeel,
  “Constrained Policy Optimization,” ICML 2017.
  https://proceedings.mlr.press/v70/achiam17a.html
- [R6] D. M. Roijers, P. Vamplew, S. Whiteson, and R. Dazeley,
  “A Survey of Multi-Objective Sequential Decision-Making,” JAIR 48, 2013.
  https://www.jair.org/index.php/jair/article/view/10836

## 18. Implementation Handoff Checklist

Before setting `Status: APPROVED`:

- [x] Scope, exclusions, and optional behavior are explicit.
- [x] Inputs and outputs define type, shape, range, and ordering.
- [x] Prohibited future, privileged, and diagnostic-only data is listed.
- [x] All three formulas are explicit.
- [x] Default, parameters, and invalid configuration behavior are explicit.
- [x] Numerical canonicalization and singular/error behavior are explicit.
- [x] State, timing, reset, termination, and truncation are explicit.
- [x] PPO, TD3, SAC, N-step, PER, and ACL interactions are explicit.
- [x] Diagnostics, reproducibility, compatibility, and migration are explicit.
- [x] Every core requirement maps to an objective acceptance criterion.
- [x] Required validation categories are selected.
- [x] Literature results and project adaptations are distinguished.
- [x] No material scientific decision remains open.
- [x] Legacy vector arity, schema, scale provenance boundary, and repository-vs-historical reproduction boundary are explicit.
- [x] N-step/PER are explicitly conditional future constraints and excluded from this implementation scope.
- [x] Codex verifies exact legacy implementation paths, values, and config names.
- [x] Codex creates or updates the ExecPlan and protected tests.
- [x] The user explicitly approves this complete specification.
- [x] Status becomes `APPROVED`.
- [x] The file is moved to `docs/specifications/` without `_UNDER_REVIEW`.
- [x] The repository-assigned ADR is approved and linked.
- [x] `project_index.md` registers the exact authoritative path and version.

### Proposed canonical filename after approval

```text
docs/specifications/rulebook_scalarization_v1.0_specification.md
```

### Proposed review filename

```text
incoming/rulebook_scalarization_v1.0_specification_UNDER_REVIEW.md
```

### Proposed `project_index.md` entry after approval

```markdown
| Rulebook scalarization v1.0 | `specifications/rulebook_scalarization_v1.0_specification.md`; ID `SCAL-V1.0`, version `1.0` | `AUTHORITATIVE`; explicit user approval `2026-07-17` | `implementation/rulebook_scalarization_v1.0_exec_plan.md` | Implementation validation and final reconciliation required |
```

The index becomes authoritative only after Codex records the actual repository
path and approval evidence.

## 19. Approval Record

- Approved by: `user`
- Approval date: `2026-07-17`
- Approval evidence: `User message: "se non resta altro da dire sì" in response to the complete updated specification review.`
- Approval notes: `Approval covers the default Rulebook implementation family v2/specification 4.7 configuration, all three modes, generic legacy vectors and scales, scalar_reward coexistence, conditional N-step/PER semantics, checkpoint compatibility, and repository-vs-historical scale provenance boundary.`
- Repository path: `docs/specifications/rulebook_scalarization_v1.0_specification.md`
- Project index updated: `YES` (`docs/project_index.md`, 2026-07-17)
