# ADR-011: Configurable Rulebook Scalarization For Scalar Baselines

- Status: `Approved`
- Date: `2026-07-17`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-17`
- Supersedes: `NONE`
- Affected specifications: `docs/specifications/rulebook_scalarization_v1.0_specification.md`, `SCAL-V1.0`, version `1.0`
- Affected ExecPlans: `docs/implementation/rulebook_scalarization_v1.0_exec_plan.md`

## Context

The repository has a rulebook monitor that produces ordered margins and scalar
baseline learners that consume one `scalar_reward` per transition. The previous
v1 reward manager combines rulebook margins, tanh normalization, sigmoid shaping,
and optional native reward mixing. Rulebook implementation family v2 instead
produces a complete four-macro-margin result and currently preserves the native
environment reward at the wrapper boundary.

Scalar baselines require a deterministic adapter while the vector rulebook and
future lexicographic learners must remain independent. Reward semantics must be
recorded in run/checkpoint artifacts so incompatible scientific conditions cannot
be resumed or pooled silently.

## Decision

1. Scalarization is always downstream of the selected rulebook and every mode
   receives an ordered margin vector.
2. The default scientific configuration is Rulebook implementation family v2,
   Rulebook specification version `4.7-final-implementation-complete`, and
   `bounded_satisfaction_rank`.
3. `bounded_centered_sigmoid` and `bounded_satisfaction_rank` require the exact
   four-macro contract shared by authoritative Rulebook v4.6 and v4.7:
   `m1:m3 ∈ [-1, 0]`, `m4 ∈ [-1, 1]`.
4. `legacy_scaled_sigmoid` may consume any explicitly adapted finite ordered
   vector of length `N`, with exactly `N` positive finite scales and exponents
   `(N, ..., 1)`. The adapter must declare the vector schema and ordering.
5. The existing `scalar_reward` interface remains the learner-facing output,
   but exactly one configured scalarization mode supplies it per run. Native
   environment reward may remain diagnostic and has zero weight in conformant
   core runs.
6. N-step replay and PER are conditional future compatibility constraints, not
   implementation scope for this feature. If later enabled, N-step targets sum
   already scalarized rewards and PER priorities use scalar critic TD error.
   `PPO.n_steps` is rollout length, not an N-step return.
7. Checkpoint and run metadata must record exact rulebook and scalarization
   identity, including mode, parameters, schema, legacy scales/provenance, and
   native reward weight. Same-run resume must reject mismatches before loading
   learner, optimizer, replay, or rollout state. Model-only transfer is a new
   run identity.
8. Repository legacy reproduction and exact historical paper reproduction are
   distinct claims. Current configuration values establish only repository
   provenance unless an external artifact verifies the historical experiment.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Use only the v2 four-margin scalarizer | Simple bounded contract | Cannot reproduce compatible legacy rulebook vectors | Legacy compatibility is required |
| Keep the old scalar manager as the only producer | Minimal code change | Does not provide bounded modes or v2 scalar reward delivery | Violates `SCAL-V1.0` |
| Deliver native and scalar rewards concurrently to learners | Supports comparison in one runtime path | Creates ambiguous learner semantics and invalid pooled results | One configured scalar reward is required |
| Implement N-step/PER in this change | Complete future replay behavior | Expands scope and architecture without an approved replay specification | Deferred as conditional compatibility semantics |

## Consequences

The runtime wiring must connect v2 rulebook results to the configured scalarizer
before transition storage while preserving the original vector and diagnostics.
Configuration and checkpoint schemas gain scalarization identity fields. Existing
checkpoints/replay states without matching semantics cannot continue the same run,
but model weights may be used for a separately labeled transfer run.

The legacy mode remains calibration-dependent and does not prove exact historical
paper reproduction without external evidence. The bounded default preserves
satisfaction-pattern dominance per transition, not lexicographic expected-return
optimality or safe exploration.

## Validation And Traceability

Affected requirements and acceptance criteria: `REQ-SCAL-001` through
`REQ-SCAL-010`; `AC-SCAL-001` through `AC-SCAL-016`. Mandatory validation covers
mode/formula references, boundary and invalid inputs, deterministic numerical
behavior, vector preservation, PPO/TD3/SAC parity, terminal/truncation handling,
resume rejection, and representative runtime smoke behavior. N-step/PER tests
are conditional on those future extensions being enabled.

## Approval Record

- Approved by: `user`
- Approval evidence: User message `"se non resta altro da dire sì"` on 2026-07-17 approving the complete updated specification.
- Notes: Approval covers the complete `SCAL-V1.0` text and the clarifications integrated before approval.
