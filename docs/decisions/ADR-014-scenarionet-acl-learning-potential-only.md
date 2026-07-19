# ADR-014: ScenarioNet ACL Learning-Potential-Only Usefulness

- Status: `Approved`
- Date: `2026-07-19`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-19`
- Amends: `docs/specifications/automatic_curriculum_learning_v1_specification.md` §28
- Affected specification: ACL v1, ScenarioNet scalar core only
- Affected ExecPlan: `docs/implementation/scalar_autonomous_driving_pipeline_audit_exec_plan.md`

## Context

ACL v1 §11 defines final usefulness as a Rulebook-first lexicographic pair.
The approved ScenarioNet scalar core instead selects, replaces, replays, and
feeds back scenarios solely from algorithm-specific learning potential. The
ScenarioNet v1.1 dataset contract also disables mutation.

## Decision

For the selected ScenarioNet scalar ACL core:

1. MAB selects exactly the six frozen ScenarioNet arms A0–A5.
2. Generate/Exploit, scenario replay, staleness, warm-up, capacity,
   replacement, checkpoint, and resume remain in scope.
3. `ScenarioUsefulness.value = LP_alg_s`; its ordering, buffer replacement,
   replay probabilities, and MAB feedback use this value only.
4. `C_safe`, Rulebook margins, Rulebook criticality, safety rank, and every
   Rulebook-derived value are diagnostic-only and must not influence
   `ScenarioUsefulness.value` or any curriculum decision.
5. Scenario mutation, mutation-generated children, and mutation configuration
   are prohibited for this core.
6. ACL v1 §12 algorithm-specific learning-potential formulas remain required.
   This ADR does not approve the current critic/value-loss proxy as a permanent
   substitute.

## Consequences

Existing scenario-ACL artifacts that encode Rulebook-first rank/order are not
compatible with the selected core and must be restarted or migrated explicitly
before use. Rulebook diagnostics may still be logged and analysed. No
transition-replay behavior changes: `n_steps=3` remains the project core and
the supported domain remains `{1,3}`.

## Approval Record

- Approved by: user
- Approval evidence: user message “va bene fai pure. Anche la mutation rimane
  off, non la usiamo.” on 2026-07-19.
