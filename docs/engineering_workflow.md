# Specification-Driven Engineering Workflow

## Purpose

This lightweight process controls how scientific requirements become tested,
traceable thesis software. It defines process, not feature behavior.

## Document Roles

- A specification defines required scientific and observable behavior. It is
  authoritative only after explicit user approval and registration in
  `project_index.md`.
- An ExecPlan maps an approved specification to the current repository, test
  strategy, milestones, findings, and validation. Its required format is
  `.agent/PLANS.md`.
- An ADR records an approved decision with scientific, architectural,
  compatibility, or experimental consequences. It does not approve itself.
- `project_index.md` identifies authoritative, candidate, missing, and
  superseded documents without inferring authority from filenames.

## Source Precedence

Apply explicit user approvals, the indexed authoritative specification,
applicable approved ADRs, existing public contracts, `AGENTS.md`, verified
repository conventions, and internal preferences, in that order.

## Lifecycle

### 1. Scientific Definition

The user approves a versioned specification only after scope, formulas,
interfaces, defaults, fallbacks, edge cases, compatibility, and acceptance
criteria are sufficiently resolved. Codex cannot approve scientific changes.

### 2. Repository Handoff

Store the approved specification under `docs/specifications/` and register its
exact path, version, approval evidence, related ADRs, and implementation status
in `project_index.md`.

### 3. Analysis And Planning

Before production changes, Codex reads the full specification and relevant ADRs,
inspects code and tests, creates or updates an ExecPlan, defines acceptance
criteria and mandatory tests, maps requirements to code and validation, and
raises evidence-backed approval gates.

### 4. Approval Gate

Implementation may start when the specification is approved, the initial test
strategy is defined, and no unresolved decision can materially change target
behavior or architecture. Independent work may continue only when it cannot
prejudice an open decision.

### 5. Incremental Implementation

Implement small milestones. Update production code, tests, documentation, and
the ExecPlan together. Add a regression test for every bug, run focused checks,
and preserve unexpected findings and deviations.

### 6. Verification And Closure

Run applicable focused tests, regressions, configured quality checks, and a
representative smoke test. Reconcile each requirement and acceptance criterion,
record unavailable checks and residual risk, inspect the diff, document
limitations, and update the index.

Use `IMPLEMENTED` when intended code exists but mandatory verification is
incomplete. Use `VERIFIED` only after full reconciliation and required validation.

## Definition Of Ready

A specification is ready when the applicable items are explicit:

- exact ID, version, path, status, and approval;
- scope, exclusions, inputs, outputs, interfaces, and compatibility;
- notation, formulas, units, shapes, frames, ranges, masks, and timing;
- applicability, missing-data behavior, errors, and fallbacks;
- state, reset, termination, truncation, and determinism;
- configuration defaults and experimentally frozen values;
- objective acceptance criteria and minimum validation categories;
- prohibited future, privileged, or diagnostic-only information;
- no unresolved material decision.

Use `Not applicable` instead of silently omitting an uncertain category.

## Protected Tests

Freeze acceptance behavior and the minimum mandatory test matrix before
implementation. Tests may be added or strengthened. Mandatory tests cannot be
removed, weakened, skipped, marked expected-to-fail, or changed to accommodate
the implementation without approval. Private-detail tests may change during an
approved refactor if the observable contract is preserved.

## Quality Tool Adoption

- Use `make lint` as the canonical Ruff lint command for owned Python code.
- Use `make format` and `make format-check` as the canonical Ruff formatter
  commands. Until the existing formatting baseline is cleaned in a dedicated
  change, restrict them with `PYTHON_QUALITY_PATHS` to new and materially modified
  files; do not mix mass formatting with semantic work.
- Global mypy enforcement is deferred. Require type annotations on new and
  materially modified public interfaces and non-trivial functions, and introduce
  module-level checking only through an approved gradual plan.
- Do not impose a global coverage threshold without an approved baseline and
  scope. Requirement-to-acceptance-test traceability takes priority over an
  arbitrary percentage.

## ADR Threshold

Use the ExecPlan for equivalent internal choices. Use an ADR for an approved
material decision such as a formula, threshold, public interface, observation
contract, dependency, fallback, data format, termination convention, simulator
limitation, or deliberate specification deviation.

## Definition Of Done

A feature is done only when approved requirements are implemented and traced to
code and tests, mandatory and regression tests pass, applicable quality and smoke
checks are recorded, documentation matches behavior, deviations are approved,
limitations are explicit, the ExecPlan matches the final implementation, the
index is current, and the diff has been reviewed.
