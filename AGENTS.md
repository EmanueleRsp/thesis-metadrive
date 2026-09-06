# Repository Development Instructions

## Purpose

This repository contains research software for a master's thesis on reinforcement
learning for autonomous driving. Prioritize scientific correctness,
reproducibility, requirement-to-code-to-test traceability, and small reviewable
changes.

## Language

- Communicate with the user in Italian, including progress, explanations,
  decision requests, and final reports.
- Write source code, identifiers, filenames, configuration keys, comments,
  docstrings, tests, logs, exceptions, and repository technical documentation in
  English.
- Preserve the notation and terminology of the selected scientific specification.
- Apply the English documentation rule to new or modified prose. Do not translate
  an approved scientific specification without approval because translation may
  alter its meaning.

## Sources Of Truth

Use this precedence order:

1. explicit user approvals;
2. the specification identified as authoritative in `docs/project_index.md`;
3. approved ADRs applicable to that specification;
4. existing public interfaces and architectural contracts;
5. this file;
6. verified repository conventions;
7. internal implementation preferences.

Scientific specifications are user-approved behavioral contracts. Do not change
formulas, semantics, values, defaults, dimensions, interfaces, experimental
behavior, dataset policy, or acceptance criteria without approval. A filename or
higher version number does not establish authority.

An `UNDER_REVIEW` specification, including one uploaded to `incoming/`, is not an
authoritative implementation contract. Read it completely, verify
repository-dependent claims, review it against the Definition of Ready, report
material gaps, and request explicit user approval. Do not begin production
implementation while its status remains `UNDER_REVIEW`.

After explicit approval, update the document to `APPROVED` and
`Authoritative: YES`, record approval evidence and date, remove `_UNDER_REVIEW`
from the canonical filename, move it to `docs/specifications/`, and update
`docs/project_index.md` plus affected links before creating or resuming the
implementation-authoritative ExecPlan.

Approved specifications belong in `docs/specifications/`; consult
`docs/project_index.md` to select the authoritative version. ExecPlans belong in
`docs/implementation/` and do not override specifications. Protocols under
`docs/protocols/` are not authoritative unless the index explicitly says so. If
`.github/copilot-instructions.md` conflicts with a higher-priority source, report
the conflict and follow the higher-priority source.

## Required Workflow

For every non-trivial feature, integration, refactor, or behavioral change:

1. read the complete selected specification and applicable ADRs;
2. inspect the current code, configuration, interfaces, and tests;
3. create or update an ExecPlan in `docs/implementation/` following
   `.agent/PLANS.md`;
4. define acceptance criteria, the mandatory test matrix, and exact available
   validation commands before changing production code;
5. identify verified facts, assumptions, conflicts, blockers, and approval gates;
6. obtain approval for unresolved choices that affect observable or scientific
   behavior, compatibility, experiments, or protected tests;
7. implement the smallest coherent change in milestones, updating code, tests,
   documentation, and the ExecPlan together;
8. add a regression test for every discovered bug;
9. run focused checks, applicable regressions, and a representative smoke test;
10. reconcile every requirement and acceptance criterion with code and tests,
    update `docs/project_index.md`, and review the final diff.

**Proportionality.** Step 3 binds when a change alters observable or scientific
behavior, a specification contract, an experimental design, or a public interface,
or when it spans more than one milestone. A bounded defect fix that changes no
approved behavior gets **no ExecPlan**: record it once in `docs/open_items.md` —
cause, fix, the evidence that the test failed before the fix, and any residual
risk — and add the regression test. Documentation weight is not evidence of care;
the same cause restated in three registers is redundancy, and it costs the review
attention the code deserves instead. `docs/project_index.md` is the register of
ExecPlans, so a change that needs no plan adds no row to it.

Do not perform unrelated cleanup. Do not add a dependency without approval. Stop
only the portion blocked by a decision; safe independent work may continue.

## Decision And Change Control

Codex may decide private names, helper decomposition, and equivalent internal
structures when they do not change behavior. Record non-obvious choices in the
ExecPlan.

Request approval before implementing:

- a specification clarification with observable, scientific, compatibility, or
  experimental impact;
- any deviation from a specification or approved acceptance behavior;
- a new fallback, dependency, public interface, data policy, metric convention,
  or termination/truncation convention;
- removal, weakening, skipping, or expectation changes for mandatory tests.

Provide evidence, alternatives, a recommendation, and consequences. Record an
approved material decision in both the ExecPlan and an ADR.

## Testing And Traceability

Use acceptance-test-first plus tests alongside implementation. The approved plan
must map stable requirement IDs to implementation locations and test IDs.

After plan approval, tests may be added or strengthened. Mandatory tests may not
be deleted, weakened, skipped, marked expected-to-fail, or changed to match the
implementation without approval. Tests tied only to private details may be
refactored if the observable contract is preserved.

Every bug fix requires a regression test. Never use a training curve as a
substitute for deterministic correctness tests. Never claim a check passed unless
it was executed successfully. For checks not run, record the reason, remaining
risk, and exact follow-up command.

## Repository Conventions

- Runtime package: `src/thesis_rl/`
- Tests: `tests/`
- Hydra configuration: `conf/`
- Repository scripts: `scripts/`
- Editable upstream projects: `third_party/`
- Docker Compose is the primary reproducible environment.
- Supported Python range: `>=3.10,<3.11`.
- Pytest configuration and Ruff settings are in `pyproject.toml`.

Preserve deterministic seed handling, train/validation/test separation, causal
policy inputs, explicit units and coordinate frames, termination/truncation
distinction, configuration logging, dataset and software versioning, and NaN or
infinity checks where applicable.

## Verified Commands

Run commands from the repository root.

- Full tests through the primary environment: `make test`
- Full tests inside an already provisioned container:
  `uv run --no-sync python -m pytest -q`
- General Ruff lint over owned code: `make lint`
- Ruff formatting over owned code: `make format`
- Ruff formatting verification: `make format-check`
- End-to-end training smoke test: `make smoke`
- Full bootstrap verification: `make verify`
- Rulebook v2 tests, scoped Ruff check, and whitespace check:
  `make rulebook-v2-check`
- Canonical EVAL-PROTOCOL v1.0 comparison-report regeneration:
  `make analyze RUN_PROFILE=<profile>` (ablation/factor-effect tables excluded
  by default per REQ-018; add `ANALYSIS_ARGS="--include-effects-tables"` to
  include them)
- Compose validation: `make config` and `make config-gpu`
- Patch whitespace validation: `git diff --check`
- CI shell validation: `bash -n setup.sh scripts/*.sh` and
  `shellcheck setup.sh scripts/*.sh`

The default lint/format scope is `src tests scripts`, excluding vendored projects,
data, outputs, and local environments. Override it for focused adoption, for
example:

`make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/module.py tests/test_module.py"`

The Ruff lint baseline is clean. The repository-wide Ruff formatting baseline is
not yet clean, so do not run mass formatting in a semantic change or treat global
`make format-check` as a completion gate until a dedicated formatting change is
approved. New and materially modified Python files must nevertheless be
formatted and checked with a focused scope.

Do not invent missing commands. There is no standard global mypy target or mypy
configuration; adopt static checking incrementally for bounded modules. New and
materially modified public interfaces and non-trivial functions require type
annotations. Coverage may be reported with existing tooling when useful, but no
global threshold is configured or may be invented.

## Completion

A change is complete only when approved requirements and acceptance criteria are
reconciled, mandatory and regression tests pass, applicable checks and a smoke
test are recorded, documentation matches behavior, no unapproved deviation
remains, the ExecPlan reflects reality, limitations are explicit, and the final
diff contains no unintended changes.

Final reports must be in Italian and include status, behavior, changed files,
executed checks and results, approved decisions, deviations, unresolved issues,
known limitations, and deferred optional work.
