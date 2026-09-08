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
7. implement the smallest coherent change in milestones, sizing each milestone so
   it is independently verifiable and fits in a single agent context window, and
   updating code, tests, documentation, and the ExecPlan together;
8. add a regression test for every discovered bug;
9. run focused checks, applicable regressions, and a representative smoke test;
10. for a change that required an ExecPlan, obtain an adversarial review from an
    agent session that did not write the change;
11. reconcile every requirement and acceptance criterion with code and tests,
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

**Review.** There is no second maintainer, so step 10 is the only review a change
gets on its way in, and it must come from a session that did not write the code:
an agent rereading its own work is rereading its assumptions, not checking them.
Give that session the ExecPlan or specification to review against — a review with
nothing written to check against inspects the code without knowing what it was
supposed to do, which is also why the plan exists before the code.

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

## Scientific Argument Standards

Never justify a choice by citing this repository's own specification. We wrote
those documents, so "REQ-xxx requires it" is circular. A repository document is
valid evidence of what the code does or what was measured; it is never evidence
that a choice is right. Arguments must rest on mechanism (algebra, invariants,
dimensional analysis), on measurement reported together with its source, or on
published literature. When the only argument for something is the specification,
say so out loud: that is a finding, and usually the specification is the thing to
fix.

A code-versus-specification divergence is therefore not automatically a code
defect. Check which side is wrong, and whether the requirement is even
self-consistent, before changing code to match it.

Do not propose a scientific constant without a derivation behind it. State the
criterion the value has to satisfy, derive or measure it, then propose one value
and its cost in the same breath — a menu of options is the wrong shape of answer
for a parameter. Prefer a criterion with a physical reading (a braking
deceleration, an episode length) over a dimensionless heuristic, and check
whether an instrument already in `scripts/` prices the alternatives, which is
cheaper and far more convincing than arguing. Before quoting a published
break-even, check which comparison actually binds.

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
- Operational recipes — remote realignment, worktree setup, GPU jobs, the
  production-path smoke, long jobs, pre-fix evidence:
  `docs/workflows/agent_operations.md`.

Preserve deterministic seed handling, train/validation/test separation, causal
policy inputs, explicit units and coordinate frames, termination/truncation
distinction, configuration logging, dataset and software versioning, and NaN or
infinity checks where applicable.

## Verified Commands

Run commands from the repository root.

- Merge gate with recorded evidence: `make gate` (whitespace, Ruff and the full
  test suite; log under `outputs/gate/`).
- Working-loop check: `make check` — the same steps without the nine
  `integration` tests, about 1m35s against the gate's 10m01s as measured on
  2026-09-08 with 16 `pytest-xdist` workers (`GATE_WORKERS`, `1` for a
  sequential run). Cheap enough to run on every change; `PARTIAL`, so never a
  gate.
- Narrow either while iterating with
  `make gate GATE_ARGS="tests/test_module.py -k case"`, which also marks the run
  `PARTIAL`.
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

## Continuous Integration And The Merge Gate

The only GitHub workflow is `.github/workflows/portability.yml`: shell syntax and
`shellcheck` over `setup.sh` and `scripts/*.sh`, `docker compose config` on the
three profile combinations, and `./setup.sh --check-only --skip-docker-checks`.
It does **not** run pytest or Ruff.

That perimeter follows from the hardware rather than from an oversight: the test
suite needs this project's architecture (`aarch64`), a CUDA GPU, the MetaDrive
simulator, and the frozen ScenarioNet panels, none of which a GitHub-hosted
runner has. The project machine is the only place the suite can run.

A pull request also reports a GitGuardian check, which is a repository integration
rather than a workflow in this repository; it scans for leaked secrets and does not
run tests either.

A green pull request is therefore not evidence that tests pass — both of its checks
can be green on a change that breaks the suite. `make gate` on the project machine
is the merge gate: it runs the whitespace checks, Ruff over owned code, and the
test suite, and it writes the executed evidence Completion requires to
`outputs/gate/<timestamp>-<commit>.log`. Run it before every merge, not only before
pushing, and cite its summary line. Narrowing the run — pytest arguments or a
narrowed `PYTHON_QUALITY_PATHS` — marks it `PARTIAL`, which is not a gate. A check
that had nothing to inspect is recorded as `NOT APPLICABLE` and named in the
verdict rather than counted as passing.

## Branching And Pull Requests

This repository has a single maintainer, so there is no reviewer to assign and no
review queue to feed. `main` is the working checkout and stays green, because
every branch starts from it.

- Branch from `main` for each unit of work, in an isolated worktree when more
  than one agent session works on the repository at once.
- Pass the merge gate, then open a pull request. It runs the portability workflow
  and records why the change was made this way — including the rejected
  alternative, which without a reviewer has nowhere else to go except the commit
  body or an ADR.
- Merge and delete the branch; this repository's history merges pull requests with
  a merge commit. Do not assign a reviewer and do not leave a pull request in
  draft, because there is no review queue that either would feed.
- Do not keep a long-lived integration branch. A local one is justified only when
  two lines of work must be exercised together while neither is mergeable yet; it
  never leaves the machine and never enters `main`. Work must never live only on a
  branch that gets rewritten.

Realign with the remote at the start of a session that will touch code, and again
before each new block of changes: a long session works from a snapshot taken at
its start, and parallel worktrees, another machine, or an earlier merge move
`origin/main` under it. The check answers two questions, and the second is the one
that gets skipped. *Is there anything to integrate?* is mechanical and binary.
*Does what changed invalidate the plan?* is not: if the files that moved intersect
the area being worked on, or if something that landed already solved the problem
being addressed, restate the plan before continuing instead of integrating and
pressing on. The commands are in `docs/workflows/agent_operations.md`.

Do without asking: branch, implement, pass the gate, open the pull request, merge
it once the gate was run and the portability check is green, delete the branch.
Ask first: rewriting published history, changing repository-wide policy, spending
money or credentials, and any architectural choice that is costly to reverse —
without a reviewer, nothing else intercepts it.

## Completion

A change is complete only when approved requirements and acceptance criteria are
reconciled, mandatory and regression tests pass, applicable checks and a smoke
test are recorded, documentation matches behavior, no unapproved deviation
remains, the ExecPlan reflects reality, limitations are explicit, an adversarial
review from a session that did not write the change is recorded whenever the
change required an ExecPlan, and the final diff contains no unintended changes.

When reporting a failure — a failed setup, a red test, a build that does not
pass, a command that blew up — classify it explicitly before proposing any
change:

- **missing on the project machine**, and installable or configurable there: a
  tool absent from `PATH`, a stopped daemon, an old runtime, network access not
  yet granted. This is not a repository problem and must not produce a code or
  documentation change;
- **a defect of the repository**, which would fail for any contributor on any
  machine: a documented quickstart that does not work, an example file unusable as
  distributed, a badly declared dependency;
- **deliberately not built yet** and declared as such. Reporting this as a bug is
  a false positive.

The categories have different destinations — a checklist to run in a terminal
versus a branch with a commit — and presenting them mixed risks committing a
change that only compensated for a local gap. Classify by where the cause is, not
by where the symptom shows: a repository defect whose symptom appears only on some
platform or configuration is still a repository defect, and the reason it stays
invisible elsewhere is a separate statement, not a substitute for the
classification.

Final reports must be in Italian and include status, behavior, changed files,
executed checks and results, approved decisions, deviations, unresolved issues,
known limitations, and deferred optional work.
