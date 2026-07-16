# Test Suite Restructure Request Template

Use this prompt in a new session only after unrelated feature changes have been
committed, reverted by their owner, or otherwise isolated from the worktree.

```text
Restructure the repository test suite under `tests/` to improve navigation and
maintenance without changing tested behavior.

Follow `AGENTS.md` and `.agent/PLANS.md` in full.

This is a mechanical test-organization refactor. Do not modify production
behavior, scientific formulas, configuration semantics, expected outcomes, or
acceptance criteria. Do not delete, weaken, skip, rename unnecessarily, or mark
tests as expected failures. Do not perform mass formatting.

Before moving files:

1. inspect `git status` and the complete current diff; stop if unrelated active
   work overlaps the test files or references that would be changed;
2. inventory all tests, fixtures, markers, helper modules, imports, Make targets,
   documentation links, ExecPlan references, and explicit test paths;
3. inspect pytest configuration and verify how recursive discovery currently
   works;
4. identify the baseline test count and the exact existing validation commands;
5. create
   `docs/implementation/test_suite_restructure_exec_plan.md` following
   `.agent/PLANS.md`;
6. define traceable requirements, acceptance criteria, a mandatory preservation
   matrix, risks, and rollback boundaries;
7. propose a domain-oriented target tree that mirrors `src/thesis_rl/` where
   useful, while avoiding directories containing only one arbitrary file;
8. distinguish domain organization from test level: prefer pytest markers for
   integration or expensive tests unless a separate cross-domain directory is
   clearly justified;
9. list every path reference that must be updated, including Makefile globs,
   documentation, protocols, and active ExecPlans;
10. report the proposed tree and any blocking decisions in Italian before moving
    files. Stop for approval if the proposed grouping or fixture ownership is
    materially ambiguous.

After approval, or immediately if no approval gate remains:

1. move files mechanically, preserving Git history where possible;
2. keep shared fixtures in `tests/fixtures/` or an appropriately scoped
   `conftest.py`; do not create imports between `test_*.py` modules;
3. update all explicit paths, Make targets, documentation links, and ExecPlan
   traceability entries;
4. do not change assertions or expected values except for path/import adjustments
   strictly required by the moves;
5. compare collection and validation results before and after the restructure;
6. run the full existing test command, `make lint`, applicable focused checks,
   and `git diff --check`;
7. record any unavailable or failing baseline check without hiding or adapting
   tests to make it pass;
8. reconcile every moved test and fixture in the ExecPlan and show the complete
   final diff;
9. do not commit unless explicitly requested.

Suggested domains to evaluate, not adopt blindly:

- `agent/`;
- `analysis/`;
- `curriculum/scenario_acl/`;
- `envs/`;
- `reward/`;
- `rulebook/v1/` and `rulebook/v2/`;
- `runtime/`;
- `scenarios/pg/` and `scenarios/scenarionet/`;
- `integration/` only for genuinely cross-domain integration tests;
- `fixtures/` for shared immutable data.
```
