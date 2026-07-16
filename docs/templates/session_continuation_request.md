# Session Continuation Request Template

Replace every `<PLACEHOLDER>` before starting a new session.

```text
This task continues work from a previous session. Do not restart from scratch.

Read first:

- `docs/implementation/<EXISTING_EXEC_PLAN>.md`;
- the authoritative specification and ADRs linked from that plan;
- `AGENTS.md` and `.agent/PLANS.md`;
- `git status` and the current diff.

Reconstruct the actual state from repository evidence. Verify the last completed
milestone, existing decisions, changed files, validation results, and remaining
work. Preserve user changes and continue from the first genuinely pending task.

If the ExecPlan is stale or conflicts with the repository, update the verified
facts and report the discrepancy before implementation. Stop for approval if a
new blocking decision affects behavior, architecture, scientific results,
interfaces, compatibility, or protected tests.

Keep the ExecPlan current throughout the resumed work. Do not commit unless
explicitly requested.
```
