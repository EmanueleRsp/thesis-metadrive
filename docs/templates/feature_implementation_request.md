# Feature Implementation Request Template

Replace every `<PLACEHOLDER>` before using this prompt.

```text
The authoritative specification for this task is:

`docs/specifications/<SPECIFICATION_FILE>.md`

The related ExecPlan must be created or resumed at:

`docs/implementation/<FEATURE>_<VERSION>_exec_plan.md`

Follow `AGENTS.md` and `.agent/PLANS.md` in full.

Before modifying production code:

1. read the complete specification, applicable ADRs, and any existing ExecPlan;
2. inspect `git status`, the current diff, implementation, configuration, and
   existing tests;
3. do not discard, overwrite, or restart existing work without verifying its
   state and provenance;
4. create or update the ExecPlan following `.agent/PLANS.md`;
5. define traceable requirements, acceptance criteria, and the initial mandatory
   test strategy before production changes;
6. identify ambiguities, technical problems, incompatibilities, and decisions
   requiring approval;
7. do not introduce silent deviations from the specification;
8. stop after planning when a blocking decision would change behavior,
   architecture, scientific results, interfaces, or compatibility.

If no blocking decision exists, implement by milestone and keep the ExecPlan
current.

Before concluding:

1. reconcile requirements, code, and tests;
2. record every command actually executed and its result;
3. update decisions, deviations, limitations, and remaining work;
4. update `docs/project_index.md` when specification or implementation status
   changes;
5. show the final diff and do not commit unless explicitly requested.
```
