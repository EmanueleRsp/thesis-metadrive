# Feature Implementation Request Template

Replace every `<PLACEHOLDER>` before using this prompt. Use either the approved
specification path or an `UNDER_REVIEW` document uploaded to `incoming/`.

```text
The specification candidate or authoritative specification for this task is:

`<incoming/FEATURE_VERSION_UNDER_REVIEW.md | docs/specifications/SPECIFICATION_FILE.md>`

The related ExecPlan must be created or resumed at:

`docs/implementation/<FEATURE>_<VERSION>_exec_plan.md`

Follow `AGENTS.md` and `.agent/PLANS.md` in full.

If the document status is `UNDER_REVIEW`:

1. read the complete document and review it against the Definition of Ready in
   `docs/engineering_workflow.md`;
2. verify repository-dependent facts directly from code, configuration, tests,
   and current documentation;
3. identify ambiguities, unresolved material decisions, incompatibilities,
   untestable acceptance criteria, and silent scientific assumptions;
4. report the findings in Italian and ask me explicitly whether I approve the
   specification; do not infer approval from the upload or filename;
5. do not create an implementation-authoritative ExecPlan or modify production
   code before explicit approval;
6. if material decisions remain open, resolve them with me before requesting
   final approval.

After I explicitly approve the specification:

1. set its metadata to `Status: APPROVED` and `Authoritative: YES`;
2. record the approval date and evidence in its approval record;
3. rename it to the canonical versioned filename without `_UNDER_REVIEW`;
4. move it from `incoming/` to `docs/specifications/`;
5. update `docs/project_index.md`, supersession links, and related document paths;
6. then continue with the implementation workflow below.

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
