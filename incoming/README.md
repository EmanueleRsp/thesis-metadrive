# Incoming Document Handoff

Use this directory to upload documents that should be inspected and placed by
Codex when working on this repository remotely.

All contents except this README are ignored by Git. For each upload, tell Codex:

- which files are new;
- whether each document is a draft, candidate, or explicitly user-approved;
- whether it replaces an existing document;
- what task should use it.

Codex must inspect the repository and the complete uploaded document before
choosing a destination. Durable approved specifications belong in
`docs/specifications/`; ExecPlans in `docs/implementation/`; approved decisions
in `docs/decisions/`; protocols in `docs/protocols/`; and historical material in
`docs/archive/`.

An uploaded filename, date, or version does not establish authority. Codex should
preserve the uploaded source until the destination and content have been
verified, then remove the processed duplicate from this directory.

When an uploaded document remains useful for provenance but is no longer an
active handoff, keep it under `incoming/archive/` rather than the top-level
incoming queue. Archived documents are historical references, not
implementation authority and not pending uploads.

For a specification named `*_UNDER_REVIEW.md`, Codex must read and review the
complete document, report material gaps, and request explicit user approval. The
upload itself is not approval. Production implementation must not start until the
user approves the specification and Codex has updated its status and approval
record, removed `_UNDER_REVIEW` from the canonical filename, moved it to
`docs/specifications/`, and updated `docs/project_index.md`.

Do not upload secrets, credentials, private keys, raw datasets, or experiment
outputs here.
