# ADR-017: Smoke-Only Replay Buffer Persistence

- Status: `Superseded` by ADR-079 (2026-09-06): replay persistence is now
  enabled on every standard profile as a periodic snapshot paired with the
  periodic checkpoint (`TRANSITION-REPLAY` v1.1). Kept for traceability.
- Date: `2026-07-21`
- Decision owner: thesis repository maintainer
- Approval evidence: explicit user approval to disable default TD3/SAC replay persistence and retain it only for `smoke` on 2026-07-21
- Related ADRs: ADR-014, ADR-016
- Affected specifications: `docs/specifications/transition_replay_v1_specification.md`
- Affected ExecPlans: `docs/implementation/run_profile_td3_diagnostics_exec_plan.md`, `docs/implementation/transition_replay_v1.0_exec_plan.md`

## Context

TD3/SAC replay persistence writes both `latest_replay_buffer.pkl` and
`final_replay_buffer.pkl`. These files are preallocated-buffer snapshots and
can consume several gigabytes even for a short run. The persisted pair is only
needed when replay continuation/resume is intentionally requested; ordinary
training and model-only final evaluation do not require it.

## Decision

Replay persistence is profile-controlled:

- the `smoke` profile enables `transition_replay.persistence.enabled` for TD3
  and SAC, preserving checkpoint/replay/resume coverage in the integration
  smoke;
- all other standard profiles disable replay persistence by default;
- an explicit Hydra override can enable it for any profile;
- transition replay itself remains active for TD3/SAC learning. This decision
  disables only serialization of replay snapshots, not in-memory learning
  behavior or replay sampling.

When persistence is disabled, model-only checkpoints remain available and a
replay-continuation resume must not be claimed for that run. A user who needs
replay continuation must enable persistence explicitly and retain the matching
checkpoint pair.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Persist replay for every run | Simplest resume workflow | Can consume multiple gigabytes per run and duplicate `latest`/`final` snapshots | Exceeds practical storage budget for normal runs |
| Disable replay persistence everywhere | Minimal storage | Removes replay/resume coverage from the mandatory smoke validation | Does not preserve the integration smoke contract |
| Profile-controlled persistence | Keeps smoke resume evidence and leaner normal runs | Resume continuation requires an explicit setting | Selected trade-off approved by the user |

## Consequences

Normal TD3/SAC runs no longer publish replay-buffer snapshot files. Their model
checkpoints, metrics, ACL state, and evaluation artifacts remain available.
Smoke runs retain the larger replay/checkpoint artifacts and therefore remain
the designated persistence/resume validation path. Explicit diagnostic runs
can opt in without changing source code.

## Validation And Traceability

- `REQ-RP-006` and `TEST-RP-006` in
  `docs/implementation/run_profile_td3_diagnostics_exec_plan.md`.
- Hydra regression coverage verifies smoke `true`, every other standard profile
  `false`, and explicit opt-in outside smoke.
- The transition replay persistence tests remain unchanged; they validate the
  persistence mechanism when the feature is enabled.

## Approval Record

- Approved by: user.
- Approval evidence: “default va tenuto disattivato ... attivo solo per smoke”
  (2026-07-21).
- Notes: This is a storage/resume-policy change; replay learning semantics are
  unchanged.
