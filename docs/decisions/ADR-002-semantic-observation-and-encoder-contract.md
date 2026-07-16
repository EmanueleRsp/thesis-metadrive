# ADR-002: Semantic Observation And Encoder Contract

- Status: `Approved`
- Date: `2026-07-16`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-16`
- Supersedes: `NONE`
- Affected specifications: `docs/specifications/observation_v1.1_specification.md`,
  ID `OBS-V1.1`, version `1.1-final-implementation-complete`;
  `docs/specifications/encoder_v1.0_specification.md`, ID `ENC-V1.0`, version
  `1.0-final-implementation-complete`
- Affected ExecPlans: `docs/implementation/semantic_observation_encoder_v1_exec_plan.md`

## Context

The repository contains a legacy semantic observation with 2363 flat features
and 107 LQ tokens, plus encoder and SB3 bridge code designed for that legacy
layout. The approved observation and encoder specifications define a different,
causal contract: a map-based route, explicit causal context, 2541 flat
features, 122 LQ tokens, strict SB3 ownership, and fail-fast checkpoint
compatibility.

## Decision

1. Adopt `OBS-V1.1` as the authoritative observation contract and
   `ENC-V1.0` as its authoritative encoder contract.
2. Treat the legacy 2363-feature/107-token layout and its checkpoints as
   incompatible with the approved semantic v1.1 configurations; do not migrate
   them implicitly.
3. Use map-based, fail-closed route construction and an environment-owned
   causal context; future SDC trajectory data remains prohibited from online
   policy inputs.
4. Use independent actor and critic encoders in TD3/SAC, with the target
   topology defined by the approved encoder specification; PPO shares one
   encoder between policy and value.
5. Publish checkpoints as complete immutable generations validated by a
   manifest and atomic `latest.json` pointer.

## Alternatives Considered

| Alternative | Benefits | Drawbacks | Reason not selected |
|---|---|---|---|
| Retain the legacy observation and encoder layout | No immediate migration work | Violates the approved causal, shape, mask, and token contracts | Incompatible with the approved specifications |
| Share actor and critic encoders in TD3/SAC | Lower memory use | Changes the specified gradient routing and SB3 ownership | Not the approved core architecture |
| Infer actor intent from future scenario tracks | More complete interaction tokens | Leaks privileged future information | Violates the anti-leakage contract |
| Save independent checkpoint and manifest paths | Simpler writer | May expose a mismatched pair after interruption | Does not meet fail-fast resume requirements |

## Consequences

Production implementation must replace or reconcile the legacy observation,
encoder, SB3 bridge, configuration, tests, and checkpoint behavior through the
linked ExecPlan. The change is intentionally incompatible with historical
semantic checkpoints. It does not authorize production implementation before
the ExecPlan's mandatory test matrix and milestones are accepted.

## Validation And Traceability

Requirements `OBS-REQ-001` through `OBS-REQ-006` and `ENC-REQ-001` through
`ENC-REQ-005`, acceptance criteria `AC-OBS-001` through `AC-OBS-006` and
`AC-ENC-001` through `AC-ENC-005`, and their mandatory tests are defined in
`docs/implementation/semantic_observation_encoder_v1_exec_plan.md`.

## Approval Record

- Approved by: user
- Approval evidence: explicit user messages in this Codex conversation:
  “approvo” and “Approvo l'encoder 1.0 comunque”.
- Notes: approval covers the final joint-review resolutions recorded in the
  two authoritative specifications.
