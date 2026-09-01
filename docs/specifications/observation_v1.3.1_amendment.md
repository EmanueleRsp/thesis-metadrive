# Specification amendment: the posted speed limit enters the semantic observation

## Metadata

- Feature: `obs_semantic_posted_speed_limit`
- Specification ID: `OBS-V1.3.1`
- Version: `1.3.1`
- Status: `APPROVED`
- Date: `2026-09-01`
- Amends: `docs/specifications/observation_v1.3_specification.md`, the
  `lane_road` group and the flat dimension `D`. Everything else — token count,
  group order, mask order, every other group's shape, and the `ENC-V1.3`
  tokenization contract apart from the `lane_road` projection width — is
  unchanged.
- Required by: `RULEBOOK-V5.1` §6, which carries `RULEBOOK-V5.0` §7
  (`REQ-RB5-OBS-01`) unchanged.
- Related ADR: `docs/decisions/ADR-068-speed-limit-normative-sub-rule.md`
- ExecPlan: `docs/implementation/rulebook_v5.1_six_level_hierarchy_exec_plan.md`
  (`RB51`, milestone `M4`)
- Approval evidence: `DEC-RB51-001`, approved 2026-08-20, option (a) — amend
  both observation contracts under `RB51` rather than deferring `M4`, because a
  rulebook without `speed_limit` is not the approved rulebook and `SCAL-V1.4`'s
  weights were calibrated with it.
- Authoritative: `YES` for §2 and §3 below; not authoritative for anything else.

## 1. Why the observation has to change at all

`RULEBOOK-V5.1` places `speed_limit` at **L3, above mission progress**. A rule
the agent is charged for must be one the agent can see: this is Test B, and the
posted limit passes its second branch, because it is an HD-map attribute that
every production autonomous-driving stack carries. It is not a quantity invented
to make a reward rule computable.

Without it the reward would penalise the agent for exceeding a number it has no
channel to read, which is the defect the whole falsification campaign exists to
remove.

## 2. Amended `lane_road` group

`lane_road` goes from **12** to **14** values. The two appended, in order:

| index | value | encoding |
|---|---|---|
| 12 | posted speed limit of the ego's **associated route lane** | `clip(v_limit / v_cap, 0, 1)`, the same scale the ego speed features use |
| 13 | availability flag | `1.0` when a posted limit is present, `0.0` otherwise |

Consequently `D` changes from **3009** to **3011**, and the `ENC-V1.3`
`lane_road` token projection from `12 -> token_dim` to `14 -> token_dim`.
**Checkpoint compatibility is intentionally broken**, which is acceptable
because the production runs have not started.

**Two values rather than one, deliberately.** A sentinel inside the normalized
channel — `0.0`, or a negative marker — is indistinguishable from a real limit
at that value once the encoder has projected it, and would train the agent to
read "no limit" as "limit zero". The flag makes absence explicit.

**Normalized by the ego speed cap, not by a fixed constant**, so the agent can
compare its own speed feature with the limit feature without a change of units.

## 3. When the feature is unavailable

The "unavailable" encoding is emitted under **exactly** the condition that makes
the sub-rule inapplicable in `RULEBOOK-V5.0` §5.7: the absence of real-map
provenance. Not for an unrecorded value, not for a sentinel, not for a failed
lane association treated separately — one condition, one encoding.

This is enforced structurally rather than by convention: the observation calls
the rulebook's own `associated_speed_limit_mps`
(`src/thesis_rl/rulebook/v2/transition.py`), the same function the reward calls,
after the same `associate_route_lane`. Two implementations of "is there a limit
here" would eventually disagree, and the failure would be silent.

**The alternative was worse than doing nothing.** Emitting a numeric limit that
the reward does not enforce — MetaDrive's own `lane.speed_limit`, say — would
make the observation assert a norm no cost backs. The agent would learn to
respect a number that costs it nothing to violate, or, worse, to ignore a channel
that sometimes matters and sometimes does not.

**On the PG panel the feature is unavailable on every step**, because no PG lane
carries real-map provenance (ADR-068). This is recorded as `RULEBOOK-V5.1`
limitation 13, and it means the two sources are under different normative regimes
for speed — visible in the observation exactly as it is in the reward.

## 4. What this does not change

The mission route, the ego history, the dynamic/static/control/interaction
groups, the context history window of 21 steps, the signal onset state, the token
count of 143, and the group and mask orders are all untouched. `dashed_line`
requires no observation change and none is made.
