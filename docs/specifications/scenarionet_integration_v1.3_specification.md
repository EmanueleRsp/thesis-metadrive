# ScenarioNet evaluation-pool amendment

## Metadata

- Feature: complete named evaluation pools and frozen diagnostic subsets
- Specification ID: `SCENARIONET-INTEGRATION`
- Version: `1.3`
- Status: `APPROVED`
- Date: `2026-07-31`
- Supersedes: version `1.2` only for its named-panel cardinalities and panel
  artifact contract. All other version `1.2` clauses remain authoritative.
- Related specification: `docs/specifications/evaluation_protocol_v1.2_specification.md`
- Related ADR: `ADR-042`
- Authoritative: `YES`

### Approval record

The user explicitly approved this amendment by requesting implementation of
the “ScenarioNet multi-panel evaluation, full-pool protocol, and terminal
monitor” ExecPlan on 2026-07-31.

## Amendment

The five canonical panel manifests are complete frozen endpoint pools, not
draws smaller than their source split:

| Panel | Split | Source | Cardinality | Role |
|---|---|---:|---:|---|
| `validation_waymo_empirical` | validation | Waymo | 150 | primary learning curve |
| `validation_pg` | validation | PG | 150 | secondary learning curve |
| `test_waymo_empirical` | test empirical | Waymo | 400 | primary final endpoint |
| `test_pg` | test empirical | PG | 300 | secondary final endpoint |
| `test_arm_stratified` | test stratified | Waymo + PG | 300 | arm-claim endpoint |

Two diagnostic scopes are also frozen as ten separate child manifests. They
are sampled once from their corresponding complete canonical manifest with a
profile-specific fixed seed independent of the learner seed:

| Scope | Validation Waymo / PG | Test Waymo / PG / stratified | Sampling seed |
|---|---:|---:|---:|
| `smoke` | 10 / 10 | 20 / 20 / 12 | 20260731 |
| `fast` | 50 / 50 | 100 / 100 / 60 | 20260732 |

`default`, `medium`, `tune`, `long`, and `thesis` resolve the canonical
complete manifests and have scope `full`. A subset manifest records its parent
panel identity and both its own hash and parent hash. The frozen index embeds
and hashes all fifteen manifests while retaining the existing selection hash
as the independent data-selection identity. Replaying a frozen dataset must
restore every embedded artifact exactly.

The profile subset is identical for all conditions and learner seeds in its
profile. A result labelled `smoke` or `fast` is diagnostic-only and shall not
be aggregated with `full` results.

