# Specification: Continuous progress as the primary reported metric

## Metadata

- Feature: `eval_protocol_progress_ratio_primary`
- Specification ID: `EVAL-PROTOCOL`
- Version: `1.3`
- Status: `APPROVED`
- Date: `2026-08-09`
- Supersedes: `docs/specifications/evaluation_protocol_v1.2_specification.md`
  only for the reporting of task progress and success and for the
  benchmark-comparability claim. v1.0's statistical protocol, v1.1, and v1.2's
  multi-panel full-pool execution and artifact contracts remain authoritative
  and unchanged.
- Related ADR: `docs/decisions/ADR-059-progress-ratio-as-primary-reported-metric.md`
- Related specification: `docs/specifications/scenarionet_integration_v1.4_specification.md`
  — released together; see §5.
- ExecPlan: `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`
- Approval evidence: explicit user approval on 2026-08-09 in the originating
  conversation (*"Scelgo B2-A"*), including the explicit acceptance that no
  ratio threshold is introduced, and separate approval of the panel-scoped
  comparability clause (*"Va bene"*).
- Authoritative: `YES` for §2–§4 below; not authoritative for anything else.

## 1. Purpose And Context

`scenarionet_integration` v1.4 removes the episode tail. A policy that drives
correctly but more slowly than the logged expert can then fail to cross the
geometric final gate, and under the previous reporting it would score exactly
zero — indistinguishable from a policy that never moved. That is an artifact of
reporting a binary event as the headline number.

The `0.95` threshold that appears natural here is not a benchmark convention: it
originates in MetaDrive's `_is_arrive_destination` and is mirrored by this
repository's `success_route_completion_threshold`. The reference closed-loop
benchmarks do not threshold progress for success at all. nuPlan reports *ego
progress along expert route ratio* as a continuous value saturated in `[0, 1]`,
entering the composite score through a weighted average, and applies exactly one
progress threshold — the *making progress* gate at `0.2`, a **multiplicative**
penalty that zeroes the whole composite score below it. Waymax and V-Max report
*SDC progression* as a ratio.

The continuous quantity already exists in this repository: the mission tracker
computes `completion_instant = clip(s / s_goal, 0, 1)` and the monotone
`completion_max`, and `route_completion` is already recorded per episode and in
aggregate.

## 2. Amended reporting contract

### 2.1 Primary metric

`route_completion` (the monotone `completion_max`) is the primary reported task
metric, continuous in `[0, 1]`, reported as mean with the confidence interval
prescribed by v1.0. **No success threshold is applied to it.**

### 2.2 Binary success

The geometric final-gate crossing remains the binary success event and remains a
terminating condition. It is reported as a secondary metric, labelled
"goal reached rate" and explicitly documented as stricter than the reference
benchmarks and not comparable with their figures.

A threshold on the ratio must not replace the gate. The station `s` is a
projection onto the route polyline: near the goal, an ego displaced laterally —
on the shoulder or on the opposing carriageway — still projects to `s ≈ s_goal`
and would be credited with success, whereas the gate requires a physical forward
swept-front-bumper crossing. `ADR-053` selected gates for this reason.

### 2.3 Making-progress gate

An episode with `route_completion < 0.2` receives a composite score of `0`,
applied multiplicatively, following nuPlan's *making progress* metric. Its
purpose is to make a degenerate standing-still policy visible in the report.

This gate belongs to evaluation only. It must not enter the reward. A per-step
stagnation or time penalty is explicitly excluded from this repository's reward
design, on the evidence that such a penalty makes the cumulative waiting cost
exceed the collision cost and induces the agent to crash rather than wait.

### 2.4 Threshold coherence

The curriculum's progress gate is aligned with the definitions above, so that
the repository carries one progress-threshold vocabulary rather than three
unrelated values.

## 3. Benchmark comparability

`route_completion = s / s_goal` is the fraction of the ego's **own assigned
route** completed. nuPlan's metric is `ego progress ÷ expert progress`.

The two coincide only where the mission goal is the expert's endpoint. This
holds for Waymo records, whose goal is the logged SDC's final position. It does
not hold for PG, whose route comes from the scenario generator and for which no
expert exists.

Therefore: **benchmark comparability may be claimed for the Waymo panel only.**
PG results are reported as own-route completion, with no comparison to
nuPlan/Waymax figures. Reports must carry this scoping explicitly.

## 4. Acceptance criteria

- `route_completion` is present, continuous and reported as the primary task
  metric in the aggregate report.
- An episode with `route_completion < 0.2` yields a composite score of `0`.
- The binary gate rate is present and labelled as stricter than the benchmarks.
- Comparability labelling appears on the Waymo panel and not on the PG panel.

## 5. Release constraint

This amendment is released together with
`docs/specifications/scenarionet_integration_v1.4_specification.md`. Neither is
scientifically acceptable alone: the zero tail without ratio reporting would
zero-score competent-but-slow policies, and ratio reporting without the zero
tail would leave the empty-world exploit in place.

## 6. Out of scope

Runtime, reward, termination conditions, mission tracking and observation
contracts are unchanged. The affected surfaces are the evaluation CSV, the
analysis and reporting layer, and the curriculum threshold.
