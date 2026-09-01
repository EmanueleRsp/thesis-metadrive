# ADR-059: Route-completion ratio as the primary reported progress metric

- Status: Approved
- Date: 2026-08-09
- Approval evidence: explicit user approval in this conversation. The user chose
  option (a) — *"Scelgo B2-A"* — asked what threshold the ratio should use
  (*"a quanto settiamo il ratio? 95%?"*), and accepted the answer that no
  threshold is required. The panel-scoped comparability clause was separately
  approved (*"Va bene"*).
- Affected specification:
  `docs/specifications/evaluation_protocol_v1.3_specification.md`
  (amends `EVAL-PROTOCOL` v1.2 reporting; no change to runtime, reward,
  termination, mission or observation contracts).
- ExecPlan: `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`
  (`DEC-RSEC-007`, `DEV-RSEC-005`).

## Context

`ADR-058` removes the episode tail. A policy slower than the logged expert can
then fail to reach the geometric final gate, and under the current reporting it
would score exactly zero — indistinguishable from a policy that never moved.
That is an artifact of reporting a binary event as the headline number, not a
property of the policy.

The `0.95` threshold that appeared natural here is not a benchmark convention.
It originates in MetaDrive's `_is_arrive_destination`
(`envs/scenario_env.py:370`: `route_completion > 0.95`), a simulator default,
and is mirrored by this repository's `success_route_completion_threshold`.

The reference closed-loop benchmarks do not threshold progress for success at
all:

- nuPlan reports *ego progress along expert route ratio* as a continuous value
  saturated in `[0, 1]`, entering the composite score through a weighted
  average with no success threshold. Its only progress threshold is the
  *making progress* gate at `0.2`, which acts as a **multiplicative** penalty:
  below it, the entire composite score becomes zero.
- Waymax and V-Max report *SDC progression* as a ratio.

The continuous quantity already exists in this repository and is already
persisted: `MissionTracker` computes
`completion_instant = clip(s / s_goal, 0, 1)` and the monotone `completion_max`
(`mission/tracker.py:229`), and `route_completion` is already a per-episode and
aggregate column of `runtime/io/csv_recorder.py:92`. Adopting the ratio is
therefore a change of which number is the headline, not new instrumentation.

A threshold on the ratio would additionally be **less** robust than the gate it
would replace. `s` is a projection onto the route polyline: near the goal, an
ego displaced laterally — on the shoulder, or on the opposing carriageway —
still projects to `s ≈ s_goal` and would be credited with success. The
geometric gate requires a physical forward swept-front-bumper crossing, which
that ego does not perform. `ADR-053` chose gates for exactly this kind of
physical unambiguity.

Finally, a comparability claim requires scoping. `route_completion = s / s_goal`
is the fraction of the ego's **own assigned route** completed. nuPlan's metric
is `ego progress ÷ expert progress`. The two coincide only where the mission
goal is the expert's endpoint — true for Waymo records, whose goal is the logged
SDC's final position — and are not comparable for PG, whose route comes from the
scenario generator and where no expert exists at all.

## Decision

1. **`route_completion` (`completion_max`) becomes the primary reported progress
   metric**, continuous, with no success threshold.
2. **The geometric final gate remains the binary success event**, unchanged, and
   remains the terminating condition. It is reported as a secondary metric
   explicitly labelled as stricter than the benchmarks and not comparable with
   them.
3. **A making-progress gate at `route_completion < 0.2` multiplies an episode's
   composite score to zero**, following nuPlan. It exists to make a degenerate
   standing-still policy visible in the report. It does **not** enter the
   reward.
4. **Benchmark comparability is claimed for the Waymo panel only.** PG results
   are reported as own-route completion with no benchmark comparison.
5. The ACL gate threshold (`route_completion_min = 0.85`,
   `curriculum/config.py:34`) is aligned with the reporting definitions so that
   the repository stops carrying three unrelated progress thresholds.

No new threshold is introduced. `0.95` is not retained as a progress-success
criterion.

## Consequences

- Reward, termination conditions, mission tracking and observation contracts are
  **unchanged**. The affected surfaces are `runtime/io/csv_recorder.py`,
  `analysis/run_analysis.py`, `curriculum/config.py` and the evaluation
  protocol document.
- A policy that drives well but slower than the expert is scored on what it
  achieved instead of receiving zero, which is what makes `ADR-058` acceptable.
  The two ADRs are released together.
- A standing-still policy scores zero through the making-progress gate rather
  than silently scoring "safe", so the degenerate regime is visible in the
  report without being encoded in the reward — consistent with the separate
  decision to add no stagnation penalty.
- Headline numbers from prior runs are not comparable, because the headline
  quantity changes.
- The comparability clause is a claim-scoping statement with no code effect on
  PG runs; without it, a comparison to nuPlan/Waymax figures would not be
  defensible in the thesis.

Regression tests: `TEST-RSEC-012` (making-progress gate zeroes a degenerate
episode) and `TEST-RSEC-013` (comparability labelling is applied to the Waymo
panel only).
