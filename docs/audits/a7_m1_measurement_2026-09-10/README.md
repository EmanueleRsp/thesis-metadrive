# A7 `M1` — The Measurement

- Date: 2026-09-10
- Status: `EVIDENCE`; not a specification and not authoritative
- Subject: `docs/implementation/rulebook_a7_five_channel_hierarchy_exec_plan.md`
  (`A7`), milestone `M1`
- Purpose: emit the six measurements `M1` exists to produce — the per-episode
  exposure distributions the A7 budgets `τ₁`–`τ₄` are read from, the A7 reward
  and its rank-preservation predicate transcribed independently of production,
  the six-member A7 grid's `fraction_below_standstill`, argmax-within-level
  frequency on `K2`/`K3`, the two `λ₄` alternatives `DEC-A7-010` pre-registers,
  and the `K2`/`K3` co-occurrence `DEC-A7-013` requires
- Artifacts: `a7_m1_summary.json` (the `a7_measurement` report section and the
  six A7 grid variants' `counterfactual_rulebooks` entries, extracted from the
  full measurement written to
  `/scratch/e.respino/thesis-metadrive/outputs/a7_m1_measurement_waymo_train_full.json`).
  The full report and the per-record top-1 % rows for every exposure channel on
  every object are inside this file already — nothing megabyte-wide is
  regenerable-but-omitted here; the full 217,189-transition per-step trace is
  what is left uncommitted, and it regenerates from the command below.

Reproduce with:

```bash
docker compose -p thesis-metadrive run --rm -T dev uv run --no-sync python scripts/measure_expert_rulebook_transition.py \
  --data-root /workspace/data/scenarionet \
  --frozen-index /workspace/thesis-metadrive/data/scenarionet/frozen/scenario_selection_index.json \
  --split train --source waymo --workers 24 \
  --output /workspace/outputs/a7_m1_measurement_waymo_train_full.json
```

Roughly 45 minutes on 24 workers. No learner and no training: this is offline
replay of logged expert trajectories through production's own
`evaluate_transition`, exactly as `RULEBOOK-V5.1` §5.5 and the 2026-09-01
comfort audit were measured.

---

## 0. A blocking defect, found before any of the six outputs could exist

The first attempt at this run measured **zero records**. `production_scalarization`
(`scripts/measure_expert_rulebook_transition.py`) constructed
`ScalarizationConfig(mode="six_level_priority_weighted_rank", priority_base=2.2, ...)`.
`priority_base=2.2` was correct when that line was written (2026-09-01); ADR-081
moved the six-level mode's *required* base to 2.5 on 2026-09-07
(`SIX_LEVEL_PRIORITY_BASE`, commit `d983c87`) without this call site following
it, so `ScalarizationConfig.__post_init__` rejected construction on every one
of the 1100 records before any per-step work ran. The replay's own exception
handler counted each as a skipped scenario, so the report read
`"scenarios_measured": 0` — silently, the identical failure shape a comment
already on that line documents for a different, earlier cause (the
four-margin/six-margin conflation fixed 2026-09-01).

**Pre-fix evidence, executed**, not inferred: the unmodified tree, run end to
end against the full 1100-record panel, reported

```json
"scenarios_measured": 0,
"scenarios_skipped": {"error:ScalarizationConfigurationError": 1100}
```

with every sampled message reading `"Mode 'six_level_priority_weighted_rank'
requires priority_base=2.5, got 2.2."`.

Classified per `AGENTS.md` as a **defect of the repository** — it would fail
identically for any contributor on any machine — rather than a local gap, and
recorded as `C54` in `docs/open_items.md` rather than carried in the A7 plan,
per that document's Proportionality rule: a bounded fix that changes no
approved behaviour. Fixed by importing `SIX_LEVEL_PRIORITY_BASE` instead of
restating it as a second literal
(`production_scalarization_config()`), so the two cannot drift apart again.
Regression test:
`tests/test_measure_expert_rulebook_transition.py::test_production_scalarization_config_matches_the_shipped_configuration`,
which reads `conf/scalarization/default.yaml` directly rather than hardcoding
a second comparison value — the same discipline
`tests/test_rulebook_v51_orderings.py`'s `_shipped_gamma()` already uses for
the discount.

**Consequence**: no output of this script is trustworthy from 2026-09-07
(ADR-081) to this fix. Nothing in that window could have been produced by a
passing run, so no committed figure from that period rests on a silently-empty
replay — the script simply could not have run.

## 1. Scope

**1100 Waymo `train` records, 217,189 transitions, 0 skipped** — the identical
scope `RULEBOOK-V5.1` §5.5 and the 2026-09-01 comfort audit were measured on.

## 2. Output 1 — per-episode exposure, all three objects, `τ₁`–`τ₄`

`X_imp` (K1, at-fault impact), `X_int` (K2, interaction risk), `X_hard` (K3,
non-negotiable compliance) and `X_soft` (K4, negotiable lane compliance), each
on the raw undiscounted realized total, the per-span total (divided by the
episode's own route length `Q`), and the `γ = 0.9982`-discounted total.

| channel | raw max (`τ_i`) | raw `p99` | per-span max | discounted max | max record |
|---|---:|---:|---:|---:|---|
| `X_imp` (`τ₁`) | **0.000000** | 0.000000 | 0.000000 | 0.000000 | (every episode ties at 0) |
| `X_int` (`τ₂`) | **110.379799** | 8.067951 | 0.720226 | 98.733955 | `waymo:training_20s:71344f609367eace` |
| `X_hard` (`τ₃`) | **21.287443** | 3.323886 | 0.264242 | 19.613411 | `waymo:training_20s:5f8217a5da0b24ff` |
| `X_soft` (`τ₄`) | **23.386514** | 2.737476 | 0.514168 | 20.197176 | `waymo:training_20s:1ef1a62b6bebe06` |

**The budget rule (ExecPlan §6.4), applied**: `τ_i` = the per-episode maximum of
the logged expert on the frozen `train` panel, minus any episode excluded for a
**declared** panel defect, with the exclusions listed per record. **No
exclusion is declared for this run** — nothing here asserts that any of the
records above is defective — so `declared_panel_defect_exclusions: []` for
every channel and `τ_i` is the unmodified maximum. The full per-record top-1 %
rows are in `a7_measurement.exposure_by_channel.<object>.<channel>.top_1pct_records`
in the committed JSON, which is what makes the rule auditable rather than
asserted: a future session with grounds to exclude a specific record (for
instance, one of the four named above, whose values sit far above their own
channel's `p99`) has the record to point at.

**`τ₁ = 0` is admissible, measured rather than assumed.** `X_imp` is exactly
`0.000000` on every one of the 1100 episodes, on all three objects: the logged
expert never records an at-fault impact anywhere on this panel. ExecPlan §6.4's
stated condition for `τ₁ = 0` is satisfied.

**Why the maximum rather than `p99`**: had `p99` been adopted instead (the
plan's own withdrawn first draft), `τ₂` would read 8.07 against the recorded
maximum of 110.38 — excluding the exact competent-human episode the rule's own
first requirement (a budget must admit the human) says a budget may not
exclude.

## 3. Output 2 — `a7_reward`/`a7_is_rank_preserving`, checked before use

Transcribed independently from ExecPlan §5.1, beside `v51_reward`/
`v51_is_rank_preserving`. Checked against the plan's own worked figures before
trusting them for this run (`tests/test_measure_expert_rulebook_transition.py`):
the rank-preservation ratios **1.1514 / 1.1478 / 1.1390** at `φ=0` and
**1.1105 / 1.0975 / 1.1390** at `φ=0.25` (at `a=2.5, σ=0.30, λ₄=2.0, w₅=0.15`),
and the fully-violated-K2 step costs **8.125** at `φ=0` / **8.375** at
`φ=0.25`. The transcription trap the plan records at §5.1 — letting `φ` reach
K4 (`w₅`) as an extra "lower priority level" — was reconstructed explicitly and
shown to reproduce the plan's own wrong figures (1.0911 / 1.0513 / 1.0225,
thinnest margin at `k=3` instead of `k=2`), so the correct implementation was
checked against a known-wrong alternative, not only against itself.

## 4. Output 3 — the six-member A7 grid, `fraction_below_standstill`

Standstill baseline under A7 is exactly 0 (`L6` is gone), verified in code and
by a dedicated test rather than left to a name-matching fallback.

| grid member | `w₅` | `φ` | `λ₄` | `fraction_below_standstill` | mean episode return |
|---|---:|---:|---:|---:|---:|
| production (`DEC-A7-002`/`-003`) | 0.15 | 0 | 2.0 | **4.64 %** | 71.34 |
| `φ` retained | 0.15 | 0.25 | 2.0 | 4.73 % | 71.20 |
| `w₅` alternative | 0.25 | 0 | 2.0 | 4.82 % | 71.17 |
| `w₅` + `φ` alternative | 0.25 | 0.25 | 2.0 | 4.91 % | 71.02 |
| `λ₄` alternative (`DEC-A7-010`) | 0.15 | 0 | 1.25 | 6.45 % | 41.12 |
| `λ₄` alternative (`DEC-A7-010`) | 0.15 | 0 | 1.9 | 4.91 % | 67.32 |

Every member is under the **7.45 %** ceiling (`AC-RB5.1-04`). `DEC-A7-003`'s
falsifier (revert `φ` to 0.25 if the ceiling is breached) does not trip: the
movement between `φ=0` and `φ=0.25` at the production `(w₅, λ₄)` is **+0.09 pp**,
against **2.81 pp** of remaining headroom to the ceiling at `φ=0`.

## 5. Output 4 — argmax-within-level frequency, `K2`/`K3`

Over all 217,189 measured steps:

| level | sub-rule | steps | frequency |
|---|---|---:|---:|
| `K2` | (level fully satisfied) | 216,325 | 99.6022 % |
| `K2` | `rss_lateral` | 615 | 0.2832 % |
| `K2` | `ttc` | 153 | 0.0704 % |
| `K2` | **`clearance`** | **96** | **0.0442 %** |
| `K3` | (level fully satisfied) | 215,795 | 99.3582 % |
| `K3` | `offroad` | 1285 | 0.5917 % |
| `K3` | `vehicle_yield` | 55 | 0.0253 % |
| `K3` | `signal` | 38 | 0.0175 % |
| `K3` | `stop` | 16 | 0.0074 % |
| `K3` | `crosswalk` | 0 | — |
| `K3` | `speed_limit` | 0 | — |

**`F12`'s question, answered**: `clearance` — whose maximum cost is 0.7094
against 1.0 for `K2`'s other two sub-rules — *is* the argmax on this panel,
96 times, rarely but not never.

**A finding neither `F12` nor the plan named**: `crosswalk` and `speed_limit`
never win `K3`'s `max` on this panel, at any frequency — not "rare", zero. Both
are known near-inert on Waymo by *applicability* already (§4.5 of the plan:
`crosswalk` 15/1100 records, and `speed_limit`'s applicability is not similarly
small), but *argmax* is a stricter question than applicability: a sub-rule can
be applicable on a step and still never be the one that sets `K3`'s cost
because another sub-rule's cost is always at least as large when both apply.
This is evidence about which sub-rules the negotiable/non-negotiable ordering
choice can ever be *decided by* on this panel, not evidence that either
sub-rule should be reweighted or removed.

**Cross-check.** The non-"none" complements above (0.3978 % for `K2`,
0.6418 % for `K3`) reproduce, to four decimal places, the per-step interaction
and non-negotiable-compliance violation rates the plan already cites from an
independent measurement (ExecPlan §6.4, §9). The two counters were built from
`v51_l2`/`v51_l3`'s own values, not from that citation, so the agreement is a
genuine cross-check rather than a restatement.

## 6. Output 5 — the two `λ₄` alternatives (`DEC-A7-010`)

Table repeated from §4 above for the reading `DEC-A7-010` pre-registers:
`λ₄` reopens only if the *approved* `λ₄ = 2.0` member breaches the 7.45 %
ceiling. Measured: `λ₄ = 2.0` gives **4.64 %**, well under the ceiling, so the
trigger condition is not met and `λ₄` is not reopened. Had it been, the
pre-registered reading holds at the measured figures: `λ₄ = 1.9` costs
**+0.27 pp** against `λ₄ = 1.25`'s **+1.81 pp** — a sixth of the cost, for the
reasons ExecPlan §6.8 gives (1.9 also buys back the §5.4 margin; 1.25 buys an
option — opposing a fast off-corridor drive — that no K4 sub-rule can fire for
under A7's approved membership).

## 7. Output 6 — `K2`/`K3` co-occurrence (`DEC-A7-013`)

| quantity | value |
|---|---:|
| steps with both `K2` and `K3` non-zero | 15 of 217,189 (0.0069 %) |
| episodes with at least one such step | 5 of 1100 (0.45 %) |
| `K2` on those 15 steps, `p50` / `p99` | 0.2197 / 1.0 |
| `K3` on those 15 steps, `p50` / `p99` | 0.0253 / 1.0 |
| Pearson correlation on those 15 steps | −0.0161 |

**A finding, not a resolution.** This does not derive `K2 ≻ K3` from a
mechanism — ExecPlan §6.9 already states plainly that no such mechanism exists
in this repository, and this measurement does not manufacture one. What it
bounds is *how often the order could matter at all* on the logged expert: on
this panel, almost never jointly, and when it does, the two channels' severity
does not move together (correlation indistinguishable from zero at this sample
size). The order is still declared as the hierarchy's least-supported step in
the A7 specification (`M2`), with this number recorded beside it.

## 8. What this licenses, and what it does not

Licensed:

- Filling in `τ₁`–`τ₄` in the A7 specification (`M2`) with the values in §2,
  citing this audit.
- Reading `DEC-A7-002`/`-003`'s falsifier and `DEC-A7-010`'s pre-registration
  as checked, on the actual A7 reward, on the full panel — not projected from
  the four-level family or from a different weight pair.
- Reporting `K2 ≻ K3`'s co-occurrence rate as a declared, measured limitation
  of the hierarchy's evidentiary basis.

Not licensed:

- Any claim about a trained policy. Nothing here evaluates one; this is
  offline replay of a logged human.
- Treating the co-occurrence measurement as settling `K2 ≻ K3`'s correctness.
  It bounds exposure, not correctness.
- Excluding any of the outlier records named in §2 from a future `τ_i` without
  a declared, specific reason for that record.

## 9. Limitations

1. **Waymo only.** As with every other calibration in this work, PG records are
   excluded from the panel: PG's logged ego is `IDMPolicy`, so replay
   establishes nothing about what a competent human would do.
2. **`train` split only**, consistent with the rulebook calibration policy of
   never falsifying against validation or test data.
3. **Heavy right tail on `X_int`/`X_hard`/`X_soft`.** Each channel's maximum sits
   far above its own `p99` (§2), consistent with a small number of genuinely
   difficult records rather than measurement noise — but not independently
   confirmed here, since no panel defect is declared. The per-record rows this
   audit commits are what would let a future session make that determination
   without a second run.
4. **`K2`/`K3` co-occurrence is measured, not derived.** §7 is explicit that
   this is a finding about the panel's exposure, not a mechanism-level argument
   for the order.
