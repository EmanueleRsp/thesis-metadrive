# Reward calibration measurement, 2026-09-07 — the evidence behind ADR-079

`rb51_calibration_a_sigma_l4.json` is the measurement ADR-079 decides from. It is
kept here rather than only under `outputs/` because an approved decision whose
supporting analysis lives on a scratch filesystem is an approved decision whose
supporting analysis will eventually be gone.

## What it measures

`RULEBOOK-V5.1` §5.5 calibrated `λ₄`, `η` and `λ₆` against the logged Waymo
expert, and held `a` and `σ` fixed while doing it. The four-level counterfactual
family in the same instrument *does* vary `a` and `σ`, but under a placeholder
`λ = 1` and with no L5/L6 tail, so its numbers cannot be read across. Neither
grid could answer whether a different `(a, σ)` is affordable **in the hierarchy
that is actually in production**.

`v51_calibration_grid` was added to
`scripts/measure_expert_rulebook_transition.py` for this decision. It prices
`(a, σ, λ₄)` jointly under the six-level reward, refusing non-rank-preserving
members exactly as the other two grids do, and emits them under `v51cal_*` keys
in `result.counterfactual_rulebooks`. The pre-existing grids are untouched, which
is what makes the file self-validating: its `v51_l42_eta1_l60.2` row reproduces
§5.5's published row exactly (mean **70.70**, p1 **−61.81**, below standstill
**3.36 %**).

## How it was produced

Same scope §5.5 was calibrated on: **1100 Waymo `train` records, 217 189
transitions, 0 skipped**.

```
docker compose -p thesis-metadrive run --rm dev uv run --no-sync \
  python scripts/measure_expert_rulebook_transition.py \
  --data-root /workspace/data/scenarionet \
  --frozen-index /workspace/thesis-metadrive/data/scenarionet/frozen/scenario_selection_index.json \
  --split train --source waymo --workers 24 \
  --output /workspace/outputs/rb51_calibration_a_sigma_l4.json
```

About 45 minutes on 24 CPU workers. Nothing is simulated: the ego pose and
velocity come from the recorded SDC track, so the costs are a property of the
*metric definition* and not of any policy, which is what makes a counterfactual
weight sweep meaningful at all.

## How to read it

Two columns decide, and only one of them is in the file.

`fraction_below_standstill` is §5.5's own acceptance column: the fraction of
logged-expert episodes scoring worse than standing still. It is in the file.

`a_req^max` is derived, not measured, and is computed from the weights:

```
a_req_max = (w2*sigma + phi) * v_ref / (2 * lambda4 * tau)
```

with `v_ref = 22.222 m/s`, `tau = 0.95 s`, `phi = 0.25`. It reads as the hardest
conflict — measured by the constant deceleration it demands — in which the
reward's local gradient still tells the agent to slow rather than accelerate. A
real vehicle brakes at about 9 m/s², so a package below that has a reward that
points the wrong way in every conflict braking could still resolve. The
derivation is in ADR-079.

| package | `a_req^max` | `w₃`/tail | below standstill | mean | p1 |
|---|---:|---:|---:|---:|---:|
| `a=2.2 σ=0 λ₄=2.0` (previous) | 1.46 | 1.04 | 3.36 % | 70.70 | −61.8 |
| **`a=2.5 σ=0.30 λ₄=2.0`** (selected) | **12.43** | **1.18** | **4.55 %** | 68.31 | −99.4 |
| `a=3.0 σ=0.30 λ₄=2.8` | 12.32 | 1.03 | 4.55 % | 98.66 | −130.7 |
| `a=3.0 σ=0.15 λ₄=2.0` | 9.36 | 1.42 | 5.73 % | 65.78 | −154.2 |
| `a=3.0 σ=0.30 λ₄=2.0` | 17.25 | 1.42 | 5.82 % | 65.14 | −160.0 |

`a = 3.0` is rejected in both directions and the file shows why: at `λ₄ = 2.8` it
recovers the below-standstill but crushes `w₃`/tail to 1.03, i.e. one step of
maximal progress would repay 97 % of running a red light — worse than the point
it replaces; at `λ₄ = 2.0` it keeps the margin but costs 2.4 pp.

## What it does not measure

The discount. `γ = 0.996` follows from `a = 2.5` by the inequality
`Δ = ln(a) / −ln(γ) > L`, with `L = 199` from §4.6's measured horizon, and needs
no panel.
