# Specification amendment: success is reported against a measured feasibility ceiling

## Metadata

- Feature: `eval_protocol_feasibility_ceiling`
- Specification ID: `EVAL-PROTOCOL-V1.3.1`
- Version: `1.3.1`
- Status: `APPROVED`
- Date: `2026-09-01`
- Amends: `docs/specifications/evaluation_protocol_v1.3_specification.md`, the
  reporting of the binary success event only. The primary continuous
  progress-ratio metric, v1.0's statistical protocol, v1.1, and v1.2's
  multi-panel execution and artifact contracts are unchanged.
- Source of the figures: `RULEBOOK-V5.1` limitation 14.
- Approval evidence: explicit user approval 2026-09-01 — keep the panels and the
  vehicle configuration unchanged, record the finding and the reason no remedy
  was adopted.
- Authoritative: `YES` for §2 and §3; not authoritative for anything else.

## 1. What was found

A small number of frozen Waymo records **cannot be completed by the agent under
the approved vehicle configuration**. They require a mean speed above
MetaDrive's `max_speed_km_h = 80` (22.22 m/s) because the logged human driver
exceeded it.

| panel | affected | maximum achievable success |
|---|---:|---:|
| Waymo `train` | 7 of 1100 | **99.36 %** |
| Waymo `validation` | 1 of 150 | **99.33 %** |
| Waymo `test` | 1 of 555 | **99.82 %** |
| every PG panel | 0 | 100 % |

**No record is excluded.** The frozen panel is not edited to improve a figure,
and a scenario the agent cannot finish is still a scenario on which its progress
ratio is meaningful — which is precisely why v1.3 made the continuous ratio the
primary metric rather than the binary event.

## 2. Amended reporting requirement

**`REQ-EVAL131-01`.** Wherever the binary success event is reported for a Waymo
panel, the panel's feasibility ceiling from the table above must be reported
alongside it, and any statement of the form "N % of scenarios succeeded" must be
read against that ceiling rather than against 100 %.

**`REQ-EVAL131-02`.** The ceiling is a property of the frozen panel and the
vehicle configuration, not of the policy. If either changes, it must be
re-measured before results are reported; it may not be carried over.

**`REQ-EVAL131-03`.** The affected records are **not** removed from any panel,
and no per-record exemption is applied when aggregating.

## 3. Why no remedy was adopted, stated rather than omitted

The obvious remedy — raising `max_speed_km_h` so the nine records become
reachable — was considered on 2026-08-20 and **rejected for a quantitative
reason, not a procedural one**.

Completing those records *would be legal*: their posted limits are 29.06 m/s
(65 mph) or 31.29 m/s (70 mph) and the required mean speed is 22.85–29.80 m/s,
inside limit plus tolerance in every case. So the obstruction is the vehicle, not
the rulebook, and the temptation to raise the cap is real.

What makes it the wrong trade is where the cap sits in the reward. `RULEBOOK-V5.1`
§4.1 normalizes the per-step advance by `D_REF = v_ref · Δt`, and the longest step
the agent can produce **is** the cap times `Δt` — so the cap is in the
*denominator* of what a whole mission is worth:

| cap | `Q` (mean mission) | episode ceiling | change |
|---:|---:|---:|---:|
| 80 km/h | 40.58 | 84.4 | — |
| 100 | 32.46 | 67.5 | −20 % |
| **110** (the least that covers all nine) | 29.51 | **61.4** | **−27 %** |
| 120 | 27.05 | 56.3 | −33 % |

Every mission would be worth **27 % less** while every per-step penalty stayed
exactly the same. That is the direction of the v5.0 pathology the whole
falsification campaign exists to correct — making progress cheaper relative to
violation is how standing still became optimal in the first place. And `λ₄`
cannot compensate, because §5.4's rank-preservation condition bounds it:
`λ₄ + 0.1·(λ₅ + λ₆) < a`. Raising `v_ref` together with the cap preserves the
telescoping identity but does not change this arithmetic, since the ratio is what
matters.

So the choice is between **nine records out of 1805 that the agent cannot finish**
and **a 27 % reduction in the worth of all 1805 missions**. The first is a
declared, bounded, measurable ceiling. The second re-opens the degeneracy this
rulebook was built to close.

**This is a limitation, not a defect.** It is a property of comparing a simulated
vehicle against logged human driving that exceeded the simulator's own speed cap,
and it is reported rather than engineered away.
