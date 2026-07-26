# ADR-027: R1 Injury-Risk Collision Cost

- Status: APPROVED
- Date: 2026-07-26
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-26
- Supersedes: NONE
- Affected specification: `docs/specifications/rulebook_v4.9_specification.md`,
  version 4.9, status `APPROVED`, authoritative (for the §5.4/§5.6/§5.8
  subset it amends; v4.7 remains authoritative for everything else)
- Affected ExecPlan: `docs/implementation/r1_injury_risk_collision_cost_v4.9_exec_plan.md`

## Context

Rulebook v4.7 §5.4 bounds the `R1` collision cost by dividing the pre-state
normal closing speed by the **configured speed normalization cap** of the
scenario:

```
q_collision,i = max{ eps_col , ( min(u_i, u_cap,i) / u_cap,i )^2 }
u_cap,i = v_max,e + v_max,i  (vehicle)  |  v_max,e  (VRU / static)
```

Those caps are scenario configuration, not physical constants. The cost of
a collision therefore depends on **which scenario it happened in**, at
equal real impact severity. Worked counterexample using the v4.7 formula:

| Scenario | `v_max,e` | `u_i` | `q_collision` |
|---|---|---|---|
| residential | 5 m/s | 4 m/s (≈14 km/h) | `(4/5)^2 = 0.640` |
| highway | 20 m/s | 15 m/s (≈54 km/h) | `(15/20)^2 = 0.5625` |

The high-energy highway impact scores as *less severe* than the low-energy
residential one. This inverts real injury risk, and it is not a marginal
edge case: it occurs whenever scenarios with different caps are compared or
aggregated — which is the normal situation for a Waymo + PG evaluation set.

Three consequences motivated acting rather than only documenting:

1. aggregate `R1` statistics over a heterogeneous evaluation set become
   uninterpretable, because they average costs normalized by different
   divisors;
2. a lexicographic learner optimizes `J_1(pi) = E[sum_t gamma^t m_1(t)]`
   **directly**, with no scalarizer in between, so it inherits the
   inversion unfiltered;
3. no other rulebook sub-metric has this property — TTC (v4.7 §6.3) and VRU
   clearance (v4.7 §6.4) use thresholds that are fixed per actor class and
   independent of scenario configuration.

v4.7 §5.6 declares the formula an *original thesis derivation* inspired by
ScenicRules, which uses unbounded kinetic-energy loss. The squared
normalized closing speed is the bounded adaptation, and the scenario cap is
precisely the piece added to obtain the bound — i.e. the defect originates
from an implementation requirement, not from a scientific source.

Neither Censi et al. nor Veer et al. support a scenario-dependent
normalizer: where they normalize at all, the constant is fixed for the
duration of the experiment.

## Decision

Replace the mapping `u_i -> q_collision,i` with the **MAIS3+F injury-risk
logistic curve** of Lubbe, Wu & Jeppsson (2022), *Traffic Safety Research*
2:000006, built on GIDAS in-depth crash data (1999–2020):

```
z_tau(u) = beta_0(tau) + beta_v(tau) * 3.6 * u + beta_a(tau) * 65
q_collision,i = max{ eps_col , 1 / (1 + exp(-z_tau(u_i))) }
```

The `R1` cost becomes the **probability that the impact produces an
at-least-serious injury**. `u_cap` leaves the cost computation entirely.

Frozen coefficients (source: Lubbe et al. Tables 2, 3, 5, `MAIS3+F` rows):

| ActorClass | `beta_0` | `beta_v` | `beta_a` |
|---|---|---|---|
| `PEDESTRIAN` | -6.190 | 0.078 | 0.038 |
| `CYCLIST` | -7.467 | 0.079 | 0.047 |
| `VEHICLE` | -7.654 | 0.041 | 0.021 |
| `STATIC_COLLIDABLE` | -7.654 | 0.041 | 0.021 |

Three points were explicitly decided by the user on 2026-07-26:

- **`DEC-R1-02` severity level `MAIS3+F`.** `MAIS2+F` already costs 0.217
  at zero speed, burning most of the usable dynamic range; `Fatal` is
  nearly flat below 10 m/s, giving no resolution in the urban regime that
  dominates the scenario pool. `MAIS3+F` also carries the strongest
  external justification: it is the criterion the source itself uses for
  its safe-speed recommendations, following the Academic Expert Group for
  the 3rd Global Ministerial Conference on Road Safety.
- **`DEC-R1-03` reference age fixed at 65 for all classes.** With
  per-class median ages (46/39/39) part of the inter-class cost difference
  would be a difference in sample age rather than in vulnerability. A
  single fixed age isolates vulnerability; 65 is the value the source uses
  for exactly this cross-user comparison (its §4.3).
- **`DEC-R1-04` `STATIC_COLLIDABLE` mapped to the car-driver curve.** In an
  ego-versus-fixed-object impact the exposed party is the ego occupant, and
  the car-driver curve is a car-occupant injury curve. Declared limitation
  and its direction: the source curve describes impacts against another
  car's deformable front, whereas a rigid obstacle concentrates load, so
  the approximation **under-estimates** risk.

Two further decisions preserve existing v4.7 invariants and are recorded
for traceability rather than being new choices:

- **`DEC-R1-05`** the curve is fed the **normal component** of closing
  speed, not its magnitude, preserving v4.7 §5.3's deliberate
  graze-versus-frontal discrimination. Since `u_i <= ||v_e - v_i||`, this
  is conservative in the under-estimating direction.
- **`DEC-R1-06`** configured speed caps remain **scenario-eligibility
  preconditions** (v4.7 §5.4) even though they no longer normalize the
  cost. Eligibility is out of scope here and the caps are still required by
  `progress` and `wrongway`.

`DEC-R1-07`: no `MOTORCYCLIST` actor class is introduced. The source
provides a curve (Table 4) and motorcyclists are substantially more
vulnerable than car occupants, so motorcycles currently classed as
`VEHICLE` have their risk under-estimated. Deferred pending an empirical
check of motorcycle frequency in the selected scenario pool.

Rejected alternative: a **fixed global cap** computed once over the
scenario pool. It removes the demonstrated inversion and needs no external
source, but the constant remains an artifact of dataset composition — it
must be recomputed and re-frozen whenever the pool changes, and it carries
no physical meaning. The injury-risk curve was preferred because it
additionally makes costs comparable *across actor classes*, which a single
scalar cap cannot do.

## Consequences

`c_1(t)`'s distribution changes. Any baseline already run and calibrated on
v4.7 `R1` is not directly comparable afterwards, and the two sets must not
be pooled as one experimental condition.

The impact is **asymmetric across algorithms** and must be reported as
such:

- under the default scalarizer `bounded_satisfaction_rank` (`SCAL-V1.0`
  §7.5) the categorical term depends on `I_1 = 1[m_1 = 0]`, which this
  change leaves untouched — any collision produced and still produces the
  same `a^3` penalty. Magnitude enters only the continuous tie-breaker,
  bounded to `±0.25`. The effect on scalar reward is small;
- a lexicographic learner optimizing `J_1` directly sees the full effect: a
  pedestrian collision weighs about 25× a vehicle-vehicle collision at the
  same closing speed (`u = 20 m/s`: 0.869 vs 0.034). This is intended and
  consistent with the source's vulnerability ranking, but it is a
  substantial change of objective.

No policy input, reward-vector shape, rule hierarchy, scalarization
contract, or scenario-eligibility rule changes. `R2`, `R3`, `R4` are
untouched.

Relationship to v4.7 §5.6, which justified omitting masses: no masses are
introduced, so that reasoning stands. What changes is its second half —
victim type is no longer only logging information, it now selects the risk
curve. The risk v4.7 sought to avoid (a light mass numerically penalizing a
VRU collision *less*) is avoided more strongly than before: the pedestrian
curve dominates the vehicle curve over the whole domain.

Mandatory-test impact: `tests/test_rulebook_v2_collision.py` asserts the
v4.7 numeric value `0.0625` in two tests. Updating those expectations is a
direct consequence of this approved specification change, not an adaptation
of tests to the implementation; recorded as `DEC-IMPL-001` in the ExecPlan.

## Approval Record

- Approved by: user
- Approval evidence: explicit message "confermo MAIS3+F, età 65,
  static→car driver" on 2026-07-26, after the three decisions were
  presented with options, numeric consequences, and a motivated
  recommendation for each. The preceding instruction to replace the
  scenario-dependent normalizer, and the requirement to record motivations
  and sources thoroughly, come from the same conversation.
