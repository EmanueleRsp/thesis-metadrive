# ADR-060: Unified memoryless `wrong_direction` sub-rule, superseding ADR-050 and ADR-056

- Status: Approved
- Date: 2026-08-09
- Approval evidence: explicit user approval in this conversation. The user
  challenged the existing pair of sub-rules on the merits — *"con wrongway
  quindi staresti illegalizzando la retromarcia, il che non ha senso"* — and,
  after the nuPlan Driving Direction Compliance anchors were produced, approved
  the recommended unification (*"Vai con la decisione consigliata"*). The
  memoryless reformulation was then approved separately after the user raised
  the Markov-observability constraint attributed to their supervisor.
- Affected specification: `docs/specifications/rulebook_v4.13_specification.md`
  §3 (amends `rulebook_v4.7_specification.md` §7.3 and the §7.3-bis introduced
  by v4.10).
- Supersedes: `ADR-050` (wrongway status deadband) and `ADR-056` (wrongway cost
  physics-noise deadband).
- ExecPlan: `docs/implementation/reward_scale_and_episode_contract_v1_exec_plan.md`
  (`DEC-RSEC-004`, `DEV-RSEC-002`, `DEV-RSEC-003`).

## Context

R3 currently carries two direction sub-rules:

- `wrongway` (`components/road.py:175`), kinematic: the ego velocity projected
  on the canonical route tangent, normalized by the configured speed cap, with a
  `0.1 m/s` deadband;
- `wrong_carriageway` (`components/road.py:121`, `ADR-049`), positional: the
  footprint-area fraction on the opposing-direction surface.

They are not redundant. Reversing inside one's own lane produces
`wrongway > 0` and `wrong_carriageway = 0`; overtaking by occupying the
oncoming carriageway while still advancing produces the opposite. Both are
referenced to the same route tangent, which is the source-declared **legal**
lane direction, deliberately preserved by `DRIVING-MISSION-V1.1.1` §6.

Three defects nevertheless stand.

**The tolerance is a noise floor, not a manoeuvre allowance.**
`WRONGWAY_SPEED_EPSILON_MPS = 0.1 m/s` was introduced (`ADR-050`) and later
extended to the cost (`ADR-056`) to suppress physics-solver residual velocity.
It was never intended as a legal tolerance, and its effect is that any real
reverse manoeuvre is charged as a road-rule violation. Reversing is a normal,
legal manoeuvre; charging it is a specification defect, not a tuning question.

**The normalization makes the rule almost never binding at full strength.** The
cost divides by `v_max = 22.2 m/s`, so `q = 1` requires reversing at 80 km/h.
In practice the sub-rule contributes a small fraction of its range.

**Two sub-rules where the reference benchmarks have one.** nuPlan exposes a
single *Driving Direction Compliance* metric; Waymax exposes a single
`wrongway`. Neither splits the concept. Because R3 aggregates by `max`, merging
them changes no numeric outcome — it only removes a distinction that has
already misled a reader of this repository.

nuPlan's anchors are: reverse displacement below **2 m per 1 s** is compliant
(score 1), between 2 and 6 m is partial (0.5), above 6 m is a violation (0).
That is precisely the manoeuvre allowance this rule lacks.

The nuPlan formulation cannot, however, be adopted literally. It requires a 1 s
history of signed route displacement, and that history is **not** recoverable
from the agent's observation: `ego_history` carries signed route-relative
velocity but is only 5 steps deep (`history_length: 5`, i.e. 0.5 s), while the
21-step `context_history` carries `hypot(velocity)` — an unsigned magnitude that
cannot distinguish forward from reverse (`causal_semantic.py:572,2425`). A cost
depending on that window would depend on information the policy does not have.
Extending either block would change the observation dimensionality and
invalidate every checkpoint, which is a disproportionate price for one sub-rule.

## Decision

Replace `wrongway` and `wrong_carriageway` with a single R3 sub-rule:

\[
q_{\mathrm{rev}}(t)=\operatorname{clip}\!\left(\frac{[-v_{\parallel}(t)]_+ - 2}{6-2},\,0,\,1\right),
\qquad
q_{\mathrm{wrong\_direction}}(t)=\max\!\big(q_{\mathrm{rev}}(t),\,q_{\mathrm{carr}}(t)\big)
\]

with speeds in `m/s`; `q_carr` is the existing `evaluate_wrong_carriageway`,
unchanged.

`q_rev` is a function of the current state alone. `RulebookMemory` gains no
field, and `v∥` is reconstructible from the current observation frame, so the
cost stays Markovian with respect to what the policy observes.

`WRONGWAY_SPEED_EPSILON_MPS` and the separate status epsilon are removed. A
`2 m/s` tolerance sits twenty times above the `0.1 m/s` physics noise floor that
`ADR-050` and `ADR-056` were patching, so both patches become unnecessary rather
than being re-tuned. This supersedes them on the merits, as the user has
previously directed that earlier approvals be re-evaluated against new evidence
rather than treated as immutable.

## Consequences

- Legal reverse manoeuvres below `2 m/s` are no longer charged. Sustained
  reverse travel is charged progressively and reaches full cost at `6 m/s`,
  a range the previous normalization never reached in practice.
- On any trajectory containing no reverse motion the reward is numerically
  unchanged, because R3 already aggregated by `max`.
- Two repository-specific patches (`ADR-050`, `ADR-056`) are deleted rather than
  maintained. One tolerance replaces three.
- **Declared deviation from nuPlan.** The anchors `2` and `6` are displacements
  per second in nuPlan and speeds here. They coincide for sustained
  constant-speed reversing and diverge for a brief high-speed burst: 0.3 s at
  5 m/s is 1.5 m, compliant under nuPlan, partially charged here. This is an
  explicit adaptation for Markov observability and must be described as such,
  not as the nuPlan metric.
- **Declared limitation.** The reference tangent is the ego's own route. Where
  the ego is far from its route the tangent is no longer the local legal
  direction, and the sub-rule's interpretation degrades accordingly. This
  limitation predates this ADR and is unchanged by it.
- Diagnostics retain both components separately so the two failure modes remain
  distinguishable in analysis even though they share one cost.

Regression tests: `TEST-RSEC-005` (R3 exposes `wrong_direction` and no
`wrongway`), `TEST-RSEC-006` (anchors: `1.9 → 0`, `4.0 → 0.5`, `6.0 → 1`),
`TEST-RSEC-007` (memorylessness: identical cost for identical states with
different histories, and no new `RulebookMemory` field). The `ADR-056` tests are
replaced rather than deleted: their intent — a stopped or noise-perturbed ego is
never charged — is re-asserted against `wrong_direction`, which satisfies it by
a margin of twenty.
