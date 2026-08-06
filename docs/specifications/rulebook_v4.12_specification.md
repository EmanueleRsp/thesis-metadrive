# Specification: Wrong-way cost physics-noise deadband

## Metadata

- Feature: `rulebook_v2_wrongway_cost_physics_noise_deadband`
- Specification ID: `rulebook-v2-wrongway-cost-deadband`
- Version: `4.12`
- Status: `APPROVED`
- Date: `2026-08-06`
- Supersedes: `docs/specifications/rulebook_v4.7_specification.md`, version
  `4.7-final-implementation-complete` (only for §7.3.2's `wrongway` cost
  formula; the rest of v4.7 remains authoritative and unchanged)
- Related specifications: `docs/specifications/rulebook_scalarization_v1.0_specification.md`
  (downstream consumer of the corrected margin); the in-review
  `incoming/rulebook_scalarization_v1.1_specification_UNDER_REVIEW.md`
  identified this defect while auditing the tolerance-ownership boundary
  between rulebook margins and the scalarizer (its `DEC-SCAL11-003`)
- Related ADR: `docs/decisions/ADR-056-wrongway-cost-physics-noise-deadband.md`
- Authoritative: `YES`

## 1. Purpose And Context

v4.7 §7.3.2 defines
\(q_{\mathrm{wrongway}}(t)=\operatorname{clip}([-v_\parallel(t)]_+/v_{\max,e},0,1)\)
and states its intended behavior in prose: *"ego fermo: 0"* (ego at rest:
zero cost). The implementation
(`components/road.py::evaluate_wrongway`) computes \(v_\parallel\) directly
from the simulator's reported ego velocity with no floor, so a stationary
ego whose physics-solver residual velocity carries a hairline reverse
longitudinal component (order `1e-3`–`1e-1 m/s`, per `ADR-050`'s
measurement of the same noise source) produces a strictly positive
`cost`, not the `0` the prose already promises.

`ADR-050` (2026-08-01) previously diagnosed the same noise source for this
component's diagnostic `status` field and added
`WRONGWAY_STATUS_SPEED_EPSILON_MPS=0.1` there, explicitly leaving `cost`
untouched: *"cost stays the same continuous function... so the
scalarizer's input... [is] unaffected."* Re-examining that claim (this
conversation, 2026-08-06) found it incomplete: the scalarizer's own
canonicalization tolerance is `1e-8` (`SCAL-V1.0` §3.3), several orders of
magnitude below the `0.1 m/s`-scale physics noise, so the same standstill
noise `ADR-050` fixed cosmetically already flips the canonicalized macro
margin `m_3` away from exactly zero today, under the currently approved
`bounded_satisfaction_rank` default — triggering a full-weight categorical
R3-violation reward term (`priority_base^1 * (pattern - 1)`, not diluted)
on a correctly stopped ego. This is a correction to make the implementation
match what §7.3.2's prose already specifies, not a new behavioral policy.

## 2. Amendment To §7.3.2 (Wrong-way Cost)

\(q_{\mathrm{wrongway}}(t)\) gains an explicit physical deadband at the same
noise floor already frozen elsewhere for this exact noise source
(`RSS_STANDSTILL_SPEED_MPS`, `ADR-048`; `WRONGWAY_STATUS_SPEED_EPSILON_MPS`,
`ADR-050`; both `0.1 m/s`):

\[
u(t) = [-v_\parallel(t)]_+,
\qquad
\varepsilon_v = 0.1\ \mathrm{m/s},
\]

\[
q_{\mathrm{wrongway}}(t)
=
\begin{cases}
0, & u(t) \le \varepsilon_v, \\[4pt]
\operatorname{clip}\!\left(
\dfrac{u(t) - \varepsilon_v}{v_{\max,e} - \varepsilon_v},\, 0, 1
\right), & u(t) > \varepsilon_v.
\end{cases}
\]

This is a continuous, monotonically non-decreasing reparameterization of
the existing formula, not a discontinuous cutoff: at \(u=\varepsilon_v\)
both branches give exactly `0`; at \(u=v_{\max,e}\) the cost reaches exactly
`1`, the same boundary behavior as before. Requires
\(v_{\max,e}>\varepsilon_v\) (a real vehicle speed cap is always well above
`0.1 m/s`; `evaluate_wrongway` must raise if this precondition is violated,
consistent with its existing fail-fast cap validation).

The diagnostic `status` field, previously computed from a separate,
duplicated epsilon check on `-longitudinal_speed`
(`WRONGWAY_STATUS_SPEED_EPSILON_MPS`, `ADR-050`), is simplified to derive
directly from the now-deadbanded `cost`:
`VIOLATED` iff `cost > 0.0`, `SATISFIED` otherwise — the same convention
`aggregate_max_component` already uses macro-level. This removes the
duplicated tolerance definition; `status` and `cost` now share exactly one
noise floor instead of two independently maintained ones.

### 2.1 Amendment to `ADR-050`

`ADR-050`'s decision to leave `cost` untouched is superseded for this
narrow point: `cost` now shares the same `0.1 m/s` floor its `status` fix
already used. `ADR-050`'s diagnosis of the noise source and its exact
magnitude remain correct and are the empirical basis for this amendment;
only its scope conclusion ("cost is out of scope, unaffected") is revised.

## 3. Compatibility

- `RuleComponentResult.cost` for `wrongway` changes for any ego state with
  `0 < u(t) \le 0.1` — previously a small positive cost, now exactly `0`.
  States with `u(t)=0` (no reverse component) or `u(t) > 0.1` (genuine
  reverse motion) are unaffected in value, though the mapping from `u` to
  cost for `u > 0.1` is now an affine reparameterization of the same range
  (still `0` at the threshold, still `1` at `v_max,e`), not the old
  `u/v_max,e` ratio — a policy in a genuine reverse-motion state at, say,
  `u=1.0 m/s` sees a slightly larger cost under the new mapping than the
  old one at the same `u`, because the deadband compresses the "meaningful
  reverse" domain into the same `[0,1]` output range.
- `status` for `wrongway` is unchanged in every case already covered by
  `ADR-050`'s deadband (it already matched `cost > 0.0`'s pre-amendment
  behavior at the `0.1 m/s` boundary by construction); this amendment only
  removes the now-redundant second epsilon constant, it does not change
  `status`'s observable value for any input.
- Episode returns from any run using the `wrongway` component before this
  version are not directly comparable to runs after it, for scenarios where
  the ego experiences standstill or near-standstill states with hairline
  reverse residual velocity — i.e., most scenarios involving a stop.
- No change to `RuleComponentResult` shape, `TrafficControlRecord`, or the
  observation vector.
- Downstream scalarization (`SCAL-V1.0`, and the in-review `SCAL-V1.1`) is
  unaffected in contract — both consume `m_3` as already defined — but the
  numeric value of `m_3` changes for the standstill states described above,
  which is the entire purpose of this amendment.

## 4. Acceptance Criteria

### AC-RBWW-001: Standstill Noise Produces Exactly Zero Cost

- Given: an ego with a residual reverse longitudinal component
  `0 < u <= 0.1 m/s`;
- When: `evaluate_wrongway` runs;
- Then: `cost == 0.0` exactly, and `status == SATISFIED`.

### AC-RBWW-002: Continuity At The Deadband Boundary

- Given: `u` approaching `0.1 m/s` from above and below;
- When: `cost` is evaluated at `u = 0.1 - δ` and `u = 0.1 + δ` for small `δ`;
- Then: `cost` at both points converges to `0.0` as `δ -> 0` (no jump
  discontinuity at the boundary).

### AC-RBWW-003: Full-Scale Boundary Preserved

- Given: `u == v_max,e` (reverse speed at the configured cap);
- When: `evaluate_wrongway` runs;
- Then: `cost == 1.0`, unchanged from the pre-amendment formula's boundary
  value.

### AC-RBWW-004: Genuine Reverse Motion Still Flagged

- Given: `u` well above `0.1 m/s` (e.g. `u = 1.0 m/s` with a `10 m/s` cap);
- When: `evaluate_wrongway` runs;
- Then: `cost > 0.0` and `status == VIOLATED`.

### AC-RBWW-005: Status Derives From Cost, No Duplicated Tolerance

- Given: any valid input;
- When: `evaluate_wrongway` runs;
- Then: `status == VIOLATED` iff `cost > 0.0`; there is no longer a
  separately configured status-only epsilon distinct from the cost
  deadband.

## 5. References

- `docs/specifications/rulebook_v4.7_specification.md` §7.3 — unamended
  base contract (velocity sign convention, applicability, diagnostics).
- `docs/decisions/ADR-048-rss-standstill-applicability.md` — origin of the
  `0.1 m/s` physics-solver noise floor, reused here rather than inventing a
  new constant.
- `docs/decisions/ADR-050-wrongway-status-deadband.md` — prior, narrower
  fix this amendment extends and partially supersedes (§2.1).
- `incoming/rulebook_scalarization_v1.1_specification_UNDER_REVIEW.md`
  §3.1 — the scalarization-amendment review that surfaced this defect while
  auditing tolerance ownership between the rulebook and the scalarizer.

## 6. Approval Record

- Approved by: `user`
- Approval date: `2026-08-06`
- Approval evidence: explicit user approval in this conversation ("sì
  procedi e correggi il problema grazie"), following a full walkthrough of
  the mechanism (residual physics velocity -> non-zero `cost` -> non-zero
  canonical `m_3` -> categorical `I_3` flip -> full-weight reward penalty
  on a correctly stopped ego) and explicit confirmation that the fix
  reuses the already-established `0.1 m/s` noise floor rather than
  introducing a new semantic threshold.
- Approval notes: this is a narrow, single-component correction; it does
  not depend on and is not blocked by the outcome of the separate
  `SCAL-V1.1` scalarization amendment review.
- Repository path: `docs/specifications/rulebook_v4.12_specification.md`
- Project index updated: `YES` (`docs/project_index.md`, 2026-08-06)
