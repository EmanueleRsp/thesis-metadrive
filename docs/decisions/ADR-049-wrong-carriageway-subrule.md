# ADR-049: New R3 sub-rule `wrong_carriageway`

- Status: Approved
- Date: 2026-08-01
- Approval evidence: explicit user approval of
  `docs/implementation/rulebook_v2_cost_activation_corrections_v1_exec_plan.md`
  (`DEC-RBCOST-004`, `DEC-RBCOST-005`) in this conversation, following
  `docs/audits/rulebook_v2_cost_activation_audit_2026-08-01/findings.md` (F1)
  and the user's original report ("quando si invade l'altra carreggiata la
  sottoregola wrongway non la cattura ... credo dovrebbe essere un po' come
  offroad in cui si penalizza la percentuale di ego che sta nella carreggiata
  sbagliata").
- Affected specification: `rulebook_v4.7_specification.md`, new §7.3-bis
  (added by `rulebook_v4.10_specification.md`).

## Context

No existing sub-rule measures lane direction. `wrong_way`
(`components/road.py:evaluate_wrongway`) is, by v4.7 §7.3.2 design, a
signed-velocity rule that penalises only reverse motion — geometric
orientation alone must not be penalised, and that remains correct.
`offroad`'s reference surface (`geometry/drivable.py`) unions every
vertically compatible lane regardless of legal direction, per v4.7 §7.2.1
("Non dipende ... dalla direzione legale"), so the opposing carriageway is
part of the drivable surface and `offroad` returns 0 there. No other
component (`ttc`, `rss`, `rss_lateral`, `clearance`, `progress`) is
direction-aware. A vehicle driving forward in the oncoming lane of an empty
road therefore carries zero cost from every sub-rule except `solid_line`,
which was separately found broken for exactly the centreline marking that
separates the two carriageways on PG maps (`ADR` not yet filed for the
marking-coverage fix, tracked as `DEC-RBCOST-006/008/009` in the ExecPlan).

## Decision

Add `wrong_carriageway` to `road_traffic_compliance` (R3):

```
q_wrong_carriageway = A(P_e ∩ (C_opp \ C_aligned)) / A(P_e)
```

`C_aligned` and `C_opp` are, over the same vertically compatible lanes
`offroad` already uses, the union of lane polygons whose centreline tangent at
the ego's projection agrees with (`cos >= 0.5`, i.e. within +-60 degrees) or
opposes (`cos <= -0.5`) the canonical route tangent at the ego's route
projection, respectively; a lane in neither cone (a crossing branch inside a
junction) contributes to neither surface. Subtracting `C_aligned` from
`C_opp` is what keeps the cost at 0 inside junctions: an ego turning left is
inside its own route-aligned junction lane, so its footprint's overlap with an
opposing through lane's polygon is excluded.

The rule is memoryless, with no time ramp. An earlier draft proposed one, as
robustness against transient junction-edge overlap; it is withdrawn because
that overlap was hypothesised, not measured, and because `offroad` — the
direct structural analogue the user asked this rule to mirror — has none
either. If transient junction false positives are observed, they are a
geometry defect in the direction-cone logic to be fixed there, not masked
with a timer.

`DEC-RBCOST-005`: v1 charges the cost unconditionally, including when the
separating marking permits passing (a broken centreline). Suppressing the
cost during a legal overtake would require a reliable nearest-separating-
marking association that does not exist yet; building it needs exactly the
diagnostic (separating-marking class, recorded per activation) this v1
already collects. Revisit once that data shows whether legal overtakes are
being systematically suppressed by the resulting cost.

Wired as `ComponentDefinition("wrong_carriageway", ROAD_TRAFFIC_COMPLIANCE,
evaluate_wrong_carriageway)` in `registry.py`, added to the
`road_traffic_compliance` aggregation group in `aggregation.py`, and fed from
a new `carriageway_surfaces_for_ego` helper in `geometry/drivable.py` that
reuses the same vertical-compatibility gate as `drivable_surface_for_ego`
without altering that function's return value (`offroad`'s reference surface
is untouched).

## Consequences

An ego occupying the opposing carriageway now carries a graded R3 cost
proportional to the invaded footprint fraction, closing the gap the user
identified. `RulebookResult.components` gains one key; `road_traffic_
compliance`'s worst-component selection can now be `wrong_carriageway`. No
existing sub-rule's behaviour changes. The observation vector and reward
scale are unaffected (this is an additional R3 cost term, not an observation
change); episode returns on scenarios where this cost activates are not
comparable with runs completed before this change.
