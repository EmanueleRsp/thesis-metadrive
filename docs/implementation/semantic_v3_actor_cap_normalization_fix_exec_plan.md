---
title: "ExecPlan: semantic_v3 actor-cap normalization fix and dead-code cleanup"
plan_id: "semantic-v3-actor-cap-fix"
specification: "docs/specifications/observation_v1.1_specification.md (§10.2, amended 2026-07-26); docs/specifications/observation_v1.2_specification.md (inherits §10.2 unchanged, no override)"
status: "VERIFIED"
created: "2026-07-26"
last_updated: "2026-07-26"
related_adrs: []
owner: "Claude Code session"
---

## 1. Objective and scope

During a requested deep review of the perception-bounded semantic observation
(`obs=semantic_v3`, the observation actually used by the active production
training runs) against MetaDrive's default `LidarStateObservation`, four
findings were reported to the user. This ExecPlan covers the two findings the
user asked to have fixed:

- **Finding 1 (bug):** the relative-velocity normalization scale for actors
  without a configured speed cap (`configured_speed_cap_mps is None`, which is
  always the case for `ActorClass.PEDESTRIAN`/`ActorClass.CYCLIST` per
  `metadrive_live.py:_actor_speed_cap`) depended on the observed actor's own
  instantaneous speed, making the scale non-stationary exactly for the
  safety-critical VRU actor class, and worst exactly during high-closing-speed
  encounters.
- **Finding 2 (dead code):** `PerceptionBoundedSemanticBatchBuilderV12`, a
  second, unreachable ~320-line implementation of the OBS-V1.2 builder, never
  imported outside its own module and already semantically drifted from the
  actually-used `PerceptionBoundedSemanticBatchBuilder` (e.g. differing static
  type-index fallback: `3` vs `4`).

Findings 3 (dynamic slot binding is a persistent-track assignment, not a live
priority rank) and 4 (non-conflict ranking heuristic ordering) were evaluated
and are explicitly **out of scope for a code change**: Finding 3 was found to
already conform to an explicit OBS-V1.2 §7 requirement ("a track that
reappears... SHOULD reuse its previous slot") and is documented here only as a
retraction of the earlier critique; Finding 4 requires an empirical overflow
check that is not retrievable from existing logs (`SemanticOverflowDiagnostics`
is not logged anywhere in production code today, confirmed by repository-wide
grep) and is deferred to a separate follow-up.

Out of scope: any change to the OBS-V1.2 flat dimension, encoder contract
(ENC-V1.1), Rulebook formulas, or the dynamic/static/control/interaction slot
capacities.

## 2. Authoritative requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-001` | The relative-speed normalization scale ("ego cap + actor cap") MUST NOT depend on the observed actor's own instantaneous velocity when that actor has no configured speed cap. | `observation_v1.1_specification.md` §10.2 (amended 2026-07-26) |
| `REQ-002` | Unreachable/duplicate observation-builder code MUST NOT remain in the production module once identified, to prevent silent behavioral drift between a "used" and an "unused" implementation of the same contract. | Repository convention (AGENTS.md: no unrelated dead code; internal implementation detail, no spec impact) |

## 3. Current repository analysis (VERIFIED)

- `src/thesis_rl/envs/observations/causal_semantic.py:658-696` (`_dynamic_features`,
  now shifted after the fix) computes `relative_velocity` and clips it to
  `ego_speed_cap + actor_cap`. Before the fix, `actor_cap` was
  `actor.configured_speed_cap_mps or max(ego_speed_cap, hypot(*actor.velocity_xy), 1.0)`.
- `src/thesis_rl/rulebook/v2/context/metadrive_live.py:86-94`
  (`_actor_speed_cap`) returns `None` unconditionally for any actor whose
  `actor_class is not ActorClass.VEHICLE` — VERIFIED by direct read, not
  inferred.
- `src/thesis_rl/envs/thesis_scenario_env.py:402-451` wires
  `PerceptionBoundedSemanticBatchBuilder` (not `CausalSemanticBatchBuilder`,
  not the now-removed `PerceptionBoundedSemanticBatchBuilderV12`) whenever a
  `SemanticStateObservationV3` instance is present among the env's
  observations — VERIFIED, this is the only builder reachable for
  `obs=semantic_v3`, i.e. the Makefile's actual `run-train` target.
  `ego_speed_cap_mps` is not passed in `builder_kwargs`, so it is always
  `None` and the builder always falls back to `ego.configured_speed_cap_mps`
  per-episode — unaffected by this fix.
- `src/thesis_rl/envs/observations/causal_semantic.py` previously also defined
  `PerceptionBoundedSemanticBatchBuilderV12` (a second, independent subclass of
  `CausalSemanticBatchBuilder`, not of `PerceptionBoundedSemanticBatchBuilder`).
  VERIFIED via `grep -rn "PerceptionBoundedSemanticBatchBuilderV12"
  src/thesis_rl` (only the definition itself, before this fix) and via
  `__all__` (it was never exported). No test file referenced it either
  (`grep -rln "PerceptionBoundedSemanticBatchBuilderV12" tests` was empty).
  Removal is therefore a pure deletion with no behavioral impact on any
  reachable code path.
- Test fixtures: `tests/test_causal_semantic_batch.py` (`_actor`, `_context`,
  `_Vehicle` helpers) build actors with a hardcoded
  `configured_speed_cap_mps=20.0` and `actor_class=ActorClass.VEHICLE` by
  default; overriding both via `dataclasses.replace` is sufficient to
  construct a VRU fixture without touching the shared helper.

## 4. Assumptions and invariants

- Ego frame convention, `dt = 0.1 s`, and all existing OBS-V1.2 shape/mask
  invariants are unaffected and unchanged by this fix.
- The fix changes a **numeric edge case only**: whenever
  `hypot(*actor.velocity_xy) <= ego_speed_cap` (the overwhelming majority of
  VRU encounters, since typical pedestrian/cyclist speeds are well below a
  vehicle's configured cap), the produced normalized value is bit-for-bit
  identical before and after the fix, because `max(ego_speed_cap,
  hypot(...), 1.0)` was already dominated by `ego_speed_cap` in that regime.
  The only observable change is for the rarer case of a VRU actor moving
  faster than the ego's configured speed cap, where the value now saturates
  at the existing `[-1, 1]` clip bound instead of silently rescaling.

## 5. Decisions

| ID | Category | Issue | Alternatives | Decision | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-001` | Specification clarification | The approved OBS-V1.1 §10.2 text ("ego cap + actor cap") does not define the fallback when the observed actor has no configured cap. | (A) fall back to `ego_speed_cap` alone; (B) keep the actor's own instantaneous speed in the fallback; (C) introduce a new fixed per-class VRU speed constant | (A) — matches the existing dominant-case behavior exactly, requires no new invented constant, and removes the non-stationary-scale defect | Numeric encoding of `relative_velocity` for VRU actors faster than the ego cap changes from rescaled to saturated | Approved by the user (2026-07-26): "le specifiche servono per tenere traccia delle componenti implementate... se verifichiamo che ci sono cose non fatte bene le possiamo modificare, aggiornando poi la specifica" |
| `DEC-002` | Implementation detail | `PerceptionBoundedSemanticBatchBuilderV12` is unreachable | (A) delete; (B) keep as documented experimental scaffold | (A) — no reference anywhere, already drifted from the used implementation, pure maintenance risk | None (dead code) | Resolved, no approval gate (does not change observable behavior) |

## 6. Proposed design

- `causal_semantic.py`, `_dynamic_features`: replace
  `actor.configured_speed_cap_mps or max(ego_speed_cap, hypot(*actor.velocity_xy), 1.0)`
  with `actor.configured_speed_cap_mps or ego_speed_cap`. No signature or
  return-shape change; `_clip` already saturates out-of-range values, so no
  new error path is introduced.
- `causal_semantic.py`: delete `PerceptionBoundedSemanticBatchBuilderV12` in
  full (was lines 1900-2224); `__all__` already did not export it, so no
  import-site changes are needed anywhere.
- `observation_v1.1_specification.md` §10.2: add a clarifying note
  documenting the resolved fallback rule and an `amendments:` frontmatter
  entry dated 2026-07-26. `observation_v1.2_specification.md` is unaffected
  text-wise (it inherits §10.2 by its own "existing field definitions...
  remain normative" clause, §5) — no edit needed there.

## 7. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-001` | `AC-001` | `src/thesis_rl/envs/observations/causal_semantic.py` (`_dynamic_features`) | `tests/test_causal_semantic_batch.py::test_vru_relative_velocity_scale_is_invariant_to_the_actors_own_speed` | VERIFIED |
| `REQ-002` | `AC-002` | `src/thesis_rl/envs/observations/causal_semantic.py` | Full `test_causal_semantic_batch.py` + `test_perception_bounded_semantic.py` (no regression) | VERIFIED |

## 8. Test strategy

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-001` | Unit | Same physical relative velocity, different VRU absolute speed (both above ego cap) | Two scenes: ego at rest + pedestrian at 25 m/s; ego at 10 m/s + pedestrian at 35 m/s (identical relative velocity 25 m/s in both) | Normalized relative-velocity feature is bit-identical (`25.0/40.0`) in both scenes | `REQ-001` |
| `TEST-002` | Regression | No behavioral change for vehicle actors (which always have a configured cap) | Full existing `test_causal_semantic_batch.py` + `test_perception_bounded_semantic.py` suites | All previously passing tests still pass unchanged | `REQ-001`, `REQ-002` |

Commands:

```bash
docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_causal_semantic_batch.py tests/test_perception_bounded_semantic.py
docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/observations/causal_semantic.py tests/test_causal_semantic_batch.py
docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/envs/observations/causal_semantic.py tests/test_causal_semantic_batch.py
```

`make rulebook-v2-check` and the full repository suite were not run for this
change (out of scope: this module has no rulebook dependency and the full
suite is a much broader, slower check not warranted for a two-line behavioral
fix plus a pure deletion — the focused suite above fully covers the changed
module).

## 9. Milestones

- [x] M1 — Implement the `actor_cap` fallback fix.
- [x] M2 — Add the VRU scale-invariance regression test.
- [x] M3 — Remove the dead `PerceptionBoundedSemanticBatchBuilderV12` class.
- [x] M4 — Amend `observation_v1.1_specification.md` §10.2 with the
      clarification and an `amendments:` entry.
- [x] M5 — Run focused tests, lint, and format-check; record results.
- [ ] M6 — Decide whether/when to restart the 3 active production runs
      (`td3-0`, `sac-0`, `ppo-0`) under the corrected observation encoding.
      Deferred: this is an operational decision (production process
      restart), separate from the engineering correctness of the fix, and
      requires explicit user confirmation.

## 10. Progress and findings log

- 2026-07-26: Implemented `REQ-001`/`REQ-002`. Confirmed via `metadrive_live.py`
  that the bug affects 100% of pedestrian/cyclist relative-velocity encodings
  whenever the VRU's own speed exceeds the ego's configured cap; in the
  dominant case (VRU slower than ego cap) the fix is a no-op bit-for-bit.
  Confirmed the dead V12 class had zero reachable references before deleting
  it. Added one regression test; ran the two directly affected test modules
  (24 passed) plus focused ruff lint (clean) and format-check (clean for the
  edited region; the file has pre-existing, unrelated formatting debt further
  down in `_build_controls_v12`/`_yellow_required_stop_distance`, consistent
  with the repository-wide formatting baseline not being clean — left
  untouched per AGENTS.md instructions against unrelated mass reformatting).

## 11. Deviations

No deviations identified.

## 12. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/envs/observations/causal_semantic.py` | Modified | `REQ-001` fix; `REQ-002` dead-code removal |
| `tests/test_causal_semantic_batch.py` | Modified | `TEST-001` regression test |
| `docs/specifications/observation_v1.1_specification.md` | Modified | `DEC-001` clarification, §10.2 + amendments |
| `docs/implementation/semantic_v3_actor_cap_normalization_fix_exec_plan.md` | New | This ExecPlan |

## 13. Validation results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_causal_semantic_batch.py tests/test_perception_bounded_semantic.py` | `PASS` | 2026-07-26 | `24 passed in 2.12s` |
| `docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/envs/observations/causal_semantic.py tests/test_causal_semantic_batch.py` | `PASS` | 2026-07-26 | `All checks passed!` |
| `docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/envs/observations/causal_semantic.py tests/test_causal_semantic_batch.py` | `PARTIAL` | 2026-07-26 | `causal_semantic.py` reported "would reformat"; `ruff format --diff` confirmed all suggested hunks are pre-existing, unrelated to the edited regions (`_build_controls_v12` line-length wrapping, `_yellow_required_stop_distance`); `test_causal_semantic_batch.py` already formatted. No action taken, per repository-wide non-clean formatting baseline documented in AGENTS.md. |

## 14. Final reconciliation

- `REQ-001`: **VERIFIED** — fix implemented, regression test added and
  passing, dominant-case bit-identical behavior confirmed by test design.
- `REQ-002`: **VERIFIED** — dead class removed, no reachable references
  existed, full focused suite still passes.
- Finding 3 (slot binding): **NOT_APPLICABLE** — retracted; already conforms
  to OBS-V1.2 §7's explicit slot-persistence requirement.
- Finding 4 (ranking heuristic): **NOT_IMPLEMENTED**, deliberately deferred —
  requires empirical overflow evidence not available from existing production
  logs (`SemanticOverflowDiagnostics` is not logged in production code today);
  proposed as separate follow-up work, not blocking this ExecPlan.
- M6 (restart of active production runs under the corrected encoding):
  **NOT_IMPLEMENTED** — explicit user decision pending, tracked as an open
  milestone above.

No unintended changes remain in the diff for this ExecPlan's scope.
