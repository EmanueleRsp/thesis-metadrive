---
title: "ExecPlan: dynamic-slot identity removal (encoder) and context-quota ranking by distance (observation)"
plan_id: "semantic-v3-dynamic-ranking-and-slot-embedding-removal"
specification: "docs/specifications/encoder_v1.2_specification.md (ENC-V1.2, new, supersedes ENC-V1.1 for the LQ variant); docs/specifications/observation_v1.1_specification.md (§8.1 Context quota, amended 2026-07-26; inherited unchanged by observation_v1.2_specification.md)"
status: "VERIFIED"
created: "2026-07-26"
last_updated: "2026-07-26"
related_adrs:
  - "docs/decisions/ADR-026-dynamic-slot-identity-removal-and-context-quota-ranking.md"
owner: "Claude Code session"
---

## 1. Objective and scope

Follow-up to `semantic_v3_actor_cap_normalization_fix_exec_plan.md`, which
deferred/retracted Findings 3 and 4 of the same review. After the user
explicitly rejected treating those findings as an accepted trade-off and
required a genuinely better-supported design (backed by literature where
possible, not a minimal patch), both findings were re-examined in depth,
including a user-run external literature search, and are implemented here:

- **Finding 3 (encoder):** `LatentQueryEncoderV3`'s learned
  `dynamic_slot_embedding` tagged the physical dynamic-actor buffer slot
  (0-15) with a per-index identity, even though OBS-V1.1 §7 assigns these
  slots sticky/first-fit — the slot index has no stable semantic meaning, so
  this could only encode spurious slot-position correlation. Removed.
- **Finding 4 (observation):** the non-conflict ("context quota") dynamic
  ranking key ordered by lane relation before Euclidean distance, letting a
  much closer actor in a different lane lose a slot to a farther same-lane
  actor. Replaced with plain distance.

Both decisions and their full literature basis are recorded in ADR-026; this
ExecPlan covers only the engineering requirements, tests, and validation.

Out of scope: the sticky/first-fit slot-persistence mechanism itself
(OBS-V1.1 §7, unaffected); the conflict-quota ranking key (CPA/TTC-based,
explicitly kept unchanged — validated, not just left alone, by the
literature reviewed in ADR-026); the MLP encoder variant (unaffected); any
change to the OBS-V1.2 flat dimension or token count.

## 2. Authoritative requirements

| ID | Requirement | Specification section |
|---|---|---|
| `REQ-ENC-01` | `LatentQueryEncoderV3` MUST NOT apply a learned per-slot identity embedding to dynamic-history tokens. | `encoder_v1.2_specification.md` §4.2 |
| `REQ-ENC-02` | `LatentQueryEncoderV3`'s output MUST be identical, within floating-point tolerance, when identical dynamic-actor content occupies a different one of the 16 physical slots. | `encoder_v1.2_specification.md` §6 |
| `REQ-ENC-03` | Route, static, control, and interaction slot embeddings MUST remain unchanged (those slot indices are semantically stable). | `encoder_v1.2_specification.md` §4.2 |
| `REQ-OBS-01` | The context-quota (non-conflict) dynamic-actor ranking key MUST order candidates by Euclidean distance to ego, then stable actor ID, with no lane-relation/route-ahead/lateral-distance precedence. | `observation_v1.1_specification.md` §8.1 (amended) |
| `REQ-OBS-02` | The conflict-quota ranking key MUST remain unchanged (CPA validity, `t_CPA`, `d_CPA`, lane relation, distance, actor ID). | `observation_v1.1_specification.md` §8.1 |

## 3. Current repository analysis (VERIFIED)

- `src/thesis_rl/agent/planners/encoders/lq_encoder.py:312,321,364-366`
  (`LatentQueryEncoderV3`, pre-fix) added
  `self.dynamic_slot_embedding = nn.Embedding(16, token_dim)` to every
  dynamic-history token, keyed by physical slot index — VERIFIED by direct
  read.
- `src/thesis_rl/envs/observations/causal_semantic.py:556-...`
  (`_assign_slots`) confirms sticky/first-fit persistent slot assignment: a
  track keeps its previously assigned slot as long as it remains selected,
  independent of its current rank — VERIFIED by direct read, consistent with
  OBS-V1.1 §7.
- `src/thesis_rl/envs/observations/causal_semantic.py:527-540` (`_dynamic_key`,
  pre-fix): `valid, t_cpa, d_cpa = self._cpa(actor, ego)` was already computed
  unconditionally for every candidate (conflict and non-conflict alike); the
  non-conflict branch discarded these values and used
  `(1, relation, ahead, abs(lateral), distance, actor_id)` instead — VERIFIED
  by direct read. `_dynamic_key` has a single definition, shared by
  `CausalSemanticBatchBuilder` and its subclass
  `PerceptionBoundedSemanticBatchBuilder` (the one wired for `semantic_v3`) —
  VERIFIED via `grep -n "def _dynamic_key"` (single match).
- `conf/agent/planner/encoder/lq_v3.yaml` declares `slot_embedding: true` and
  `architecture_version: 1.1-perception-bounded`, but `factory.py`
  (`build_encoder`) does not read or enforce either field — VERIFIED by
  direct read of `factory.py`; these are descriptive-only, so the config
  change below is a documentation-accuracy fix, not a behavioral one.
- `src/thesis_rl/runtime/io/metadata.py:17` defines a single, global
  `ENCODER_SPECIFICATION_ID = "ENC-V1.1"` constant logged into run metadata
  for provenance; it is not used as a runtime compatibility gate anywhere
  found by repository-wide grep — VERIFIED. No caller in `src/thesis_rl`
  passes an explicit `encoder_architecture_version="1.1-perception-bounded"`
  to `build_checkpoint_manifest`; this appears to be a pre-existing,
  unrelated gap (the manifest's declared default is `"1.0-final"`), flagged
  as a separate follow-up (not fixed here — out of scope, unrelated to
  Findings 3/4).
- No test file asserted the presence of `dynamic_slot_embedding` or the
  old lane-precedence context-quota ordering — VERIFIED via repository-wide
  grep across `tests/`.
- **Operational finding (discovered during implementation, not anticipated
  in planning):** three active production training runs (`td3-0`, `sac-0`,
  `ppo-0`) share this same repository checkout. Editing
  `lq_encoder.py` while they were running caused their async-evaluation
  workers to rebuild `LatentQueryEncoderV3` from the now-modified source and
  then fail to load their existing checkpoints (`strict=True` state-dict
  match, missing `dynamic_slot_embedding.weight` keys) — `sac-0` and `ppo-0`
  crashed (`AsyncEvaluationError`, tmux session terminated) during this
  ExecPlan's implementation, before the ExecPlan or ADR were finalized. This
  is the exact incompatibility documented in ADR-026's Consequences section,
  but its *timing* (occurring mid-implementation against live processes,
  rather than at a planned restart) was not anticipated or checked for
  beforehand. See Section 11 (Deviations).

## 4. Assumptions and invariants

- The observation tensor shape/dimensions (`(16,5,22)` dynamic group, OBS-V1.2
  flat dim 3064) are unaffected by either change.
- `REQ-ENC-01`/`REQ-ENC-02` change the LQ encoder's learned-parameter count and
  break checkpoint compatibility for the LQ variant by design (no migration
  path), consistent with the ENC-V1.0 -> ENC-V1.1 precedent.
- `REQ-OBS-01` changes which actors occupy context-quota slots under overflow;
  it does not change any tensor shape or the encoder's expected input.
- Both changes materially alter the effective input distribution seen by a
  policy trained under the old code; existing checkpoints are not silently
  compatible and MUST NOT be loaded across this boundary by implicit
  conversion.

## 5. Decisions

| ID | Category | Issue | Alternatives | Decision | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-ENC-01` | Architecture / specification deviation | ENC-V1.1 §4.2 normatively required a 16-slot dynamic embedding | (A) keep as-is (accepted trade-off); (B) add an explicit rank/priority feature alongside it; (C) remove it entirely (root-cause fix) | (C) — supported by Set Transformer/DETR precedent for permutation-invariant set encoding; (B) rejected as treating a symptom without removing the spurious-correlation source | New `ENC-V1.2` specification; LQ-variant checkpoint break; parameter count decreases by `16 * token_dim` | Approved by the user (2026-07-26), see ADR-026 Approval Record |
| `DEC-OBS-01` | Specification deviation | OBS-V1.1 §8.1 Context quota normatively used lane precedence before distance | (A) keep as-is; (B) extend CPA/TTC to the whole context quota; (C) plain Euclidean distance | (C) — (B) rejected after properly accounting for the interactivity-vs-distance crossover for larger N (Sun et al. 2021) and CPA's constant-velocity fragility for exactly the agents that matter most in an uncurated pool | OBS-V1.1 §8.1 amended in place; changes which actors occupy context-quota slots under overflow | Approved by the user (2026-07-26), see ADR-026 Approval Record |

## 6. Proposed design

- `lq_encoder.py`, `LatentQueryEncoderV3.__init__`: remove
  `self.dynamic_slot_embedding = nn.Embedding(16, token_dim)` and its entry in
  the `nn.init.normal_` initialization loop.
- `lq_encoder.py`, `LatentQueryEncoderV3.tokenize_structured`: remove the
  `dynamic = dynamic + self.dynamic_slot_embedding(...)` addition; dynamic
  tokens keep their group projection, type embedding, and
  `history_time_embedding` only.
- `LatentQueryEncoderV2` (ENC-V1.0/OBS-V1.1 legacy route, explicitly retained
  only for historical reproducibility per `docs/project_index.md`) is
  deliberately left unchanged — out of scope.
- `causal_semantic.py`, `_dynamic_key`: replace the non-conflict return value
  `(1, relation, ahead, abs(self._route_lateral(actor)), distance, actor.actor_id)`
  with `(1, distance, actor.actor_id)`. The now-unused `ahead` computation is
  removed; `relation` is retained (still used by the conflict-quota branch).
- `conf/agent/planner/encoder/lq_v3.yaml`: update `slot_embedding: false` and
  `architecture_version: 1.2-perception-bounded` for documentation accuracy
  (these fields are not read by `factory.py`, so this is a non-behavioral
  consistency fix).
- `src/thesis_rl/runtime/io/metadata.py`: bump
  `ENCODER_SPECIFICATION_ID = "ENC-V1.2"` so future run metadata accurately
  reports the encoder specification actually in effect.

## 7. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-ENC-01` | `AC-ENC-01` | `src/thesis_rl/agent/planners/encoders/lq_encoder.py` (`LatentQueryEncoderV3`) | `tests/test_encoders_v11.py::test_lq_v11_dynamic_output_is_invariant_to_which_slot_holds_an_actor` (asserts `not hasattr(encoder, "dynamic_slot_embedding")`) | VERIFIED |
| `REQ-ENC-02` | `AC-ENC-02` | same | same test (identical output across slot 0 vs slot 9) | VERIFIED |
| `REQ-ENC-03` | `AC-ENC-03` | same | `tests/test_encoders_v11.py` (existing tests, unchanged, still pass) | VERIFIED |
| `REQ-OBS-01` | `AC-OBS-01` | `src/thesis_rl/envs/observations/causal_semantic.py` (`_dynamic_key`) | `tests/test_causal_semantic_batch.py::test_context_quota_ranks_by_distance_not_lane_precedence` | VERIFIED |
| `REQ-OBS-02` | `AC-OBS-02` | same | `tests/test_causal_semantic_batch.py::test_conflict_quota_ranking_is_unchanged_by_the_context_quota_amendment` | VERIFIED |

## 8. Test strategy

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-ENC-01` | Unit | Encoder output invariant to which physical slot holds identical dynamic-actor content | Two flat OBS-V1.2 tensors, identical dynamic content in slot 0 vs slot 9, all other groups zero | `torch.testing.assert_close` on the two `(1,256)` outputs; `not hasattr(encoder, "dynamic_slot_embedding")` | `REQ-ENC-01`, `REQ-ENC-02` |
| `TEST-ENC-02` | Regression | Existing LQ v3/v3-lite/v3-micro/MLP contract tests | Full `test_encoders_v11.py` + `test_encoders_v10.py` | All previously passing tests still pass unchanged | `REQ-ENC-03` |
| `TEST-OBS-01` | Unit | Closer, different-lane actor outranks a farther same-lane actor in the context quota | `_dynamic_key` called directly on two non-conflict candidates | `key(closer, other-lane) < key(farther, same-lane)` | `REQ-OBS-01` |
| `TEST-OBS-02` | Unit | Conflict-quota ranking is CPA/TTC-based and unaffected by the context-quota change | `_dynamic_key` called with the candidate in `conflict_ids` | Returned tuple's rank-0 tier matches `_cpa(...)` directly | `REQ-OBS-02` |
| `TEST-OBS-03` | Regression | No unintended change to unrelated observation-builder behavior | Full `test_causal_semantic_batch.py` + `test_perception_bounded_semantic.py` | All previously passing tests still pass unchanged | `REQ-OBS-01`, `REQ-OBS-02` |

Commands:

```bash
docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_encoders_v10.py tests/test_encoders_v11.py tests/test_causal_semantic_batch.py tests/test_perception_bounded_semantic.py
docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/agent/planners/encoders/lq_encoder.py src/thesis_rl/envs/observations/causal_semantic.py tests/test_encoders_v11.py tests/test_causal_semantic_batch.py
docker compose run --rm dev uv run --no-sync ruff format --check src/thesis_rl/agent/planners/encoders/lq_encoder.py src/thesis_rl/envs/observations/causal_semantic.py tests/test_encoders_v11.py tests/test_causal_semantic_batch.py
```

`make rulebook-v2-check` was not run: neither changed module has a rulebook
dependency (the encoder consumes only the flat observation tensor; the
observation builder's ranking key does not touch rulebook code paths). The
full repository suite (`make test`) was not run for the same
proportionality reason documented in the preceding ExecPlan; the focused
suite above fully covers both changed modules and their direct consumers.

## 9. Milestones

- [x] M1 — Remove `dynamic_slot_embedding` from `LatentQueryEncoderV3`.
- [x] M2 — Add the slot-invariance regression test.
- [x] M3 — Replace the context-quota ranking key with plain distance in
      `_dynamic_key`.
- [x] M4 — Add context-quota/conflict-quota ranking regression tests.
- [x] M5 — Write `ENC-V1.2` specification (new file, supersedes ENC-V1.1 for
      the LQ variant).
- [x] M6 — Amend `observation_v1.1_specification.md` §8.1 with the new
      Context quota rule, amendments entry, and rationale/citations footnote.
- [x] M7 — Write ADR-026 covering both decisions with full literature basis.
- [x] M8 — Update `conf/agent/planner/encoder/lq_v3.yaml` and
      `ENCODER_SPECIFICATION_ID` for documentation/provenance accuracy.
- [x] M9 — Update `docs/project_index.md` registry rows.
- [x] M10 — Run focused tests, lint, and format-check; record results.
- [ ] M11 — Restart the active production runs under the corrected
      encoder/observation. **Not resolved by this ExecPlan**: `sac-0` and
      `ppo-0` already crashed mid-implementation (unplanned, see Section 3
      and Section 11) due to checkpoint incompatibility; `td3-0` remains at
      risk of the same crash on its next checkpoint reload. The user
      explicitly chose, mid-incident, to leave `sac-0` stopped for now and
      not preemptively intervene on `td3-0`/`ppo-0`. Whether/when to restart
      any of the three runs remains open.

## 10. Progress and findings log

- 2026-07-26: Implemented `REQ-ENC-01`/`REQ-ENC-02`/`REQ-ENC-03` and
  `REQ-OBS-01`/`REQ-OBS-02`. Added one encoder regression test and two
  observation-ranking regression tests. Ran the four directly affected test
  modules (39 passed). Ran focused ruff lint (clean) and format-check (one
  self-introduced formatting nit in the new context-quota test, fixed
  immediately; remaining "would reformat" hunks in `causal_semantic.py`
  confirmed via `ruff format --diff` to be the same pre-existing, unrelated
  formatting debt already documented in the preceding ExecPlan — outside the
  edited region, left untouched).
- 2026-07-26 (mid-implementation, unplanned): while `lq_encoder.py` was being
  edited, the active production runs `sac-0` and `ppo-0` (sharing this
  repository checkout) crashed on checkpoint reload with
  `RuntimeError: ... Unexpected key(s) in state_dict:
  "...encoder.dynamic_slot_embedding.weight"` via
  `AsyncEvaluationError`. Reported to the user immediately (push notification
  plus in-conversation explanation); user decided to leave `sac-0` stopped
  and not preemptively intervene on `td3-0`/`ppo-0`.

## 11. Deviations

- The implementation caused two live production training runs to crash
  during the editing window, before the change was fully documented or the
  user had confirmed a restart plan. This was not identified as a risk
  during planning: no check was made for actively running processes sharing
  this repository checkout before editing shared source code. This is
  recorded as a process gap for future changes to files consumed by live
  training/evaluation workers, not as a defect in the code change itself
  (the crash is the correct, intended failure mode for an incompatible
  checkpoint — `strict=True` state-dict matching did exactly what it should).

## 12. Files

| Path | Action | Purpose |
|---|---|---|
| `src/thesis_rl/agent/planners/encoders/lq_encoder.py` | Modified | `REQ-ENC-01`/`REQ-ENC-02`/`REQ-ENC-03` |
| `tests/test_encoders_v11.py` | Modified | `TEST-ENC-01` |
| `src/thesis_rl/envs/observations/causal_semantic.py` | Modified | `REQ-OBS-01`/`REQ-OBS-02` |
| `tests/test_causal_semantic_batch.py` | Modified | `TEST-OBS-01`/`TEST-OBS-02` |
| `docs/specifications/encoder_v1.2_specification.md` | New | ENC-V1.2, supersedes ENC-V1.1 for the LQ variant |
| `docs/specifications/observation_v1.1_specification.md` | Modified | §8.1 amendment, amendments entry |
| `docs/decisions/ADR-026-dynamic-slot-identity-removal-and-context-quota-ranking.md` | New | Both decisions, full literature basis |
| `conf/agent/planner/encoder/lq_v3.yaml` | Modified | Documentation-accuracy: `slot_embedding: false`, version bump |
| `src/thesis_rl/runtime/io/metadata.py` | Modified | `ENCODER_SPECIFICATION_ID` bump to `ENC-V1.2` |
| `tests/test_run_metadata.py` | Modified | Updated expected `encoder.specification_id` to `ENC-V1.2` |
| `docs/project_index.md` | Modified | Registry rows for ENC-V1.2 and the OBS-V1.1 amendment |
| `docs/implementation/semantic_v3_dynamic_ranking_and_slot_embedding_removal_exec_plan.md` | New | This ExecPlan |

## 13. Validation results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_encoders_v10.py tests/test_encoders_v11.py tests/test_causal_semantic_batch.py tests/test_perception_bounded_semantic.py` | `PASS` | 2026-07-26 | `39 passed in 3.61s` |
| `docker compose run --rm dev uv run --no-sync ruff check ...` (changed files) | `PASS` | 2026-07-26 | `All checks passed!` |
| `docker compose run --rm dev uv run --no-sync ruff format --check ...` (changed files) | `PARTIAL` | 2026-07-26 | One self-introduced nit in the new test fixed immediately; remaining `causal_semantic.py` "would reformat" hunks confirmed pre-existing/unrelated via `ruff format --diff` (same debt as the preceding ExecPlan, outside the edited region). `lq_encoder.py` and both test files format-clean. |

## 14. Final reconciliation

- `REQ-ENC-01`, `REQ-ENC-02`, `REQ-ENC-03`: **VERIFIED**.
- `REQ-OBS-01`, `REQ-OBS-02`: **VERIFIED**.
- M11 (production run restart): **NOT_IMPLEMENTED** — `sac-0`/`ppo-0` already
  crashed unplanned during implementation; user decided not to intervene for
  now. Remains an explicit open decision, not resolved by this ExecPlan.
- The process gap identified in Section 11 (no check for live processes
  sharing the repository checkout before editing shared source) is recorded
  here for future changes; no corrective mechanism was added to the
  repository itself, as that would be a distinct, unscoped change.

No other unintended changes remain in the diff for this ExecPlan's scope.
