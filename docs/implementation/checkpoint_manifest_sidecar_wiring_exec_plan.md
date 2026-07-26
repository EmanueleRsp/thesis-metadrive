# ExecPlan: checkpoint-manifest sidecar wiring

- Status: `VERIFIED`
- Date: 2026-07-26
- Authoritative specification: none (no scientific/behavioral deviation;
  additive diagnostic tooling only, see "Decision control" below)
- Related: ADR-026 (the encoder change that triggered the operational
  incident motivating this ExecPlan)

## Objective

Give checkpoint incompatibility a clear, early, field-named error instead
of PyTorch's raw `RuntimeError: ... Unexpected key(s) in state_dict: ...`,
without migrating the existing flat-`.zip` checkpoint convention to a
heavier scheme.

## Motivation (operational incident)

While implementing ADR-026 (removal of `dynamic_slot_embedding` from
`LatentQueryEncoderV3`), editing `lq_encoder.py` while `sac-0`/`ppo-0`
(and, moments later, `td3-0`) were live caused all three to crash on
their next checkpoint resume/eval: their saved state_dict no longer
matched the just-edited encoder architecture. The crash itself was the
correct outcome (the weights genuinely cannot load), but the error
message gave no indication of *why*, in domain terms.

Investigating a narrower gap reported via a spawned task
("`encoder_architecture_version` default is never overridden") revealed a
larger finding: the entire manifest-based compatibility system already
present in the codebase (`CheckpointManifest`, `build_checkpoint_manifest`,
`assert_checkpoint_compatible`, `Agent.save_generation`,
`load_planner_backend_generation`, the whole
`sb3_extensions/checkpointing.py` "atomic generation" scaffolding) has
zero production callers. Production save/load uses SB3's native flat
`.zip` files exclusively, via `Agent.save()` and
`runtime/wiring/builders.py::load_planner()`.

## Decision control

This ExecPlan does not introduce a new scientific behavior, specification
deviation, or data policy requiring a fresh approval gate beyond what was
already given: the user explicitly requested this wiring
("Ah no allora implementalo") after a plain-language explanation of
exactly the design implemented here (manifest sidecar + fail-open
pre-load check). No ADR is required: this is a diagnostic/reliability
reinforcement, not a change to any approved formula, semantics, or
acceptance criterion.

## Requirements

| ID | Behavior |
|---|---|
| REQ-CKPT-01 | `Agent.save()` writes a `<stem>.manifest.json` sidecar when a manifest has been set via `set_checkpoint_manifest`, and writes nothing when it has not been set (no behavior change for existing callers) |
| REQ-CKPT-02 | `load_planner()` rejects, before any backend `state_dict` load, a checkpoint whose sidecar manifest differs from the current run's manifest on any state_dict-shape-relevant field, naming the first mismatched field |
| REQ-CKPT-03 | The compatibility check ignores differences in provenance-only fields (`git_commit`, `sb3_version`, `sb3_commit`, `seed`) and in `rulebook_*`/`scalarization_*`/`legacy_*` fields (already covered by the pre-existing `assert_reward_semantics_compatible`), so normal development (new commits, dependency bumps) does not spuriously block a resume |
| REQ-CKPT-04 | The compatibility check is a no-op when no sidecar file exists next to the checkpoint, so `td3-0`/`sac-0`/`ppo-0`'s pre-existing checkpoints (saved before this feature) remain resumable |
| REQ-CKPT-05 | The manifest is built from `cfg` and an already-constructed `env` only, identically at save time and load time, with no encoder/planner instantiation required |

## Design

Mirrors the existing, already-wired `contracts/reward_semantics.py`
pattern (`build_reward_semantics_identity` → `write_reward_semantics_sidecar`
→ `assert_reward_semantics_compatible`, wired into `Agent.save()` and
`load_planner()`), applied to the pre-existing but unwired
`contracts/checkpoint_manifest.py` contract.

**Design correction made during planning**: the existing
`assert_checkpoint_compatible` compares all 35
`CHECKPOINT_MANIFEST_FIELDS`, including `git_commit`/`sb3_version`/
`sb3_commit`/`seed`. Reusing it unmodified would reject a resume after
almost any commit in this actively-developed repository. A new,
narrower `CHECKPOINT_MANIFEST_SHAPE_FIELDS` subset (the fields that
determine `state_dict` shape) and a new comparison function
(`assert_checkpoint_manifest_compatible_if_present`) were introduced
instead; `assert_checkpoint_compatible` itself is untouched and remains
used only by the unwired "atomic generation" system, where all-field
strict equality is the intended, correct contract.

## Files

| File | Change |
|---|---|
| `src/thesis_rl/contracts/checkpoint_manifest.py` | Added `CHECKPOINT_MANIFEST_SHAPE_FIELDS`, `checkpoint_manifest_sidecar_path`, `write_checkpoint_manifest_sidecar`, `assert_checkpoint_manifest_compatible_if_present` (fail-open). No changes to existing functions/dataclass. |
| `src/thesis_rl/runtime/wiring/checkpoint_identity.py` (new) | `build_current_checkpoint_manifest(cfg, env)`: pure function sourcing every manifest field from `cfg`/`env`; `flat_dim` from `env.observation_space.shape[0]` (not a cfg-derived schema lookup, so it is correct for any `obs.type`, including `lidar_state`); `encoder_architecture_version` from the encoder config's own `architecture_version` field (already present and correctly bumped in every `conf/agent/planner/encoder/*.yaml`), not from `BaseEncoder`'s never-overridden `ClassVar`; rulebook/scalarization/legacy fields reused from `build_reward_semantics_identity(cfg)`. |
| `src/thesis_rl/agent/agent.py` | Added `checkpoint_manifest` attribute + `set_checkpoint_manifest`; `Agent.save()` writes the sidecar when set, mirroring the existing `checkpoint_identity`/reward-semantics sidecar. `save_generation` untouched. |
| `src/thesis_rl/runtime/loops/train_loop.py` | One line after the existing `agent.set_checkpoint_identity(...)` call: `agent.set_checkpoint_manifest(build_current_checkpoint_manifest(cfg, env))`. Covers all 7 `agent.save(...)` call sites with no further changes. |
| `src/thesis_rl/runtime/wiring/builders.py` | `load_planner()` calls `assert_checkpoint_manifest_compatible_if_present(...)` right after the existing `assert_reward_semantics_compatible(...)` call and before `load_planner_backend(...)`. Covers all three production load call sites (`train_loop.py` resume, `train_loop.py` `_make_eval_agent`, `async_evaluation.py`), since all three call `load_planner()`. |
| `tests/test_checkpoint_manifest_sidecar.py` (new) | 6 regression tests, see below. |

No changes to `sb3_extensions/checkpointing.py`, `load_planner_backend_generation`, or `Agent.save_generation` — confirmed-unused scaffolding, left as-is.

## Traceability

| REQ | Test |
|---|---|
| REQ-CKPT-01 | `test_agent_save_writes_manifest_sidecar_when_set`, `test_agent_save_skips_manifest_sidecar_when_unset` |
| REQ-CKPT-02 | `test_load_planner_rejects_shape_incompatible_sidecar_before_backend_load` |
| REQ-CKPT-03 | `test_load_planner_ignores_sidecar_mismatch_on_provenance_only_fields` |
| REQ-CKPT-04 | `test_load_planner_skips_manifest_check_when_sidecar_absent` |
| REQ-CKPT-05 | `test_build_current_checkpoint_manifest_flat_dim_from_env_observation_space` |

## Validation results

```
docker compose run --rm dev uv run --no-sync python -m pytest -q \
    tests/test_checkpoint_manifest_sidecar.py tests/test_checkpointing.py tests/test_run_metadata.py
# 18 passed

docker compose run --rm dev uv run --no-sync python -m pytest -q tests/ \
    -k "checkpoint or agent_save or reward_semantics or builders"
# 27 passed, 969 deselected

docker compose run --rm dev uv run --no-sync python -m pytest -q \
    tests/test_reward_semantics.py tests/test_agent_pipeline.py
# 13 passed

docker compose run --rm dev uv run --no-sync ruff check \
    src/thesis_rl/contracts/checkpoint_manifest.py \
    src/thesis_rl/runtime/wiring/checkpoint_identity.py \
    src/thesis_rl/runtime/wiring/builders.py \
    src/thesis_rl/agent/agent.py \
    src/thesis_rl/runtime/loops/train_loop.py \
    tests/test_checkpoint_manifest_sidecar.py
# All checks passed!

docker compose run --rm dev uv run --no-sync ruff format --check \
    src/thesis_rl/contracts/checkpoint_manifest.py \
    src/thesis_rl/runtime/wiring/checkpoint_identity.py \
    tests/test_checkpoint_manifest_sidecar.py
# 3 files already formatted
```

`ruff format --check` on `agent.py`, `train_loop.py`, and `builders.py`
reports pre-existing, unrelated formatting debt (verified via
`ruff format --diff`: none of the reported hunks touch lines added by
this change) — consistent with AGENTS.md's documented non-clean
repository-wide formatting baseline; not addressed here per the
established precedent of not mass-formatting unrelated code in a
semantic change.

## Final reconciliation

All 5 requirements implemented and covered by a passing regression test.
No impact on the currently-stopped `sac-0`/`ppo-0`/`td3-0` checkpoints
(no sidecar present next to them → the new check no-ops, identical
behavior to before this change). The restart decision for those three
runs remains a separate, still-open item (tracked in the ADR-026
ExecPlan, not here).
