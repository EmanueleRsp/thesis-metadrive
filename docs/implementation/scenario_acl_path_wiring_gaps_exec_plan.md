# ExecPlan: ScenarioNet-ACL training-path wiring gaps

- Status: `DEFERRED` (documented, not implemented)
- Date: 2026-07-27
- Authoritative specification: `specifications/evaluation_protocol_v1.0_specification.md`
  (`REQ-014` / `DEC-014`, amended 2026-07-25) for the tracked-subset gap
- Related: `implementation/checkpoint_manifest_sidecar_wiring_exec_plan.md`
  (the feature affected by gap B), ADR-020 (evaluation video diagnostics)

## Objective

Record, precisely and with line references, a class of defect discovered
on 2026-07-27: features implemented in `runtime/loops/train_loop.py` are
unreachable for every production run, because the ScenarioNet-ACL
curriculum takes an early-return branch into a separate driver that does
not carry the same wiring.

Nothing here is implemented. This document exists so the gaps are not
rediscovered from scratch, and so the affected claims are not made in the
thesis without qualification.

## Root cause (shared by all gaps below)

`train_loop.py:791` dispatches ScenarioNet-ACL runs into a separate
driver and returns:

```python
if curriculum_cfg.is_scenario_acl:
    run_scenario_acl_training(...)
    return
```

Every production run uses `curriculum=scenario_acl_scenarionet`, so all
`train_loop.py` code after that line is dead for real runs. Wiring added
to `train_loop.py` in good faith therefore silently does nothing, with
no error and no log line.

## Gap A: tracked-subset GIF rendering is never produced (`REQ-014`/`DEC-014`)

`EVAL-PROTOCOL v1.0` `REQ-014` requires a small, frozen tracked subset of
scenario UIDs, drawn once from the frozen validation/test panels and
identical across periodic validation and final test, rendered
unconditionally at every evaluation so that visual progression over
training is comparable episode-for-episode.

Implementation status: the *selection* machinery is complete and tested
(`scripts/build_panel_manifest.py`, `scenarios/panel_manifest.py`
`build_balanced_panel` / `_select_diverse_uids`, feature-diversity greedy
selection over 12 scenario feature keys, default 4 UIDs/arm for
`validation` and 10 UIDs/arm for `test`). The *rendering* is not reachable.

Three independent blockers, in increasing order of cost:

1. **Never configured.** `provider.panel_manifest_path` appears nowhere
   in `conf/`, and no `*_panel_manifest_*.json` exists on disk. With no
   manifest, `_tracked_subset_uids_from_resolved_env_cfg` returns `()`
   and `maybe_build_periodic_tracked_subset_recorder_factory` returns
   `None` (`runtime/io/eval_artifacts.py:617`) — deliberately never a
   fallback draw. The scenario catalog required to build a manifest does
   exist (`$SCENARIONET_DATA_ROOT/catalog/scenario_catalog.parquet`).

2. **Unreachable from the ACL path.** The tracked-subset wiring lives at
   `train_loop.py:1695` (periodic) and `train_loop.py:2714` (final),
   both after the early return. `curriculum/scenario_acl/driver.py`
   contains no reference to `tracked_subset` / `tracked_scenario_uids`.

3. **The ACL periodic-eval path cannot record artifacts at all.** ACL
   periodic validation is asynchronous: it goes through
   `async_evaluation_manager.enqueue(...)` (`driver.py:1019`), not
   through `agent.evaluate(..., artifact_recorder_factory=...)`.
   `runtime/async_evaluation.py` has zero references to
   `artifact_recorder_factory` or any recorder. Adding tracked-subset
   rendering to periodic ACL evaluation therefore requires giving the
   asynchronous evaluation subsystem artifact-recording capability
   (recorder construction across a process boundary, GIF rendering
   inside evaluation workers) — a feature, not a port.

Consequence for current runs: `video.record_intermediate_evals` has **no
effect** on ACL runs, for reason (3), independently of its value. Only
final evaluation records GIFs, via
`maybe_build_live_final_eval_recorder_factory` at `driver.py:1262` and
`driver.py:2255`.

Additional semantic mismatch to resolve before implementing: ACL periodic
evaluation samples `scenario_source_schedule=("waymo",) * n` with a
waymo-only arm schedule (`driver.py:1030-1034`), whereas the tracked
subset is drawn from a frozen panel balanced across sources. The two
must be reconciled, not merely connected.

## Gap B: checkpoint-manifest sidecar is never written for ACL runs

`implementation/checkpoint_manifest_sidecar_wiring_exec_plan.md`
(`VERIFIED`) added `agent.set_checkpoint_manifest(...)` at
`train_loop.py:880`, immediately after the pre-existing
`agent.set_checkpoint_identity(...)` at `train_loop.py:879`. Both lines
are after the early return.

`driver.py:1493` carries only the `set_checkpoint_identity` half:

```python
agent.set_checkpoint_identity(build_reward_semantics_identity(cfg))
```

So the reward-semantics sidecar is written for ACL runs and the
checkpoint-manifest sidecar is not — confirmed empirically: production
checkpoint directories contain `latest.reward_semantics.json` but no
`latest.manifest.json`. The fail-open design of
`assert_checkpoint_manifest_compatible_if_present` means this degrades
silently to "no check", which is exactly the pre-existing behaviour the
ExecPlan set out to improve.

Unlike Gap A, this is a one-line fix: `env` is already in scope at
`driver.py:1482-1484`, so

```python
agent.set_checkpoint_manifest(build_current_checkpoint_manifest(cfg, env))
```

can be added directly after `driver.py:1493`. It needs a regression test
asserting the sidecar is written on the ACL path specifically, so the
same class of defect cannot recur unnoticed.

## Systemic follow-up

Both gaps share one preventable cause: there is no test asserting that
the ACL path and the non-ACL path agree on per-run setup. A parity test
enumerating the setup calls both paths must perform would have caught
Gap B at authoring time and Gap A at specification time. Recommended
before either gap is closed individually.

## Documentation drift found while investigating (not a gap, but adjacent)

- `conf/video/default.yaml` carries a `selection:` block
  (`best`, `median`, `worst_ev`, `collision`, `out_of_road`) that no code
  reads. The similarly-named logic in
  `analysis/videos/select_video_episodes.py` hardcodes its own category
  list and is invoked only from the offline
  `thesis_rl.analysis.run_analysis` CLI, never during training.
- `docs/protocols/live_eval_video_protocol.md` states its five-category
  scheme is superseded by `REQ-014`/`DEC-008`'s four categories
  (`representative_success`, `representative_failure`,
  `severe_rule_violation`, `algorithm_disagreement`), but
  `select_video_episodes.py` still implements the old five.

Neither was introduced by this investigation; both are recorded here
because they affect any future work on qualitative video selection.

## Impact on thesis claims

Until Gap A is closed, the thesis must not claim `REQ-014` tracked-subset
conformance. Qualitative material is limited to final-evaluation
episodes. This is a documented limitation, not a silent one.
