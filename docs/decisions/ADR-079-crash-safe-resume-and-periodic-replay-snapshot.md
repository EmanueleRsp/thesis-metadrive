# ADR-079: Crash-safe resume — periodic replay snapshot on every profile, SIGTERM as Ctrl+C, explicit replay reset

- Status: **Approved**
- Date: 2026-09-06
- Supersedes: ADR-017 (smoke-only replay buffer persistence)
- Approval evidence: explicit user approval on 2026-09-06 of the four decision
  gates `DEC-RES-002`, `DEC-RES-003`, `DEC-RES-004`, `DEC-RES-005` of
  `RESUME-ABRUPT-001` after a written illustration of each ("approvo tutto"),
  plus one addition in the same message: "magari a fine run si elimina ciò che
  poi non serve più per liberare spazio" (`DEC-RES-006`), and the instruction to
  implement everything ("procedi l'implementazione completa di tutto").
- Affected specifications:
  `docs/specifications/transition_replay_v1_specification.md` amended by
  `docs/specifications/transition_replay_v1.1_amendment.md`
  (`TRANSITION-REPLAY-V1.1`: REQ-023/024/024a/025/025a/033, §8.9, §9.1).
- Related: issue [#3](https://github.com/EmanueleRsp/thesis-metadrive/issues/3),
  `open_items` `C13`/`V2`/`D4`, ADR-016 (ACL resume restart policy, unchanged),
  ADR-024 (quarantine as resume state, unchanged),
  `implementation/resumable_training_after_abrupt_interruption_v1_exec_plan.md`.

## Context

A training process killed abruptly (SIGKILL, OOM-killer, node crash, `docker
stop`) could not be resumed as a valid continuation: only SIGINT reached the
save handler; `latest.zip`, the training state and the RNG state were written
in place, so a kill during a chunk-boundary save left a torn snapshot with no
previous good copy; the replay buffer was paired only with `final`
(`TRANSITION-REPLAY` v1.0 forbade periodic replay artifacts) and persistence
was off on every production profile (ADR-017), so a resume either failed closed
or silently continued on an empty buffer. Two production runs were lost this
way (`D4`). The storage cost that motivated ADR-017 is now measured: ≈ 7.2 GB
per off-policy run at full buffer, 12 writes per `thesis` run when the snapshot
follows `checkpoint.periodic_interval_steps`.

## Decisions

| ID | Decision |
|---|---|
| `DEC-RES-002` | SIGTERM is handled like Ctrl+C: the training CLI installs a handler that raises `KeyboardInterrupt`; `compose.yaml` raises `stop_grace_period` to 120 s. |
| `DEC-RES-003` | `TRANSITION-REPLAY` v1.1: `persistence.trigger=periodic_and_final` pairs the replay buffer with every periodic model checkpoint (cadence `checkpoint.periodic_interval_steps`, crossing semantics), `keep_last=1` for the replay artifact. `run_profile.replay_persistence` becomes `true` on every standard profile. This supersedes ADR-017. |
| `DEC-RES-004` | REQ-025 made explicit: a model-only resume of an off-policy learner fails before training unless `checkpoint.resume.allow_replay_reset=true`, in which case `replay_reset=true` is logged and recorded and β progress restarts. |
| `DEC-RES-005` | The non-ACL scenario-provider draw position is **not** persisted (deferred): the default configuration uses the ACL loop, whose selection state is persisted. |
| `DEC-RES-006` | After the `final` pair is committed, intermediate replay artifacts (periodic pairs, legacy `latest_*` pair) are deleted to free disk. |
| `DEC-RES-007` | Implementation decision recorded for review: the interrupt handlers no longer write a mid-chunk snapshot. Mid-chunk the model is ahead of `current_global_step`, the curriculum state and β progress, so the snapshot written there was internally inconsistent (the same defect made the asynchronous-evaluation callback write `latest` with the evaluated step instead of the live one). The resumable snapshot is the last chunk-boundary one, complete and atomic; Ctrl+C/SIGTERM therefore lose at most one chunk, exactly like a kill. |

## Alternatives Considered

| Alternative | Reason not selected |
|---|---|
| Keep v1.0 and rely on the empty-segment resume only (`DEC-RES-004` alone) | Zero storage, but not a continuation: the trained policy refills an empty buffer and SAC/TD3 restart from `learning_starts`; unfit for seed comparisons. |
| Persist the replay at every chunk boundary | Triples the writes (30 per `thesis` run) for at most 50 k fewer steps lost. |
| A separate `snapshot` checkpoint name at its own cadence | Adds a second frequency knob and a new artifact family; pairing with the periodic checkpoint that already exists at the wanted cadence reuses one concept. |
| Adopt the unused `checkpointing.py` generation/pointer layout | Changes the public checkpoint layout consumed by evaluation, videos and analysis (`DEC-EP-002`); tmp + `os.replace` on the existing names gives the same crash safety. |
| Keep writing a mid-chunk snapshot on Ctrl+C with `planner.num_timesteps` as the counter | Fixes the step counter but not the curriculum/ACL state, which is chunk-level by design. |

## Consequences

- Every standard profile now writes ≈ 7.2 GB of replay state at each periodic
  checkpoint (14.4 GB peak during the atomic replace), deleted at `final`.
  Disk budgets for concurrent runs must account for it; `run_profile.replay_persistence=false`
  remains an explicit opt-out.
- A crashed run resumes with `checkpoint.resume.enabled=true
  checkpoint.resume.run_dir=<dir> paths.run_dir=<dir>
  checkpoint.resume.checkpoint_name=periodic`, losing at most
  `periodic_interval_steps` steps; `checkpoint_name=latest` with
  `allow_replay_reset=true` loses at most one chunk but starts an empty replay
  segment, recorded as such.
- Periodic checkpoints now also exist on the scenario ACL loops, which
  previously wrote none, and their curriculum state is frozen per snapshot.
- Ctrl+C no longer produces a mid-chunk checkpoint (`DEC-RES-007`).
- Existing tests `test_replay_persistence_is_enabled_only_for_smoke_by_default`,
  `test_periodic_replay_persistence_is_rejected` and the `final_or_manual`
  assertion in `test_final_scalar_pipeline_defaults_compose` encoded ADR-017 /
  v1.0 and were changed accordingly under this approval.

## Validation And Traceability

`RESUME-ABRUPT-001` §8–§9: `tests/test_resume_snapshot.py` (`TEST-RES-001`,
`002`, `005`, `006`, `007`, `008`, `009` unit form, `011`, `013`, ACL snapshot
tests), `tests/test_transition_replay_config.py`,
`tests/test_hydra_preset_run_configs.py`; kill-and-resume smoke (`TEST-RES-010`)
recorded in the ExecPlan §11.
