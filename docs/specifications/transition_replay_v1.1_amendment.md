# Specification amendment: periodic replay snapshot paired with the periodic checkpoint

## Metadata

- Feature: `transition_replay_periodic_snapshot`
- Specification ID: `TRANSITION-REPLAY-V1.1`
- Version: `1.1`
- Status: `APPROVED`
- Date: `2026-09-06`
- Amends: `docs/specifications/transition_replay_v1_specification.md`
  (`TRANSITION-REPLAY` v1.0): REQ-023, REQ-024, REQ-025 (made observable),
  §8.9 serialization timing, §9.1 configuration, §9 validation table rows for
  `persistence.trigger` / `persistence.periodic_frequency_steps`. Everything
  else in v1.0 (N-step core, PER, reward compatibility, pairing identity
  REQ-033, load validation REQ-026) is unchanged.
- Driven by: issue
  [#3](https://github.com/EmanueleRsp/thesis-metadrive/issues/3), ExecPlan
  `implementation/resumable_training_after_abrupt_interruption_v1_exec_plan.md`
  (`RESUME-ABRUPT-001`), decisions `DEC-RES-003` and `DEC-RES-004`.
- Approval evidence: explicit user approval 2026-09-06 ("approvo tutto, magari
  a fine run si elimina ciò che poi non serve più per liberare spazio. procedi
  l'implementazione completa di tutto"), recorded in ADR-079.
- Authoritative: `YES` for §2–§4; not authoritative for anything else.

## 1. Why

Under v1.0 the replay buffer was serialized only at the `final` checkpoint or
on a manual (Ctrl+C) stop. A run killed abruptly never reaches either, so its
periodic `latest` checkpoint could not be continued with its replay state, and
`open_items` `V2`/`D4` record production runs lost this way. ADR-017 had
declined per-run persistence when its cost was "several gigabytes"; the cost is
now measured (§4) and accepted at a bounded cadence.

## 2. Amended requirements

**`REQ-023` (amended) — persistence disabled by default becomes profile-controlled.**
Replay persistence remains a configuration switch
(`transition_replay.persistence.enabled`). Its default value is decided by the
run profile (ADR-079: enabled on every standard profile). The invariant
"periodic checkpoint callbacks shall not produce replay artifacts" is
**withdrawn** and replaced by REQ-024 below.

**`REQ-024` (amended) — trigger set.** `transition_replay.persistence.trigger`
takes one of:

| value | replay artifact written at |
|---|---|
| `final_or_manual` | the `final` checkpoint only (v1.0 behaviour) |
| `periodic_and_final` | every **periodic model checkpoint** and the `final` checkpoint |

Under `periodic_and_final`:

- the replay snapshot **rides the periodic model checkpoint**
  `checkpoints/periodic/step_XXXXXXXX.zip`; its cadence is therefore
  `checkpoint.periodic_interval_steps` by construction, and
  `checkpoint.save_periodic` must be `true` with a positive interval (validated
  before training starts);
- `persistence.periodic_frequency_steps` is **not a setting** and must stay
  `null`; a value is rejected with a message naming
  `checkpoint.periodic_interval_steps`;
- the periodic snapshot is *due* when the run **crosses** a multiple of the
  interval within a chunk (`⌊step/I⌋` increases), not only when a chunk ends
  exactly on the multiple — with `eval_interval = 50 000` and
  `periodic_interval_steps = 125 000` the exact test would skip every other
  snapshot;
- the snapshot consists of `step_X.zip` (+ sidecars), `step_X_replay_buffer.pkl`,
  `step_X_checkpoint_pair.json`, `step_X_rng_state.pkl`,
  `step_X_quarantine_state.json`, `step_X_training_state.yaml` and, for the
  scenario ACL loops, a frozen copy of the curriculum state under
  `step_X_acl/`; every file is published atomically (temporary sibling +
  `os.replace`) and the training/curriculum state file is written **last** as
  the commit marker;
- `keep_last = 1` applies to the replay artifact: committing a new pair deletes
  the previous periodic pair (replay + pair manifest); the older periodic
  **model** zips follow `checkpoint.keep_last_periodic` as before, and their
  companion files are removed together with the zip;
- the `latest` checkpoint stays **model-only** at every chunk boundary.

**`REQ-024a` (new) — cleanup after `final`.** Once the `final` model/replay pair
is committed, every intermediate replay artifact (`periodic/*_replay_buffer.pkl`,
`periodic/*_checkpoint_pair.json`, legacy `latest_replay_buffer.pkl` /
`latest_checkpoint_pair.json`) is removed and the removal is logged
(`DEC-RES-006`). Model zips and the small state files are kept.

**`REQ-025` (clarified, now observable).** A resume whose checkpoint carries no
complete pair is **not** silent:

- by default it **fails before training** with a message naming both remedies;
- with `checkpoint.resume.allow_replay_reset=true` it proceeds as a **new empty
  replay segment**: the event `replay_reset` (`replay_reset=true`) is appended
  to `events.jsonl`, `run_metadata.yaml` gets `replay_reset: true`, a warning is
  logged, and `beta_progress_env_steps` restarts from 0;
- a pair manifest whose replay artifact is missing is a partially committed
  pair and fails regardless of the flag (REQ-033).

**`REQ-025a` (new) — resume selectors.** `checkpoint.resume.checkpoint_name`
accepts, besides `latest`, `final` and an explicit stem, the alias `periodic`,
which resolves to the **newest periodic checkpoint carrying a complete pair**
(model + replay + pair manifest). When none exists the run fails before
training with the list of periodic checkpoints found.

**`REQ-033` (extended) — torn-snapshot check.** The pair manifest and the
training/curriculum state record `model_num_timesteps`, the number of
environment steps the saved model was trained on. On resume it must equal the
loaded model's own counter; a mismatch is a torn snapshot and fails before
training. Snapshots written before this amendment lack the field and are
accepted with a warning. The recorded `seed` must equal the configured seed.

## 3. Amended serialization timing (§8.9)

```text
chunk boundary:
    latest.zip (model only), latest_rng_state.pkl, latest_quarantine_state.json,
    latest_training_state.yaml   <- written last, commit marker

chunk boundary crossing k * periodic_interval_steps  (trigger=periodic_and_final):
    periodic/step_X.zip, step_X_replay_buffer.pkl, step_X_checkpoint_pair.json,
    step_X_rng_state.pkl, step_X_quarantine_state.json, [step_X_acl/],
    step_X_training_state.yaml   <- written last, commit marker
    then: previous periodic replay pair removed (keep_last = 1)

final:
    final.zip, final_replay_buffer.pkl, final_checkpoint_pair.json
    then: intermediate replay artifacts removed (REQ-024a)

SIGINT / SIGTERM:
    no snapshot is written; the last chunk-boundary snapshot is the resumable one
    (`DEC-RES-007`: mid-chunk the model is ahead of every chunk-level counter)
```

## 4. Cost accepted

At `buffer_size = 300 000`, `semantic_v3` flat dim 3011, float32 observations
and next observations, float64 raw priorities and a bool validity mask, one
replay artifact is ≈ 7.2 GB when the buffer is full (≈ 24.1 kB per stored
transition; `__getstate__` truncates to the active rows before that), with a
≈ 14.4 GB peak during the atomic replacement. At the `thesis` profile's
`periodic_interval_steps = 125 000` a 1.5 M-step run writes 12 snapshots and
loses at most 125 000 steps on a crash.

## 5. Configuration (§9.1 delta)

```yaml
transition_replay:
  persistence:
    enabled: ${run_profile.replay_persistence}   # true on every standard profile (ADR-079)
    trigger: periodic_and_final                   # v1.0 value final_or_manual still accepted
    periodic_frequency_steps: null                # not a setting; must stay null
    keep_last: 1

checkpoint:
  resume:
    checkpoint_name: latest      # latest | final | periodic | <stem>
    allow_replay_reset: false    # REQ-025 empty segment only when explicitly allowed
```

## 6. Out of scope

Bitwise-identical continuation (v1.0 §8.10 unchanged). Persisting the in-flight
asynchronous-evaluation queue. The non-ACL scenario-provider draw position
(`DEC-RES-005`, deferred).
