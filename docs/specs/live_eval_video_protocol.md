# Live Evaluation Video Protocol

## Objective

Make the videos used for analysis, reporting, and the thesis consistent with
the same episodes from which evaluation metrics are computed, instead of
treating a post-hoc reconstructed replay as "official."

## Problem

The current codebase includes an **offline replay** video pipeline:

- episode selection from `csv/eval_episodes.csv`
- environment rerun with checkpoint + seed
- render into `videos/final_eval/*.gif`

This pipeline is useful for qualitative material, but it does **not guarantee**
that the trajectory exactly matches the one observed during the original
evaluation. In MetaDrive, divergence may depend on:

- seed and scenario window (`start_seed`, `num_scenarios`)
- curriculum stage / `eval_env`
- reactive or IDM traffic vs log replay
- wrapper stack
- termination flags
- reward and rulebook mode
- initial state and internal dynamics

Conclusion: offline replay should be treated as **qualitative / exploratory**,
not as the official source supporting quantitative metrics.

## Current Status In The Repository

### Existing Evaluation Logging

The training runtime already saves:

- aggregated metrics in `csv/evals.csv`
- per-episode metrics in `csv/eval_episodes.csv`
- per-rule metrics in `csv/rule_metrics.csv`
- final metrics in `csv/final_eval.csv`

Main hooks:

- `src/thesis_rl/runtime/loops/train_loop.py`
- `src/thesis_rl/agent/agent.py`
- `src/thesis_rl/runtime/io/csv_recorder.py`

### Existing Post-Hoc Video Replay

The current video pipeline uses:

- `conf/video/default.yaml`
- `src/thesis_rl/analysis/videos/select_video_episodes.py`
- `src/thesis_rl/analysis/videos/render_selected_videos.py`
- `src/thesis_rl/analysis/videos/render_qualitative_videos.py`

In `conf/video/default.yaml` it is already annotated that:

- `mode: offline_replay`
- `live_final_eval` is a future option

This is the natural extension point.

## Architectural Decision

For the "official" artifacts of the project:

1. the video should be recorded **during evaluation**
2. the video should be saved together with a manifest and numeric log
3. offline replay should remain available, but labeled as a
   **non-authoritative replay**

## Official Evaluation Bundle

For each recorded evaluation episode, the run should produce a coherent bundle:

- rendered video
- JSON manifest
- JSONL trajectory log or NPZ/Parquet equivalent
- links to the existing CSV files

Proposed structure inside `paths.videos_dir`:

```text
videos/
  final_eval/
    eval_0001/
      episode_0001.gif
      episode_0001.manifest.json
      episode_0001.trajectory.jsonl
      episode_0002.gif
      episode_0002.manifest.json
      episode_0002.trajectory.jsonl
      ...
```

Alternatively, if you prefer to separate heavy assets from qualitative ones:

```text
artifacts/
  evaluation_bundle/
    final_eval/
      eval_0001/
        episode_0001.manifest.json
        episode_0001.trajectory.jsonl
videos/
  final_eval/
    eval_0001/
      episode_0001.gif
```

The second option is cleaner if you want to distinguish:

- `videos/` = media
- `artifacts/` = metadata and numeric logs

## Per-Episode Manifest

Each official video should have an `episode_<id>.manifest.json` file with at
least:

```json
{
  "schema_version": 1,
  "run_id": "20260602_120000",
  "eval_id": 1,
  "eval_type": "final",
  "scenario_set": "test",
  "episode_id": 1,
  "checkpoint_path": "checkpoints/final.zip",
  "checkpoint_type": "final",
  "checkpoint_global_step": 500000,
  "seed": 7,
  "scenario_seed": 1007,
  "scenario_id": "seed_1007",
  "deterministic": true,
  "curriculum_stage": "stage3",
  "stage_index": 3,
  "env_config_resolved": {},
  "map_config": {},
  "traffic_density": 0.2,
  "traffic_mode": "reactive",
  "termination_flags": {},
  "reward_type": "rulebook",
  "reward_behavior": "scalar_reward",
  "rulebook_config": "selection",
  "wrappers": [
    "RuleRewardWrapper"
  ],
  "git_commit": "abc123",
  "metadrive_version": "x.y.z",
  "episode_metrics": {
    "reward": 12.3,
    "env_reward": 9.8,
    "scalar_rule_reward": 1.1,
    "hybrid_reward": 12.3,
    "episode_length": 243,
    "success": true,
    "collision": false,
    "out_of_road": false,
    "timeout": false,
    "route_completion": 0.83,
    "top_rule_violation_rate": 0.04,
    "error_value": 0.12,
    "violated_rules": "none",
    "violation_pattern": "none"
  }
}
```

### Practical Note

`env_config_resolved` should contain the effective configuration used for
evaluation, after merging:

- base config
- optional `curriculum.eval_env`
- test split through `apply_eval_scenario_seed_split(...)`

This is more useful than saving only an abstract reference.

## Per-Episode Trajectory Log

Each recorded episode should also save the real numeric log, for example in
JSONL:

```json
{"t":0,"action":[0.1,-0.2],"reward":0.03,"done":false,"truncated":false,"ego":{"x":1.2,"y":3.4,"heading":0.5,"speed_kmh":21.0},"rule":{"vector":[0.2,1.0],"violated":[]}}
{"t":1,"action":[0.1,-0.1],"reward":0.04,"done":false,"truncated":false,"ego":{"x":1.4,"y":3.8,"heading":0.5,"speed_kmh":22.1},"rule":{"vector":[0.3,1.0],"violated":[]}}
```

Recommended minimum fields per step:

- `t`
- `action`
- `reward`
- `env_reward`
- `scalar_rule_reward`
- `hybrid_reward`
- `done`
- `truncated`
- `ego`:
  - position
  - heading
  - speed
- `route_completion`
- `rule_reward_vector`
- `rule_metadata`
- `violated_rules`
- `top_rule_violation`
- reduced / sanitized `info`

If the full `info` payload is too verbose, it is better to save:

- a stable and useful subset for auditing
- plus an optional `raw_info_path` field

## Minimal CSV Schema Extensions

The current CSV infrastructure is already very close. The minimum useful
extensions are:

### `eval_episodes.csv`

Add:

- `video_authoritative_path`
- `video_manifest_path`
- `trajectory_log_path`
- `video_recorded_live`
- `replay_warning`

Note:

- the current `video_path` can remain for compatibility
- but it should be treated as a legacy or generic video-path field

### `final_eval.csv`

Optionally add:

- `official_video_bundle_dir`
- `official_video_count`
- `official_trajectory_count`

## Proposed Software Changes

### Phase 1 - Runtime Support For Live Recording

#### 1. Config

Extend `conf/video/default.yaml`:

```yaml
enabled: false
mode: offline_replay            # offline_replay | live_final_eval
record_intermediate_evals: false
record_final_eval: true
max_final_videos: 5
save_manifest: true
save_trajectory_log: true
official_only: true
```

Semantics:

- `offline_replay`: current behavior
- `live_final_eval`: record during final evaluation
- `official_only`: files produced in this mode are the official source

#### 2. Runtime Recorder

Introduce a new module, for example:

- `src/thesis_rl/runtime/io/eval_artifacts.py`

Responsibilities:

- open the video writer
- open the trajectory-log writer
- accumulate episode metadata
- finalize and save the manifest
- return the paths to be written to the CSV files

Proposed API:

```python
class EvalEpisodeArtifactRecorder:
    def on_episode_start(...)
    def on_step(frame, action, reward, done, truncated, info, ...)
    def on_episode_end(metrics, ...)
    def close()
```

### Phase 2 - Hook Into The Evaluation Loop

The best integration point is inside `Agent.evaluate(...)` in
`src/thesis_rl/agent/agent.py`, because that is where the episode/step loop
already has access to:

- reset seed
- chosen actions
- reward
- done/truncated
- `step_info`
- final episode metrics

Proposed signature extension:

```python
def evaluate(
    ...,
    artifact_recorder_factory: Callable[[dict[str, Any]], Any] | None = None,
)
```

Flow:

1. create a recorder with episode context at episode start
2. at each step:
   - render frame
   - save trajectory row
3. at episode end:
   - save manifest
   - return artifact paths
4. `metrics["per_episode"]` also includes artifact paths

New fields in `metrics["per_episode"]`:

- `video_authoritative_path`
- `video_manifest_path`
- `trajectory_log_path`
- `video_recorded_live`

### Phase 3 - Runtime CSV Writing

Update the points in `src/thesis_rl/runtime/loops/train_loop.py` that already
iterate over `per_episode` to write `eval_episodes.csv`.

Main zones:

- intermediate evaluation
- final evaluation

For the first iteration, it is better to record **final evaluation only**:

- `video.mode=live_final_eval`
- `video.record_final_eval=true`
- `video.record_intermediate_evals=false`

This minimizes overhead and data volume.

### Phase 4 - Standalone `cli.evaluate`

Also align `src/thesis_rl/runtime/loops/eval_loop.py`, so that an evaluation
launched separately produces the same official bundle.

This matters for experimental consistency:

- training-time final eval
- standalone eval from checkpoint

should produce artifacts with the same schema.

## Rendering During Evaluation

For live rendering, you can reuse the same top-down logic already present in:

- `src/thesis_rl/analysis/videos/render_selected_videos.py`

In particular:

- `_render_topdown_frame(...)`
- `_save_gif(...)`

It is worth extracting these helpers into a shared module, for example:

- `src/thesis_rl/runtime/io/video_utils.py`

That way the runtime and analysis pipeline use the same implementation.

## Compatibility With The Current Pipeline

The offline replay pipeline should not be removed immediately. It should only
be reclassified:

- **official**: live evaluation bundle
- **unofficial**: offline replay / qualitative replay

Proposed rule:

- quantitative reports and the thesis should use only
  `video_recorded_live=true`
- offline replays can still be used for debugging or illustrative figures, but
  only with an explicit label

## Impact On Analysis Modules

### `select_video_episodes.py`

It should no longer select episodes to rerender for official cases. Instead it
should:

- read `eval_episodes.csv`
- filter episodes with `video_recorded_live=true`
- choose which bundles to promote into the report

### `render_selected_videos.py`

It should be split logically:

- legacy path: offline replay
- new path: if the official video exists, **do not rerender**

Preferred behavior:

1. use `video_authoritative_path` if present
2. fall back to offline replay only when explicitly requested

## Incrementally Safe Strategy

### Milestone 1

Implement only:

- live recording for `final_eval`
- JSON manifest
- artifact paths in `eval_episodes.csv`

without a full trajectory log yet.

### Milestone 2

Add:

- step-by-step trajectory log
- richer metadata
- links from `final_eval.csv`

### Milestone 3

Update the analysis pipeline to always prefer live artifacts.

## Validation

Recommended new tests:

- `tests/test_live_eval_video_protocol.py`
- `tests/test_eval_artifact_recorder.py`

Minimum checks:

1. `Agent.evaluate(...)` with a recorder produces one manifest per episode
2. `train_loop.py` writes the new fields into `eval_episodes.csv`
3. `video.mode=live_final_eval` does not require a later replay
4. `render_selected_videos.py` uses the official file when it exists
5. `offline_replay` remains backward compatible

## Recommended Final Decision

### Experimental Policy

- checkpoint = training artifact
- `final_eval.csv` + `eval_episodes.csv` + `rule_metrics.csv` = quantitative
  truth
- live video + manifest + trajectory log = qualitative / forensic truth
  consistent with the quantitative side

### Operational Rule

For important runs:

1. train
2. save checkpoint
3. run deterministic final evaluation
4. during that evaluation save:
   - metrics
   - CSV files
   - video
   - manifest
   - trajectory log
5. use only these artifacts in the report

This approach is consistent with the design already present in the repository
and requires a localized extension, not a complete pipeline rewrite.
