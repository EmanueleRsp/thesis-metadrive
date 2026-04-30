# Validation Commands (Phase 1)

Run all commands from project root:

```bash
cd /home/e.respino/main/thesis/thesis-metadrive
```

Note:
- In `conf/config.yaml` the default is `run_profile=fast`.
- Use `run_profile=...` only when you intentionally want a different run budget.

## 0) Setup

What this step validates:
- Environment and dependencies are consistent before running expensive experiments.
- Core code contracts still pass after recent refactors.

```bash
uv sync --extra dev
uv pip install -e .
uv run --no-sync python -m pytest -q
```

Expected:
- Tests pass.
- No import/runtime wiring errors.

Visual/manual checks:
- Scan test output summary: no skipped-critical suites, no intermittent errors.

## 1) Smoke Test End-to-End (train + eval + checkpoint)

What this step validates:
- Minimal end-to-end train/eval/checkpoint flow works on the current codebase.
- TD3 update path is active with minimal targeted overrides for a short smoke run.

```bash
uv run --no-sync python -m thesis_rl.train --config-name presets/td3/td3_scalar_def_no_curr \
  experiment.total_timesteps=2000 \
  experiment.eval_interval=1000 \
  experiment.log_interval=100 \
  experiment.eval_episodes=2 \
  experiment.final_eval_episodes=2 \
  planner.learning_starts=100 \
  planner.batch_size=32
```

Expected:
- `n_updates > 0`.
- `actor_loss` and `critic_loss` are finite (not `NaN`).
- Intermediate eval runs on validation split, final eval on test split.
- `checkpoints/final.zip` exists.
- Run directory contains `logs/`, `csv/`, `checkpoints/`, `artifacts/`.

Visual/manual checks:
- Open `outputs/<run>/logs/` and confirm training progresses (no repeated reset/crash patterns).
- Open `outputs/<run>/csv/final_eval.csv` and confirm it has one coherent row.

## 2) Baseline Validation (curriculum OFF, scalar native)

What this step validates:
- Baseline behavior without curriculum/reward-wrapper confounders.
- Core agent-planner-adapter path on native scalar reward.

```bash
for s in 0 1 2; do
  uv run --no-sync python -m thesis_rl.train \
    --config-name presets/td3/td3_scalar_native_no_curr \
    run_profile=medium seed=$s
done
```

Expected:
- All runs complete (`status=completed` in run metadata).
- Final eval metrics are produced for each seed.

Visual/manual checks:
- Compare `final_eval.csv` across seeds: metrics should vary but stay in plausible ranges.
- Spot-check `evals.csv` curves: no flatlined or exploding reward/error trends.

## 3) Curriculum Validation (curriculum ON, scalar native)

What this step validates:
- Curriculum progression logic and stage-aware train/eval splitting.

```bash
for s in 0 1 2; do
  uv run --no-sync python -m thesis_rl.train \
    --config-name presets/td3/td3_scalar_native_curr \
    run_profile=medium seed=$s
done
```

Expected:
- Curriculum logs/events appear (`promotions.csv` when promotions happen).
- Stage transitions are coherent (no train/eval pool overlap warnings as hard errors).

Visual/manual checks:
- Inspect `promotions.csv`: promotion events should be temporally coherent (increasing steps/eval ids).
- In plots/tables, check if stage transitions align with metric changes (no impossible jumps).

## 4) Rulebook Validation (curriculum ON, scalar rulebook)

What this step validates:
- Rulebook scalar reward path under curriculum.
- Rule components/margins are available for diagnostics and later scale tuning.

```bash
for s in 0 1 2; do
  uv run --no-sync python -m thesis_rl.train \
    --config-name presets/td3/td3_scalar_rulebook_curr \
    run_profile=medium seed=$s \
    reward.rule_margin_log_path='${paths.logs_dir}/rule_margins.jsonl'
done
```

Expected:
- `csv/rule_metrics.csv` is produced.
- `logs/rule_margins.jsonl` exists and is non-empty.
- Rule-based best checkpoints can be produced when metrics improve.

Visual/manual checks:
- Open a sample of `rule_margins.jsonl`: confirm `rule_components` is populated (not always empty).
- Inspect `rule_metrics.csv`: per-rule margins should show variation across evaluations.

## 4.1) Run Comparison Sanity Checks (core metrics)

What this step validates:
- Baseline/curriculum/rulebook runs are all numerically sane before deeper analysis.

Run after steps 2-4 (baseline/curriculum/rulebook):

```bash
python - << 'PY'
import glob, os, pandas as pd
files = sorted(glob.glob("outputs/**/csv/final_eval.csv", recursive=True))
if not files:
    raise SystemExit("No final_eval.csv found")
df = pd.concat([pd.read_csv(f).assign(_src=f) for f in files], ignore_index=True)
cols = [c for c in [
    "reward_mode","seed","success_rate","collision_rate","out_of_road_rate",
    "route_completion","top_rule_violation_rate","avg_error_value","max_error_value"
] if c in df.columns]
print(df[cols].sort_values(["reward_mode","seed"]).to_string(index=False))
PY
```

Expected:
- Metrics are finite and in valid ranges (`rate` fields in `[0,1]`).
- No obviously broken regime (example: all-zero success with all-one collision).

Visual/manual checks:
- Compare rows side-by-side: confirm rulebook is not trivially worse on all axes.

## 5) Scale-Tuning Pass (rule margins -> suggested scales)

What this step validates:
- Logged rule margins can be converted into stable initial scale suggestions.

Collect dedicated diagnostics run:

```bash
uv run --no-sync python -m thesis_rl.train \
  --config-name presets/td3/td3_scalar_rulebook_scale_tuning_no_curr \
  # uses default run_profile=fast
```

Extract suggested scales:

```bash
uv run --no-sync python -m thesis_rl.reward.scale_tuning \
  --input outputs/<RUN_PATH>/logs/rule_margins.jsonl \
  --percentile 90 \
  --min-scale 1e-6
```

Expected:
- Command prints a `scales:` block with one value per rule.
- No missing-rule error (`No rule_components found in margin log`).

Then update `conf/reward/base_rulebook.yaml` (`reward.scales`) and re-run step 4 for confirmation.

Visual/manual checks:
- Confirm suggested scales are not degenerate (all identical by accident, or extreme outliers without reason).
- After updating scales and re-running step 4, compare `rule_metrics.csv` to verify reduced saturation/imbalance.

## 6) CSV and Artifact Presence Checks

What this step validates:
- Mandatory outputs exist for completed runs.

Quick check for required CSVs/checkpoints:

```bash
find outputs -type f \( \
  -name train_chunks.csv -o -name evals.csv -o -name eval_episodes.csv -o -name promotions.csv -o -name rule_metrics.csv -o -name final_eval.csv -o \
  -name final.zip -o -name latest.zip -o -name checkpoint_index.csv -o -name best_checkpoints.yaml -o \
  -name latest_replay_buffer.pkl -o -name latest_training_state.yaml -o -name latest_rng_state.pkl \
\) | sort
```

Expected:
- Required files exist for completed runs.
- `promotions.csv` may be empty/absent only when curriculum is disabled or no promotion event occurs.

Visual/manual checks:
- Browse one run directory tree manually to verify outputs are organized and not partially missing.

## 6.1) CSV Schema Validation vs Objectives

What this step validates:
- Output contracts match documented schema (`docs/csv_evaluation_objectives.md`).

Validate CSV headers against `docs/csv_evaluation_objectives.md` (including V2 fields).

Quick header dump:

```bash
python - << 'PY'
import csv, glob, os
targets = [
    "train_chunks.csv",
    "evals.csv",
    "eval_episodes.csv",
    "promotions.csv",
    "rule_metrics.csv",
    "final_eval.csv",
]
for path in sorted(glob.glob("outputs/**/csv/*.csv", recursive=True)):
    name = os.path.basename(path)
    if name not in targets:
        continue
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    print(f"{path}\n  -> {header}\n")
PY
```

Expected:
- Headers match the expected objective schema for each CSV type.
- Key identifiers are present and non-null: `algorithm`, `seed`, `run_id`, `stage`, `stage_index`, `global_step`.

Visual/manual checks:
- Open 1-2 CSVs directly and verify header readability and consistent naming conventions.

## 6.2) CSV Granularity Validation

What this step validates:
- Row-level granularity rules (per chunk / per eval / per episode / per rule) are respected.

```bash
python - << 'PY'
import glob, os, pandas as pd
runs = sorted(glob.glob("outputs/**/csv", recursive=True))
for csv_dir in runs:
    files = {os.path.basename(p): p for p in glob.glob(os.path.join(csv_dir, "*.csv"))}
    if "evals.csv" in files and "eval_episodes.csv" in files:
        e = pd.read_csv(files["evals.csv"])
        ep = pd.read_csv(files["eval_episodes.csv"])
        if {"eval_id"}.issubset(e.columns) and {"eval_id"}.issubset(ep.columns):
            counts = ep.groupby("eval_id").size()
            print(csv_dir, "eval_episodes per eval_id:", counts.to_dict())
PY
```

Expected:
- `train_chunks.csv`: one row per training chunk.
- `evals.csv`: one row per evaluation aggregate.
- `eval_episodes.csv`: N rows per `eval_id` (N = eval episodes for that run).
- `promotions.csv`: rows only when promotion events happen.
- `rule_metrics.csv`: one row per rule per evaluation.
- `final_eval.csv`: one row per run.

Visual/manual checks:
- Open grouped counts output and verify no irregular holes in `eval_id` progression.

## 6.3) Run-Selection Filters and Dedupe Validation

What this step validates:
- Aggregation honors selection policy (completed + include_in_comparison + dedupe + expected seeds).

```bash
uv run --no-sync python -m analysis.run_analysis --only aggregate --seed-list 0,1,2
```

Expected:
- Aggregation includes only `status=completed` and `analysis.include_in_comparison=true`.
- Dedupe keeps latest run for identical comparison key.
- Missing-seed warnings are emitted when expected seeds are absent.

Visual/manual checks:
- Confirm scale-tuning runs are excluded from final comparisons.

## 7) Analysis Pipeline Validation

What this step validates:
- End-to-end aggregate/tables/plots/orchestrator behavior.

```bash
uv run --no-sync python -m analysis.run_analysis --only aggregate
uv run --no-sync python -m analysis.run_analysis --only tables
uv run --no-sync python -m analysis.run_analysis --only plots
uv run --no-sync python -m analysis.run_analysis --only all --no-videos
uv run --no-sync python -m analysis.run_analysis --only all --video-max 3
```

Expected:
- Aggregated CSVs under `analysis/aggregated/*_all_runs.csv`.
- Tables generated under `analysis/tables` (`.csv` and `.md`).
- Plots generated under `analysis/plots` (`.png`).
- Video pipeline skips gracefully when dependencies/checkpoints are missing.
- No crashes when partial datasets are present.
- Learning curves use `global_step` on x-axis.
- Curriculum promotion markers are rendered when promotion data exists.
- Tables compute `mean ± 95% CI`; with one seed, warning/no crash behavior is acceptable.

Visual/manual checks:
- Open key plots and verify trend continuity (no impossible zig-zag from bad indexing).
- Confirm x-axis label/values are `global_step`.
- In curriculum plots, verify promotion markers align with expected promotion steps.
- Open generated markdown tables and verify CI formatting is readable and coherent.

## 7.1) Video-Pipeline Output Validation

What this step validates:
- Episode selection + replay rendering + CSV linkage.

```bash
find outputs -type f \( \
  -name video_selection.json -o \
  -name video_index.csv -o \
  -name "*.gif" \
\) | sort
```

Expected:
- Episode selection artifacts exist (`video_selection.json`, `video_index.csv`).
- Replay render emits GIF outputs when rendering dependencies/checkpoints are available.
- `eval_episodes.csv` contains `video_path` updates for rendered episodes.
- Replay-fidelity fields/warnings (including `replay_match` diagnostics) are present when provided by pipeline.

Visual/manual checks:
- Watch sampled GIFs (best/median/worst/collision/out_of_road) and confirm label semantics match behavior.
- Cross-check a GIF path against `eval_episodes.csv.video_path`.

## 8) Idempotency / Reproducibility (analysis)

What this step validates:
- Re-running analysis with identical inputs is stable and non-destructive.

```bash
uv run --no-sync python -m analysis.run_analysis --only all --no-videos
uv run --no-sync python -m analysis.run_analysis --only all --no-videos
```

Expected:
- Re-running does not corrupt outputs.
- Aggregated numbers/tables remain stable for identical input runs.

Visual/manual checks:
- Compare timestamps/file counts: rerun may update files but should not change metric values unexpectedly.

## 9) Resume Validation

What this step validates:
- Resume restores planner/adapter/replay/state/RNG and continues without silent resets.

Use a real completed run path:

```bash
uv run --no-sync python -m thesis_rl.train \
  --config-name presets/td3/td3_scalar_native_curr \
  run_profile=medium \
  checkpoint.resume.enabled=true \
  checkpoint.resume.run_dir='/absolute/path/to/previous/run_dir' \
  checkpoint.resume.checkpoint_name=latest
```

Expected:
- Resume log includes restored checkpoint/state/replay/RNG.
- Training continues with increasing global step.
- New checkpoints and CSV rows append coherently.

Visual/manual checks:
- Compare pre-resume and post-resume CSV tails: `global_step` and `chunk_id` should continue, not restart.

## 9.1) Checkpoint Lifecycle Validation

What this step validates:
- Expected checkpoint families (`latest`, `periodic`, `final`) are created and retained correctly.

```bash
find outputs -type f \( \
  -path "*/checkpoints/latest.zip" -o \
  -path "*/checkpoints/final.zip" -o \
  -path "*/checkpoints/periodic/step_*.zip" \
\) | sort
```

Expected:
- `latest.zip` appears after first chunk and is updated on subsequent chunks.
- `final.zip` exists for completed runs.
- Periodic checkpoints respect `checkpoint.periodic_interval_steps`.
- Retention respects `checkpoint.keep_last_periodic` (older periodic snapshots pruned).

Visual/manual checks:
- Sort periodic checkpoints by step and verify monotonic naming with pruning of older files.

## 9.2) Checkpoint Metadata and Final-Eval Fields

What this step validates:
- Metadata coherence between checkpoint registry and final-eval references.

```bash
python - << 'PY'
import glob, pandas as pd, os
for p in sorted(glob.glob("outputs/**/checkpoints/metadata/checkpoint_index.csv", recursive=True)):
    df = pd.read_csv(p)
    must = [c for c in ["checkpoint_path","type","global_step","reason","timestamp"] if c in df.columns]
    print(p, "rows=", len(df), "cols=", must)
for p in sorted(glob.glob("outputs/**/csv/final_eval.csv", recursive=True)):
    df = pd.read_csv(p)
    cols = [c for c in ["checkpoint_path","checkpoint_type","checkpoint_global_step"] if c in df.columns]
    print(p, "final_eval checkpoint cols:", cols)
PY
```

Expected:
- `checkpoint_index.csv` rows are coherent (`type/path/step/reason/timestamp`).
- `best_checkpoints.yaml` is present and points to current best/final artifacts.
- `final_eval.csv` includes valid `checkpoint_path`, `checkpoint_type`, `checkpoint_global_step`.

Visual/manual checks:
- Open `best_checkpoints.yaml` and verify referenced files exist on disk.

## 9.3) Resume Fail-Fast Validation (negative test)

What this step validates:
- Invalid resume configuration fails explicitly (no silent fallback to fresh run).

```bash
uv run --no-sync python -m thesis_rl.train \
  --config-name presets/td3/td3_scalar_native_curr \
  checkpoint.resume.enabled=true \
  checkpoint.resume.run_dir='/tmp/this_path_should_not_exist' \
  checkpoint.resume.checkpoint_name=latest
```

Expected:
- Run fails fast with explicit missing-checkpoint/state error (no silent fallback).

## 10) Best-Checkpoint Policy Validation

What this step validates:
- Best-checkpoint policies (lexicographic and rulebook variants) actually trigger and persist.

For at least one rulebook run, verify best-checkpoint artifacts:

```bash
find outputs -type f \( \
  -name best_lexicographic.zip -o \
  -name best_lexicographic_rulebook.zip -o \
  -name best_thresholded_lexicographic_rulebook.zip \
\) | sort
```

Expected:
- At least one completed rulebook run emits these files when metrics improve.
- `checkpoints/metadata/best_checkpoints.yaml` points to current best/final paths consistently.

Visual/manual checks:
- Inspect `checkpoint_index.csv`: improvement reasons should align with saved best checkpoint types.

## 10.1) Replay Checkpoint Selection Validation

What this step validates:
- Video replay respects explicit checkpoint selection (`video.replay_checkpoint`).

When rendering videos, explicitly set replay checkpoint target:

```bash
uv run --no-sync python -m thesis_rl.train \
  --config-name presets/td3/td3_scalar_rulebook_curr \
  video.enabled=true \
  video.replay_checkpoint=final
```

Expected:
- Replay uses the configured checkpoint target (`final`, not ambiguous implicit latest).

Visual/manual checks:
- Compare replay outputs from `final` vs `latest` on same run when available; ensure selection changes behavior as expected.

## Notes

- Default `run_profile` is `fast` (quick diagnostics/iteration).
- `run_profile=medium` is the default validation profile.
- `run_profile=long` is for final comparisons and thesis-quality runs.
- MetaDrive uses `reset(seed=...)` as scenario index; scenario split separation is configured in `conf/config.yaml`.
- For fair comparisons, keep a fixed seed set across presets (example: `0,1,2` for all compared configs).
- Keep tuning runs out of final comparisons (`analysis.include_in_comparison=false`).
