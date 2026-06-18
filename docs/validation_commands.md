# Validation Commands (Phase 1)

Start a tmux session and start a container with GPU access. Then run the following commands from the container shell to validate core training/evaluation/checkpointing behavior and CSV output contracts before scaling up to longer runs and deeper analysis.

Run all commands from project root.

Note:
- In `conf/config.yaml` the default is `run_profile=fast`.
- Use `run_profile=...` only when you intentionally want a different run budget.
- The examples below assume `OUTPUTS_ROOT=/scratch/$USER/thesis-metadrive/outputs`.

Optional helper for path-heavy commands:

```bash
export OUTPUTS_ROOT=/scratch/$USER/thesis-metadrive/outputs
```

## 0) Setup

What this step validates:
- Environment and dependencies are consistent before running expensive experiments.
- Core code contracts still pass after recent refactors.

```bash
uv sync --extra dev
uv pip install -e .
uv run --no-sync python -m pytest -q
```

If you are inside the Docker Compose container, this works because the service
mounts `../third-party/metadrive` at `/workspace/third-party/metadrive` for
the local `metadrive` source with write access for build metadata, while
`torch` is inherited from the NVIDIA base image instead of being installed by
`uv`.

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
uv run --no-sync python -m thesis_rl.cli.train --config-name presets/td3/td3_monitor_only_no_curr \
  run_profile=smoke
```

Expected:
- `n_updates > 0`.
- `actor_loss` and `critic_loss` are finite (not `NaN`).
- Intermediate eval runs on validation split, final eval on test split.
- `checkpoints/final.zip` exists.
- Run directory contains `logs/`, `csv/`, `checkpoints/`, `artifacts/`.

Visual/manual checks:
- Open `<run_dir>/logs/` and confirm training progresses (no repeated reset/crash patterns).
- Open `<run_dir>/csv/final_eval.csv` and confirm it has one coherent row.

## 2) Baseline Validation (curriculum OFF, native / behavior=off)

What this step validates:
- Baseline behavior without curriculum/reward-wrapper confounders.
- Core agent-planner-adapter path on native scalar reward.

```bash
for s in 0 1 2; do
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/td3/td3_native_no_curr \
    run_profile=medium seed=$s
done
```

Expected:
- All runs complete (`status=completed` in run metadata).
- Final eval metrics are produced for each seed.

Visual/manual checks:
- Compare `final_eval.csv` across seeds: metrics should vary but stay in plausible ranges.
- Spot-check `evals.csv` curves: no flatlined or exploding reward/error trends.

## 2.1) Baseline Validation (curriculum OFF, native / behavior=monitor_only)

What this step validates:
- Rulebook wrapper active with curriculum disabled while training signal remains native env scalar reward.
- `monitor_only` path (`lambda_env=1.0`, `lambda_rule=0.0`) is stable in non-curriculum runs.

```bash
for s in 0 1 2; do
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/td3/td3_monitor_only_no_curr \
    run_profile=medium seed=$s
done
```

Expected:
- All runs complete (`status=completed` in run metadata).
- Final eval metrics are produced for each seed.
- Aggregate behavior is broadly aligned with step 2 (allowing seed noise), with no systematic regressions.

Visual/manual checks:
- Inspect run config snapshot and confirm `reward.type=native`, `reward.behavior=monitor_only`, `reward.lambda_env=1.0`, `reward.lambda_rule=0.0`.
- Compare `final_eval.csv` from steps 2 and 2.1: no systematic collapse/drift introduced by wrapper activation.

## 3) Curriculum Validation (curriculum ON, native / behavior=off)

What this step validates:
- Curriculum progression logic and stage-aware train/eval splitting.

```bash
for s in 0 1 2; do
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/td3/td3_native_curr \
    run_profile=medium seed=$s
done
```

Expected:
- Curriculum logs/events appear (`promotions.csv` when promotions happen).
- Stage transitions are coherent (no train/eval pool overlap warnings as hard errors).

Visual/manual checks:
- Inspect `promotions.csv`: promotion events should be temporally coherent (increasing steps/eval ids).
- In plots/tables, check if stage transitions align with metric changes (no impossible jumps).

## 3.1) Curriculum Validation (curriculum ON, native / behavior=monitor_only)

What this step validates:
- Curriculum progression with rulebook wrapper active while training signal remains the native env scalar reward.
- `monitor_only` path (`lambda_env=1.0`, `lambda_rule=0.0`) does not regress core train/eval behavior.

```bash
for s in 0 1 2; do
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/td3/td3_monitor_only_curr \
    run_profile=medium seed=$s
done
```

Expected:
- Runs complete and produce standard train/eval/checkpoint artifacts.
- Curriculum events remain coherent as in step 3.
- Final returns should be broadly aligned with step 3 (allowing seed noise), since reward passed to replay is env-native.

Visual/manual checks:
- Inspect run config snapshot and confirm `reward.type=native`, `reward.behavior=monitor_only`, `reward.lambda_env=1.0`, `reward.lambda_rule=0.0`.
- Compare `final_eval.csv` from steps 3 and 3.1: no systematic collapse/drift introduced by wrapper activation.

## 4) Rulebook Validation (curriculum ON, rulebook / behavior=scalar_reward)

What this step validates:
- Rulebook scalar reward path under curriculum.
- Rule components/margins are available for diagnostics and later scale tuning.

```bash
for s in 0 1 2; do
  uv run --no-sync python -m thesis_rl.cli.train \
    --config-name presets/td3/td3_scalar_reward_curr \
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
uv run --no-sync python -m thesis_rl.analysis.run_analysis --run-profile medium --only aggregate
uv run --no-sync python -m thesis_rl.analysis.run_analysis --run-profile medium --only tables
```

Expected:
- Metrics are finite and in valid ranges (`rate` fields in `[0,1]`).
- No obviously broken regime (example: all-zero success with all-one collision).

Visual/manual checks:
- Open `$OUTPUTS_ROOT/analysis/medium/tables/final_evaluation.md` and compare rows grouped by curriculum/reward behavior.

## 5) Scale-Tuning Pass (rule margins -> suggested scales)

What this step validates:
- Logged rule margins can be converted into stable scale suggestions.
- Sparse rules (for example collision-related) are handled with explicit coverage checks (no default fallback).

Collect diagnostics logs from normal rulebook behavior:

```bash
uv run --no-sync python -m thesis_rl.cli.train \
  --config-name presets/td3/td3_scalar_reward_scale_tuning_no_curr \
```

Collect forced diagnostics logs to activate rare rules:

```bash
for i in $(seq 0 699); do
  uv run --no-sync python src/thesis_rl/tools/debug/force_rule_scenarios.py \
    --start-seed $((10000 + i)) \
    --seed $((42 + i)) \
    --map 5 \
    --traffic-density 0.5 \
    --out "$OUTPUTS_ROOT/forced_rule_scenarios_${i}.json"
done
```

Aggregate all margin logs into one dataset:

```bash
uv run --no-sync python -m thesis_rl.tools.calibration.aggregate_rule_margins \
  --input "$OUTPUTS_ROOT/**/logs/rule_margins.jsonl" "$OUTPUTS_ROOT/debug_rule_margins_forced_scenarios.jsonl" \
  --output "$OUTPUTS_ROOT/scale_calibration/aggregated_rule_margins.jsonl"
```

Run strict scale tuning with minimum active-sample requirements:

```bash
uv run --no-sync python -m thesis_rl.tools.calibration.scale_tuning \
  --input "$OUTPUTS_ROOT/scale_calibration/aggregated_rule_margins.jsonl" \
  --percentile 90 \
  --min-scale 1e-6 \
  --min-active-margin 1e-9 \
  --min-samples 300 \
  --strict \
  --output-json "$OUTPUTS_ROOT/scale_calibration/scale_report.json"
```

Optional one-command loop helper (aggregate + strict check):

```bash
uv run --no-sync python -m thesis_rl.tools.calibration.scale_calibration_loop \
  --inputs "$OUTPUTS_ROOT/**/logs/rule_margins.jsonl" "$OUTPUTS_ROOT/debug_rule_margins_forced_scenarios.jsonl" \
  --min-samples 300
```

Expected:
- If any rule has insufficient active samples, strict tuning fails explicitly.
- Once all rules pass coverage, `scale_report.json` contains final suggested scales and coverage stats.

Then update `conf/reward/rulebook_defaults.yaml` (`reward.scales`) and re-run step 4 for confirmation.

Visual/manual checks:
- Confirm suggested scales are not degenerate (all identical by accident, or extreme outliers without reason).
- After updating scales and re-running step 4, compare `rule_metrics.csv` to verify reduced saturation/imbalance.

## 6) Analysis Pipeline Validation

What this step validates:
- End-to-end aggregate/tables/plots/orchestrator behavior.

```bash
uv run --no-sync python -m thesis_rl.analysis.run_analysis --run-profile medium --only aggregate
uv run --no-sync python -m thesis_rl.analysis.run_analysis --run-profile medium --only tables
uv run --no-sync python -m thesis_rl.analysis.run_analysis --run-profile medium --only plots
uv run --no-sync python -m thesis_rl.analysis.run_analysis --run-profile medium --only all --no-videos
uv run --no-sync python -m thesis_rl.analysis.run_analysis --run-profile medium --only all --video-max 3
```

Expected:
- Aggregated CSVs under `$OUTPUTS_ROOT/analysis/medium/aggregated/*_all_runs.csv`.
- Tables generated under `$OUTPUTS_ROOT/analysis/medium/tables` (`.csv` and `.md`).
- Plots generated under `$OUTPUTS_ROOT/analysis/medium/plots` (`.png`).
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

## 6.1) Video-Pipeline Output Validation

What this step validates:
- Episode selection + replay rendering + CSV linkage.

```bash
find "$OUTPUTS_ROOT" -type f \( \
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

## 7) Idempotency / Reproducibility (analysis)

What this step validates:
- Re-running analysis with identical inputs is stable and non-destructive.

```bash
uv run --no-sync python -m thesis_rl.analysis.run_analysis --run-profile medium --only all --no-videos
uv run --no-sync python -m thesis_rl.analysis.run_analysis --run-profile medium --only all --no-videos
```

Expected:
- Re-running does not corrupt outputs.
- Aggregated numbers/tables remain stable for identical input runs.

Visual/manual checks:
- Compare timestamps/file counts: rerun may update files but should not change metric values unexpectedly.

## 8) Resume Validation

What this step validates:
- Resume restores planner/adapter/replay/state/RNG and continues without silent resets.

Use a real completed run path:

```bash
uv run --no-sync python -m thesis_rl.cli.train \
  --config-name presets/td3/td3_native_curr \
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

## 9) Best-Checkpoint Policy Validation

What this step validates:
- Best-checkpoint policies (lexicographic and rulebook variants) actually trigger and persist.

For at least one rulebook run, verify best-checkpoint artifacts:

```bash
find "$OUTPUTS_ROOT" -type f \( \
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
uv run --no-sync python -m thesis_rl.cli.train \
  --config-name presets/td3/td3_scalar_reward_curr \
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
