# Validation Commands (Phase 1)

Start a tmux session and start a container with GPU access. Then run the following commands from the container shell to validate core training/evaluation/checkpointing behavior and CSV output contracts before scaling up to longer runs and deeper analysis.

Run all commands from project root.

Note:
- In `conf/config.yaml` the default is `run_profile=fast`.
- Use `run_profile=...` only when you intentionally want a different run budget.
- `run_profile=tune` is the preferred profile for local hyperparameter tuning loops.
- The examples below assume `OUTPUTS_ROOT=/workspace/outputs`.
- For scalar planner baselines, the current preferred presets are the
  fork-backed ones under `presets/agent/*_sb3`.
- For the current baseline SB3-fork closure, the active gate is:
  tests + smoke runs + medium train runs.
- Standalone `load+eval` and `resume` validation are currently deferred and are
  not required to consider the scalar migration functionally closed.
- `presets/agent/*_lq_sb3` assume `obs=semantic_state`.

Optional helper for path-heavy commands:

```bash
export OUTPUTS_ROOT=/workspace/outputs
```

## 0) Setup

What this step validates:
- Environment and dependencies are consistent before running expensive experiments.
- Core code contracts still pass after recent refactors.

```bash
uv run --no-sync python -m pytest -q
```

If you are inside the Docker Compose container, this works because the service
mounts the whole repository at `/workspace/thesis-metadrive`, including the
submodules under `third_party/`, while `torch` is installed from the dedicated
CUDA wheel index after the platform-neutral `uv` synchronization.

Expected:
- Tests pass.
- No import/runtime wiring errors.

Visual/manual checks:
- Scan test output summary: no skipped-critical suites, no intermittent errors.

## Code Quality Commands

Run the canonical Ruff lint target over project-owned Python code:

```bash
make lint
```

Formatting commands use the same default scope (`src tests scripts`):

```bash
make format
make format-check
```

The lint baseline is clean. The repository-wide formatting baseline is not yet
clean, so do not run mass formatting as part of a semantic change. Until a
dedicated formatting-only change establishes the baseline, format and check only
new or materially modified files:

```bash
make format PYTHON_QUALITY_PATHS="src/thesis_rl/module.py tests/test_module.py"
make format-check PYTHON_QUALITY_PATHS="src/thesis_rl/module.py tests/test_module.py"
```

Global mypy enforcement and a mandatory coverage threshold are intentionally
deferred. Do not invent either as an implementation gate.

## 0.2) Rulebook v2 F10: calibrazione e pilot

I target `make` automatizzano soltanto la parte riproducibile. Le prove di
frenata devono essere raccolte con la configurazione ego effettiva, su strada
rettilinea e piana, senza traffico. Non inserire valori sintetici: servono 10
prove valide per ciascuna velocità target `5, 10, 15, 20 m/s`.

The repository provides the canonical source configuration at
`conf/rulebook_v2/ego_calibration.json`. `make rulebook-v2-init` installs it
non-destructively at:

```text
data/scenarionet/rulebook_v2/ego_config.json
```

`ego_config.json` è il JSON canonico della configurazione fisica ego congelata
derivata dai default MetaDrive/ScenarioNet correnti e descrive anche il banco
di prova rettilineo:

```json
{
  "map_config": {"type": "block_sequence", "config": "SSSSSSSSSS"},
  "traffic_density": 0.0,
  "random_agent_model": false,
  "num_agents": 1,
  "vehicle_config": {"vehicle_model": "default"}
}
```

Il campo `vehicle_config` deve corrispondere al modello ego effettivo del run
finale. Se il run finale introduce override fisici, sostituire il file con il
JSON risolto del run prima della raccolta: lo split del dataset contiene gli
scenari e le mappe, non i parametri dinamici dell'ego, quindi non può generare
questo file in modo affidabile. Il target `make rulebook-v2-collect-trials` usa questa configurazione
per eseguire automaticamente le prove e generare
`data/scenarionet/rulebook_v2/braking_trials.json`. Ogni elemento prodotto
contiene:

```json
{
  "target_speed_mps": 5.0,
  "reached_speed_mps": 5.02,
  "collided": false,
  "left_lane": false,
  "mean_deceleration_mps2": 3.1
}
```

La sequenza raccomandata è:

```bash
make rulebook-v2-prepare
make rulebook-v2-pilot
```

`rulebook-v2-prepare` uses the CPU-only dataset pipeline container. When the
calibration is missing it always recollects all 40 real braking trials before
creating the hash-bound artifact; it never rebinds stale trials or creates a
synthetic calibration. The full ScenarioNet pipeline invokes this preparation
automatically only when the required artifacts are missing.

`rulebook-v2-collect-trials` esegue 10 prove per target sulla pista `S` senza
traffico, applica acceleratore/frenata massimi e misura la decelerazione tra il
primo campione sotto il 90% e quello sotto il 10% della velocità iniziale.
`rulebook-v2-calibrate` calcola automaticamente l'hash canonico di
`ego_config.json`, applica il protocollo `Q_0.05^lower`, floor a `0.1` e cap a
`4.0 m/s²`, quindi scrive:

```text
data/scenarionet/rulebook_v2/calibration_b_e.json
```

### Filtro Rulebook prima degli split

La pipeline ScenarioNet ora esegue il filtro Rulebook v2 **prima** di assegnare
gli split. Il catalogo grezzo resta un artifact di audit; soltanto gli scenari
che superano map-matching, geometrie, segnali pertinenti e contratti hash
possono entrare in `train`, `validation` o `test`.

Per eseguire soltanto questo passaggio sul catalogo grezzo esistente:

```bash
make rulebook-v2-filter-catalog
```

Produce:

```text
data/scenarionet/catalog/scenario_catalog_rulebook_v2.parquet
data/scenarionet/rulebook_v2/catalog_eligibility.json
```

Il JSON contiene ogni scenario analizzato, l'esito, le cause di esclusione,
l'hash geometrico e l'hash della calibrazione. Per rigenerare l'intera pipeline
con il filtro a monte, inclusi split e runtime view:

```bash
make scenarionet-pipeline
```

La configurazione `conf/scenarios/pipeline_v1.yaml` abilita il filtro v2. Il
comando richiede quindi `ego_config.json` e `calibration_b_e.json` validi;
se uno dei due manca, la pipeline termina senza costruire split non conformi.
The `rulebook_v2.workers` parameter controls the process count for static
filtering; its default is 8. The CLI shows a Rich dashboard with completed
records, rate, elapsed time, and ETA while keeping artifacts deterministic and
ordered by `scenario_uid`.
The catalog-building stage uses the independent `waymo.workers` and
`pg.workers` values for per-file loading and feature extraction, with separate
Rich progress tasks and a single parent writer for the final artifacts.

Il pilot preliminare non dichiara l'eleggibilità finale: misura conversione
statica e task-route eligibility. Il report viene scritto in:

```text
data/scenarionet/rulebook_v2/pilot_offline.json
```

Per il pilot finale serve anche l'hash della configurazione geometrica
congelata:

```bash
make rulebook-v2-pilot-final GEOMETRY_CONFIG_HASH='<sha256-geometria>'
```

Questo target richiede l'artifact `b_e` validato e passa lo stesso hash ego al
contratto di calibrazione. Il report finale è:

```text
data/scenarionet/rulebook_v2/pilot_final.json
```

Per eseguire in un'unica tranche raccolta delle 40 prove, calibrazione,
validazione, pilot finale e controlli v2:

```bash
make rulebook-v2-f10 GEOMETRY_CONFIG_HASH='<sha256-geometria>'
```

Se mancano `ego_config.json`, `braking_trials.json`, l'artifact oppure
`GEOMETRY_CONFIG_HASH`, il target termina volontariamente con errore esplicito;
non sono previsti fallback o artifact sintetici.

Per i controlli ordinari restano disponibili:

```bash
make rulebook-v2-check
```

che esegue test Rulebook v2/CLI, Ruff e `git diff --check` nel container `dev`.

## 0.1) Fork-Backed Planner Checks

What this step validates:
- The local SB3 fork is the effective algorithmic core for the current scalar
  planner path.
- Canonical fork-backed presets and thesis-to-SB3 bridges still compose and
  build correctly.

```bash
uv run --no-sync python -m pytest -q \
  tests/test_hydra_agent_presets.py \
  tests/test_sb3_extensions.py \
  tests/test_sb3_direct_backends.py
```

Expected:
- Fork-backed preset composition passes.
- SB3 bridge tests pass.
- Direct fork-backed backend tests pass.

Visual/manual checks:
- If failures mention `legacy` presets only, check whether the failure is in an
  archival compatibility path or in the canonical fork-backed path.

## 1) Smoke Test End-to-End (train + eval + checkpoint)

What this step validates:
- Minimal end-to-end train/eval/checkpoint flow works on the current codebase.
- TD3 update path is active with minimal targeted overrides for a short smoke run.

```bash
uv run --no-sync python -m thesis_rl.cli.train --config-name presets/test/smoke_train
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
- Resume restores planner/adapter/replay/state/RNG and continues without silent resets
  (`RESUME-ABRUPT-001`, `TRANSITION-REPLAY` v1.1).

Resume **in place** (same `run_dir`, so CSVs and checkpoints continue in one directory)
from the newest periodic snapshot, which carries the replay buffer:

```bash
uv run --no-sync python -m thesis_rl.cli.train \
  --config-name presets/td3/td3_native_curr \
  run_profile=medium \
  paths.run_dir='/absolute/path/to/previous/run_dir' \
  checkpoint.resume.enabled=true \
  checkpoint.resume.run_dir='/absolute/path/to/previous/run_dir' \
  checkpoint.resume.checkpoint_name=periodic
```

`checkpoint_name` accepts `latest` (model-only, every chunk), `final`, `periodic`
(newest `periodic/step_X` with a complete model/replay pair) or an explicit stem
such as `periodic/step_00125000`. A model-only resume of TD3/SAC fails before
training unless `checkpoint.resume.allow_replay_reset=true`, which starts an
empty replay segment and records `replay_reset=true` (REQ-025).

Expected:
- Resume log includes restored checkpoint/state/replay/RNG.
- Training continues with increasing global step.
- New checkpoints and CSV rows append coherently.
- A torn snapshot (model and state from different saves) or a different `seed`
  fails before training with a message naming both values.

Kill-and-resume check (`TEST-RES-010`): run a smoke, `docker kill -s KILL` its
container after the first `replay_snapshot_written` event, relaunch with the
command above; the resumed run must finish with `final.zip`, and
`checkpoints/metadata/checkpoint_index.csv` must show monotone `global_step`.

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
- `run_profile=tune` is for algorithm-focused hyperparameter search.
- `run_profile=long` is for final comparisons and thesis-quality runs.
- MetaDrive uses `reset(seed=...)` as scenario index; scenario split separation is configured in `conf/config.yaml`.
- For fair comparisons, keep a fixed seed set across presets (example: `0,1,2` for all compared configs).
- Keep tuning runs out of final comparisons (`analysis.include_in_comparison=false`).
