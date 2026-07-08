# Analysis Pipeline

This is the unified pipeline for aggregating multi-seed runs, generating
tables/plots, and building A/B/C comparisons. Every analysis is scoped by
`run_profile`.

Conventions used in this document:

- `outputs-root` = `/workspace/outputs` by default
- `analysis-root` = `<outputs-root>/analysis`
- `/scratch/...` is only an optional host mount on the remote VM

## Useful Flags

- `--analysis-root` default `<outputs-root>/analysis`
- `--outputs-root` default `<outputs-root>`
- `--run-profile` required (`smoke|fast|medium|long|...`)
- `--comparison-dimension` `none|curriculum|reward|algorithm|task_contract`
- `--comparison-id` regenerate one comparison view only
- `--algorithm`
- `--reward-type`
- `--reward-behavior`
- `--rulebook-config`

## Main Commands

```bash
python -m thesis_rl.analysis.run_analysis --run-profile medium --only all --no-videos
python -m thesis_rl.analysis.run_analysis --run-profile medium --only aggregate
python -m thesis_rl.analysis.run_analysis --run-profile medium --only tables
python -m thesis_rl.analysis.run_analysis --run-profile medium --only plots
```

## A/B/C Comparisons

```bash
python -m thesis_rl.analysis.run_analysis --run-profile medium --only all --no-videos \
  --comparison-dimension curriculum \
  --algorithm sb3_td3 \
  --reward-type native \
  --reward-behavior monitor_only \
  --rulebook-config selection

python -m thesis_rl.analysis.run_analysis --run-profile medium --only all --no-videos \
  --comparison-dimension reward \
  --algorithm sb3_td3 \
  --curriculum-name stages \
  --rulebook-config selection

python -m thesis_rl.analysis.run_analysis --run-profile medium --only all --no-videos \
  --comparison-dimension algorithm \
  --curriculum-name stages \
  --reward-type rulebook \
  --reward-behavior scalar_reward \
  --rulebook-config selection
```

## Main Outputs

### `<outputs-root>/analysis/<run_profile>/aggregated/`

- `train_chunks_all_runs.csv`
- `evals_all_runs.csv`
- `eval_episodes_all_runs.csv`
- `promotions_all_runs.csv`
- `rule_metrics_all_runs.csv`
- `final_eval_all_runs.csv`
- `selected_runs.csv`

### `<outputs-root>/analysis/<run_profile>/tables/`

- `final_evaluation.*`
- `curriculum_efficiency.*`
- `sample_efficiency_thresholds.*`
- `generalization_train_vs_eval.*`
- `rulebook_compliance.*`

### `<outputs-root>/analysis/<run_profile>/plots/`

- `learning_success_vs_global_step.png`
- `learning_collision_vs_global_step.png`
- `learning_out_of_road_vs_global_step.png`
- `learning_route_completion_vs_global_step.png`

### `<outputs-root>/analysis/<run_profile>/comparisons/<dimension>/<comparison_id>/`

- `aggregated/*.csv`
- `tables/*`
- `plots/*`
- `qualitative/video_manifest.csv` when requested
- `qualitative/gifs/*.gif` when requested
