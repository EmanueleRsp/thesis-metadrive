# Analysis Pipeline

Questa cartella contiene gli script post-run per aggregare risultati multi-seed e generare tabelle/plot.

## Script disponibili

- `run_analysis.py`: orchestratore unico (`aggregate + tables + plots`).
- `aggregate_runs.py`: unisce i CSV prodotti nelle run in file aggregati `*_all_runs.csv`.
- `make_final_tables.py`: genera tabella finale `mean ± 95% CI` in `csv` e `md`.
- `make_curriculum_tables.py`: genera tabella curriculum/sample-efficiency in `csv` e `md`.
- `make_rulebook_tables.py`: genera tabelle rulebook compliance globali e per-regola in `csv` e `md`.
- `make_plots.py`: genera i grafici comparativi principali in formato immagine (`.png`).

## Uso rapido

```powershell
python -m analysis.run_analysis
```

Parametri principali:

- `--outputs-root`: root delle run (default: `outputs`)
- `--analysis-root`: root output analisi (default: `analysis`)
- `--only`: `all | aggregate | tables | plots`
- `--seed-list`: lista seed ufficiale (default `0..9`)
- `--total-timesteps`, `--eval-episodes`, `--final-eval-episodes`: filtri protocollo opzionali

## Output

```text
analysis/aggregated/
  train_chunks_all_runs.csv
  evals_all_runs.csv
  eval_episodes_all_runs.csv
  promotions_all_runs.csv
  rule_metrics_all_runs.csv
  final_eval_all_runs.csv

analysis/tables/
  final_evaluation.csv
  final_evaluation.md
  curriculum_efficiency.csv
  curriculum_efficiency.md
  rulebook_compliance.csv
  rulebook_compliance.md
  rule_violation_by_rule.csv
  rule_violation_by_rule.md

analysis/plots/
  learning_success_vs_global_step.png
  learning_collision_vs_global_step.png
  learning_out_of_road_vs_global_step.png
  learning_route_completion_vs_global_step.png
  learning_rule_top_violation_vs_global_step.png
  learning_avg_error_value_vs_global_step.png
  curriculum_stage_index_vs_global_step.png
  safety_performance_tradeoff.png
  rule_metrics_violation_rate_by_rule.png
  episode_error_distribution_boxplot.png
```
