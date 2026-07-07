# Docs Index

Questa cartella contiene documentazione operativa, note architetturali e
materiale storico.

## Convenzioni correnti

Le convenzioni portabili del repository sono:

- dipendenze esterne sotto `third_party/`
- clone con submodule inizializzati
- configurazione host-specifica via `.env`
- path stabili nel container:
  - `/workspace/thesis-metadrive`
  - `/workspace/outputs`
  - `/workspace/data`
  - `/workspace/data/scenarionet`
  - `/workspace/data/metadrive`

`/scratch/...` resta un'opzione valida per la VM remota, ma non e' piu' un
requisito di default.

## Operativi correnti

- `docs/comparison_run_commands.md`
  - comandi correnti per smoke run e run di confronto nel setup Docker
- `docs/algorithm_selection_playbook.md`
  - sequenza operativa per selezione algoritmo e qualification run
- `docs/validation_commands.md`
  - checklist e comandi di validazione leggeri
- `docs/analysis_pipeline.md`
  - pipeline analitica e convenzioni sugli output
- `docs/thesis_experiment_roadmap.md`
  - roadmap sperimentale corrente

## Riferimento corrente

- `docs/algorithm_comparison_protocol.md`
- `docs/csv_evaluation_objectives.md`
- `docs/live_eval_video_protocol.md`
- `docs/curriculum_learning_specification.md`
- `docs/scenario_acl_implementation_plan.md`

## Design / architettura

- `docs/architecture.md`
- `docs/sb3_fork_migration_plan.md`
- `docs/initial_design_decisions.md`
- `docs/metadrive_assumptions.md`

## Storici / da rivedere

- `docs/hydra_config_strategy.md`
- `docs/implementation_plan.md`
- `docs/open_questions.md`

## Regola pratica

Se un documento e' in conflitto con:

- `README.md`
- il comportamento reale del codice
- il layout `third_party/` + `.env` + `/workspace/{outputs,data}`

trattalo come da aggiornare, non come fonte di verita' finale.
