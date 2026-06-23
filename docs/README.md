# Docs Index

Questa cartella contiene sia documentazione operativa corrente sia documenti
di design/storici accumulati durante l'evoluzione del progetto.

Per evitare ambiguità:

- usa prima i documenti sotto **Operativi correnti**
- tratta i documenti sotto **Design / riferimento** come supporto concettuale
- tratta i documenti sotto **Storici / da rivedere** come non autoritativi finché
  non vengono riallineati allo stato attuale del repo

## Operativi correnti

- `docs/comparison_run_commands.md`
  - comandi correnti per smoke run e run di confronto nel setup Docker attuale
- `docs/algorithm_selection_playbook.md`
  - sequenza operativa e comandi per selezione algoritmo, qualification run e
    passaggio successivo a osservazioni/encoder
- `docs/thesis_experiment_roadmap.md`
  - roadmap sperimentale corrente fase-per-fase, con budget, seed, componenti
    bloccati e criteri decisionali
- `docs/validation_commands.md`
  - checklist e comandi di validazione; utile, ma alcune parti vanno lette alla
    luce del workflow Docker attuale
  - per la chiusura corrente della migrazione SB3 fork, considera come gate
    attivo test, smoke run e medium run; `load+eval` e `resume` restano
    validazioni rimandate
- `docs/analysis_pipeline.md`
  - pipeline analitica e convenzioni sugli output sotto `/scratch/$USER/...`

## Riferimento corrente

- `docs/algorithm_comparison_protocol.md`
  - obiettivi e logica dei confronti sperimentali
- `docs/csv_evaluation_objectives.md`
  - schema e significato degli artefatti CSV
- `docs/live_eval_video_protocol.md`
  - stato e obiettivi della pipeline video/evaluation
- `docs/curriculum_learning_specification.md`
  - specifica tecnica del curriculum scenario-level
  - nota importante: alcune flag previste per `ScenarioEnv` replay non sono
    oggi supportate 1:1 dalla versione di MetaDrive nel container; vedi anche
    `docs/scenario_acl_implementation_plan.md`

## Design / architettura

- `docs/architecture.md`
- `docs/scenario_acl_implementation_plan.md`
- `docs/sb3_fork_migration_plan.md`
- `docs/initial_design_decisions.md`
- `docs/metadrive_assumptions.md`

Questi documenti aiutano a capire il razionale del progetto, ma non vanno
interpretati come guida operativa definitiva per il setup corrente.

Nota:
- `docs/sb3_fork_migration_plan.md` e` anche il riferimento corrente per:
  - confine `thesis_rl` vs fork SB3
  - strategia di integrazione encoder/decoder/policy custom
  - criteri per rimuovere i backend legacy
  - stato della chiusura funzionale della migrazione baseline

## Storici / da rivedere

- `docs/hydra_config_strategy.md`
- `docs/implementation_plan.md`
- `docs/open_questions.md`

Questi documenti riflettono fasi precedenti del progetto. Possono essere ancora
utili come contesto storico, ma al momento non sono la fonte di verità per:

- comandi da eseguire
- layout attuale dei path
- workflow Docker/Compose
- stato effettivo delle decisioni implementative

## Regola pratica

Se un documento in `docs/` è in conflitto con:

- `README.md`
- il comportamento reale del codice
- i comandi in `docs/comparison_run_commands.md`

considera il documento da aggiornare o verificare, non come riferimento finale.
