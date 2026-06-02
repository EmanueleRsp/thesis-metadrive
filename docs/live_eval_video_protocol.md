# Live Evaluation Video Protocol

## Obiettivo

Rendere i video usati per analisi, report e tesi coerenti con gli episodi da cui vengono calcolate le metriche di evaluation, evitando di trattare come "ufficiale" un replay ricostruito post-hoc.

## Problema

Nel codice attuale esiste una pipeline video **offline replay**:

- selezione episodi da `csv/eval_episodes.csv`
- rerun dell'ambiente con checkpoint + seed
- render in `videos/final_eval/*.gif`

Questa pipeline e` utile per materiale qualitativo, ma **non garantisce** che la traiettoria coincida esattamente con quella osservata nella evaluation originale. In MetaDrive la divergenza puo` dipendere da:

- seed e scenario window (`start_seed`, `num_scenarios`)
- curriculum stage / `eval_env`
- traffico reactive/IDM vs log-replay
- wrapper stack
- termination flags
- reward/rulebook mode
- stato iniziale e dinamica interna

Conclusione: il replay offline va considerato **qualitativo/esplorativo**, non la fonte ufficiale per supportare metriche quantitative.

## Stato attuale nel repo

### Logging evaluation gia` presente

Il training runtime salva gia`:

- metriche aggregate in `csv/evals.csv`
- metriche episodio per episodio in `csv/eval_episodes.csv`
- metriche per-regola in `csv/rule_metrics.csv`
- metriche finali in `csv/final_eval.csv`

Hook principali:

- `src/thesis_rl/runtime/loops/train_loop.py`
- `src/thesis_rl/agent/agent.py`
- `src/thesis_rl/runtime/io/csv_recorder.py`

### Replay video post-hoc gia` presente

La pipeline video attuale usa:

- `conf/video/default.yaml`
- `src/thesis_rl/analysis/videos/select_video_episodes.py`
- `src/thesis_rl/analysis/videos/render_selected_videos.py`
- `src/thesis_rl/analysis/videos/render_qualitative_videos.py`

In `conf/video/default.yaml` e` gia` annotato:

- `mode: offline_replay`
- `live_final_eval` come opzione futura

Questo e` il punto naturale da estendere.

## Decisione architetturale

Per gli artefatti "ufficiali" del progetto:

1. il video va registrato **durante la evaluation**
2. il video va salvato insieme a manifest e log numerico
3. il replay offline resta disponibile, ma etichettato come **non authoritative replay**

## Bundle ufficiale di evaluation

Per ogni episodio registrato durante evaluation, il run deve produrre un bundle coerente:

- video renderizzato
- manifest JSON
- trajectory log JSONL o NPZ/Parquet
- collegamento ai CSV gia` esistenti

Struttura proposta dentro `paths.videos_dir`:

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

In alternativa, se si preferisce separare asset pesanti da quelli qualitativi:

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

La seconda opzione e` piu` pulita se si vogliono distinguere:

- `videos/` = media
- `artifacts/` = metadata e log numerici

## Manifest per episodio

Ogni video ufficiale deve avere un file `episode_<id>.manifest.json` con almeno:

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

### Nota pratica

`env_config_resolved` deve contenere la config effettiva usata per la evaluation, dopo merge di:

- config base
- eventuale `curriculum.eval_env`
- split test via `apply_eval_scenario_seed_split(...)`

Questo e` piu` utile di salvare solo un riferimento astratto.

## Trajectory log per episodio

Ogni episodio registrato deve salvare anche il log numerico reale, ad esempio in JSONL:

```json
{"t":0,"action":[0.1,-0.2],"reward":0.03,"done":false,"truncated":false,"ego":{"x":1.2,"y":3.4,"heading":0.5,"speed_kmh":21.0},"rule":{"vector":[0.2,1.0],"violated":[]}}
{"t":1,"action":[0.1,-0.1],"reward":0.04,"done":false,"truncated":false,"ego":{"x":1.4,"y":3.8,"heading":0.5,"speed_kmh":22.1},"rule":{"vector":[0.3,1.0],"violated":[]}}
```

Campi minimi consigliati per step:

- `t`
- `action`
- `reward`
- `env_reward`
- `scalar_rule_reward`
- `hybrid_reward`
- `done`
- `truncated`
- `ego`:
  - posizione
  - heading
  - speed
- `route_completion`
- `rule_reward_vector`
- `rule_metadata`
- `violated_rules`
- `top_rule_violation`
- `info` ridotto / sanitizzato

Se il payload completo di `info` e` troppo verboso, conviene salvare:

- un sottoinsieme stabile e utile per audit
- piu` un eventuale campo opzionale `raw_info_path`

## Estensioni minime al CSV schema

L'infrastruttura CSV attuale e` gia` molto vicina. Le estensioni minime utili sono:

### `eval_episodes.csv`

Aggiungere:

- `video_authoritative_path`
- `video_manifest_path`
- `trajectory_log_path`
- `video_recorded_live`
- `replay_warning`

Nota:

- l'attuale `video_path` puo` restare per compatibilita`
- ma dovrebbe essere trattato come campo legacy / generic video path

### `final_eval.csv`

Aggiungere opzionalmente:

- `official_video_bundle_dir`
- `official_video_count`
- `official_trajectory_count`

## Modifiche software proposte

### Fase 1 — supporto runtime per live recording

#### 1. Config

Estendere `conf/video/default.yaml`:

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

Semantica:

- `offline_replay`: comportamento attuale
- `live_final_eval`: registra durante la final evaluation
- `official_only`: i file prodotti in questa modalita` sono la fonte ufficiale

#### 2. Recorder runtime

Introdurre un modulo nuovo, per esempio:

- `src/thesis_rl/runtime/io/eval_artifacts.py`

Responsabilita`:

- aprire writer video
- aprire writer trajectory log
- accumulare metadata episodio
- chiudere/salvare manifest finale
- restituire i path da scrivere nei CSV

API proposta:

```python
class EvalEpisodeArtifactRecorder:
    def on_episode_start(...)
    def on_step(frame, action, reward, done, truncated, info, ...)
    def on_episode_end(metrics, ...)
    def close()
```

### Fase 2 — hook nell'evaluation loop

Il punto migliore e` dentro `Agent.evaluate(...)` in `src/thesis_rl/agent/agent.py`, perche' li` esiste gia` il loop episodio/step e si vedono:

- reset seed
- azioni scelte
- reward
- done/truncated
- `step_info`
- metriche finali episodio

Estensione proposta della signature:

```python
def evaluate(
    ...,
    artifact_recorder_factory: Callable[[dict[str, Any]], Any] | None = None,
)
```

Flusso:

1. all'inizio episodio crea recorder con contesto episodio
2. ad ogni step:
   - render frame
   - salva trajectory row
3. a fine episodio:
   - salva manifest
   - ritorna path artefatti
4. `metrics["per_episode"]` include anche i path degli artefatti

Nuovi campi in `metrics["per_episode"]`:

- `video_authoritative_path`
- `video_manifest_path`
- `trajectory_log_path`
- `video_recorded_live`

### Fase 3 — scrittura nei CSV run-time

Aggiornare i punti in `src/thesis_rl/runtime/loops/train_loop.py` che gia` iterano su `per_episode` per scrivere `eval_episodes.csv`.

Zone principali:

- evaluation intermedia
- final evaluation

Per la prima iterazione conviene registrare **solo final evaluation**, quindi:

- `video.mode=live_final_eval`
- `video.record_final_eval=true`
- `video.record_intermediate_evals=false`

Questo minimizza overhead e volume dati.

### Fase 4 — standalone `cli.evaluate`

Allineare anche `src/thesis_rl/runtime/loops/eval_loop.py`, in modo che una evaluation lanciata separatamente produca lo stesso bundle ufficiale.

Questo e` importante per coerenza sperimentale:

- training-time final eval
- standalone eval da checkpoint

devono produrre artefatti con lo stesso schema.

## Render durante evaluation

Per il rendering live si puo` riusare la stessa logica topdown gia` presente in:

- `src/thesis_rl/analysis/videos/render_selected_videos.py`

in particolare:

- `_render_topdown_frame(...)`
- `_save_gif(...)`

Conviene estrarre questi helper in un modulo condiviso, ad esempio:

- `src/thesis_rl/runtime/io/video_utils.py`

cosi` il runtime e la pipeline analysis usano la stessa implementazione.

## Compatibilita` con la pipeline attuale

La pipeline offline replay non va rimossa subito. Va solo riclassificata:

- **ufficiale**: live evaluation bundle
- **non ufficiale**: offline replay / qualitative replay

Regola proposta:

- i report quantitativi e la tesi devono usare solo `video_recorded_live=true`
- i replay offline possono essere usati per debug o figure illustrative, ma con etichetta esplicita

## Impatto sui moduli analysis

### `select_video_episodes.py`

Non deve piu` selezionare episodi da rerenderizzare per i casi ufficiali. Deve invece:

- leggere `eval_episodes.csv`
- filtrare gli episodi che hanno `video_recorded_live=true`
- scegliere quali bundle promuovere nel report

### `render_selected_videos.py`

Va sdoppiato logicamente:

- percorso legacy: replay offline
- percorso nuovo: se il video ufficiale esiste, **non rerenderizzare**

Comportamento preferito:

1. usa `video_authoritative_path` se presente
2. fallback a replay offline solo se richiesto esplicitamente

## Strategia incrementalmente sicura

### Milestone 1

Implementare solo:

- live recording per `final_eval`
- manifest JSON
- path in `eval_episodes.csv`

senza trajectory log completo.

### Milestone 2

Aggiungere:

- trajectory log step-by-step
- metadati piu` ricchi
- link da `final_eval.csv`

### Milestone 3

Aggiornare analysis pipeline per preferire sempre gli artefatti live.

## Validazione

Nuovi test consigliati:

- `tests/test_live_eval_video_protocol.py`
- `tests/test_eval_artifact_recorder.py`

Verifiche minime:

1. `Agent.evaluate(...)` con recorder produce un manifest per episodio
2. `train_loop.py` scrive i nuovi campi in `eval_episodes.csv`
3. `video.mode=live_final_eval` non richiede replay successivo
4. `render_selected_videos.py` usa il file ufficiale se esiste
5. `offline_replay` resta backward compatible

## Decisione finale consigliata

### Policy sperimentale

- checkpoint = artefatto del training
- `final_eval.csv` + `eval_episodes.csv` + `rule_metrics.csv` = verita` quantitativa
- video live + manifest + trajectory log = verita` qualitativa/forense coerente con la quantitativa

### Regola operativa

Per run importanti:

1. train
2. salva checkpoint
3. esegui final evaluation deterministica
4. durante quella evaluation salva:
   - metriche
   - CSV
   - video
   - manifest
   - trajectory log
5. usa nel report solo questi artefatti

Questo approccio e` coerente con il design gia` presente nel repo e richiede un'estensione localizzata, non una riscrittura completa della pipeline.
