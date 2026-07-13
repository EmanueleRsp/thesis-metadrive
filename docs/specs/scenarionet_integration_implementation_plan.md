# Piano di implementazione e tracker — ScenarioNet v1

## Scopo

Questo documento traduce la specifica
[`scenarionet_integration_spec_v1.md`](scenarionet_integration_spec_v1.md) in un
piano operativo e mantiene lo stato di avanzamento dell'integrazione.

La specifica rimane la fonte normativa. Questo file registra:

- ordine e stato delle attività;
- criteri di completamento ed evidenze;
- problemi tecnici incontrati;
- decisioni implementative data-dependent;
- eventuali deviazioni dalla specifica e relativa approvazione.

Non si modifica la specifica per riflettere scorciatoie temporanee. Una modifica
del comportamento richiesto viene prima registrata nella sezione **Deviazioni**
e, se accettata, riportata in una nuova revisione della specifica.

---

## Stato complessivo

**Stato:** implementazione in corso  
**Fase corrente:** F9 — smoke end-to-end, pilot e freeze
**Ultimo aggiornamento:** 2026-07-12
**Specifica di riferimento:** ScenarioNet integration spec v1

### Legenda

| Stato | Significato |
|---|---|
| `NON_INIZIATA` | Nessuna attività implementativa eseguita |
| `IN_CORSO` | Attività in lavorazione |
| `BLOCCATA` | Non può avanzare; esiste un problema aperto bloccante |
| `PRONTA_PER_VERIFICA` | Implementazione terminata, evidenze ancora incomplete |
| `COMPLETATA` | Criteri di uscita soddisfatti ed evidenze registrate |
| `DEFERITA` | Fuori dalla v1 o rinviata esplicitamente dalla specifica |

### Dashboard delle fasi

| Fase | Obiettivo | Stato | Dipende da |
|---|---|---|---|
| F0 | Bootstrap, dipendenze e API locali | `COMPLETATA` | — |
| F1 | Modello dati, manifest e configurazione | `COMPLETATA` | F0 |
| F2 | Provider, catalogo e split core | `COMPLETATA` | F1 |
| F3 | Feature extraction, soglie e arms | `COMPLETATA` | F1 |
| F4 | Generazione PG offline e pilot | `COMPLETATA` | F0, F1, F3 |
| F5 | Conversione e preparazione Waymo | `IN_CORSO` | F0, F1 |
| F6 | Validazione e runtime database | `IN_CORSO` | F2–F5 |
| F7 | `ThesisScenarioEnv` e scene context | `IN_CORSO` | F0, F2 |
| F8 | Vectorization, logging e wiring training/eval | `COMPLETATA` | F2, F7 |
| F9 | Smoke end-to-end, pilot e freeze | `IN_CORSO` | F3–F8 |
| F10 | Costruzione dataset completo | `NON_INIZIATA` | F9; esecuzione utente |

---

## Principi di esecuzione

1. Il codice resta eseguibile e testabile dopo ogni fase.
2. Le API vengono adattate ai commit locali; i submodule non vengono aggiornati
   arbitrariamente.
3. I test unitari usano fixture sintetiche piccole. Waymo e la generazione PG
   completa non devono essere necessari per eseguire la suite ordinaria.
4. I test che richiedono simulatori o dataset reali sono marcati come integration
   test e falliscono con un messaggio esplicito quando manca la fixture richiesta.
5. Tutti i path persistiti nel catalogo sono relativi a
   `SCENARIONET_DATA_ROOT`.
6. Il catalogo usa un'unica tassonomia semantica canonica A0–A5. Gli arms
   generator-level dell'ACL restano invece profili procedurali: non vengono
   rinominati in A0–A5 perché un profilo di generazione non garantisce la
   categoria dello scenario realizzato e PG non genera il contesto VRU A4.
   L'integrazione avviene tramite il filtro del provider, senza confondere i
   due livelli.
7. Nessun valore data-dependent viene inventato: viene ispezionato, calcolato o
   lasciato `unknown` come richiesto dalla specifica.
8. Il test finale della tesi non viene usato per calibrazione, tuning o decisioni
   implementative.

### Allineamento ACL e staged

La distinzione operativa è la seguente:

| Livello | Identificatore | Significato |
|---|---|---|
| Generazione ACL | `broad_random`, `simple_low_risk`, ... | Distribuzione di parametri PG scelta dal MAB; sono 7 profili storici (`arm_space=generator`). |
| ScenarioNet | `A0_simple_low_traffic` ... `A5_critical_mixed` | Classe primaria assegnata dopo l'estrazione delle feature e la classificazione. |
| Staged ScenarioNet | stage nominato `A0` ... `A5` | Selezione progressiva di uno degli arms semantici tramite `provider.arm`. |
| ACL ScenarioNet | MAB su `A0` ... `A5` | Selezione adattiva di record già classificati tramite `arm_space=scenario`. |

È stato aggiunto `conf/curriculum/stages_scenarionet.yaml`, selezionabile con
`curriculum=stages_scenarionet`. Ogni stage conserva lo stesso filtro per train,
validation e test. Il bilanciamento 50/50 Waymo/PG resta attivo dove entrambe le
sorgenti sono disponibili; A4 usa esplicitamente Waymo-only perché PG non
contiene VRU. Il provider rifiuta arm sconosciuti e non applica fallback: se una
combinazione source × split × arm è vuota l'errore è esplicito.

ACL ora supporta entrambi i percorsi. Il percorso storico mantiene i generator
arms PG; il percorso `scenario_acl_scenarionet` usa direttamente il catalogo,
aggiorna il MAB sui sei arms semantici e usa il replay buffer sui record esatti
selezionati, senza generare o rilabelizzare scenari. La mutazione resta
disabilitata. Il percorso semantico è volutamente strict e, per il catalogo
congelato, usa PG-only per A0 e Waymo-only per A4, perché le altre combinazioni
sono vuote.

---

## Layout software previsto

Il layout indicativo della specifica viene adattato al package Python esistente:

```text
src/thesis_rl/
├── scenarios/
│   ├── catalog.py
│   ├── records.py
│   ├── splits.py
│   ├── runtime_database.py
│   ├── validation.py
│   ├── features.py
│   ├── arms.py
│   ├── provider.py
│   ├── manifests.py
│   └── pg/
│       ├── profiles.py
│       ├── generator.py
│       ├── exporter.py
│       └── report.py
├── envs/
│   ├── thesis_scenario_env.py
│   ├── scenario_env_factory.py
│   ├── episode_control.py
│   ├── scene_context_adapter.py
│   └── rulebook_reward_mixin.py
└── cli/scenarios/
    ├── convert_waymo.py
    ├── validate_database.py
    ├── generate_pg_dataset.py
    ├── build_catalog.py
    ├── build_splits.py
    ├── build_runtime_databases.py
    ├── compute_arm_thresholds.py
    └── smoke_test_scenarios.py

conf/
├── env/scenarionet.yaml
└── scenarios/
    ├── dataset_v1.yaml
    ├── pg_profiles_v1.yaml
    └── features_v1.yaml
```

I nomi definitivi delle CLI possono essere accorpati sotto un singolo entrypoint
con sottocomandi, purché ogni operazione resti riproducibile e invocabile in modo
indipendente.

---

## F0 — Bootstrap, dipendenze e API locali

**Stato:** `COMPLETATA`

### Obiettivo

Rendere riproducibile l'ambiente e congelare il contratto concreto offerto dai
commit locali di ScenarioNet e MetaDrive.

### Attività

- [x] Registrare commit project, MetaDrive e ScenarioNet.
- [x] Aggiungere ScenarioNet alle dipendenze effettive del progetto e aggiornare
      `uv.lock`.
- [x] Aggiungere le dipendenze minime comuni per Parquet. Le dipendenze Waymo
      pesanti restano differite a F5, perché dipendono dalla release disponibile.
- [x] Verificare import di ScenarioNet, MetaDrive, `ScenarioDescription` e
      `ScenarioEnv` nel container ufficiale.
- [x] Inventariare chiavi e hook reali di `ScenarioEnv`:
      reset/selezione scenario, `done_function`, data manager, mapping/summary,
      export e validazione.
- [x] Inventariare road block PG nativi e struttura `ScenarioDescription`; la
      valutazione dell'affidabilità dei metadata topologici resta in F3.
- [x] Individuare le primitive locali utilizzabili per distinguere linea continua
      e uscita fisica dalla superficie stradale.
- [x] Individuare una fixture ScenarioDescription Waymo minima già inclusa negli
      asset MetaDrive, senza duplicarla nella data root.
- [x] Eseguire primo reset e rollout headless di 10 step.
- [x] Creare il manifest iniziale non congelato.

### Criteri di uscita

- ambiente ricostruibile da lockfile;
- import essenziali riusciti;
- un singolo scenario caricato con reset e 10 step finiti;
- tabella delle API locali compilata;
- manifest iniziale creato senza valori inventati.

### Evidenze

- commit progetto: `b77427ea5e937a67f80fbb54ca6c832a3c113518`
- MetaDrive: `0.4.3`, commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- ScenarioNet: `0.0.1`, commit `d4acdb5f5a844744fc85cb2dc3880d7d4a6eb170`
- ambiente: Python `3.10.20`, NumPy `1.26.4`, pandas `2.3.3`,
  PyArrow `25.0.0`, GeoPandas `0.14.4`
- build: `docker compose build dev`; `uv pip check` senza incompatibilità
- bootstrap: `python -m thesis_rl.cli.scenarios.bootstrap`
- smoke: `python -m thesis_rl.cli.scenarios.smoke ... --steps 10`
- fixture: `third_party/metadrive/metadrive/assets/waymo`
- risultato smoke: scenario `2a1e44d405a6833f`, lunghezza 91,
  observation shape `(161,)`, action shape `(2,)`, 10 step finiti
- test mirati: 4 passati; Ruff e mypy mirati passati
- regressione: 175 test passati, 2 failure preesistenti/non correlate registrate
  in ISS-008
- artifact locali: `${SCENARIONET_DATA_ROOT}/manifest.yaml` e
  `catalog/local_api_inventory.json`

---

## F1 — Modello dati, manifest e configurazione

**Stato:** `COMPLETATA`

### Attività

- [x] Implementare `ScenarioRecord` immutabile secondo lo schema v1.
- [x] Implementare `ScenarioFeatures` con valori topologici tri-state.
- [x] Implementare modelli e validazione di `manifest.yaml` e
      `split_manifest.yaml`.
- [x] Implementare risoluzione sicura dei path relativi alla data root.
- [x] Implementare creazione idempotente della struttura directory richiesta.
- [x] Aggiungere configurazioni Hydra per dataset, feature e ScenarioEnv.
- [x] Definire versioni esplicite di feature e profili PG; la versione
      ScenarioDescription resta `null` fino alla prima conversione reale.
- [x] Testare serializzazione, round-trip e rifiuto dei path fuori dalla root.

### Criteri di uscita

- modelli pubblici stabili e tipizzati;
- manifest validabile e aggiornabile senza perdere campi;
- nessun path assoluto persistito nel catalogo;
- unit test verdi.

### Evidenze

- modelli: `src/thesis_rl/scenarios/records.py`
- path/layout: `src/thesis_rl/scenarios/paths.py`
- manifest: `src/thesis_rl/scenarios/manifests.py`
- configurazioni: `conf/env/scenarionet.yaml`, `conf/scenarios/*.yaml`
- verifica: 22 test F0/F1 passati; Ruff e mypy mirati passati
- layout locale creato idempotentemente sotto `SCENARIONET_DATA_ROOT`

---

## F2 — Catalogo, split e provider core

**Stato:** `COMPLETATA`

### Attività

- [x] Implementare lettura/scrittura atomica del catalogo Parquet.
- [x] Garantire unicità di `scenario_uid` e `runtime_index` per split.
- [x] Implementare split deterministici raggruppati per la chiave più forte.
- [x] Implementare controlli di overlap Waymo e seed PG.
- [x] Impedire l'uso del test set nelle API di calibrazione/soglie (completato
      in F3 nelle API train-only).
      vengono introdotte tali API).
- [x] Implementare `ScenarioProvider`.
- [x] Implementare `UniformScenarioProvider` con source sampling 50/50,
      modalità strict e nessun fallback.
- [x] Implementare `FixedSequenceScenarioProvider` senza ripetizione implicita.
- [x] Implementare RNG per worker con `SeedSequence([global_seed, worker_id])`.
- [x] Testare bilanciamento statistico, riproducibilità, filtri, errori e
      esaurimento della sequenza.

### Criteri di uscita

- catalogo sintetico round-trip senza perdita di dati;
- split disgiunti verificati;
- provider deterministici e strict;
- test unitari previsti dalla specifica verdi.

### Evidenze

- catalogo: `src/thesis_rl/scenarios/catalog.py`
- split: `src/thesis_rl/scenarios/splits.py`
- provider: `src/thesis_rl/scenarios/provider.py`
- verifica cumulativa F0–F2: 34 test passati; Ruff e mypy mirati verdi
- bilanciamento provider: 10.000 reset sintetici, frazione Waymo entro 0,47–0,53
- politica conteggi: modalità esatta fallisce se gruppi indivisibili rendono
  impossibili i target; modalità automatica seleziona gruppi interi e registra
  nel manifest gli scenari esclusi dal catalogo finale

---

## F3 — Feature extraction, soglie e arms

**Stato:** `COMPLETATA`

### Attività

- [x] Ispezionare schema e metadata prodotti dai converter locali.
- [x] Implementare estrazione comune Waymo/PG da `ScenarioDescription`.
- [x] Implementare route length e route relevance senza riconoscitore geometrico
      avanzato.
- [x] Implementare topologia `simple`, `merge_or_roundabout`, `intersection`,
      `mixed`, `unknown`.
- [x] Implementare conteggi temporali Q90 per agenti e veicoli rilevanti.
- [x] Implementare distanze veicolo/VRU e `vru_interaction` a 8 metri.
- [x] Implementare affidabilità semaforica e tag derivati.
- [x] Calcolare Q40/Q75 soltanto su train candidate bilanciato per sorgente.
- [x] Implementare assegnazione A0–A5 esattamente nell'ordine specificato.
- [x] Produrre statistiche, soglie e distribuzione per arm/sorgente.
- [x] Aggiungere fixture sintetiche per casi unknown, mixed, A0–A5 e segnali
      incompleti.

### Criteri di uscita

- feature finite e deterministiche sulle fixture;
- A1 basato su `relevant_vehicles_q90 > 0`;
- profilo PG non usato come prova della topologia;
- soglie evaluation sempre lette dal file train congelato;
- unit test verdi.

### Evidenze

- feature: `src/thesis_rl/scenarios/features.py`
- arms/tag: `src/thesis_rl/scenarios/arms.py`
- soglie: `src/thesis_rl/scenarios/thresholds.py`
- report: `src/thesis_rl/scenarios/reports.py`
- fixture: sintetiche più scenario Waymo bundled reale
- verifica cumulativa F0–F3: 51 test passati; Ruff e mypy mirati verdi
- guard train-only e source balance testate; evaluation rifiutata
- politica signal relevance registrata in DEC-012

---

## F4 — Generazione PG offline e pilot

**Stato:** `COMPLETATA`

### Attività

- [x] Mappare P0, P1, P2, P3 e P5 sui road block nativi disponibili.
- [x] Fallire esplicitamente se un blocco richiesto non è supportato.
- [x] Usare `IDMPolicy` per la traiettoria nominale ed esportare
      `ScenarioDescription`.
- [x] Implementare seed/range, `--count`, profilo e output root. La gestione
      degli split PG resta in F6 tramite seed disgiunti.
- [x] Salvare un generation manifest per ogni scenario.
- [x] Eseguire validazione, feature extraction e arm assignment per ogni export.
- [x] Implementare fallimento esplicito con report per ogni generazione fallita,
      tentativi e senza riusare seed fra split.
- [x] Produrre matrice profilo-arm e invalid rate.
- [x] Eseguire smoke automatico con 2 scenari temporanei per profilo.
- [x] Preparare il comando per il pilot utente da 20 scenari per profilo.

### Criteri di uscita

- ogni profilo supportato genera almeno uno scenario valido e ricaricabile;
- manifest sufficiente alla rigenerazione;
- seed disgiunti verificati;
- report diagnostico prodotto;
- nessuna generazione on-the-fly nel training principale.

### Evidenze

- profili/API: `src/thesis_rl/scenarios/pg/profiles.py`
- generator/exporter: `src/thesis_rl/scenarios/pg/generator.py`,
  `src/thesis_rl/scenarios/pg/exporter.py`
- validazione/report: `src/thesis_rl/scenarios/pg/validation.py`,
  `src/thesis_rl/scenarios/pg/report.py`
- CLI: `python -m thesis_rl.cli.scenarios.generate_pg_dataset`
- pilot minimo: 1 per profilo, 5/5 validi, invalid rate 0%
- pilot sviluppo: 2 per profilo, 10/10 validi, invalid rate 0%
- matrice pilot sviluppo:
  `P0→A0`, `P1→A0/A1`, `P2→A2`, `P3→A3`, `P5→A5`
- artifact: `${SCENARIONET_DATA_ROOT}/pg/pilot/pg_pilot_report.json`
- test cumulativi F0–F4: 57 passati; Ruff e mypy verdi

---

## F5 — Conversione e preparazione Waymo `training_20s`

**Stato:** `IN_CORSO`

### Responsabilità

Il codice e gli smoke test di conversione fanno parte dell'implementazione.
Download, licenze, credenziali e disponibilità del dataset Waymo sono a carico
dell'utente.

### Attività

- [x] Implementare wrapper CLI riproducibile sul converter ScenarioNet locale.
- [x] Validare che la sorgente sia esclusivamente `training_20s`.
- [x] Estrarre e documentare il più forte identificativo di gruppo disponibile.
- [ ] Conservare release, directory sorgente, converter commit e output nel
      manifest (da completare insieme alla conversione del pool definitivo).
- [x] Supportare conversione limitata per smoke test (un shard reale convertito
      in 61 scenari; ambiente TensorFlow dedicato pronto).
- [x] Eseguire il controllo ufficiale existence/integrity sul campione reale;
      simulation e overlap restano da integrare/eseguire.
- [x] Estrarre feature e record di catalogo senza usare future track online.
- [x] Implementare split interni raggruppati e verifica overlap; da eseguire sul
      database reale.

### Criteri di uscita

- almeno un piccolo campione Waymo convertito, validato e ricaricato;
- provenienza `training_20s` verificata;
- grouping key registrata;
- split sintetici/reali disgiunti;
- CLI completa pronta per l'esecuzione sul dataset intero.

### Evidenze

- wrapper: `src/thesis_rl/scenarios/waymo.py`
- CLI: `python -m thesis_rl.cli.scenarios.convert_waymo`
- source guard: richiede file `training_20s.tfrecord*`
- grouping preference: `source_log_id → segment_id → source_file_id → source_file → scenario id`
- fixture converted: 3 scenari Waymo bundled caricati come `training_20s`
- immagine dedicata: `Dockerfile.waymo` + profilo `compose.waymo.yaml`, build
  riuscito con TensorFlow `2.11.0`, MetaDrive e ScenarioNet importabili
- CLI container: `--help` riuscito; preflight raw inesistente rifiutato prima del
  converter con errore diagnostico
- mount raw host→`/workspace/waymo_raw` verificato tramite `make waymo-convert`
- automazione download: `scripts/prepare_waymo.sh`, `make waymo-auth` e
  `make waymo-pipeline`; configurazione in `.env.example`, nessun secret nel
  repository
- smoke reale: 1 shard Waymo convertito in 61 scenari; output ScenarioNet con
  `dataset_summary.pkl`, `dataset_mapping.pkl` e sottodirectory `database_0`
- compatibilità converter: fissato `protobuf==3.20.3` nel container dedicato;
  import dei binding `scenario_pb2` verificato
- grouping reale: `source_file` normalizzato al basename del TFRecord, senza
  path assoluti del container
- check ufficiale ScenarioNet existence/integrity: 61/61 scenari caricati
- validazione applicativa: 61 `valid`, 0 `warning`, 0 `invalid`
- runtime mapping temporaneo verificato per 61/61 file senza copie fisiche
- verifica F5/F6 mirata: 80 test ScenarioNet/PG passati; Ruff e mypy verdi
- conversione completa del pool: differita a F10; smoke reale limitato a 1 shard
  autorizzato, sufficiente per verificare converter e runtime senza scaricare i
  1000 shard nel repository

---

## F6 — Validazione e runtime database

**Stato:** `IN_CORSO`

### Attività

- [x] Integrare i wrapper per existence/integrity, simulation e overlap check
      ufficiali; [x] eseguire existence/integrity su 61 scenari; simulation e
      overlap sul pool finale restano pendenti.
- [x] Implementare validazione applicativa della tesi.
- [x] Distinguere `valid`, `warning` e `invalid` con warning strutturati.
- [x] Escludere automaticamente gli invalid dai runtime database (builder
      runtime filtra `valid`/`warning`).
- [x] Costruire viste `runtime/train`, `runtime/validation`, `runtime/test`
      evitando copie fisiche quando supportato.
- [x] Assegnare `runtime_index` nel builder finale e verificarlo contro
      summary/mapping effettivi.
- [x] Verificare mapping runtime e presenza dei file risolti.
- [x] Salvare `validation_summary.json` e hash del catalogo quando il CLI riceve
      `--catalog`.

### Criteri di uscita

- viste runtime caricabili per entrambe le sorgenti;
- mismatch catalogo/runtime bloccante e diagnosticabile;
- validazione di reset e 10 step su campioni Waymo e PG;
- report di validazione persistito.

### Evidenze

- validazione: `src/thesis_rl/scenarios/validation.py`
- runtime mapping: `src/thesis_rl/scenarios/runtime_database.py`
- CLI mapping: `python -m thesis_rl.cli.scenarios.validate_database`
- controllo locale: `compileall` e `git diff --check` passati
- test container F0–F6 mirati: 80 passati; Ruff e mypy verdi
- regressione completa: 258 passati, 2 failure baseline già registrati in ISS-008
- `validation_summary.json` e hash catalogo: collegati al CLI
  `validate_database --catalog ...`; senza catalogo resta disponibile il check
  del mapping runtime; smoke reale: 61 record, 61 valid, 0 warning/invalid,
  runtime files 61
- CLI check ufficiale: `python -m thesis_rl.cli.scenarios.check_database
  existence|simulation|overlap ...` disponibile; simulation/overlap restano da
  eseguire sul catalogo definitivo
- check ufficiale existence/integrity sulla vista mista: 82/82 scenari caricati
  correttamente
- check ufficiale simulation sulla vista mista: 82/82 scenari simulati senza
  errori
- check ufficiale overlap tra viste Waymo e PG smoke: nessuna sovrapposizione
- dataset Waymo completo: esplicitamente deferito a F10; il campione di 1 shard
  resta una fixture smoke e non viene trattato come dataset finale

---

## F7 — `ThesisScenarioEnv`, episode control e scene context

**Stato:** `IN_CORSO`

### Attività

- [x] Implementare `SceneContextAdapter` sulle API locali.
- [x] Riutilizzare reward manager/rulebook wrapper esistenti tramite il wiring
      `maybe_wrap_env_with_reward_manager`.
- [x] Collegare `ThesisScenarioEnv` alla factory per split e worker; verifica
      end-to-end dei processi vectorized completata in F8.
- [x] Applicare configurazione ScenarioEnv richiesta, inclusi 10 Hz,
      `reactive_traffic=true` e cache disabilitate.
- [x] Integrare il provider nell'hook usato da ogni reset, incluso auto-reset.
- [x] Verificare UID atteso contro scenario effettivamente caricato.
- [x] Preservare collisioni e destinazione native.
- [x] Implementare separazione tra linea continua e uscita fisica usando soltanto
      primitive native locali.
- [x] Ricalcolare la termination dopo l'override di `OUT_OF_ROAD`.
- [x] Implementare truncation a `scenario.length + extra_steps`, inclusi 0 e 50.
- [x] Esporre identificativi, source, arm, dimensioni ego e ragione terminale
      nell'`info`, non nell'osservazione della policy.
- [x] Verificare shape uniforme tra Waymo e PG nel mixed smoke vectorized.

### Criteri di uscita

- matrice termination/truncation della specifica completamente testata;
- reward e observation finite sulle due sorgenti;
- nessun metadata di catalogo esposto alla policy;
- reset provider-driven verificato.

### Evidenze

- `src/thesis_rl/envs/scene_context.py` e
  `src/thesis_rl/envs/thesis_scenario_env.py`
- test unitari della matrice extra-step e dei predicati line/boundary: passati
- smoke headless su shard Waymo reale: reset `(161,)` e step riusciti
- smoke provider-driven su runtime database ordinato: UID/scenario_id verificati
- shape Waymo/PG e wiring completo del catalogo nei worker: verificati nello
  smoke misto vectorized

---

## F8 — Vectorization, logging e wiring training/evaluation

**Stato:** `COMPLETATA`

### Attività

- [x] Collegare la nuova factory al builder Hydra senza rompere `MetaDriveEnv`.
- [x] Creare configurazione `env=scenarionet` e preset smoke.
- [x] Verificare un environment per processo con start method `spawn`.
- [x] Integrare provider locale e seed per worker; la partizione runtime viene
      derivata dal catalogo quando `num_scenarios=-1`.
- [x] Usare sequenza fissa o partizione statica per evaluation.
- [x] Verificare auto-reset, chiusura e assenza di subprocess orfani nello smoke
      a 2 worker.
- [x] Aggiungere logging episodico completo richiesto dalla specifica tramite
      info di scenario e ragione terminale.
- [x] Aggiungere conteggi run-level per reset/source, step/source ed episodi/arm;
      raccolta, aggregazione train multi-chunk e persistenza metadata sono attive.
- [x] Conservare riferimento/copia di manifest, split e soglie negli artifact run
      quando i file sono disponibili; snapshot e SHA-256 vengono scritti nei
      metadata ScenarioNet.
- [x] Verificare bootstrap corretto dei timeout nei backend on/off-policy.

### Criteri di uscita

- training ed evaluation selezionabili da config;
- vector env stabile su reset ripetuti;
- fixed evaluation sequence riproducibile;
- schema di logging completo;
- nessuna regressione della pipeline MetaDrive esistente.

### Evidenze

- configurazione `conf/env/scenarionet.yaml` con catalog path, seed e provider;
- `_worker_env_overrides` supporta la partizione ScenarioNet e il catalogo;
- smoke Hydra factory + catalogo Parquet Waymo: reset riuscito, observation
  shape `(161,)`, UID verificato;
- smoke vector `spawn` a 2 worker con auto-reset: passato;
- `get_runtime_stats` aggrega worker e la valutazione persiste i contatori nei
  metadata del run;
- snapshot manifest/split/soglie, aggregazione train multi-chunk e stats eval:
  attivi; il controllo catalogo/runtime ora fallisce preventivamente in caso di
  vista incompatibile; resta da validare il catalogo finale congelato.

---

## F9 — Smoke end-to-end, pilot e freeze

**Stato:** `IN_CORSO`

### Attività

- [x] Eseguire pipeline completa con conteggi piccoli Waymo + PG.
- [x] Eseguire breve training smoke provider-driven (native reward monitor-only,
      20 step single-worker e 20 step vectorized) e con reward custom.
- [x] Eseguire random-policy smoke headless sulla fixture Waymo inclusa.
- [x] Verificare la fixed sequence nel vector smoke; evaluation con checkpoint
      resta da completare.
- [x] Eseguire audit causale statico delle opzioni observation: future trajectory
      e future signal phase sono rifiutate dalla factory; test passati.
- [x] Eseguire suite unit, integration e regressione; i 2 failure baseline restano
      registrati in ISS-008.
- [x] Eseguire simulation e overlap ufficiali sul database smoke misto.
- [x] Profilare RAM, tempo di reset e cleanup sullo smoke headless.
- [ ] Eseguire pilot visivo Waymo per scegliere 50 oppure 0 extra step.
- [x] Eseguire pilot PG di default e, al massimo, una revisione manuale dei
      profili secondo i criteri della specifica.
- [x] Aggiungere output Rich alla pipeline: stadi, spinner, progress bar per
      generazione/validazione e riepiloghi leggibili senza alterare i report JSON.
- [ ] Congelare manifest software, profili PG e decisioni data-dependent.
- [x] Documentare i comandi destinati all'utente per la generazione completa.

### Criteri di uscita

- smoke train completo senza errori sistematici;
- tutte le evidenze della definition of done disponibili salvo la numerosità del
  dataset completo;
- nessun problema bloccante aperto;
- versioni e decisioni congelate.

### Evidenze

- mixed smoke: 61 Waymo + 21 PG, runtime mapping 82/82, provider Uniform 50/50,
  reset/step e shape uniforme passati;
- simulation ufficiale: 82/82 scenari caricati e simulati senza errori;
- overlap ufficiale Waymo↔PG: nessuna sovrapposizione;
- regressione: 259 test passati, 2 failure baseline non correlati;
- pilot PG default: 100/100 generati, invalid rate 0%, matrice profilo-arm
  persistita in `data/scenarionet/pg/pilot/pg_pilot_report.json`;
- training-only smoke 20 step con reward custom passato; audit causale statico e
  smoke vectorized 2 worker passati;
- random policy passato; profiling smoke passato; pilot visivo Waymo e freeze
  finale restano pendenti.
- CLI finali catalogo/split/soglie/runtime e pipeline unica verificate sullo
  smoke (182 record, tre runtime view); la pipeline completa richiede i conteggi
  target in `.env` e seleziona solo i gruppi necessari, registrando gli esclusi.

---

## F10 — Costruzione dataset completo e accettazione

**Stato:** `NON_INIZIATA`

Questa fase viene eseguita dall'utente tramite le CLI validate in F9; Codex può
assistere nel monitoraggio e nella diagnosi.

### Attività

- [ ] Convertire il pool Waymo necessario.
- [ ] Generare i pool PG con seed disgiunti.
- [ ] Costruire split con conteggi baseline o riduzione esplicitamente registrata.
- [ ] Calcolare soglie soltanto sul train candidate bilanciato.
- [ ] Costruire catalogo e viste runtime finali.
- [ ] Validare tutto il dataset e salvare distribuzioni/report.
- [ ] Eseguire smoke finale sul dataset congelato.
- [ ] Verificare uno per uno i 28 punti della definition of done.

### Criteri di uscita

- dataset, manifest, split, catalogo e soglie congelati;
- definition of done completamente soddisfatta;
- comandi e artifact associati alla prima run principale.

---

## Correzione della copertura Waymo negli arm

**Stato:** `V2 IMPLEMENTATA; AUDIT VISIVO PENDENTE` (2026-07-13)

### Revisione v2: da tassonomia semantica a scala curricolare

L'audit v1 ha mostrato che una tassonomia esclusiva per tipo di scenario non
realizza l'obiettivo della tesi: A3 assorbiva 1427 Waymo perché indicava ogni
intersezione, A5 ne assorbiva 742 perché precedeva A4 quando VRU e junction
coesistevano, mentre A4 rimaneva con 12 VRU fuori dalle junction. La regola era
internamente coerente, ma gli arm non erano ordinati per difficoltà.

La v2 separa pertanto:

- **tag semantici multi-label**: traffico, merge/roundabout, intersezione,
  segnale, VRU e conflitti;
- **primary arm esclusivo**: livello curricolare A0–A5, progressivo ma non
  definito dalla mera presenza di una categoria.

La scala adottata è:

| Arm | Nome v2 | Criterio operativo |
|---|---|---|
| A0 | `simple_low_traffic` | topologia semplice, nessun VRU route-relevant, al massimo 8 veicoli rilevanti Q90 |
| A1 | `traffic` | strada semplice con traffico superiore ad A0, oppure junction semplice con al massimo 8 agenti Q90 e 1 conflitto |
| A2 | `junction` | merge/intersezione senza VRU route-relevant e sotto le soglie di complessità A3 |
| A3 | `complex_junction` | junction senza VRU route-relevant, con almeno 25 agenti Q90 o 4 conflitti veicolari |
| A4 | `vru` | VRU route-relevant, ma senza la combinazione critica richiesta da A5 |
| A5 | `critical_mixed` | junction con conflitto VRU; topologia mista con almeno 3 conflitti veicolari; oppure junction con almeno 30 agenti Q90 e 6 conflitti |

La densità usata dall'arm è un valore fisico versionato, non un quantile calcolato anche sugli split di
evaluation. Le soglie train-only Q40/Q75 restano valide per i tag analitici
`low_traffic`/`dense_traffic`, ma non determinano il primary arm e non creano
una dipendenza circolare tra classificazione e split.

I conflitti sono calcolati offline dalle traiettorie e non sono osservazioni
fornite alla policy. Parametri v2 iniziali: orizzonte CPA 5 s, distanza CPA
veicolo 4 m, distanza CPA VRU 3 m, raggio di rilevanza 50 m e tolleranza
verticale 3 m. Un conflitto richiede moto relativo in avvicinamento, CPA entro
l'orizzonte e distanza sotto la soglia del tipo di agente.

`relevant_agents_q90` (abbreviato nei report come **agenti Q90**) non è il
numero totale di track nel file. A ogni timestep conta gli agenti dinamici
non-ego (veicoli, pedoni e ciclisti) validi entro 50 m dall'ego e sullo stesso
livello stradale (tolleranza verticale 3 m), quindi prende il percentile 90 nel
tempo. Per esempio Q90=11 significa che nel 90% dei timestep il conteggio è al
massimo 11; il valore attenua picchi brevissimi senza nascondere una densità
persistente. `relevant_vehicles_q90` applica la stessa misura ai soli veicoli.

Il filtro ScenarioNet `object_number` conta tutti i track, ego incluso: zero
oggetti è impossibile e `<2` seleziona sostanzialmente scenari ego-only. Per A0
si usa invece il numero Q90 di veicoli route-relevant. `object_number` non viene
usato né per classificare né per acquisire i candidati, perché non esprime la
difficoltà route-local richiesta dagli arm.

La selezione v2 accetta soltanto `signal_reliability=complete` oppure
`not_applicable`; `partial` e `missing` restano catalogabili per audit ma non
entrano negli split/runtime. Gli stati mancanti non vengono imputati. Questa
policy riduce il pool Waymo disponibile e il relativo deficit deve essere
colmato con shard incrementali mai elaborati, non riusando scene scartate.

### Problema osservato

La prima pipeline completa ha prodotto una distribuzione Waymo limitata ad A1
e A4, mentre PG copre A0, A1, A2, A3 e A5. Questa distribuzione non dimostra
che Waymo sia privo di merge, intersezioni o contesti misti: il loader Waymo
invocava l'estrattore senza metadata topologici, ottenendo
`has_intersection=None`, `has_merge_or_roundabout=None` e `topology_tag=unknown`.
Con la precedenza A0–A5 corrente, uno scenario Waymo poteva quindi raggiungere
soltanto A1 tramite veicoli rilevanti oppure A4 tramite VRU.

La pipeline selezionava inoltre i gruppi train/validation/test prima
dell'assegnazione finale degli arm. La selezione conosceva i soli target totali
per sorgente e non poteva controllare la matrice `source × arm × split`.

### Evidenze disponibili

- gli shard Waymo non portano etichette compatibili con gli arm della tesi e
  non possono essere scaricati "arm per arm";
- gli scenari ScenarioNet convertiti espongono grafo delle lane
  (`entry_lanes`, `exit_lanes`), geometrie di crosswalk/stop e stati dinamici
  dei semafori con lane/stop point;
- un audit esplorativo su 100 scenari convertiti ha trovato indizi topologici
  vicino alla traiettoria ego nel 91% dei casi per diramazioni, 55% per
  crosswalk e 65% per semafori. Sono indizi, non ground truth, ma provano che
  l'informazione prima ignorata è disponibile;
- i filtri ScenarioNet standard sono utili per prefiltrare qualità, luci e
  oggetti, ma non forniscono una classificazione route-aware di merge,
  intersezioni e contesti misti.

### Decisioni implementative v1 (storiche, superate dalla scala v2)

1. La v1 conservava A0–A5 come categorie per merge/intersezione/VRU; questa
   decisione è superata da DEC-023 e resta qui per spiegare gli audit storici.
2. Inferire la topologia Waymo dalla mappa rispetto alla traiettoria ego, senza
   usare stati futuri come input online alla policy. Il calcolo è offline e
   serve esclusivamente a catalogazione, split e curriculum.
3. Considerare intersezione la presenza route-local di controlli/attraversamenti
   (semaforo, stop sign o crosswalk). Considerare merge la presenza route-local
   di lane con almeno due ingressi in assenza di evidenza d'intersezione; una
   diramazione in uscita da sola non prova un merge. Un ciclo nel grafo locale
   non è prova sufficiente di roundabout e non viene classificato come tale in
   `v1`.
4. Usare valori tri-state conservativi: in assenza di lane route-local
   sufficienti la topologia resta `unknown`, non `simple`.
5. Assegnare il `primary_arm` già durante la costruzione del catalogo. Le soglie
   Q40/Q75 train-only modificano soltanto i tag low/dense traffic e non il
   `primary_arm`, quindi non impediscono la selezione arm-aware.
6. Selezionare sempre gruppi indivisibili per evitare leakage. Quote e cap sono
   applicati al catalogo/runtime, senza duplicare file né cancellare il pool
   convertito eccedente.
7. Non fissare quote Waymo arbitrarie prima della riclassificazione del pool già
   convertito. Prima si misura la nuova matrice; solo gli arm insufficienti
   giustificano il download incrementale di shard mai elaborati.

### Parametri iniziali versionati

| Parametro | Valore iniziale | Motivazione |
|---|---:|---|
| distanza lane–route | 6 m | associa la centerline alla traiettoria senza includere l'intera scena |
| distanza controllo–route | 15 m | include stop point/crosswalk immediatamente pertinenti al percorso |
| ingressi minimi per merge | 2 | richiede evidenza di convergenza, non una semplice uscita |
| copertura sorgenti | A2/A3/A5 entrambe; A4 Waymo; A0 Waymo best effort | rispetta assenza intenzionale di VRU PG e rarità naturale di A0 reale |

Questi valori sono una policy di catalogazione `v1`, devono essere sottoposti a
verifica visiva stratificata e possono essere congelati o corretti sulla base
dell'audit, registrando una nuova decisione/versione.

I minimi v2 `source × split × arm` sono versionati in
`conf/scenarios/pipeline_v1.yaml`. Per Waymo sono A0=10/3/5, A1=10/3/5,
A2=50/15/25, A3=160/40/80, A4=80/20/40 e A5=100/25/50
(train/validation/test). A0 è intenzionalmente non soddisfatto dal pool
corrente e rende misurabile il fabbisogno di nuovi shard. Dopo la calibrazione
delle junction leggere, per PG sono A0=300/75/150, A1=250/60/125,
A2=100/25/50, A3=20/5/10 e A5=30/8/15. PG A4 resta escluso per la scelta
esplicita di non generare VRU. I minimi sono inferiori alle disponibilità
osservate e lasciano margine all'assegnazione di gruppi indivisibili.

### Flusso operativo adottato

`converti → classifica → inventaria source×arm → seleziona per quote/cap →`
`scarica un nuovo batch soltanto se restano deficit`.

Il primo passo dopo l'implementazione è ricostruire il catalogo sui 35 shard già
convertiti, senza riscaricare o riconvertire. La pipeline deve produrre nel
manifest i target richiesti, i conteggi effettivi e gli eventuali deficit per
arm. Se il pool è insufficiente, i batch successivi devono usare shard non già
presenti e fermarsi quando i minimi sono soddisfatti; l'overshoot dell'ultimo
gruppo è ammesso e tracciato.

### Stato e risultato

- diagnosi: `CONFERMATA`;
- documentazione e decisioni: `REGISTRATE`;
- estrattore topologico Waymo: `IMPLEMENTATO` con indice spaziale route-local;
- classificazione pre-split: `IMPLEMENTATA`;
- selezione group-aware con minimi e report deficit: `IMPLEMENTATA`;
- riclassificazione del pool: `ESEGUITA` su 2409 Waymo + 1750 PG;
- feature CPA/conflitto: `IMPLEMENTATE` e persistite nel catalogo;
- audit visivo stratificato: `PENDENTE` prima del freeze scientifico;
- ricostruzione del catalogo/runtime ufficiale: `PENDENTE`, eseguibile con
  `make scenarionet-recatalog` senza download, conversione o rigenerazione PG.

Il primo audit tecnico (prima della disambiguazione intersezione/merge) ha
prodotto Waymo A1=72, A2=156, A3=160, A4=12, A5=2009 su 2409 scenari. Il dato
A5 è stato rifiutato come non plausibile: le convergenze interne agli incroci
controllati erano contate anche come merge. La correzione e il secondo audit
sono parte dello stato `IN_IMPLEMENTAZIONE`, non una modifica opportunistica
delle quote.

Dopo la correzione, il secondo audit ha prodotto Waymo A1=72, A2=156,
A3=1427, A4=12 e A5=742. PG è rimasto invariato: A0=395, A1=305, A2=497,
A3=375, A5=178. L'assenza Waymo A0 è accettata come proprietà del pool
osservato, non compensata ricatalogando interazioni reali come scenari semplici.

L'audit v2 iniziale con soglie 8 agenti/2 conflitti ha prodotto Waymo A5=1742
ed è stato rifiutato perché descriveva la mediana del pool. Dopo la
ricalibrazione, la distribuzione Waymo è A0=0, A1=71, A2=394, A3=998, A4=395,
A5=551. La distribuzione PG finale, inclusa la clausola topologia mista, è
A0=665, A1=35, A2=849, A3=119, A4=0, A5=82. A4 è ora sostanziale e A5 è una
coda composta; l'assenza A0 Waymo resta un deficit di acquisizione esplicito.

La successiva calibrazione richiesta dall'audit porta A0 da 3 a 8 veicoli Q90,
ammette in A1 junction leggere con al massimo 8 agenti Q90 e 1 conflitto, e
porta A3 da 20/3 a 25 agenti Q90 oppure 4 conflitti. Sul catalogo selezionato
precedente la proiezione esatta, prima del filtro sui segnali, è Waymo A0=2,
A1=116, A2=398, A3=572, A4=294, A5=406 e PG A0=700, A1=653, A2=263,
A3=52, A5=82. Il cambiamento risolve in particolare l'accumulo PG in A2.
Applicando al vecchio sottoinsieme Waymo la policy sui segnali restano 1116
scenari eleggibili: A0=2, A1=90, A2=246, A3=324, A4=195 e A5=259. I numeri
definitivi richiedevano quindi l'espansione del pool convertito: il vecchio
sottoinsieme non poteva raggiungere il target Waymo 1750 dopo l'esclusione di
`partial`/`missing`.

L'espansione automatica del 2026-07-13 ha riconosciuto i 35 shard iniziali e
convertito in un singolo batch gli 8 shard inediti `00035`--`00042`. Il pool è
passato da 2409 a 2981 scenari convertiti e da 1508 a 1865 eleggibili, superando
il target 1750 senza un secondo download. La distribuzione eleggibile prima
dello split è A0=2, A1=153, A2=389, A3=584, A4=314, A5=423; affidabilità:
`complete`=657, `not_applicable`=1208, `partial`=1115, `missing`=1. I 1116
scenari `partial`/`missing` restano nell'audit pool ma non possono entrare negli
split. I TFRecord del batch sono stati rimossi dopo la conversione. Il target
numerico è soddisfatto; A0 Waymo resta un deficit semantico esplicito (2 contro
18 minimi aggregati), non correggibile alterando retroattivamente le etichette.

La ricatalogazione finale ha selezionato 1829 Waymo e 1750 PG. Il target Waymo
1750 viene superato perché gli shard sono gruppi indivisibili anti-leakage:
train=1031, validation=282, test=516; PG resta esattamente 1000/250/500. La
distribuzione selezionata Waymo è A0=2, A1=150, A2=383, A3=573, A4=305,
A5=416. Tutti i minimi non-A0 sono soddisfatti; il deficit totale 16 coincide
con A0 (train 10, validation 1, test 5). Le soglie train-only finali sono
`tau_low=4`, `tau_dense=22`, bilanciate su 1000 scenari per sorgente e marcate
`feature_version=v2`. Existence, simulation e overlap ufficiali hanno concluso
con exit code 0 su tutti gli split; pipeline completata in 1794 s.

Non vengono applicati cap per rendere uniforme l'istogramma: con target Waymo
1750/2409 e soli 860 scenari circa in A0/A1/A2/A4, almeno metà della selezione
deve necessariamente provenire da A3/A5. Correggere questo rapporto richiede
nuovi shard a bassa complessità oppure un target Waymo minore, non la
ricatalogazione arbitraria di scene complesse.

---

## Matrice dei test obbligatori

La colonna evidenza deve contenere il nome reale del test una volta implementato.

| Area | Copertura richiesta | Stato | Evidenza |
|---|---|---|---|
| Record/catalogo | serializzazione, path, UID, runtime index | `NON_INIZIATA` | — |
| Split | grouping, overlap Waymo, seed PG, test isolation | `NON_INIZIATA` | — |
| Feature | unknown, mixed, Q90, VRU, segnali | `NON_INIZIATA` | — |
| Arms | A0–A5 e A1 vehicle relevance | `NON_INIZIATA` | — |
| Soglie | Q40/Q75 train-only e source balance | `NON_INIZIATA` | — |
| Provider | strict, 50/50, seed, exhaustion, auto-reset | `NON_INIZIATA` | — |
| Termination | collision, destination, line, road, route | `NON_INIZIATA` | — |
| Truncation | extra step 0/50 e bootstrap | `NON_INIZIATA` | — |
| Reward/context | adapter e dimensioni ego | `NON_INIZIATA` | — |
| Integration | Waymo/PG, space, reset, rollout | `NON_INIZIATA` | — |
| Vectorization | reset, fixed eval, close/cleanup | `NON_INIZIATA` | — |
| Causalità | nessuna informazione futura alla policy | `NON_INIZIATA` | — |
| Regressione | pipeline MetaDrive esistente | `NON_INIZIATA` | — |

---

## Registro problemi

### Regole

- Severità: `BLOCCANTE`, `ALTA`, `MEDIA`, `BASSA`.
- Stato: `APERTO`, `IN_ANALISI`, `RISOLTO`, `ACCETTATO`.
- Un problema risolto conserva causa, soluzione, test di regressione e commit.

| ID | Severità | Stato | Problema | Impatto / prossima azione |
|---|---|---|---|---|
| ISS-001 | ALTA | RISOLTO | `scenarionet` era una source `uv`, ma non una dipendenza effettiva e non compariva nel lockfile. | Aggiunti ScenarioNet, pandas, PyArrow e PyYAML; build e import verificati. |
| ISS-002 | MEDIA | RISOLTO | Nessun container attivo o virtualenv host disponibile durante l'analisi iniziale. | Immagine ricostruita e comandi eseguiti con container effimeri. |
| ISS-003 | MEDIA | ACCETTATO | Data root inizialmente senza manifest o dataset ScenarioNet. | Manifest creato; assenza del dataset reale attesa fino a F5/F10. |
| ISS-004 | ALTA | RISOLTO | `ScenarioEnv` locale aggrega route deviation, linea/road state e usa una guardia truthy per `allowed_more_steps`. | `ThesisScenarioEnv` separa linea continua/uscita fisica, ricalcola la termination e applica il limite esplicito anche con `extra_steps_after_scenario=0`; test e smoke headless passati. |
| ISS-005 | MEDIA | RISOLTO | Record e generator arms dell'ACL esistente hanno semantica diversa dai nuovi record e arms A0–A5. | Aggiunti due arm space espliciti: `generator` mantiene i 7 profili PG, `scenario` usa il MAB sui sei arms A0–A5 del catalogo; staged usa lo stesso filtro semantico. |
| ISS-006 | MEDIA | RISOLTO | Dipendenze Waymo sono intenzionalmente escluse dal `setup.py` ScenarioNet locale. | Definito e verificato il container dedicato `Dockerfile.waymo`; il runtime RL non viene appesantito. |
| ISS-007 | BASSA | RISOLTO | L'immagine runtime non include il client `git`, necessario solo per rilevare commit e worktree state. | Implementato fallback read-only per commit; dirty state resta `null` senza client. |
| ISS-008 | BASSA | APERTO | La suite completa ha 2 failure non correlate (258 test passati): component name mancante nel tool forced-rule e preset test che attende `td3` mentre la config usa `td3_sb3`. | Non correggere nella pipeline ScenarioNet; aprire issue separata sul baseline. |
| ISS-017 | MEDIA | RISOLTO | Le funzioni core per catalogo/split/runtime/soglie non erano ancora esposte come pipeline CLI unica. | Aggiunte quattro CLI, `make scenarionet-pipeline`, configurazione `.env`, controlli finali e test smoke. |
| ISS-018 | BASSA | RISOLTO | `.env` mescolava override macchina e default scientifici della pipeline. | Aggiunto `conf/scenarios/pipeline_v1.yaml` con resolver CLI; `.env` ora contiene solo override opzionali, path e configurazione host. |
| ISS-019 | ALTA | RISOLTO | Waymo raggiungeva soltanto A1/A4 perché la topologia convertita non veniva analizzata e gli split precedevano la classificazione. | Implementati topologia route-aware, arm pre-split e selezione `source × arm` per gruppi; audit quantitativo riuscito, audit visivo ancora richiesto prima del freeze. |
| ISS-020 | MEDIA | APERTO | Le label route-aware Waymo sono state validate quantitativamente ma non ancora con ispezione visiva stratificata. | Campionare A2/A3/A4/A5 e falsi negativi A1 prima di congelare DEC-021; non cambiare soglie per ottenere un istogramma desiderato. |
| ISS-021 | ALTA | RISOLTO | Gli arm v1 erano categorie semantiche non ordinali: la precedenza A5 svuotava A4 e intersezioni/VRU dominavano la distribuzione. | Implementati livelli curricolari v2, tag multi-label e conflitti CPA offline; distribuzione ricalibrata, audit visivo resta ISS-020. |
| ISS-009 | MEDIA | RISOLTO | I token `numpy.str_` dei profili complessi non erano serializzabili da PyYAML nei generation manifest. | Normalizzati a `str` in `GenerationSpec`; pilot complesso 5/5 e 10/10 riusciti. |
| ISS-010 | MEDIA | ACCETTATO | Il converter Waymo locale richiede TensorFlow e file raw `training_20s`; il dataset raw resta esterno al repository. | TensorFlow è isolato nel container dedicato; la conversione reale resta subordinata a dataset/licenza dell'utente. |
| ISS-011 | BASSA | RISOLTO | L'ultima esecuzione Docker dei test F6 era stata rifiutata dal limite di approvazioni dell'ambiente. | Test mirati rieseguiti: 80 passati, Ruff e mypy verdi. |
| ISS-012 | MEDIA | ACCETTATO | Il download Waymo richiede un account Google autorizzato e il Google Cloud CLI; queste credenziali non possono essere generate dal repository. | `make waymo-auth` guida il login una tantum; `make waymo-pipeline` automatizza il download senza salvare token o chiavi in `.env`. |
| ISS-013 | MEDIA | RISOLTO | I binding protobuf Waymo generati da ScenarioNet richiedono l'API 3.20, mentre TensorFlow 2.11 risolveva 3.19.6. | Il container Waymo installa `protobuf==3.20.3` dopo il resolver; import `scenario_pb2` e conversione reale verificati. |
| ISS-014 | MEDIA | RISOLTO | I PG esportati hanno un id `PGMap-<seed>`, mentre il summary runtime locale espone `metadata.scenario_id` come `<seed>`. | Il controllo UID/runtime accetta solo questa normalizzazione PG esplicita; mismatch diversi restano errori bloccanti. Mixed smoke reset/step passato. |
| ISS-015 | MEDIA | RISOLTO | Il loop training/seeding assumeva `env.config.start_seed` e i reset forzati bypassavano il provider ScenarioNet. | Selezione split provider-aware, reset senza seed per Uniform/FixedSequence e risoluzione preventiva di `num_scenarios` nel motore; training single/vectorized smoke passati. |
| ISS-016 | MEDIA | RISOLTO | Un catalogo misto poteva essere avviato con una runtime view incompleta, lasciando a MetaDrive un'asserzione poco diagnostica sul numero di scenari. | La factory confronta catalogo e `dataset_summary`/mapping prima della costruzione dell'ambiente e indica la `data_directory` corretta; mismatch e training smoke validi verificati. |

---

## Registro decisioni data-dependent

| ID | Stato | Decisione da congelare | Valore / motivazione | Evidenza |
|---|---|---|---|---|
| DEC-001 | CONFERMATA | Package applicativo | `thesis_rl.scenarios`, coerente con il layout Python del repository | struttura esistente `src/thesis_rl` |
| DEC-002 | CONFERMATA | MetaDrive commit/versione | `0.4.3` / `85e5dadc6c7436d324348f6e3d8f8e680c06b4db` | manifest e API inventory F0 |
| DEC-003 | CONFERMATA | ScenarioNet commit/versione | `0.0.1` / `d4acdb5f5a844744fc85cb2dc3880d7d4a6eb170` | manifest e API inventory F0 |
| DEC-004 | CONFERMATA | Waymo release | Motion Dataset v1.2.0, bucket `waymo_open_dataset_motion_v_1_2_0`, variante `training_20s` | smoke reale F5; manifest completo ancora da aggiornare |
| DEC-005 | CONFERMATA | Grouping key Waymo | `source_file` normalizzato al basename dello shard; nessun `source_log_id`/`segment_id` esposto nel campione | 61 scenari dello shard; test grouping |
| DEC-006 | DA_VERIFICARE | Metadata topologici affidabili | da schema converter/PG esportato | — |
| DEC-007 | DA_VERIFICARE | Blocchi PG nativi | da API MetaDrive locale | — |
| DEC-008 | DA_CALCOLARE | `tau_low`, `tau_dense` | Q40/Q75 sul train candidate bilanciato | — |
| DEC-009 | DA_VERIFICARE | Extra step Waymo | 50 oppure 0 dopo pilot visivo | — |
| DEC-010 | DA_VERIFICARE | Dimensioni ego in observation | includere sempre se variabili, omettere sempre se standardizzate | — |
| DEC-011 | CONFERMATA | Split con gruppi incompatibili con conteggi esatti | fallire esplicitamente; una riduzione richiede aggiornamento documentato del manifest | test split F2 e requisito no-leakage |
| DEC-012 | CONFERMATA | Semafori presenti ma route relevance non dimostrabile | `signal_reliability=partial` e route-light tri-state `None`; non inferire applicabilità globale | schema Waymo locale, test F3, principio no-heuristic |
| DEC-013 | CONFERMATA | PG native topology mapping | `S/C`; `y/r/R/O`; `X/T` verificati headless sul commit MetaDrive locale | smoke API F4 |
| DEC-014 | CONFERMATA | PG pilot development size | 2 scenari per profilo; pilot utente configurato a 20 per profilo | report F4, invalid rate 0% |
| DEC-015 | CONFERMATA | Ambiente conversione Waymo | Container separato `Dockerfile.waymo` + profilo `compose.waymo.yaml` per TensorFlow 2.11 e converter; runtime RL leggero | build riuscito; import e CLI verificati; ISS-006 |
| DEC-016 | CONFIRMED | Waymo authentication/download | Google Cloud CLI with one-time user OAuth; URI, pattern, and paths in `.env`; no API key (it does not replace IAM) and no JSON contents in the repository | `make waymo-auth`, `scripts/prepare_waymo.sh`, setup documentation |
| DEC-017 | CONFIRMED | Final split counts | Default targets 1000/250/500 per source, whole-group assignment, and effective counts persisted in the manifest; edit `conf/scenarios/pipeline_v1.yaml` directly | Avoids duplication between YAML and `.env`; avoids impossible Waymo group counts without silent reductions |
| DEC-018 | CONFIRMED | Optional non-interactive credentials | Supports `GOOGLE_APPLICATION_CREDENTIALS` as a path to an external, authorized service-account JSON; gcloud OAuth remains the default; API keys are unsupported because they do not grant IAM on the bucket | Enables automation without placing the secret or JSON contents in the repository or `.env` |
| DEC-019 | CONFERMATA | Semantica arm durante il riequilibrio | Conservare A0–A5; A5 resta multi-semantico e la difficoltà rimane espressa da tag separati | Evita di ottenere equilibrio numerico falsando le classi scientifiche |
| DEC-020 | CONFERMATA | Acquisizione Waymo per arm | Classificare il pool convertito, applicare quote/cap al catalogo e scaricare batch incrementali solo per deficit; nessun download arm-specifico | Gli shard remoti non contengono gli arm locali e i file eccedenti non devono essere duplicati nel runtime |
| DEC-021 | PROVVISORIAMENTE_CONFERMATA | Topologia Waymo route-aware v1 | Lane-route 6 m, controlli-route 15 m, merge con almeno 2 ingressi e senza controlli d'intersezione; intersezione da controlli/attraversamenti route-local; nessuna inferenza roundabout da semplice ciclo | Distribuzione quantitativa plausibile sul pool completo; conferma finale subordinata al campione visivo stratificato |
| DEC-022 | CONFERMATA | Minimi per sorgente/split/arm | Valori in `pipeline_v1.yaml`, derivati dalle disponibilità Waymo 72/156/1427/12/742 e PG 395/305/497/375/178 | Garantisce copertura senza forzare A0 Waymo o A4 PG e registra deficit causati da gruppi indivisibili |
| DEC-023 | CONFERMATA | Significato degli arm v2 | Primary arm = livello di difficoltà A0–A5; semantiche scenario conservate come tag indipendenti | Allinea il bandit a un curriculum progressivo ed evita che una precedenza nominale svuoti una classe |
| DEC-024 | PROVVISORIAMENTE_CONFERMATA | Parametri difficoltà v2 | A0 max 8 veicoli Q90; A1 ammette junction ≤8 agenti/≤1 conflitto; A3 da 25 agenti o 4 conflitti; A5 da conflitto VRU, topologia mista+3 conflitti oppure 30 agenti+6 conflitti; CPA 5 s, 4 m veicoli, 3 m VRU | Sul catalogo selezionato A0≤8 recupera 2 Waymo, A1-junction circa 70 e A3 743→circa 572; conferma finale subordinata ad audit visivo |
| DEC-026 | CONFERMATA | Gestione dello sbilanciamento residuo | Minimi e deficit espliciti, nessun cap artificiale A3/A5; acquisizione incrementale per A0 Waymo | Il target 1750 richiede comunque molti A3/A5 dal pool corrente; un cap non può creare scenari semplici mancanti |
| DEC-027 | CONFERMATA | Affidabilità segnale ammessa negli split | Accettare `complete` e `not_applicable`; escludere `partial`/`missing` senza imputazione | `not_applicable` significa assenza di semaforo route-relevant; stati incompleti restano disponibili solo per audit |
| DEC-028 | CONFERMATA | Arresto acquisizione incrementale Waymo | Contare solo `complete`/`not_applicable` e acquisire shard remoti mai convertiti in batch da 8 fino al target totale 1750, con cap di sicurezza a 128 nuovi shard per esecuzione | Il rendimento eleggibile non è noto prima della conversione; un numero fisso di shard non garantisce il target dopo l'esclusione di `partial`/`missing` |
| DEC-029 | CONFERMATA | Persistenza dei batch Waymo | Un database ScenarioNet separato per batch, catalogazione ricorsiva, TFRecord del batch eliminati dopo successo e scenari convertiti esclusi conservati per audit | Il converter non è append-safe sul medesimo database; questa struttura evita overwrite, duplicazioni raw/converted e corruzione del pool preesistente |
| DEC-025 | SUPERATA | Minimi v1 per sorgente/split/arm | I valori DEC-022 non vengono riutilizzati automaticamente dopo il cambio semantico | Le disponibilità per arm cambiano; le quote v2 saranno fissate soltanto dopo il nuovo audit |

---

## Deviazioni dalla specifica

Nessuna deviazione approvata.

Usare questo schema per ogni proposta:

| ID | Stato | Requisito originale | Modifica proposta | Motivo tecnico | Impatto scientifico | Approvazione |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

Stati ammessi: `PROPOSTA`, `IN_VALUTAZIONE`, `APPROVATA`, `RIFIUTATA`,
`SUPERATA`. Una deviazione non viene implementata finché non è approvata, salvo
un fallback strettamente temporaneo coperto da test e chiaramente marcato.

---

## Registro avanzamento

Aggiornare questa sezione al termine di ogni sessione che modifica codice,
dataset o decisioni.

| Data | Fase | Modifica | Verifica | Problemi/decisioni |
|---|---|---|---|---|
| 2026-07-11 | Pianificazione | Creato piano operativo e tracker | Confronto con sezioni 1–30 della specifica | Registrati ISS-001–006 e DEC-001–010 |
| 2026-07-11 | F0 | Avviato bootstrap; aggiunte dipendenze applicative ScenarioNet e Parquet | Lockfile e runtime ancora da verificare | ISS-001 in lavorazione |
| 2026-07-11 | F0 | Completati build, inventario API, manifest iniziale e smoke ScenarioEnv | 4 test mirati, Ruff/mypy verdi; smoke 10 step; suite 175 pass/2 failure non correlate | ISS-001/002/007 risolti; ISS-008 aperto; DEC-002/003 confermate |
| 2026-07-11 | F1 | Implementati record/features immutabili, validazione manifest, path sicuri, layout e config v1 | 22 test F0/F1, Ruff e mypy verdi | Nessuna deviazione; ScenarioDescription version resta data-dependent |
| 2026-07-11 | F2 | Implementati catalogo Parquet, split raggruppati e provider uniform/fixed strict | 34 test cumulativi F0–F2, Ruff/mypy verdi | DEC-011 confermata; calibrazione test-set guard differita a F3 |
| 2026-07-11 | F3 | Implementate feature comuni, Q90/VRU, soglie train-only, arms A0–A5, tag e report | 51 test cumulativi F0–F3, Ruff/mypy verdi | DEC-012 confermata; nessuna euristica topologica complessa |
| 2026-07-11 | F4 | Implementati profili PG nativi, export IDM, validation, generation manifest, CLI e pilot | 57 test cumulativi; pilot 10/10 validi; Ruff/mypy verdi | ISS-009 risolto; DEC-013/014 confermate |
| 2026-07-11 | F5 | Implementati wrapper strict Waymo, grouping key, loader catalogo e CLI converter | 62 test cumulativi; source guard e fixture converted verificati | ISS-010 aperto; DEC-015 da decidere |
| 2026-07-11 | F6 | Implementati validazione applicativa e runtime summary/mapping senza copie fisiche | compileall e diff check passati; test container pendenti | ISS-011 aperto |
| 2026-07-12 | F5 | Aggiunti container/profilo Compose dedicati, Makefile e guida operativa Waymo; corretti dipendenze e preflight CLI | Build immagine riuscito; TensorFlow 2.11, ScenarioNet/MetaDrive importabili; `--help` e source guard verificati | DEC-015 confermata; conversione reale attende raw Waymo |
| 2026-07-12 | F6 | Collegati esclusione invalid e runtime index nel builder; corretti fixture runtime | 80 test mirati passati; Ruff/mypy verdi | ISS-011 risolto; wiring CLI completato nella sessione successiva |
| 2026-07-12 | F5 | Montato il raw host in sola lettura come `/workspace/waymo_raw` e collegato il target Makefile | Preflight su directory host vuota rifiutato correttamente (`training_20s.tfrecord*` richiesto) | Nessuna modifica alla semantica della specifica |
| 2026-07-12 | Regressione | Eseguita suite completa dopo F6/F7/F8/F9 | 255 passati, 2 failure baseline | ISS-008 resta aperto e non correlato |
| 2026-07-12 | F5 | Aggiunta pipeline unica download→build→conversione e guida autenticazione Google Cloud | `bash -n`, Compose config e guardia `gcloud` verificati; senza CLI il comando si ferma senza modificare dati | DEC-016 confermata; serve login utente una tantum |
| 2026-07-12 | F5 | Corretto protobuf converter e convertito il primo shard Waymo reale | 61 scenari convertiti; loader ricorsivo e grouping verificati; 8 test mirati passati, Ruff/mypy verdi | F5 resta aperta per validazione ufficiale, catalogo completo e split reali |
| 2026-07-12 | F5/F6 | Eseguiti check ufficiale existence/integrity, validazione applicativa e runtime mapping temporaneo | 61/61 caricabili; 61 validi, 0 warning/invalid; runtime mapping 61/61 verificato | Simulation/overlap ufficiali e dataset completo restano pendenti |
| 2026-07-12 | F6 | Collegati CLI catalog-driven, hash catalogo, report JSON e wrapper dei check ufficiali ScenarioNet | Test CLI/API e Ruff passati; execution simulation/overlap sul pool finale pendente | F6 resta in corso per dataset/catalogo definitivo |
| 2026-07-12 | F7 | Implementati `SceneContextAdapter`, `ThesisScenarioEnv`, factory branch e provider reset hook | 10 test mirati passati; reset/step Waymo reale e reset provider-driven verificati nel container | Shape Waymo/PG e wiring catalogo nei worker passano a F8 |
| 2026-07-12 | F6 | Eseguito il CLI catalog-driven con hash e report persistente sul runtime Waymo smoke | 61 record, 61 valid, 0 warning/invalid, 61 file runtime | Simulation/overlap ufficiali e catalogo definitivo restano pendenti |
| 2026-07-12 | F8 | Collegati worker `spawn`, provider fixed-sequence, auto-reset deterministico e contatori runtime | Smoke reale 2 worker passato; stats aggregati e persistiti nei metadata train/eval | Catalogo finale congelato resta da validare |
| 2026-07-12 | F9 | Generato pilot PG minimo e costruito catalogo/runtime misto Waymo+PG | 61 Waymo + 21 PG caricati; runtime mapping 82/82; mixed Uniform 50/50 reset+step e shape uniforme passati | Dataset completo Waymo e simulation/overlap finali restano pendenti |
| 2026-07-12 | F9 | Corretto il wiring train/eval provider-driven e aggiunta modalità training-only | Training smoke 20 step single-worker e vectorized 2 worker `spawn` passati | Evaluation con split validation/test e random policy scientifica restano pendenti |
| 2026-07-12 | F6/F9 | Eseguito verifier ufficiale existence/integrity sul runtime misto | 82/82 scenari caricabili; CLI ora crea automaticamente la directory errori | Simulation/overlap finali restano pendenti |
| 2026-07-12 | F6 | Eseguiti verifier ufficiali simulation e overlap sullo smoke misto | Simulation 82/82; overlap Waymo↔PG senza sovrapposizioni | Evidenza valida sul campione; ripetere sul catalogo finale |
| 2026-07-12 | F9 | Eseguito pilot PG configurato dalla specifica | 100/100 scenari validi (20 per profilo), invalid rate 0%; matrice profilo-arm persistita | Seed pilot separati; dataset completo resta F10 |
| 2026-07-12 | F8/F9 | Aggiunto controllo preventivo catalogo/runtime e verificato reward custom | Runtime mismatch diagnosticato prima del simulatore; training smoke 20 step riuscito con vista mista | `env.config.data_directory` deve puntare alla runtime view del catalogo |
| 2026-07-12 | F9 | Aggiunto ed eseguito random-policy smoke headless | Fixture Waymo: 10/10 step, observation `(161,)`, action `(2,)`, valori finiti | Percorso CLI `scenarios.smoke --policy random` |
| 2026-07-12 | F9 | Profilato smoke headless e cleanup env | 3 reset+2 step: `1.933/0.488/0.684 s`, max RSS `755220 KB`; ogni env chiuso correttamente | Misura indicativa della fixture, non del dataset completo |
| 2026-07-12 | Regressione | Rieseguita suite completa dopo guardia catalogo/runtime e random smoke | 256 passati, 2 failure baseline | ISS-008 resta aperto e non correlato |
| 2026-07-12 | F9 | Aggiunte CLI finali e pipeline unica configurabile | Smoke 182 record: catalogo, split, soglie train-only, runtime train/validation/test e help CLI verificati | ISS-017 risolto; dataset completo resta F10 |
| 2026-07-12 | F9 | Added gcloud installer and single-command pipeline configuration | `make install-gcloud` verified; `make scenarionet-pipeline` resolves counts directly from YAML and fails if YAML is incomplete | OAuth remains interactive for safety; `.env` contains at most an external credential path |
| 2026-07-12 | Regressione | Rieseguita suite completa dopo l’orchestratore finale | 257 passati, 2 failure baseline | ISS-008 resta aperto e non correlato |
| 2026-07-12 | F9 | Reso automatico lo split per gruppi interi verso i target baseline | Target 1000/250/500 configurabili; conteggi effettivi persistiti nel manifest; test di disgiunzione passati | Evita conteggi manuali incompatibili con shard Waymo |
| 2026-07-12 | F9 | Corretto il pipeline per selezionare il sottoinsieme target dal pool convertito | I gruppi non selezionati restano sul disco ma non entrano nel catalogo/runtime finale; esclusioni persistite nel manifest | Evita di usare accidentalmente tutti gli scenari dei 1000 shard |
| 2026-07-12 | F9 | Separati default scientifici e override macchina | Aggiunto `conf/scenarios/pipeline_v1.yaml` e resolver; supportato service account tramite path esterno | ISS-018 risolto; login OAuth interattivo resta il percorso predefinito |
| 2026-07-12 | Regressione | Rieseguita suite completa dopo split automatico | 258 passati, 2 failure baseline | ISS-008 resta aperto e non correlato |
| 2026-07-12 | F9 | Made `conf/scenarios/pipeline_v1.yaml` the single source for scientific parameters | `compileall`, `bash -n`, and `git diff --check` passed | Removed duplicate overrides from `.env.example`, Compose, and orchestration; credentials remain OAuth or an external service-account path; API keys excluded because they do not grant IAM on the bucket |
| 2026-07-12 | F5/F10 | Added remote inventory and optional raw cleanup | `make waymo-inventory` queries count/size without downloading; `WAYMO_CLEANUP_RAW_AFTER_CONVERSION=true` deletes TFRecords only after a non-empty database is verified | Avoids keeping raw and converted data simultaneously while preserving a conservative default |
| 2026-07-12 | F5/F10 | Fixed `WAYMO_NUM_FILES` with wildcard patterns | The limit now selects shards before download and is also applied during conversion | Prevents accidental full-bucket downloads when a subset is requested |
| 2026-07-12 | F9 | Added Rich observability to the ScenarioNet pipeline | Stage timing/config summary, PG and validation progress bars, runtime/check spinners, and human summaries are sent to stderr while JSON stdout remains stable; Ruff/mypy and 259 tests pass with the 2 known baseline failures | Makes long dataset preparation runs inspectable without changing pipeline semantics |
| 2026-07-13 | F3/F5/F9 | Implementati topologia Waymo route-aware, evidenze/confidenza, arm pre-split e selezione group-aware con minimi/deficit | 42 test ScenarioNet mirati verdi, Ruff e mypy verdi; audit su 2409 Waymo + 1750 PG | ISS-019 risolto; DEC-019–022; audit visivo ISS-020 pendente |
| 2026-07-13 | Regressione | Eseguita suite completa nel container runtime dopo la correzione arm | 264 test passati; 2 failure baseline ISS-008 e 2 smoke su artifact catalogo/runtime preesistenti non allineati | Ricostruire catalogo/runtime ufficiali con la nuova pipeline prima degli smoke data-dependent |
| 2026-07-13 | F3/F9 | Sostituita tassonomia arm v1 con scala curricolare v2; aggiunti CPA offline, conflitti, tag, quote e deficit espliciti | 45 test mirati, Ruff e mypy verdi; audit 2409 Waymo + 1750 PG; suite 267 pass/4 failure preesistenti o data-dependent | ISS-021 risolto; DEC-023/024/026; A0 Waymo richiede mining aggiuntivo, audit visivo ISS-020 pendente |
| 2026-07-13 | F5/F10 | Implementata espansione Waymo automatica deficit-driven | Pool status distingue totale/eleggibile e distribuzione arm; selezione di shard mai convertiti, database append-safe per batch, cleanup raw e stop a 1750 eleggibili | DEC-027–029; `partial`/`missing` non vengono imputati né selezionati; il cap 128 impedisce download senza limite |
| 2026-07-13 | F9/F10 | Eseguita ricatalogazione finale dopo l'espansione Waymo | 1829 Waymo + 1750 PG selezionati; feature v2; existence/simulation/overlap exit 0; 61 test dataset passati | Deficit residuo soltanto A0 Waymo=16; overshoot Waymo dovuto ai gruppi shard indivisibili |

---

## Prossima sessione

La preparazione tecnica del dataset e l'integrazione ScenarioNet sono concluse:
catalogo, split, soglie, runtime e verifier ufficiali sono stati eseguiti sul
pool finale. Restano attività di validazione scientifica opzionali prima del
freeze definitivo:

1. eseguire evaluation a sequenza fissa con checkpoint sul catalogo con split
   `validation`/`test`;
2. eseguire il pilot visivo Waymo e congelare `extra_steps_after_scenario`;
3. congelare manifest, profili PG e decisioni data-dependent;
4. decidere se accettare il deficit A0 Waymo oppure intraprendere un mining
   dedicato di scenari reali semplici.
