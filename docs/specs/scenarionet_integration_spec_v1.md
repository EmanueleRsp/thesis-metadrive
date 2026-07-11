# Specifica di implementazione — Integrazione ScenarioNet nel progetto di tesi

**Stato:** specifica architetturale e implementativa v1 — pronta per l’implementazione  
**Ambito:** integrazione di scenari reali Waymo/ScenarioNet e scenari procedurali MetaDrive in una pipeline unificata di training ed evaluation  
**Training principale previsto:** 1.500.000 environment steps  
**Setting:** single-agent ego control, post-perception / mid-to-end  
**Formato canonico degli scenari:** `ScenarioDescription`

---

## 1. Scopo del documento

Questo documento definisce in modo completo e vincolante la prima versione dell’integrazione di ScenarioNet nel progetto.

La specifica copre:

1. sorgenti e formato dei dati;
2. organizzazione delle directory;
3. costruzione dei dataset e degli split;
4. generazione offline degli scenari procedurali;
5. validazione e catalogazione;
6. estrazione delle feature di scenario;
7. definizione e assegnazione degli arms;
8. caricamento e sampling;
9. configurazione di `ScenarioEnv`;
10. horizon, termination/truncation e reward custom;
11. esecuzione vectorized e riproducibilità;
12. logging, test e criteri di completamento.

Il documento non definisce ancora nel dettaglio:

- la formula finale del rulebook;
- la struttura finale della semantic observation;
- l’encoder neurale;
- il funzionamento interno dell’Automatic Curriculum Learning;
- le varianti lessicografiche o distribuzionali degli algoritmi.

Tali componenti verranno integrate dopo che la pipeline ScenarioNet sarà stabile. Questo documento prepara però tutte le interfacce necessarie affinché quelle fasi possano essere aggiunte senza ristrutturare il caricamento degli scenari.

---

## 2. Decisioni architetturali congelate

### 2.1 Setting del problema

La pipeline opera in modalità:

```text
single-agent ego control
post-perception / mid-to-end
continuous control
closed-loop simulation
```

La policy controlla soltanto il veicolo ego.

La policy non riceve raw camera o LiDAR come sorgente percettiva primaria. Riceverà successivamente una rappresentazione semantica strutturata derivata dallo stato corrente del simulatore.

Sono ammesse soltanto informazioni causalmente disponibili al tempo corrente:

- stato dell’ego;
- route dell’ego;
- mappa locale;
- corsie e connettività;
- agenti osservati fino al tempo corrente;
- traffic controls correnti;
- oggetti statici rilevanti.

Sono vietati:

- future tracks reali;
- future route degli altri veicoli;
- future fasi semaforiche;
- esito futuro dell’episodio;
- etichette o metadati del curriculum passati alla policy;
- informazioni ground-truth future utilizzate come feature.

#### Route ego e causalità

Nella v1 la route ego resa disponibile dal modulo di navigazione di `ScenarioEnv`
può essere usata internamente dall’environment per navigazione, progresso,
destinazione, successo e valutazione della rilevanza degli elementi di scena. Non
si implementa in questa fase un nuovo route planner o un nuovo map-matching
avanzato.

La futura osservazione semantica potrà esporre alla policy soltanto una
rappresentazione **geometrica di navigazione** della route ego, per esempio
checkpoint relativi o route tokens privi di informazione temporale. Non devono
essere esposti come feature:

```text
timestamp futuri della traiettoria registrata
velocità o accelerazioni future registrate
action future dell’ego registrato
validity mask future
traiettorie future time-indexed dell’ego o degli altri agenti
```

La definizione concreta dei route tokens appartiene alla successiva specifica della
`SemanticStateObservation`; l’integrazione ScenarioNet deve soltanto preservare
una route ego valida e impedire leakage temporale.

### 2.2 Sorgenti degli scenari

La configurazione principale usa:

```text
50% Waymo / ScenarioNet
50% MetaDrive procedural generation
```

Il rapporto 50/50 è il default sperimentale della v1: è una scelta di bilanciamento ragionevole e riproducibile, non un valore dichiarato universalmente ottimale.

Il 50/50 si riferisce a:

1. composizione del pool di training;
2. probabilità della sorgente nei reset di training.

Non garantisce il 50/50 esatto delle transizioni, perché le durate episodiche possono differire.

Il logging deve quindi distinguere:

- quota dei reset per sorgente;
- quota degli environment step per sorgente.

### 2.3 Dataset reale

La sorgente reale principale è la variante Waymo `training_20s`, convertita tramite ScenarioNet.

La pipeline deve restare dataset-agnostic a livello di `ScenarioDescription`, così da poter aggiungere in futuro altri dataset ScenarioNet senza cambiare environment, catalogo o curriculum.

### 2.4 Dataset procedurale

Gli scenari procedurali vengono:

1. generati offline;
2. esportati come `ScenarioDescription`;
3. validati;
4. classificati;
5. congelati prima degli esperimenti principali.

Non si usa generazione procedurale on-the-fly durante il training principale.

### 2.5 Formato comune

Waymo e PG devono essere rappresentati con lo stesso formato canonico:

```text
ScenarioDescription
```

Tutte le operazioni successive devono dipendere da questo formato e non dalla sorgente originale.

### 2.6 Mutation

La mutation degli scenari è disattivata nella v1.

```yaml
scenario_mutation:
  enabled: false
```

### 2.7 Curriculum nativo di ScenarioNet

Non si usa il curriculum nativo di `ScenarioEnv`.

Il futuro curriculum sarà implementato esternamente tramite l’interfaccia `ScenarioProvider` e lavorerà sugli arms definiti in questo documento. La v1 implementa soltanto un provider uniforme; l’ACL esistente verrà adattato successivamente senza modificare l’environment.

---

## 3. Versioni, commit e manifest

L’implementazione usa il submodule ScenarioNet già presente e checkoutato nel progetto.

Codex deve:

1. ispezionare il commit corrente del submodule e le sue dipendenze;
2. verificare le API concrete di ScenarioNet e MetaDrive disponibili localmente;
3. installare o adeguare le dipendenze compatibili necessarie;
4. non aggiornare arbitrariamente il submodule a una versione diversa;
5. congelare commit e versioni dopo la prima conversione, il primo reset e lo smoke test riusciti.

Non è necessario scegliere manualmente tutte le versioni prima di iniziare. La compatibilità viene risolta sull’installazione effettiva e poi registrata.

La v1 usa come sorgente reale esclusivamente la variante Waymo:

```text
training_20s
```

coerente con la costruzione del database Waymo descritta da ScenarioNet. Gli split train, validation e test della tesi vengono costruiti internamente sul database convertito a partire da tale variante.

File obbligatorio, sotto la data root configurata nel `.env`:

```text
${SCENARIONET_DATA_ROOT}/manifest.yaml
```

Schema minimo:

```yaml
dataset_id: scenarionet_v1

software:
  project_commit: null
  metadrive_version: null
  metadrive_commit: null
  scenarionet_commit: null
  python_version: null

waymo:
  release: null
  source_variant: training_20s
  converter_commit: null
  source_directory: null
  converted_directory: null

procedural:
  generator_commit: null
  exporter_commit: null
  profiles_version: pg_profiles_v1

scenario_description:
  version: null

creation:
  created_at: null
  created_by_command: null
```

Dopo il congelamento del manifest:

- non si cambiano versioni durante il confronto principale;
- un cambiamento incompatibile di converter o generatore crea un nuovo `dataset_id`;
- ogni run conserva il riferimento o una copia del manifest usato.

## 4. Organizzazione delle directory

La posizione dei dati è configurata tramite il `.env` già previsto dal progetto:

```dotenv
SCENARIONET_DATA_ROOT=/percorso/scelto/dall_utente/scenarionet_v1
```

Il codice non deve incorporare path assoluti. Tutti i path nel catalogo sono relativi a `SCENARIONET_DATA_ROOT`.

Struttura logica:

```text
${SCENARIONET_DATA_ROOT}/
├── manifest.yaml
│
├── waymo/
│   ├── database/
│   ├── metadata/
│   └── validation/
│
├── pg/
│   ├── database/
│   ├── generation_manifests/
│   ├── pilot/
│   └── validation/
│
├── runtime/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── catalog/
│   ├── scenario_catalog.parquet
│   ├── feature_statistics.json
│   ├── arm_thresholds.json
│   ├── arm_distribution.json
│   └── validation_summary.json
│
└── splits/
    ├── train.json
    ├── validation.json
    ├── test.json
    └── split_manifest.yaml
```

Le directory `runtime/<split>` costituiscono la vista caricabile da `ScenarioEnv`. Devono essere costruite usando i summary e mapping supportati dalla versione ScenarioNet installata, evitando copie fisiche degli scenari quando possibile.

I file originali Waymo e PG restano separati; il catalogo e le viste runtime forniscono l’interfaccia unificata.

---

## 5. Dimensioni previste e budget

### 5.1 Configurazione principale

La prima configurazione seria prevista è:

```yaml
training:
  total_timesteps: 1_500_000

dataset:
  train:
    waymo: 1000
    pg: 1000
  validation:
    waymo: 250
    pg: 250
  test:
    waymo: 500
    pg: 500
```

Questi valori sono la baseline pianificata.

Possono essere ridotti soltanto per vincoli tecnici documentati. Una riduzione deve modificare il manifest dello split e non deve essere effettuata silenziosamente.

### 5.2 Profili di sviluppo

Per debug e smoke test sono ammesse configurazioni più piccole:

```yaml
development:
  train:
    waymo: 250
    pg: 250
  validation:
    waymo: 100
    pg: 100
  test:
    waymo: 100
    pg: 100
```

Le run di sviluppo non sostituiscono la run principale.

---

## 6. Split e prevenzione del leakage

### 6.1 Waymo

La v1 usa esclusivamente gli scenari provenienti da:

```text
Waymo training_20s
```

convertiti tramite ScenarioNet in `ScenarioDescription`.

Il database risultante viene diviso internamente in tre sottoinsiemi disgiunti:

```text
Waymo training_20s convertito
├── thesis train
├── thesis validation
└── thesis final test
```

Gli split ufficiali Waymo `validation` e `testing` non vengono utilizzati nella v1, poiché non corrispondono necessariamente alla stessa variante di scenari completi da 20 secondi usata dal database ScenarioNet di riferimento.

La suddivisione interna deve essere:

- deterministica;
- congelata prima degli esperimenti principali;
- raggruppata per il più forte identificativo di provenienza disponibile;
- priva di log, segmenti o gruppi condivisi fra gli split.

Ordine di preferenza per la chiave di gruppo:

```text
source_log_id
segment_id
source_file_id / TFRecord shard
strongest equivalent metadata available
```

Se il formato `training_20s` contiene scenari già indipendenti e non sovrapposti e non espone un identificativo superiore, il gruppo può coincidere con lo scenario originale. La scelta effettiva deve essere registrata nel manifest.

Il thesis final test non deve essere usato per:

- scelta degli iperparametri;
- selezione dei checkpoint;
- modifica degli arms;
- calibrazione delle soglie;
- revisione dei profili PG;
- scelte sul rulebook o sull’osservazione.

Nel catalogo:

```text
official_split = "training_20s"
split = "train" | "validation" | "test"
```

I due campi rappresentano rispettivamente la provenienza Waymo e lo split interno della tesi.

### 6.2 Procedural generation

Per PG si usano seed disgiunti fra train, validation e test.

Non sono ammessi:

- lo stesso seed in split differenti;
- lo stesso scenario rigenerato in più split;
- in future versioni, parent e child mutato in split differenti.

Non è richiesta una deduplicazione geometrica sofisticata nella v1.

### 6.3 Verifica degli split

La pipeline deve verificare:

```text
intersection(train_group_ids, validation_group_ids) == empty
intersection(train_group_ids, test_group_ids) == empty
intersection(validation_group_ids, test_group_ids) == empty
```

Deve inoltre eseguire il controllo di overlap fornito dalla versione ScenarioNet installata, ove applicabile.

### 6.4 Manifest degli split

`split_manifest.yaml` deve contenere almeno:

```yaml
split_seed: 0

source_policy:
  waymo_source: waymo_training_20s

  thesis_train: internal_grouped_split
  thesis_validation: internal_grouped_split
  thesis_test: internal_grouped_split

  excluded:
    waymo_official_validation:
      reason: not_used_in_v1

    waymo_official_testing:
      reason: not_used_in_v1

grouping:
  waymo: strongest_available_source_group
  pg: generation_seed

counts:
  train:
    waymo: 1000
    pg: 1000
  validation:
    waymo: 250
    pg: 250
  test:
    waymo: 500
    pg: 500

catalog_hash: null
created_at: null
```

## 7. Scenario catalog

Il catalogo rappresenta l’interfaccia unificata tra dataset, viste runtime e training.

Formato:

```text
Parquet
```

### 7.1 Record minimo

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ScenarioRecord:
    scenario_uid: str                 # chiave globale univoca
    scenario_id: str                  # id originale
    source: str                       # "waymo" | "pg"
    relative_path: str                # relativo a SCENARIONET_DATA_ROOT

    official_split: Optional[str]
    source_log_id: Optional[str]
    source_scenario_id: Optional[str]

    dataset_version: str
    converter_version: Optional[str]

    split: str                        # train | validation | test
    runtime_index: Optional[int]
    length: int

    pg_profile: Optional[str]
    pg_seed: Optional[int]
    map_id: Optional[str]

    primary_arm: str
    tags: tuple[str, ...]

    signal_reliability: str           # not_applicable | complete | partial | missing
    validation_status: str            # valid | warning | invalid
    validation_warnings: tuple[str, ...]
```

Identificativo consigliato:

```python
scenario_uid = f"{source}:{dataset_version}:{scenario_id}"
```

### 7.2 Campi feature aggiuntivi

Il Parquet conserva soltanto le feature necessarie per catalogazione, curriculum e diagnosi:

```text
route_length_m
topology_tag
has_intersection
has_merge_or_roundabout
has_route_traffic_light
has_route_stop_sign
has_route_crosswalk
has_vehicle
has_pedestrian
has_cyclist
relevant_agents_q90
relevant_vehicles_q90
min_vehicle_distance_m
min_vru_distance_to_route_m
low_traffic
dense_traffic
vru_interaction
```

Valori ammessi per `topology_tag`:

```text
simple
merge_or_roundabout
intersection
mixed
unknown
```

`mixed` viene usato quando lo stesso scenario contiene sia una struttura
`intersection` sia una struttura `merge_or_roundabout` rilevanti per la route
ego. La categoria evita di forzare arbitrariamente uno scenario topologicamente
misto in una sola delle due classi; il `primary_arm` resta comunque determinato
dalla logica A0–A5 descritta più avanti.

---

## 8. Arms di scenario

Gli arms descrivono lo scenario risultante, non il profilo usato per generarlo.

### 8.1 Arms definitivi

```text
A0_simple_lane_follow
A1_vehicle_interaction
A2_merge_or_roundabout
A3_intersection
A4_vru_interaction
A5_complex_mixed
```

### 8.2 Significato

#### A0 — Simple lane following

- nessuna topologia complessa rilevante;
- nessuna interazione veicolare persistente secondo `relevant_vehicles_q90`;
  sono comunque ammessi veicoli lontani o presenze brevi che non rendono
  positivo il novantesimo percentile temporale;
- nessun VRU rilevante;
- compito dominante: route following e controllo base.

#### A1 — Vehicle interaction

- interazione con traffico veicolare;
- strada ordinaria;
- nessun merge, roundabout o incrocio rilevante;
- compiti: distanza, adattamento velocità, accodamento e cambio corsia.

#### A2 — Merge or roundabout

- merge, diverge, bottleneck, ramp o roundabout;
- struttura rilevante per la route ego.

Non è obbligatorio distinguere automaticamente i sottotipi se la versione convertita non li espone direttamente.

#### A3 — Intersection

Comprende tutte le junction rilevanti per la route:

- incroci semaforizzati;
- incroci non semaforizzati;
- incroci con stop;
- T-junction e forme equivalenti disponibili.

Il tipo di controllo resta un tag secondario e sarà utilizzato dal rulebook e dall’evaluation.

#### A4 — VRU interaction

- pedone o ciclista potenzialmente rilevante per la route ego;
- può essere alimentato prevalentemente o esclusivamente da Waymo nella v1.

#### A5 — Complex mixed

- almeno due fattori semantici distinti tra:
  - intersection;
  - merge o roundabout;
  - VRU interaction.

Il solo traffico denso non rende automaticamente uno scenario A5: rimane un tag di difficoltà.

### 8.3 Tag indipendenti

Ogni scenario mantiene più tag anche se ha un solo `primary_arm`.

Tag minimi:

```text
has_dense_traffic
has_merge_or_roundabout
has_intersection
has_signalized_intersection
has_unsignalized_intersection
has_traffic_light
has_stop_sign
has_crosswalk
has_vru
has_unknown_signal
has_static_obstacle
```

---

## 9. Profili di generazione PG

I profili servono a produrre diversità controllata, ma non determinano direttamente il `primary_arm`.

Pipeline:

```text
PG profile
→ MetaDrive generation
→ ScenarioDescription
→ validation
→ feature extraction
→ tags
→ primary arm
```

Codex deve implementare i profili usando esclusivamente le configurazioni e i road block nativi disponibili nella versione MetaDrive locale. Non deve sviluppare nella v1 un generatore stradale, un sistema di precedenza, semafori o VRU custom.

Se un blocco richiesto non è disponibile, lo script deve segnalarlo chiaramente e non sostituirlo con un’euristica complessa non prevista.

### 9.1 Parametri generali

```yaml
pg_generation:
  offline: true
  max_episode_length: 500
  ego_generation_policy: IDMPolicy
  accident_prob_default: 0.0
  random_lane_num: true
  random_lane_width: true
```

L’ego viene controllato da una policy rule-based durante la generazione offline per produrre una traiettoria nominale valida. Durante il training viene controllato dalla policy RL tramite `EnvInputPolicy`.

I nomi esatti delle opzioni MetaDrive devono essere ricavati dalla versione locale.

### 9.2 P0 — Simple

```yaml
profile: P0_simple
native_topology:
  one_of: [straight, curve]
traffic_density:
  min: 0.00
  max: 0.05
num_blocks:
  choices: [2, 3]
accident_prob: 0.0
max_episode_length: 500
```

Target prevalente:

```text
A0_simple_lane_follow
```

### 9.3 P1 — Vehicle interaction

```yaml
profile: P1_vehicle_interaction
native_topology:
  one_of: [straight, curve]
traffic_density:
  min: 0.07
  max: 0.18
num_blocks:
  choices: [2, 3, 4]
accident_prob: 0.0
max_episode_length: 500
```

Target prevalente:

```text
A1_vehicle_interaction
```

### 9.4 P2 — Merge or roundabout

```yaml
profile: P2_merge_or_roundabout
required_native_block:
  one_of: [merge, ramp, bottleneck, roundabout]
traffic_density:
  min: 0.08
  max: 0.18
num_blocks:
  choices: [2, 3, 4]
accident_prob: 0.0
max_episode_length: 500
```

La route ego deve attraversare il blocco richiesto, quando questa verifica è resa disponibile dal generatore nativo.

Target prevalente:

```text
A2_merge_or_roundabout
```

### 9.5 P3 — Intersection

```yaml
profile: P3_intersection
required_native_block:
  one_of: [intersection, t_intersection]
traffic_density:
  min: 0.08
  max: 0.18
num_blocks:
  choices: [2, 3, 4]
accident_prob: 0.0
max_episode_length: 500
```

Il profilo usa l’intersezione nativa disponibile. Non è richiesto generare semafori, VRU o una negoziazione della precedenza custom.

Target prevalente:

```text
A3_intersection
```

### 9.6 P5 — Complex mixed

```yaml
profile: P5_complex_mixed
native_complex_blocks:
  min_count: 2
  pool: [merge, ramp, bottleneck, roundabout, intersection, t_intersection]
traffic_density:
  min: 0.15
  max: 0.25
num_blocks:
  choices: [4, 5]
accident_prob: 0.0
max_episode_length: 500
```

Se la versione locale non permette di imporre due blocchi complessi, il profilo usa una sequenza nativa più varia e traffico più alto. L’assegnazione finale ad A5 dipende comunque dalle feature dello scenario risultante.

### 9.7 A4 e PG

Non si richiede nella v1 un profilo PG specifico per `A4_vru_interaction`.

La copertura di pedoni e ciclisti può provenire principalmente da Waymo. Un generatore PG custom per VRU è una possibile estensione, non un requisito di completamento.

### 9.8 Ostacoli statici

`accident_prob` rimane a zero nei profili principali.

Gli ostacoli statici non definiscono un arm autonomo nella v1. Possono essere conservati tramite il tag:

```text
has_static_obstacle
```

---

## 10. Manifest di generazione PG

Ogni scenario PG deve conservare:

```yaml
scenario_id: null

generation:
  profile: P2_merge_or_roundabout
  seed: null
  block_sequence: []
  traffic_density: null
  lane_num: null
  lane_width: null
  accident_prob: 0.0
  max_episode_length: 500

software:
  generator_commit: null
  exporter_commit: null

output:
  scenario_path: null
  scenario_length: null
  validation_status: null
```

Il seed e i parametri devono consentire di rigenerare lo scenario.

---

## 11. Pilot PG e generazione del dataset

Codex implementa gli script di generazione e i relativi smoke test. La generazione completa del dataset viene successivamente avviata dall’utente.

### 11.1 Pilot leggero

Prima della generazione finale si esegue un pilot configurabile, con default:

```text
20 scenari per profilo
```

Profili:

```text
P0, P1, P2, P3, P5
```

Totale di default:

```text
100 scenari
```

Per ogni scenario:

```text
generate
→ export
→ official validation
→ thesis smoke validation
→ feature extraction
→ arm assignment
```

Durante lo sviluppo automatico sono sufficienti 2–3 scenari temporanei per profilo per verificare che il codice funzioni.

### 11.2 Report diagnostico

Il pilot produce una matrice profilo-arm e il tasso di scenari invalidi.

La percentuale di scenari finiti nell’arm target è una diagnostica, non un requisito bloccante. Non viene ottimizzata sistematicamente.

È ammessa una sola revisione manuale dei parametri se:

- un profilo non produce quasi mai la topologia prevista;
- più del 10% degli scenari è invalido;
- il blocco richiesto non viene effettivamente usato;
- un profilo fallisce per incompatibilità con la versione MetaDrive installata.

Dopo l’eventuale revisione:

```text
pg_profiles_v1 = frozen
```

### 11.3 Dataset PG finale

Si genera un numero prefissato di scenari validi, senza inseguire quote rigide per arm.

Target principale:

```text
1.000 scenari PG validi per il training
```

Più gli scenari previsti per validation e test.

Procedura:

1. generare il pool PG con seed registrati;
2. scartare gli scenari invalidi;
3. costruire gli split tramite seed disgiunti;
4. unire il train candidate PG al train candidate Waymo;
5. calcolare Q40 e Q75 sul train candidate;
6. assegnare gli arms;
7. produrre il report finale della distribuzione.

Non si rigenera il dataset soltanto per rendere uguali gli arms. Un arm raro viene documentato e gestito dal sampler/curriculum successivo.

---

## 12. Feature extraction comune

La funzione deve accettare qualunque `ScenarioDescription`:

```python
def extract_scenario_features(
    scenario: dict,
    source: str,
    realized_generation_metadata: dict | None = None,
) -> ScenarioFeatures:
    ...
```

### 12.1 Principio di semplicità

L’estrattore topologico usa, in ordine:

1. metadata topologici effettivamente presenti nello scenario convertito;
2. tipi e connessioni delle `map_features` già disponibili;
3. per PG, metadata dei blocchi effettivamente realizzati ed esportati;
4. `unknown` quando la classificazione non è ricavabile in modo affidabile.

Il nome del profilo PG non costituisce prova della topologia dello scenario. Per esempio, `P3_intersection` non implica automaticamente `has_intersection=true`: la feature deve risultare dai blocchi effettivamente realizzati, dai metadata esportati o dalle `map_features` dello scenario.

`pg_profile` resta nel catalogo e nel manifest esclusivamente per diagnosi e riproducibilità.

Non si implementa nella v1 un riconoscitore geometrico complesso basato soltanto sulle polylines.

Codex deve ispezionare il converter e lo schema realmente presenti nel submodule per usare i campi corretti; non deve inventare nomi di metadata.

### 12.2 Struttura suggerita

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ScenarioFeatures:
    scenario_id: str
    source: str
    length: int

    route_length_m: float
    topology_tag: Literal[
        "simple",
        "merge_or_roundabout",
        "intersection",
        "mixed",
        "unknown",
    ]

    has_intersection: bool | None
    has_merge_or_roundabout: bool | None

    has_route_traffic_light: bool | None
    has_route_stop_sign: bool | None
    has_route_crosswalk: bool | None
    signal_reliability: Literal[
        "not_applicable", "complete", "partial", "missing"
    ]

    has_vehicle: bool
    has_pedestrian: bool
    has_cyclist: bool

    relevant_agents_q90: float
    relevant_vehicles_q90: float

    min_vehicle_distance_m: float | None
    min_vru_distance_to_route_m: float | None

    low_traffic: bool
    dense_traffic: bool
    vru_interaction: bool
```

Regola minima per `topology_tag`:

```python
if has_intersection is True and has_merge_or_roundabout is True:
    topology_tag = "mixed"
elif has_intersection is True:
    topology_tag = "intersection"
elif has_merge_or_roundabout is True:
    topology_tag = "merge_or_roundabout"
elif (
    has_intersection is False
    and has_merge_or_roundabout is False
):
    topology_tag = "simple"
else:
    topology_tag = "unknown"
```

Per i due campi topologici:

```text
True  = presenza rilevata
False = assenza determinata
None  = non valutabile
```

La motivazione è mantenere distinta l'incertezza (`unknown`) dalla presenza
effettiva di più strutture topologiche (`mixed`), senza introdurre un
riconoscitore geometrico complesso.

### 12.3 Route relevance

Una feature topologica o un traffic control è rilevante soltanto se interessa la route ego o una zona di conflitto direttamente collegata a essa.

Non basta che l’elemento sia presente in un punto qualsiasi della mappa.

Se la route relevance non può essere determinata in modo affidabile per una specifica feature, il relativo valore resta `unknown` o non valutabile invece di essere inferito con euristiche complesse.

## 13. Agenti rilevanti e densità

Per ogni timestep:

\[
n_t =
\#\left\{
j:
valid_{j,t}
\land
dynamic_j
\land
\|p_{j,t}-p_{ego,t}\|_2 \leq 50\text{ m}
\land
|z_{j,t}-z_{ego,t}| < 3\text{ m}
\right\}.
\]

La metrica episodica è:

\[
N_s^{rel}=Q_{0.90}(n_0,\dots,n_{T-1}).
\]

Motivazione:

- il massimo è troppo sensibile a un singolo frame;
- la media può nascondere una fase breve ma critica;
- il novantesimo percentile rappresenta un livello alto ma persistente di interazione.

Configurazione congelata:

```yaml
relevant_agents:
  radius_m: 50.0
  vertical_tolerance_m: 3.0
  temporal_quantile: 0.90
```

Una futura versione potrà aggiungere filtraggio topologico più sofisticato. Non è requisito della v1, salvo che il pilot mostri falsi positivi evidenti su strade sovrapposte o non connesse.

---

## 14. VRU interaction

Un pedone o ciclista presente lontano dalla route non rende automaticamente lo scenario A4.

Si calcola:

\[
d_{VRU,route}
=
\min_{j,t}
d(p_{j,t}, \mathcal R_{ego}).
\]

Configurazione:

```yaml
vru_interaction:
  max_distance_to_route_m: 8.0
```

Definizione:

```python
vru_interaction = (
    has_pedestrian_or_cyclist
    and min_vru_distance_to_route_m <= 8.0
)
```

La soglia è fisicamente interpretabile e verrà verificata visivamente su alcuni casi di confine durante il pilot, ma non ottimizzata per bilanciare artificialmente gli arms.

---

## 15. Calcolo delle soglie di traffico

Le soglie non vengono scelte arbitrariamente.

Vengono calcolate esclusivamente sul train set congelato.

### 15.1 Procedura

1. usare lo stesso numero di scenari Waymo e PG;
2. calcolare `relevant_agents_q90`;
3. calcolare:

\[
\tau_{low}=Q_{0.40}(N_s^{rel})
\]

\[
\tau_{dense}=Q_{0.75}(N_s^{rel});
\]

4. arrotondare a interi;
5. salvare le soglie;
6. applicare gli stessi valori a validation e test.

Definizioni:

```python
low_traffic = relevant_agents_q90 <= tau_low
dense_traffic = relevant_agents_q90 >= tau_dense
```

`tau_low` e `tau_dense` definiscono esclusivamente i tag quantitativi
`low_traffic` e `dense_traffic`. Non determinano direttamente il confine
A0/A1, che dipende dalla presenza persistente di almeno un veicolo rilevante
secondo `relevant_vehicles_q90`.

In sintesi:

```text
A0/A1 → rilevanza veicolare persistente
low/dense traffic → tag quantitativi relativi al train set
```

### 15.2 File prodotto

`arm_thresholds.json`:

```json
{
  "feature_version": "v1",
  "relevant_radius_m": 50.0,
  "vertical_tolerance_m": 3.0,
  "temporal_quantile": 0.90,
  "low_traffic_quantile": 0.40,
  "dense_traffic_quantile": 0.75,
  "tau_low": null,
  "tau_dense": null,
  "computed_on_split": "train",
  "balanced_sources": true
}
```

Le soglie non vengono mai ricalcolate sui set di evaluation.

---

## 16. Assegnazione del primary arm

### 16.1 Fattori

```python
merge_roundabout = f.has_merge_or_roundabout is True
intersection = f.has_intersection is True
vru_context = f.vru_interaction
vehicle_interaction = f.relevant_vehicles_q90 > 0
```

`vehicle_interaction` usa `relevant_vehicles_q90` anziché la sola presenza di
un veicolo nello scenario. In questo modo A1 richiede che almeno un veicolo sia
effettivamente vicino all'ego per una parte non trascurabile dell'episodio:
un veicolo lontano, presente soltanto nel file, non rende lo scenario
un'interazione veicolare.

Le informazioni sul tipo di controllo dell’intersezione restano tag secondari:

```text
has_signalized_intersection
has_unsignalized_intersection
has_stop_sign
has_unknown_signal
```

### 16.2 Complessità mista

A5 richiede almeno due fattori semantici distinti:

```python
semantic_count = sum([
    merge_roundabout,
    intersection,
    vru_context,
])

complex_mixed = semantic_count >= 2
```

Il traffico denso e l'interazione veicolare rimangono tag e misure di
difficoltà, ma non spostano da soli uno scenario in A5. Questa scelta impedisce
che quasi ogni incrocio trafficato venga assorbito dall'arm misto.

### 16.3 Algoritmo

```python
def assign_primary_arm(f: ScenarioFeatures) -> str:
    merge_roundabout = f.has_merge_or_roundabout is True
    intersection = f.has_intersection is True
    vru_context = f.vru_interaction
    vehicle_interaction = f.relevant_vehicles_q90 > 0

    semantic_count = sum([
        merge_roundabout,
        intersection,
        vru_context,
    ])

    if semantic_count >= 2:
        return "A5_complex_mixed"

    if vru_context:
        return "A4_vru_interaction"

    if intersection:
        return "A3_intersection"

    if merge_roundabout:
        return "A2_merge_or_roundabout"

    if vehicle_interaction:
        return "A1_vehicle_interaction"

    return "A0_simple_lane_follow"
```

Uno scenario con `topology_tag="unknown"` può comunque essere classificato
come A1, A4 o A0 sulla base delle altre feature. Non viene forzata una
classificazione topologica non affidabile.

### 16.4 Controllo distribuzione

Dopo la catalogazione:

- si salva la distribuzione per arm e sorgente;
- un arm raro viene ispezionato, ma non si modificano automaticamente soglie o dataset per renderlo grande quanto gli altri;
- A4 può essere prevalentemente Waymo;
- eventuali aggregazioni future vengono decise prima del training principale.

## 17. Validazione degli scenari

La validazione avviene su due livelli.

### 17.1 Validazione ufficiale ScenarioNet

Usare gli strumenti forniti dalla versione installata per le operazioni disponibili, quali:

```text
existence check
database integrity
simulation check
overlap check
creazione di un database ripulito dagli errori
```

I nomi esatti dei comandi vengono verificati sulla versione locale.

Gli scenari falliti vengono esclusi dal catalogo valido o dal database runtime.

### 17.2 Validazione applicativa della tesi

Controlli minimi:

```text
SDC presente
SDC valido al reset
ScenarioDescription coerente
array temporali della lunghezza prevista
map features presenti
route ego disponibile e non degenere
nessun NaN/Inf nelle quantità essenziali
reset del ThesisScenarioEnv riuscito
10 step di rollout senza eccezioni
reward custom calcolabile
observation con shape valida
termination e truncation calcolabili
```

La topologia `unknown` non rende automaticamente invalido lo scenario.

### 17.3 Affidabilità dei semafori

Valori:

```text
signal_reliability = not_applicable
signal_reliability = complete
signal_reliability = partial
signal_reliability = missing
```

Semantica:

- `not_applicable`: nessun semaforo route-relevant è presente;
- `complete`: i semafori route-relevant e gli stati necessari sono disponibili;
- `partial`: esiste almeno un semaforo route-relevant, ma alcuni stati temporali o
  riferimenti sono incompleti;
- `missing`: un semaforo route-relevant è stato identificato, ma lo stato dinamico
  necessario non è disponibile.

Tag derivato:

```python
has_unknown_signal = signal_reliability in {"partial", "missing"}
```

`not_applicable` non genera warning e non rende lo scenario non valutabile. Le
regole dipendenti dai semafori restituiscono `NOT_EVALUABLE` soltanto nei casi
`partial` o `missing`. Lo scenario può comunque restare utilizzabile per le altre
regole e per collision avoidance.

### 17.4 Stati di validazione

```text
valid
warning
invalid
```

- `valid`: utilizzabile senza anomalie note;
- `warning`: utilizzabile, ma con feature non valutabili;
- `invalid`: non entra negli split runtime.

---

## 18. ScenarioEnv factory

Si implementa una factory unica:

```python
def make_thesis_scenario_env(
    *,
    data_root: str,
    split: str,
    worker_id: int,
    provider,
    config: dict,
):
    ...
```

`data_root` viene letto da `SCENARIONET_DATA_ROOT`; la directory caricata è:

```text
${SCENARIONET_DATA_ROOT}/runtime/<split>
```

### 18.1 Configurazione base

I nomi esatti delle chiavi devono essere verificati sulla versione installata,
ma la semantica richiesta è:

```yaml
scenario_env:
  data_directory: ${SCENARIONET_DATA_ROOT}/runtime/<split>
  curriculum_level: 1

  horizon: null
  allowed_more_steps: null

  start_scenario_index: 0
  num_scenarios: -1

  worker_index: 0
  num_workers: 1
  sequential_seed: false

  agent_policy: EnvInputPolicy
  discrete_action: false
  set_static: false

  no_map: false
  need_lane_localization: true
  cull_lanes_outside_map: true
  map_region_size: 1024

  no_traffic: false
  no_static_vehicles: false
  no_light: false

  reactive_traffic: true
  filter_overlapping_car: true
  skip_missing_light: true
  static_traffic_object: true

  store_data: false
  store_map: false

  physics_world_step_size: 0.02
  decision_repeat: 5

  crash_vehicle_done: true
  crash_object_done: true
  crash_human_done: true

  out_of_route_done: false
  relax_out_of_road_done: false
  truncate_as_terminate: false

thesis_episode_control:
  extra_steps_after_scenario: 50
```

I time limit nativi `horizon` e `allowed_more_steps` vengono disattivati.
L’unica truncation temporale è quella implementata da `ThesisScenarioEnv`
tramite `extra_steps_after_scenario`.

`relax_out_of_road_done=false` viene mantenuto come configurazione di base,
ma la semantica finale della tesi separa esplicitamente:

```text
uscita fisica dalla superficie stradale → terminale
solo contatto/attraversamento di una linea continua → non terminale
```

La separazione viene realizzata localmente in `ThesisScenarioEnv` usando i
predicati e i flag già disponibili nella versione MetaDrive installata. Non si
introduce un nuovo riconoscitore geometrico avanzato della drivable area.

Le logiche episodiche aggiunte dal subclass sono quindi:

1. l'adattamento circoscritto della condizione `OUT_OF_ROAD`;
2. la truncation esplicita basata su
   `scenario.length + extra_steps_after_scenario`.

Il curriculum nativo resta disattivato dal punto di vista logico: la selezione è controllata dal provider esterno. Ogni processo possiede un environment autonomo; il partizionamento non viene affidato al meccanismo worker interno di ScenarioNet.

`reactive_traffic=true` introduce reattività parziale per i veicoli idonei, prevalentemente tramite comportamento longitudinale rule-based. Non implica replanning multi-agent completo, libera negoziazione delle route o comportamento reattivo completo di pedoni e ciclisti.

### 18.2 Cache

Default:

```yaml
store_data: false
store_map: false
```

Dopo profiling è ammesso abilitare `store_map` soltanto come ottimizzazione prestazionale, senza cambiare i risultati.

### 18.3 Frequenza

```text
physics step = 0,02 s
decision repeat = 5
control step = 0,1 s
control frequency = 10 Hz
```

La frequenza resta uniforme fra Waymo e PG.

### 18.4 Stato fisico dell’ego

Il scene context deve esporre:

```text
ego_length
ego_width
```

La scelta di includere `ego_length` ed `ego_width` nell’osservazione è globale per l’intero esperimento:

- se le dimensioni variano fra gli scenari, entrambe le feature sono sempre presenti per Waymo e PG;
- se l’ego viene standardizzato e le dimensioni sono costanti, entrambe vengono sempre omesse.

La shape dell’osservazione non può variare fra episodi.

## 19. Horizon episodico

Non si imposta un horizon differente in base alla sorgente.

Il limite episodico è:

\[
H_s = length(s) + E
\]

dove:

```text
E = extra_steps_after_scenario
```

Il valore iniziale è:

```yaml
extra_steps_after_scenario: 50
```

A 10 Hz corrisponde a 5 secondi aggiuntivi.

Il limite viene implementato esplicitamente da `ThesisScenarioEnv`:

```python
time_limit_reached = (
    episode_steps
    >= current_scenario_length + extra_steps_after_scenario
)
```

Quando il limite viene raggiunto:

```text
terminated = false
truncated = true
```

La configurazione deve supportare correttamente anche:

```yaml
extra_steps_after_scenario: 0
```

che significa troncare l’episodio alla lunghezza effettiva dello scenario, non disabilitare il limite.

Non si fa affidamento sulla semantica truthy/falsy di `ScenarioEnv.allowed_more_steps` per rappresentare il valore zero.

Il replay buffer deve conservare separatamente `terminated` e `truncated`; la truncation temporale non viene trattata automaticamente come terminale per il bootstrap.

Prima del congelamento definitivo si esegue un pilot visivo su un piccolo campione Waymo `training_20s`:

- si mantiene `50` se il comportamento post-traccia è accettabile;
- si imposta `0` se il traffico oltre la traccia rende lo scenario incoerente;
- non si implementano extrapolatori custom nella v1.

Gli scenari Waymo `training_20s` avranno normalmente una lunghezza vicina a 200 step, ma il codice usa sempre la lunghezza effettivamente esportata nello `ScenarioDescription`.

Gli scenari PG hanno `max_episode_length=500`, ma il limite effettivo resta basato sulla lunghezza esportata dello scenario.

## 20. Termination semantics

`ThesisScenarioEnv` preserva le termination native di `ScenarioEnv` per
collisioni e raggiungimento della destinazione, ma applica una semantica custom
circoscritta alla condizione `OUT_OF_ROAD`.

Configurazione di base:

```yaml
relax_out_of_road_done: false
out_of_route_done: false
truncate_as_terminate: false
```

Il valore di `relax_out_of_road_done` non è la fonte finale della semantica
applicativa: serve come base tecnica, mentre il subclass separa il boundary
fisico dalle linee continue.

### 20.1 Semantica terminale finale

```text
collisione con veicolo                 → terminated
collisione con oggetto                 → terminated
collisione con essere umano / VRU      → terminated
raggiungimento della destinazione       → terminated
uscita fisica dalla superficie stradale → terminated
solo contatto/attraversamento linea continua → non terminated
deviazione dalla route                 → non terminated
limite temporale custom                → truncated
```

La linea continua resta disponibile al rulebook e al logging tramite
`crossed_continuous_line`, ma da sola non termina l'episodio.

### 20.2 Implementazione dell'out-of-road custom

L'implementazione deve distinguere due predicati indipendenti:

```python
crossed_continuous_line = (
    self.scene_context.is_on_continuous_line(vehicle)
)
physical_out_of_road = (
    self.scene_context.is_physically_out_of_road(self, vehicle)
)
```

`is_physically_out_of_road` deve riusare i flag, le collisioni con boundary o le
primitive di lane localization disponibili nella versione MetaDrive locale. Non
deve introdurre nella v1 un nuovo algoritmo geometrico avanzato della superficie
stradale.

Schema concettuale:

```python
native_done, done_info = super().done_function(vehicle_id)
vehicle = self.agents[vehicle_id]

crossed_continuous_line = (
    self.scene_context.is_on_continuous_line(vehicle)
)
physical_out_of_road = (
    self.scene_context.is_physically_out_of_road(self, vehicle)
)

# La sola linea continua non termina l'episodio.
if crossed_continuous_line and not physical_out_of_road:
    done_info[TerminationState.OUT_OF_ROAD] = False

# Dopo avere rimosso OUT_OF_ROAD è obbligatorio ricalcolare il booleano
# complessivo: native_done potrebbe essere True proprio e soltanto a causa
# della linea continua. L'helper deve rispecchiare le terminal keys della
# versione ScenarioEnv locale ed escludere MAX_STEP.
done = self._recompute_terminated_from_info(done_info)

# L'uscita fisica resta terminale, anche se nello stesso step è stata
# attraversata una linea continua.
if physical_out_of_road:
    done_info[TerminationState.OUT_OF_ROAD] = True
    done = True

return done, done_info
```

Non è ammessa la scorciatoia:

```python
physical_out_of_road = native_out_of_road and not crossed_continuous_line
```

perché potrebbe annullare erroneamente una vera uscita fisica quando i due eventi
si verificano nello stesso step. Analogamente, dopo avere disattivato
`OUT_OF_ROAD` per la sola linea continua, non si può restituire invariato il
`native_done`: il booleano terminale deve essere ricalcolato dai singoli motivi
terminali rimasti attivi.

`out_of_route_done=false` mantiene non terminale la sola deviazione dalla route.
La configurazione concreta e la precedenza dei restanti eventi devono essere
lette dal codice locale di `ScenarioEnv`.

### 20.3 Truncation aggiunta dalla tesi

Il limite temporale custom è:

```python
time_limit_reached = (
    episode_steps
    >= current_scenario_length + extra_steps_after_scenario
)
```

Quando il limite viene raggiunto senza che sia già avvenuta una termination:

```text
terminated = false
truncated = true
```

L'implementazione deve preservare il risultato terminale già corretto dalla
semantica precedente e aggiungere soltanto la truncation. Concettualmente:

```python
done, done_info = self._apply_thesis_termination_semantics(vehicle_id)

if not done and self._time_limit_reached(vehicle_id):
    done_info[TerminationState.MAX_STEP] = True

return done, done_info
```

`BaseEnv` ricava così `truncated=true` da `TerminationState.MAX_STEP`, mentre
collisioni, successo e uscita fisica continuano a produrre `terminated=true`.
L'hook concreto deve essere adattato alla firma della versione installata.

## 21. Reward custom e scene context

Il reward già implementato per `MetaDriveEnv` deve essere riutilizzato.

Non si reimplementano in questa fase:

- rulebook;
- reward vector;
- scalarizzazione;
- metriche episodiche delle regole.

### 21.1 Struttura consigliata

```python
class RulebookRewardMixin:
    def reward_function(self, vehicle_id):
        return self.compute_rulebook_reward(vehicle_id)


class ThesisMetaDriveEnv(RulebookRewardMixin, MetaDriveEnv):
    pass


class ThesisScenarioEnv(RulebookRewardMixin, ScenarioEnv):
    pass
```

Se il progetto usa già un wrapper esterno, il wrapper può ricevere `ScenarioEnv` senza cambiare l’interfaccia pubblica.

### 21.2 Adapter di contesto

Le modifiche previste riguardano soltanto l’accesso uniforme ai dati ambientali:

```python
class SceneContextAdapter:
    def get_ego_vehicle(self, env, vehicle_id):
        ...

    def get_route_completion(self, env, vehicle):
        ...

    def is_on_continuous_line(self, vehicle) -> bool:
        ...

    def is_destination_reached(self, env, vehicle) -> bool:
        ...

    def is_physically_out_of_road(self, env, vehicle) -> bool:
        ...

    def get_native_out_of_road(self, env, vehicle) -> bool:
        ...

    def get_termination_reason(self, env, vehicle) -> str | None:
        ...

    def get_traffic_controls(self, env):
        ...

    def get_nearby_agents(self, env, vehicle):
        ...

    def get_ego_dimensions(self, vehicle) -> tuple[float, float]:
        ...
```

`get_native_out_of_road` espone la condizione aggregata restituita dal
`ScenarioEnv` locale. `is_physically_out_of_road` separa la reale uscita dalla
superficie stradale dal solo contatto con una linea continua, riusando primitive
native locali. `is_on_continuous_line` resta disponibile per rulebook e logging e
non è, da sola, una condizione terminale.

### 21.3 Compatibilità semantica da verificare

Differenze probabili:

```text
MetaDriveEnv → NodeNetworkNavigation
ScenarioEnv  → TrajectoryNavigation
```

Devono essere verificati:

- route completion;
- lane identifiers;
- lateral position;
- traffic controls;
- collision flags;
- predicati nativi necessari a distinguere uscita fisica e linea continua;
- continuous-line state diagnostico;
- destination reached;
- termination reason;
- dimensioni dell’ego.

Il reward nativo di `ScenarioEnv` può essere usato soltanto per lo smoke test del caricamento. Il training finale usa il reward custom esistente.

## 22. ScenarioProvider e sampling

La selezione degli scenari resta separata dall’environment.

Interfaccia minima:

```python
class ScenarioProvider:
    def sample(
        self,
        *,
        split: str,
        worker_id: int,
        source: str | None = None,
        arm: str | None = None,
    ) -> ScenarioRecord:
        ...
```

### 22.1 Implementazione v1

```python
class UniformScenarioProvider(ScenarioProvider):
    ...
```

Sampling di training:

```text
source ~ Bernoulli(0.5)
scenario ~ Uniform(valid train scenarios of source)
```

Il provider restituisce il `ScenarioRecord`; l’environment usa il relativo `runtime_index` per il reset.

Il provider opera in modalità stretta:

```yaml
strict: true
allow_fallback: false
```

Se una sorgente richiesta, un arm richiesto o lo split selezionato non contiene
scenari validi, il provider genera un errore esplicito. Non sono ammesse
rinormalizzazioni, sostituzioni di sorgente o fallback silenziosi.

### 22.2 Integrazione con `reset()`

`ThesisScenarioEnv` deve interrogare il provider a ogni reset automatico, incluso l’auto-reset eseguito dal vectorized environment.

Contratto concettuale:

```python
def _select_next_scenario(self, force_runtime_index=None):
    if force_runtime_index is not None:
        runtime_index = force_runtime_index
        record = self.catalog.get_by_runtime_index(runtime_index)
    else:
        record = self.scenario_provider.sample(
            split=self.split,
            worker_id=self.worker_id,
        )
        runtime_index = record.runtime_index

    self.current_scenario_record = record
    return runtime_index
```

L’hook concreto da sovrascrivere deve essere scelto dopo aver ispezionato la versione locale di `ScenarioEnv`.

A ogni reset deve essere verificato che:

```text
runtime_index selezionato
→ scenario caricato
→ scenario_uid atteso
```

Un disallineamento fra catalogo e runtime database deve generare un errore esplicito, non un warning silenzioso.

### 22.3 Evaluation

Validation e test usano un provider dedicato:

```python
class FixedSequenceScenarioProvider(ScenarioProvider):
    def __init__(
        self,
        records: Sequence[ScenarioRecord],
        *,
        repeat: bool = False,
    ):
        ...
```

Contratto:

- lista fissa di `scenario_uid`;
- ordine deterministico;
- ogni scenario eseguito esattamente una volta quando `repeat=false`;
- nessun curriculum;
- nessun sampling casuale;
- nessun fallback;
- nessuna modifica del catalogo;
- errore esplicito quando la sequenza è terminata o contiene record non
  risolvibili.

Nella v1 l'evaluation usa un solo environment oppure una partizione statica
precomputata della sequenza tra worker. Non si usa un coordinatore dinamico.

### 22.4 Futuro ACL

Il futuro:

```python
class ACLScenarioProvider(ScenarioProvider):
    ...
```

userà la stessa interfaccia e potrà ricevere:

```text
scenario_uid
source
primary_arm
tags
learning_potential
last_sampled_step
```

L’adattamento dell’ACL esistente al multi-environment non è requisito di questa integrazione v1. Non si implementa ora un coordinatore distribuito del curriculum.

## 23. Vectorized environments e seed

### 23.1 Architettura

Usare:

```text
un ScenarioEnv per processo
SubprocVecEnv o equivalente process-based
```

Metodo di avvio consigliato:

```text
spawn
```

Ogni processo possiede il proprio environment e il proprio provider/sampler locale nella v1.

Non si usa il meccanismo multi-worker interno di `ScenarioEnv` come sistema principale di partizionamento.

### 23.2 Seed per worker

```python
worker_rng = np.random.default_rng(
    np.random.SeedSequence([global_seed, worker_id])
)
```

La sequenza esatta è garantita a parità di seed, numero di worker e configurazione. Non si richiede determinismo bitwise fra macchine o versioni differenti.

### 23.3 Duplicati simultanei

È ammesso che due worker selezionino occasionalmente lo stesso scenario.

Non si introducono lock, code globali o sincronizzazione per impedirlo.

### 23.4 Cleanup

Devono esistere test che verifichino:

- chiusura corretta degli environment;
- terminazione dei subprocess;
- assenza di errori sistematici dopo reset ripetuti.

---

## 24. Logging minimo

Il logging deve conservare le informazioni necessarie alle analisi della tesi e alla riproduzione dei fallimenti.

### 24.1 Per episodio

```text
scenario_uid
scenario_id
source
split
primary_arm
worker_id
episode_length
episode_return
success
collision
out_of_road
crossed_continuous_line
route_completion
termination_reason
terminated
truncated
```

Quando l’ACL sarà attivo si aggiungeranno:

```text
learning_potential
sampling_mode
```

### 24.2 Per run

```text
reset per source
step per source
episodi per arm
```

### 24.3 Diagnostica dataset PG

La pipeline di preparazione, separata dai log di training, produce:

```text
profile → resulting arm matrix
invalid rate by profile
arm distribution by source
```

Non si aggiungono metriche ridondanti che non verranno utilizzate nelle analisi, ma gli identificativi necessari a riprodurre uno scenario o una sequenza di sampling sono obbligatori.

## 25. Moduli software previsti

Struttura indicativa:

```text
src/
├── scenarios/
│   ├── catalog.py
│   ├── records.py
│   ├── splits.py
│   ├── runtime_database.py
│   ├── validation.py
│   ├── features.py
│   ├── arms.py
│   ├── provider.py
│   └── manifests.py
│
├── scenarios/pg/
│   ├── profiles.py
│   ├── generator.py
│   ├── exporter.py
│   └── report.py
│
├── envs/
│   ├── thesis_scenario_env.py
│   ├── scenario_env_factory.py
│   ├── episode_control.py
│   ├── scene_context_adapter.py
│   └── rulebook_reward_mixin.py
│
└── scripts/
    ├── convert_waymo.py
    ├── validate_database.py
    ├── generate_pg_dataset.py
    ├── build_catalog.py
    ├── build_splits.py
    ├── build_runtime_databases.py
    ├── compute_arm_thresholds.py
    └── smoke_test_scenarios.py
```

Gli script devono essere CLI riproducibili e supportare conteggi piccoli per lo sviluppo. Codex non deve eseguire la generazione completa come parte obbligatoria dell’implementazione.

## 26. Sequenza di implementazione

### Fase 1 — Bootstrap

1. usare il submodule ScenarioNet corrente;
2. verificare le API e dipendenze MetaDrive/ScenarioNet;
3. creare il manifest incompleto;
4. caricare un singolo scenario;
5. effettuare reset e breve rollout;
6. congelare le versioni risultanti.

### Fase 2 — Waymo

1. acquisire Waymo `training_20s`;
2. convertire tramite ScenarioNet;
3. eseguire la validazione ufficiale disponibile;
4. estrarre il più forte identificativo di gruppo disponibile;
5. costruire gli split interni train, validation e test;
6. verificare l’assenza di overlap;
7. creare record del catalogo e viste runtime.

### Fase 3 — PG e pilot

1. implementare P0, P1, P2, P3 e P5 tramite configurazioni native;
2. implementare `--count` e intervalli di seed;
3. generare pochi scenari smoke durante lo sviluppo;
4. permettere all’utente di avviare il pilot da 20 scenari per profilo;
5. produrre la matrice diagnostica profilo-arm.

### Fase 4 — Dataset e split

1. generare successivamente il numero prefissato di scenari PG validi;
2. costruire train, validation e test tramite seed disgiunti;
3. costruire le viste runtime ScenarioNet;
4. validare assenza di overlap.

### Fase 5 — Feature e arms

1. estrarre le feature Waymo + PG;
2. calcolare Q40/Q75 esclusivamente sul train candidate bilanciato per sorgente;
3. assegnare tag e primary arm;
4. salvare distribuzioni e soglie;
5. non modificare il dataset per rendere uguali gli arms.

### Fase 6 — Environment unificato

1. implementare `ThesisScenarioEnv`;
2. collegare il reward esistente;
3. preservare le termination native di collisione e destinazione;
4. configurare `relax_out_of_road_done=false` e `out_of_route_done=false`;
5. separare, tramite i predicati locali disponibili, l'uscita fisica dalla sola
   linea continua;
6. rendere terminale l'uscita fisica e non terminale la sola linea continua;
7. implementare la truncation custom
   `scenario.length + extra_steps_after_scenario`;
8. distinguere termination e truncation;
9. implementare la factory.

La modifica è circoscritta all'adapter e al subclass e non introduce un nuovo
riconoscitore geometrico avanzato della drivable area.

### Fase 7 — Provider e vectorization

1. implementare `UniformScenarioProvider` in modalità stretta e senza fallback;
2. implementare `FixedSequenceScenarioProvider`;
3. implementare il 50/50 per reset;
4. integrare il provider nell’hook di reset di `ScenarioEnv`;
5. verificare `runtime_index → scenario_uid`;
6. verificare l’auto-reset con `SubprocVecEnv`;
7. collegare gli environment process-based;
8. verificare seed, cleanup ed evaluation deterministica.

### Fase 8 — Smoke training

1. random policy;
2. policy nominale dove applicabile;
3. training breve con reward custom;
4. controllo reset errors;
5. controllo del logging minimo;
6. controllo RAM e chiusura processi.

## 27. Test obbligatori

### 27.1 Unit test

```text
catalog serialization/deserialization
relative-path resolution
split grouping
no split overlap
runtime-index resolution
runtime_index → scenario_uid consistency
feature extraction con topology unknown
feature extraction con topology mixed
arm assignment
A1 basato su relevant_vehicles_q90
signal_reliability not_applicable senza warning
has_unknown_signal vero soltanto per partial/missing
threshold computation
seed reproducibility
provider source balance
provider invoked on automatic reset
extra_steps_after_scenario=0
extra_steps_after_scenario=50
physical out-of-road predicate
continuous-line non-terminal override
native collision/destination termination passthrough
provider strict no-fallback behavior
fixed-sequence provider exhaustion
reward adapter
ego dimensions availability
```

### 27.2 Integration test

```text
load Waymo training_20s scenario
load PG scenario
same action-space shape
same observation-space shape
same reward interface
same terminated/truncated schema
native collision/destination termination preserved
custom physical-out-of-road/continuous-line separation
custom scenario-length truncation
vectorized reset and close
fixed evaluation sequence
```

### 27.3 Causal leakage test

Il codice dell’observation e del reward deve verificare che:

- nessun indice temporale maggiore del current step venga letto per costruire
  feature dinamiche di ego, agenti o traffic controls;
- nessuna future track venga passata alla policy;
- la geometria di navigazione futura della route ego possa essere letta soltanto
  come path geometrico, senza preservarne l'indicizzazione temporale;
- la route ego eventualmente esposta sia priva di timestamp, velocità,
  accelerazioni, azioni o validity mask future;
- nessuna etichetta di arm o sorgente venga inserita nell’osservazione;
- nessuna future signal phase venga usata.

Le tracce complete restano ammesse soltanto per feature extraction offline e catalogazione.

### 27.4 Termination e truncation test

Casi minimi:

```text
collisione → terminated=true secondo la termination nativa
raggiungimento destinazione → terminated=true secondo la termination nativa
linea continua senza uscita fisica → terminated=false
uscita fisica dalla superficie stradale → terminated=true
linea continua + uscita fisica nello stesso step → terminated=true
deviazione dalla route con out_of_route_done=false → terminated=false
max episode steps → truncated=true e terminated=false, se non già terminato
extra_steps_after_scenario=0 → truncation alla lunghezza esportata
extra_steps_after_scenario=50 → truncation alla lunghezza esportata + 50
```

I test devono verificare la semantica applicativa della tesi e non limitarsi a
replicare il comportamento aggregato nativo di `ScenarioEnv`.

### 27.5 Split test

```text
all Waymo records originate from training_20s
no source group shared across thesis train/validation/test
thesis test never used by threshold or profile calibration scripts
```

### 27.6 Validation test

Per un campione Waymo e PG:

```text
official simulation check passed, se disponibile
reset passed
10-step rollout passed
reward finite
observation finite
termination/truncation finite
```

## 28. Definition of done

L’integrazione ScenarioNet v1 è completata quando:

1. Waymo e PG vengono caricati tramite la stessa API applicativa.
2. Entrambi usano la stessa action space e observation interface.
3. Il reward custom esistente funziona su entrambe le sorgenti.
4. La data root è configurabile tramite `.env`.
5. Le viste runtime train, validation e test risolvono correttamente i record del catalogo.
6. Tutti gli scenari Waymo provengono da `training_20s` e gli split interni non condividono gruppi.
7. Gli split PG usano seed disgiunti.
8. Gli scenari inclusi hanno superato la validazione richiesta.
9. Le feature topologiche usano soltanto dati disponibili; i casi incerti diventano `unknown` e quelli realmente misti diventano `mixed`.
10. Gli arms definitivi A0–A5 vengono assegnati senza dipendere dal profilo PG dichiarato.
11. A1 usa `relevant_vehicles_q90 > 0`, non la sola presenza di un veicolo nel file.
12. Q40 e Q75 sono calcolati soltanto sul train candidate e riutilizzati in evaluation.
13. `ThesisScenarioEnv` usa `reactive_traffic=true` e preserva le termination native di collisione e destinazione.
14. La sola linea continua è non terminale, mentre l'uscita fisica dalla superficie stradale è terminale.
15. La separazione usa primitive native locali e non introduce un riconoscitore geometrico avanzato della drivable area.
16. `relax_out_of_road_done=false`, `out_of_route_done=false` e `truncate_as_terminate=false` sono applicati secondo le API locali e corretti dalla semantica del subclass.
17. L’horizon è `scenario.length + extra_steps_after_scenario`, con supporto corretto sia per `50` sia per `0`.
18. `terminated` e `truncated` restano distinti nel training loop e nel replay buffer.
19. Ogni scenario selezionato dal provider corrisponde allo `scenario_uid` caricato.
20. `UniformScenarioProvider` realizza il 50/50 per reset in modalità stretta e senza fallback.
21. `FixedSequenceScenarioProvider` realizza validation e test in ordine fisso senza fallback.
22. Gli environment vectorizzati funzionano a processi e si chiudono correttamente.
23. L’evaluation usa una sequenza fissa.
24. Il logging conserva le metriche e gli identificativi minimi definiti.
25. Nessuna informazione futura o metadato del curriculum viene esposto alla policy.
26. Gli script di conversione, generazione PG, validazione, catalogazione e split sono eseguibili con conteggi piccoli e completi.
27. Una smoke run con il training loop completo termina senza errori sistematici.
28. Manifest, catalogo, split e soglie sono salvati e associati alla run.

## 29. Elementi deliberatamente data-dependent

La sorgente Waymo della v1 è già congelata come:

```text
training_20s
```

I seguenti valori vengono invece determinati durante l’implementazione o la preparazione del dataset:

```text
MetaDrive version e commit
ScenarioNet commit già checkoutato
Waymo release effettivamente disponibile
converter version
nomi esatti delle chiavi API
road block PG nativi disponibili
topology metadata realmente prodotti dal converter
strongest available Waymo grouping identifier
tau_low
tau_dense
extra_steps_after_scenario = 50 oppure 0 dopo il pilot visivo
semantica concreta del done_function nativo della versione MetaDrive installata
```

Non sono decisioni scientifiche aperte.

Le regole sono:

```text
dettaglio API → ispezionare la versione locale e adattare il codice;
feature non disponibile in modo affidabile → usare unknown;
funzionalità PG non nativa → non implementare un sottosistema custom nella v1;
valore data-dependent → calcolarlo o verificarlo con la procedura prevista e congelarlo nel manifest.
```

Una volta prodotti manifest e dataset, tali valori restano congelati durante gli esperimenti principali.

## 30. Configurazione riepilogativa

```yaml
project:
  setting: single_agent_post_perception
  action_space: continuous

data:
  root_env_variable: SCENARIONET_DATA_ROOT
  format: ScenarioDescription
  runtime_views: [train, validation, test]

software:
  use_checked_out_scenarionet_submodule: true
  freeze_after_first_successful_smoke_test: true
  allow_arbitrary_submodule_update: false

dataset:
  source_reset_probability:
    waymo: 0.5
    pg: 0.5

  waymo_split_policy:
    source: training_20s
    train: internal_grouped_split
    validation: internal_grouped_split
    test: internal_grouped_split
    exclude_official_validation: true
    exclude_official_testing: true
    group_by: strongest_available_source_group

  train:
    waymo: 1000
    pg: 1000
  validation:
    waymo: 250
    pg: 250
  test:
    waymo: 500
    pg: 500

training:
  total_timesteps: 1_500_000

scenario_env:
  reactive_traffic: true
  control_frequency_hz: 10
  horizon: null
  allowed_more_steps: null
  out_of_route_done: false
  relax_out_of_road_done: false
  truncate_as_terminate: false
  crash_vehicle_done: true
  crash_object_done: true
  crash_human_done: true

  scenario_selection:
    start_scenario_index: 0
    num_scenarios: -1
    worker_index: 0
    num_workers: 1
    sequential_seed: false

  episode_control:
    extra_steps_after_scenario: 50

  termination:
    preserve_native_collision_and_destination: true
    use_custom_out_of_road_separation: true
    continuous_line_only_is_terminal: false
    physical_out_of_road_is_terminal: true
    collision_vehicle: native
    collision_object: native
    collision_human: native
    destination_reached: native

  cache:
    store_data: false
    store_map: false

reward:
  use_existing_rulebook_implementation: true
  use_native_scenario_env_reward: false

scene_context:
  expose_ego_length_width: true
  include_in_policy_observation_if_variable: true
  observation_shape_constant_across_episodes: true
  route:
    use_scenario_env_navigation_geometry: true
    expose_time_indexed_future_trajectory: false
    semantic_route_tokens_defined_later: true

pg:
  generation: offline
  use_native_metadrive_blocks_only: true
  max_episode_length: 500
  accepted_train_target: 1000
  pilot_scenarios_per_profile: 20
  profiles:
    - P0_simple
    - P1_vehicle_interaction
    - P2_merge_or_roundabout
    - P3_intersection
    - P5_complex_mixed
  mutation: false
  accident_prob_core_profiles: 0.0
  enforce_minimum_scenarios_per_arm: false

arms:
  names:
    - A0_simple_lane_follow
    - A1_vehicle_interaction
    - A2_merge_or_roundabout
    - A3_intersection
    - A4_vru_interaction
    - A5_complex_mixed

  relevant_agents:
    radius_m: 50.0
    vertical_tolerance_m: 3.0
    temporal_quantile: 0.90

  traffic_thresholds:
    low_quantile: 0.40
    dense_quantile: 0.75
    compute_on: train_candidate_only
    balance_sources: true

  vru_interaction:
    max_distance_to_route_m: 8.0

provider:
  strict: true
  allow_fallback: false
  training: UniformScenarioProvider
  evaluation: FixedSequenceScenarioProvider

vectorization:
  process_based: true
  start_method: spawn
  external_provider: true
  allow_simultaneous_duplicate_scenarios: true

logging:
  per_episode:
    - scenario_uid
    - scenario_id
    - source
    - split
    - primary_arm
    - worker_id
    - episode_length
    - episode_return
    - success
    - collision
    - out_of_road
    - crossed_continuous_line
    - route_completion
    - termination_reason
    - terminated
    - truncated
  per_run:
    - resets_per_source
    - steps_per_source
    - episodes_per_arm

validation:
  official_scenarionet_when_available: true
  thesis_smoke_validation: true
```

La configurazione riepilogativa rappresenta lo schema applicativo del progetto. La factory deve passare a MetaDrive soltanto le chiavi supportate dalla versione locale; sezioni descrittive come `termination` ed `episode_control` non devono essere inoltrate direttamente come chiavi sconosciute.

## 31. Decisione finale

La pipeline v1 è definita come:

```text
checked-out ScenarioNet submodule + compatible local MetaDrive
→ Waymo training_20s
→ ScenarioNet conversion
→ internal grouped train/validation/test split
→ MetaDrive PG offline with native blocks only
→ ScenarioDescription
→ official validation where available
→ thesis smoke validation
→ unified ScenarioCatalog with relative paths
→ runtime train/validation/test views
→ feature extraction based on realized scenario metadata
→ train-only Q40/Q75 thresholds
→ tags + six primary arms
→ strict UniformScenarioProvider controlling training resets
→ FixedSequenceScenarioProvider for validation/test
→ ThesisScenarioEnv
→ existing custom reward
→ native collision/destination termination
→ custom physical-out-of-road / continuous-line separation
→ explicit scenario-length truncation
→ separate terminated/truncated semantics
→ process-based vectorized training
```

Questa specifica congela l’integrazione ScenarioNet v1 mantenendo un’implementazione fattibile e proporzionata alla tesi.

Non fanno parte della v1:

```text
riconoscimento geometrico topologico avanzato
nuovo riconoscitore geometrico avanzato custom della drivable area
generazione custom di semafori, VRU o precedenze
mutation degli scenari
ACL multi-worker
ottimizzazione sistematica della purezza dei profili PG
```

Su questa base verranno successivamente integrati:

```text
scene context completo
rulebook definitivo
semantic observation
encoder
baseline scalarizzata
automatic curriculum learning
algoritmi lessicografici
algoritmi distribuzionali
```
