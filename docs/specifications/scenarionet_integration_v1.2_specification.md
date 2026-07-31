# Specifica di emendamento — Holdout empirici, doppio panel di test e grouping Waymo per co-locazione di mappa

## Metadata

- **Feature:** empirical holdout policy, dual test panels, arm-minimum training pool, map-based Waymo grouping
- **Specification ID:** `SCENARIONET-INTEGRATION`
- **Version:** `1.2`
- **Status:** `APPROVED`
- **Date:** `2026-07-31`
- **Supersedes:** `docs/specifications/scenarionet_integration_v1.1_specification.md`, version `1.1` (solo per i sottoinsiemi §5, §6.1, §6.2, §6.4, §6.5, §6.6, §6.8 modificati da questa versione; il resto di v1.1 resta autoritativo e invariato, incluse §1-§4, §6.3, §6.7, §7-§12)
- **Related specifications:** `docs/specifications/evaluation_protocol_v1.1_specification.md` (emenda separatamente `EVAL-PROTOCOL`, introducendo i nuovi panel usati da questo documento)
- **Related ADRs:** ADR-001, ADR-009, ADR-012 (materialmente superseduta per gli split di validation/test); ADR-037, ADR-038
- **Related ExecPlan:** `docs/implementation/empirical_holdout_split_and_dual_test_panels_v1.2_exec_plan.md`
- **Authoritative:** `YES`

### Registro di approvazione

- **Date:** `2026-07-31`
- **Evidence:** explicit user approval in this conversation ("mi sembra vada
  tutto bene") after review of the drafted `UNDER_REVIEW` document and its
  companion `EVAL-PROTOCOL` amendment.
- **Approved scope:** the complete v1.2 amendment as drafted, including the
  decisions recorded in §2 (`DEC-001`..`DEC-007`).

## 1. Motivazione

L'analisi che precede questo documento (registrata nella sessione di
conversazione del 2026-07-31 e nell'ExecPlan collegato) ha verificato tre
difetti nella v1.1:

1. **Gli holdout non sono empirici.** La politica `balanced_arm_source`
   impone quote near-uniform `A0`–`A5` a validation e test esattamente come
   al training (v1.1 §5.1, §6.6). Validation e test misurano quindi una
   macro-performance uniforme sulle competenze, non la prestazione attesa
   sulla distribuzione eligible osservata di ciascuna sorgente. Questo è un
   benchmark legittimo, ma finora non esiste alcun endpoint di
   generalizzazione empirica accanto ad esso.
2. **I panel di valutazione ribilanciano una seconda volta.** Il draw dei
   panel di validation/test (100/300 episodi, `EVAL-PROTOCOL` v1.0 §14) è
   già bilanciato per arm indipendentemente dallo split sottostante. Un
   cambiamento della sola politica di split, senza toccare i panel, non
   avrebbe alcun effetto osservabile.
3. **Il group-disjointness Waymo è nella pratica assente.** v1.1 §6.1
   prevede la chiave di gruppo `source_log_id`/`segment_id` con fallback
   `f"scenario:{uid}"` quando non è disponibile un identificativo superiore.
   Nei dati convertuti disponibili, il campo `source_log_id` coincide sempre
   con il nome dello shard TFRecord, cosa che il codice tratta esplicitamente
   come non affidabile e scarta, ricadendo sul fallback per-scenario — cioè
   nessun raggruppamento reale. Questo è conforme a v1.1 §6.1 (che anticipava
   l'assenza di un identificativo migliore), ma rende qualunque endpoint
   empirico ottimista per leakage geografico.

Questo documento introduce: una politica di riserva "holdout-first" che
seleziona validation e test empirici prima di guardare le etichette
`A0`–`A5`; un pool di test stratificato per competenza come endpoint
separato; un pool di training con minimi per arm invece di quote esatte; e
una chiave di raggruppamento Waymo basata sull'identità di mappa, adottata
incondizionatamente per prudenza scientifica (non subordinata a una stima di
magnitudine — vedi §3.5).

## 2. Decisioni di revisione approvate

Le decisioni seguenti sono state esplicitamente approvate dall'utente il
2026-07-31, nella sessione di conversazione che ha originato questo
documento, e sono registrate anche nell'ExecPlan collegato (`DEC-001`..
`DEC-007`):

- **`DEC-001`**: doppio endpoint di test — `test_waymo_empirical` (primario,
  senza bilanciamento per arm) e `test_arm_stratified` (endpoint di
  competenza, bilanciato), più `test_pg` (secondario, generalizzazione
  procedurale). Non un test empirico soltanto, né lo status quo bilanciato
  soltanto.
- **`DEC-002`**: la chiave di raggruppamento Waymo per co-locazione di mappa
  è adottata incondizionatamente, indipendentemente dalla stima di quanti
  scenari fossero effettivamente a rischio di leakage. Il costo di
  costruirla è lo stesso costo dello strumento usato per *misurarla*; non
  esiste uno scenario in cui attendere la misura cambierebbe la decisione
  implementativa.
- **`DEC-003`**: la miscela dei profili di generazione PG per gli holdout è
  **equiprobabile**: 20% ciascuno per `P0_simple`, `P1_vehicle_interaction`,
  `P2_merge_or_roundabout`, `P3_intersection`, `P5_complex_mixed`. La
  distribuzione di arm che ne risulta è una conseguenza osservata del
  generatore sotto questa miscela dichiarata, mai un target di selezione.
- **`DEC-004`**: `checkpoints/final.zip` resta l'unico checkpoint ufficiale
  (`EVAL-PROTOCOL` `DEC-002`/`REQ-006`). Questo documento non tocca la
  politica di checkpoint.
- **`DEC-005`**: dimensioni di pool e panel come da tabella in §3.3.
- **`DEC-006`**: cadenza di validazione periodica differenziata per panel —
  `validation_waymo_empirical` ogni 25.000 step (curva primaria),
  `validation_pg` ogni 100.000 step (diagnostica).
- **`DEC-007`**: risolto per via fattuale — il pool convertito
  (`catalog/scenario_catalog.parquet`, database Waymo e PG) è confermato
  presente e riusabile; non è richiesta una nuova campagna di acquisizione.

## 3. Emendamenti

### 3.1 Emendamento a §5 — Dimensioni previste e budget

Il paragrafo §5.1 e la relativa tabella di v1.1 sono sostituiti da:

```yaml
training:
  total_timesteps: 1_500_000

dataset:
  test_empirical:
    waymo: 400
    pg: 300
  test_stratified:
    waymo: ~180   # vincolato dalla capacità eligible Waymo per A0/A1/A2, vedi §3.6
    pg: ~120
  validation:
    waymo: 150
    pg: 150
  train:
    waymo: residuo dopo la riserva degli holdout
    pg: residuo dopo la riserva degli holdout
```

| Pool | Waymo | PG | Panel derivati |
|---|---:|---:|---|
| `test_empirical` | 400 | 300 | `test_waymo_empirical` 300 (primario), `test_pg` 200 (secondario) |
| `test_stratified` | ~180 | ~120 | `test_arm_stratified` 300 (50 per arm) |
| `validation` | 150 | 150 | `validation_waymo_empirical` 100 (curva primaria), `validation_pg` 100 (diagnostica) |
| `train` | residuo | residuo | minimi per arm, non quote esatte — vedi §3.5 |

Il budget totale resta 3500 scenari salvo espansione futura approvata. A
differenza di v1.1, dove 700 dei 1000 scenari di test congelati non erano
mai valutati (i panel pescano solo 300 scenari su 1000), questa
configurazione dimensiona i pool sulla base di ciò che viene effettivamente
valutato, restituendo capacità al training.

Il §5.2 (profili di sviluppo) resta invariato.

### 3.2 Emendamento a §6.1 — Waymo

Il paragrafo "Ordine di preferenza per la chiave di gruppo" di v1.1 §6.1 è
sostituito da:

```text
1. source_log_id / segment_id, se dimostra di identificare un log o segmento
   condiviso (non un nome di shard TFRecord)
2. identità di mappa: fingerprint deterministico calcolato sulle polilinee di
   map_features dello scenario convertito, quantizzate spazialmente
3. scenario_id, soltanto se né 1 né 2 sono disponibili
```

Il fingerprint di mappa (livello 2) è la chiave di gruppo effettiva per la
popolazione Waymo `training_20s` disponibile a questo progetto, poiché il
converter checkoutato non espone un identificativo di log/segmento
affidabile (v1.1 §6.1, invariato). La definizione esatta del fingerprint
(campi usati, quantizzazione, funzione di hash) è fissata nell'ExecPlan
collegato e registrata nel manifest dello split insieme all'evidenza che ne
giustifica l'adozione (§3.6).

Il resto di §6.1 (esclusione degli split ufficiali Waymo `validation`/
`testing`, uso esclusivo di `training_20s`, divieti sull'uso del test) resta
invariato.

### 3.3 Emendamento a §6.2 — Procedural generation

A v1.1 §6.2 si aggiunge, prima di "Non è richiesta una deduplicazione
geometrica sofisticata":

```text
La miscela dei profili di generazione per gli holdout PG (validation e
ciascun pool di test) è dichiarata e congelata prima della generazione:
20% per ciascuno dei cinque profili (P0_simple, P1_vehicle_interaction,
P2_merge_or_roundabout, P3_intersection, P5_complex_mixed). Questa miscela
è indipendente da qualunque deficit di arm osservato; la distribuzione di
arm risultante è riportata post hoc, mai usata per aggiustare la miscela.

Il pool di training PG può continuare a usare il replenishment mirato ai
deficit (ADR-008, ADR-009), perché il training non richiede una
distribuzione empirica.

I seed degli holdout PG occupano un intervallo disgiunto da ogni altro
intervallo di seed usato dal progetto (train, run precedenti, sviluppo).
```

### 3.4 Emendamento a §6.4 — Manifest degli split

Lo schema di `split_manifest.yaml` di v1.1 §6.4 è esteso con i campi
seguenti (i campi esistenti restano invariati salvo dove indicato):

```yaml
split_seed: 0
split_policy: holdout_first_empirical_then_stratified_then_balanced_train

holdout_policy:
  order: [test_empirical, validation_empirical, test_stratified, train]
  empirical_draw:
    permutation_seed: 0
    label_blind: true          # l'allocatore non legge primary_arm/tags
  pg_holdout_mixture:
    P0_simple: 0.20
    P1_vehicle_interaction: 0.20
    P2_merge_or_roundabout: 0.20
    P3_intersection: 0.20
    P5_complex_mixed: 0.20
  pg_holdout_seed_range: null   # popolato alla generazione

grouping:
  waymo: map_identity_fingerprint   # sostituisce source_log_or_segment_else_scenario_id
  waymo_fingerprint_evidence: null  # popolato dall'audit di co-locazione
  pg: generation_seed

counts:
  test_empirical:
    waymo: 400
    pg: 300
  test_stratified:
    waymo: null   # vincolato dalla capacità eligible, popolato alla selezione
    pg: null
  validation:
    waymo: 150
    pg: 150
  train:
    waymo: null   # residuo
    pg: null      # residuo

balancing:
  train_arm_policy: minimum_per_arm   # sostituisce near_uniform per il train
  train_arm_minimums: {}              # popolato dalla configurazione pipeline
  stratified_test_arm_targets: near_uniform  # invariato, solo sul pool stratificato
  max_arm_count_difference: 1         # si applica solo al pool stratificato
  source_target_within_arm: best_effort_50_50   # si applica solo al pool stratificato
  preserve_exact_source_totals: true  # si applica a tutti i pool
  structural_empty_cells:
    A4_vru:
      pg: true
  allow_cross_source_fill_within_same_arm: true
  allow_relabeling: false
  allow_duplicate_records: false
  allow_quality_filter_relaxation: false

selection_report:
  requested_by_split_source_arm: {}
  selected_by_split_source_arm: {}
  deficits_by_split_source_arm: {}
  source_compensation_by_split_arm: {}
  total_arm_deficit: null
  observed_arm_distribution_by_pool: {}   # nuovo: reporting post hoc obbligatorio

waymo_acquisition:
  ordering_seed: 0
  batch_size_shards: 64
  max_new_shards: 256
  processed_shards: []
  stop_reason: null

catalog_hash: null
created_at: null
```

### 3.5 Emendamento a §6.5/§6.6 — Politica di selezione

Il paragrafo "Candidate pool, eligible pool e selected split" di v1.1 §6.5
resta valido come descrizione delle tre popolazioni, con l'aggiunta di uno
stadio intermedio:

```text
candidate pool
→ quality, reliability and Rulebook eligibility filters
→ eligible pool
→ deterministic pseudo-random permutation of GROUPS (per source)
→ reserve test_empirical groups, labels ignored
→ reserve validation_empirical groups, labels ignored
→ FREEZE empirical holdouts
→ reserve test_stratified groups from the residual, arm-balanced (v1.1 §6.6 policy, unchanged)
→ FREEZE stratified holdout
→ build training pool from the residual, per-arm MINIMUMS + exact per-source totals
→ selected train / validation / test pools
```

Il §6.6 "Politica `balanced_arm_source`" di v1.1 resta autoritativo e
invariato **come politica del pool `test_stratified`**. Per il pool di
`train`, il vincolo 5 ("conteggio quasi uniforme A0–A5") è sostituito da:

```text
5'. minimo configurato per cella source × arm, non uguaglianza dei conteggi
```

I vincoli 1-4 e 6 di §6.6 restano invariati per il train. La classificazione
`A0`–`A5` continua a precedere qualunque selezione (v1.1 §6.5, invariato); la
riserva degli holdout empirici, in particolare, non legge mai
`primary_arm`, `tags`, `low_traffic`, o `dense_traffic` — l'allocatore
riceve una proiezione del record priva di quei campi.

Ordine di congelamento vincolante: test empirico, poi validation empirico,
poi test stratificato, poi train. Nessuno stadio successivo può spostare un
record in uno stadio precedente. Un'acquisizione successiva al congelamento
può popolare soltanto il training pool (v1.1 §6.7, invariato).

### 3.6 Emendamento a §6.8 — Promozione del test Waymo empirico

Il paragrafo "Test Waymo-natural opzionale" di v1.1 §6.8 è sostituito da:

```text
Il pool test_empirical Waymo (§3.1) è un endpoint di valutazione primario,
non più un holdout diagnostico opzionale disabilitato di default. Il suo
contratto resta quello già definito da v1.1 §6.8:

- costruito esclusivamente da gruppi Waymo eleggibili non presenti in
  nessun altro pool primario;
- nessun ribilanciamento per arm nella selezione;
- selezione deterministica e group-disjoint, secondo la chiave di gruppo
  di §3.2;
- nessun uso per calibrazione, checkpoint selection o tuning;
- metriche riportate separatamente dal test stratificato.

Prima della sua costruzione, un audit di co-locazione di mappa (registrato
nell'ExecPlan collegato) verifica quanti gruppi Waymo dell'eligible pool
condividono geometria stradale sotto la chiave di raggruppamento
precedente (source_log_id degradato a per-scenario). Il numero risultante
è riportato come evidenza descrittiva nella sezione limitazioni della tesi;
non è un prerequisito per l'adozione della nuova chiave di gruppo (§3.2,
DEC-002), che si applica incondizionatamente.
```

## 4. Capacità nota e vincolo dichiarato

Il pool eligible Waymo attualmente disponibile contiene, per gli arm a bassa
numerosità:

| Arm | Scenari Waymo eligible |
|---|---:|
| `A0_simple_low_traffic` | 26 |
| `A1_traffic` | 143 |
| `A2_junction` | 144 |

Questi valori sono derivati dal dataset congelato v1.1 (`selection_hash`
`7c0d6f3b...`): per questi tre arm il selettore bilanciato era
capacity-limited, non quota-limited, quindi ha selezionato l'intera
capacità disponibile.

Conseguenza dichiarata: il pool `test_stratified` con target 50 scenari per
arm **non potrà raggiungere 50 scenari Waymo per `A0`**; la cella
`A0 × Waymo` resterà strutturalmente scarsa e sarà compensata da PG secondo
la politica invariata di §6.6 (vincolo 6, primo rilassabile). Questo va
riportato nel manifest, non corretto riclassificando scenari.

## 5. Verifica degli split (estensione di §6.3)

Oltre alle intersezioni già richieste da v1.1 §6.3, la pipeline deve
verificare:

```text
intersection(test_empirical_group_ids, validation_empirical_group_ids) == empty
intersection(test_empirical_group_ids, test_stratified_group_ids) == empty
intersection(test_empirical_group_ids, train_group_ids) == empty
intersection(validation_empirical_group_ids, test_stratified_group_ids) == empty
intersection(validation_empirical_group_ids, train_group_ids) == empty
intersection(test_stratified_group_ids, train_group_ids) == empty
```

cioè disgiunzione a coppie fra tutti e quattro i pool primari, non solo fra
i tre split nominali di v1.1.

## 6. Compatibilità e impatto

Rottura per costruzione: il dataset risultante ha un nuovo `selection_hash`
e un nuovo `split_policy`. Le run prodotte contro il dataset v1.1 congelato
(`selection_hash` `7c0d6f3b15ca69ff79dbb3a0daefefdae326254065a9792c0b73af1ffa730fb5`,
creato 2026-07-31) non possono essere riprese né aggregate con run sul
dataset v1.2. Questo è accettabile soltanto perché gli esperimenti ufficiali
non sono ancora iniziati; va riconfermato dall'utente al momento
dell'approvazione finale, non assunto da questo documento.

## 7. Non modificato da questa revisione

Restano invariati e non toccati da questo emendamento: §1-§4 (formato dati,
directory, generazione PG offline salvo §3.3), §6.3 (verifica base degli
split, estesa non sostituita da §5), §6.6 come politica del pool
stratificato, §6.7 (acquisizione incrementale Waymo), §7-§12 (feature
extraction, arm assignment, caricamento/sampling, configurazione
`ScenarioEnv`, horizon/termination, esecuzione vectorized, logging/test).
Le soglie fisse degli arm (`A0`-`A5`) e le soglie diagnostiche Q40/Q75 non
sono toccate da questo documento.
