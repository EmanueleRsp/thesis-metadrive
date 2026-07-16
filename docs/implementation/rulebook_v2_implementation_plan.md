# Piano di implementazione e tracker — Rulebook v2

## Scopo

Questo documento traduce
[`rulebook_v2_spec.md`](../specifications/rulebook_v2_spec.md) in un piano operativo e mantiene lo
stato di avanzamento dell'implementazione.

La specifica resta la fonte normativa per formule, dominio delle regole,
parametri e comportamento di errore. Questo tracker registra:

- ordine e stato delle attività;
- decisioni implementative congelate;
- criteri di completamento ed evidenze;
- problemi tecnici e rischi data-dependent;
- eventuali deviazioni dalla specifica;
- decisioni che richiedono una nuova discussione.

Le decisioni della sezione **Decisioni congelate** risolvono dettagli lasciati
impliciti dalla specifica senza cambiarne la semantica. Dovranno essere riportate
in una successiva revisione della specifica prima del freeze finale.

---

## Stato complessivo

**Stato:** implementazione in corso  
**Fase corrente:** F4 — snapshot live e contact-onset hook
**Ultimo aggiornamento:** 2026-07-15  
**Specifica di riferimento:** Rulebook v2, versione
`4.6-final-implementation-complete`

### Legenda

| Stato | Significato |
|---|---|
| `NON_INIZIATA` | Nessuna attività implementativa eseguita |
| `IN_CORSO` | Attività in lavorazione |
| `BLOCCATA` | Non può avanzare; esiste un problema aperto bloccante |
| `PRONTA_PER_VERIFICA` | Codice terminato, evidenze o test ancora incompleti |
| `COMPLETATA` | Criteri di uscita soddisfatti ed evidenze registrate |
| `DEFERITA` | Attività esplicitamente fuori dallo scope corrente |

### Dashboard delle fasi

| Fase | Obiettivo | Stato | Dipende da |
|---|---|---|---|
| F0 | Allineamento normativo, scope e contratti | `COMPLETATA` | — |
| F1 | Tipi canonici, configurazione e registry v2 | `COMPLETATA` | F0 |
| F2 | Primitive geometriche canoniche e 2.5D | `COMPLETATA` | F1 |
| F3 | Task route, adapter statici e validazione offline | `COMPLETATA` | F1, F2 |
| F4 | Snapshot live e contact-onset hook | `COMPLETATA` | F1 |
| F5 | Eventi, memoria, cache e zone lifecycle | `COMPLETATA` | F2–F4 |
| F6 | R1 collisione e R2 interazione dinamica | `COMPLETATA` | F4, F5 |
| F7 | R3 strada, controlli e precedenze | `COMPLETATA` | F3, F5 |
| F8 | R4 progresso, aggregazione e monitor transazionale | `COMPLETATA` | F6, F7 |
| F9 | Wrapper, wiring e output del rule vector | `COMPLETATA` | F8 |
| F10 | Conformità, calibrazione, pilot PG/Waymo e freeze | `IN_CORSO` | F3–F9 |
| D1 | Estensione osservazione semantica | `DEFERITA` | Rulebook v2 stabile |
| D2 | Scalarizzazione e learner lessicografico/distribuzionale | `DEFERITA` | Rule vector stabile |

---

## Scope corrente

### Incluso

- costruzione causale del contesto canonico richiesto dal monitor;
- valutazione transazionale `pre_state -> post_state`;
- produzione del vettore ordinato
  `(m_collision, m_interaction, m_compliance, m_progress)`;
- costi macro, risultati delle sottocomponenti e diagnostica raw;
- memoria episodica e cache delle conflict zone;
- validazione offline e smoke validation al reset;
- fail-fast senza output parziali;
- integrazione monitor-only con l'API Gymnasium e con `info`;
- test di conformità previsti dalla specifica.

### Esplicitamente fuori scope

- scelta della scalarizzazione per gli algoritmi baseline;
- critic, replay buffer o policy vettoriali;
- algoritmo lessicografico strict o thresholded;
- algoritmo distribuzionale;
- modifica immediata dell'osservazione semantica;
- tuning dei parametri fisici o geometrici congelati dalla specifica.

Il monitor non deve fare assunzioni su come il vettore verrà consumato. Il
reward scalare restituito dall'ambiente resta invariato nella prima integrazione.

---

## Decisioni congelate

### DEC-001 — La route ego è un input noto del task

**Stato:** `ACCETTATA`  
**Origine:** decisione utente del 2026-07-15

La destinazione e il percorso pianificato dell'ego sono informazioni disponibili
al sistema, analogamente a un navigatore reale. Non sono invece disponibili la
traiettoria che l'ego eseguirà realmente, i suoi tempi futuri o le future
traiettorie degli altri attori.

Per ScenarioNet si adotta questo contratto:

1. offline, la track SDC completa può essere usata esclusivamente per map-match
   e ricavare una sequenza topologica ordinata di lane;
2. l'output è un `TaskRouteRecord` statico, privo di timestamp, velocità,
   accelerazioni e posizioni future dell'ego;
3. il record contiene almeno `scenario_uid`, lane ID ordinati, provenienza,
   versione dell'adapter e hash della geometria sorgente;
4. a runtime la `RoutePolyline` viene costruita dalle centerline 3D delle lane
   del record, non dalla `TrajectoryNavigation.current_sdc_route`;
5. monitor e, in futuro, policy devono consumare lo stesso task route;
6. il monitor online non consulta mai la track SDC completa.

Per PG il task route viene ricavato direttamente dalla route/topologia generata,
senza passare da una traiettoria futura eseguita.

Una route non map-matchabile in modo univoco rende lo scenario non eleggibile
per il rulebook v2.

### DEC-002 — Un solo writer per il lifecycle delle conflict zone

**Stato:** `ACCETTATA`

Si introduce un componente infrastrutturale puro `ZoneLifecycleEvaluator`,
eseguito dopo la creazione delle zone pending e prima di crosswalk/vehicle-yield.

È l'unico writer di:

```text
preexisting_ego_occupancy_zone_ids
```

Restituisce inoltre una vista immutabile `ZoneLifecycleView` contenente:

- zone occupate nel pre-state;
- zone occupate nel post-state;
- ingressi rilevati nella transizione;
- uscite complete rilevate nella transizione;
- zone pending già occupate nel pre-state e quindi da considerare preesistenti.

Regola per le zone lazy:

- se una zona nuova interseca già l'ego nel `pre_state`, viene marcata
  preesistente prima del giudizio normativo;
- se l'ego è fuori nel `pre_state` ed entra nel `post_state`, l'ingresso viene
  valutato normalmente usando la nuova geometria;
- la sola occupazione nel `post_state` non basta a cancellare un ingresso
  realmente avvenuto nella stessa transizione.

Crosswalk e vehicle-yield leggono `ZoneLifecycleView` e scrivono soltanto i
propri flag di illegalità. Non possono modificare l'insieme condiviso.

### DEC-003 — Il rulebook produce il vettore, non decide come ottimizzarlo

**Stato:** `ACCETTATA`  
**Origine:** decisione utente del 2026-07-15

L'output normativo è:

```python
RulebookResult(
    margins=(m1, m2, m3, m4),
    costs=(c1, c2, c3),
    raw_progress_m=delta_s,
    components=...,
    complete_evaluation=True,
)
```

Nella prima integrazione:

- `info["rule_reward_vector"]` contiene esattamente i quattro margini;
- `info["rule_components"]` contiene le macro-regole nominate;
- `info["rulebook"]` contiene sottocomponenti e diagnostics;
- il reward Gymnasium restituito non viene modificato;
- non viene chiamata `lexicographic` alcuna scalarizzazione esistente.

Una futura scalarizzazione o un learner vettoriale leggeranno il risultato senza
modificare il monitor.

### DEC-004 — Registry modulare, composizione normativa fissa

**Stato:** `ACCETTATA`

L'idea del registro esistente viene mantenuta tramite un registry v2 tipizzato.
Ogni definizione dichiara:

```python
ComponentDefinition(
    name=...,
    macro_rule=...,
    evaluator=...,
    owned_memory_fields=...,
)
```

Il registry serve a costruire, testare e validare i componenti. In modalità
conforme v2 non è consentito:

- cambiare l'ordine delle quattro macro-regole;
- omettere una componente normativa;
- cambiare la macro-regola proprietaria di una componente;
- configurare due writer per lo stesso campo di memoria.

L'applicabilità resta dinamica, ma la presenza del componente nel monitor è
fissa.

### DEC-005 — Predicati dinamici di ingresso: pre-state per l'evento

**Stato:** `ACCETTATA`

Per un ingresso dell'ego in una conflict zone:

- l'evento geometrico usa il swept front bumper `pre -> post`;
- il predicato `altro veicolo già nella zona` usa il `pre_state`, cioè lo stato
  disponibile prima dell'azione che ha prodotto l'ingresso;
- gli altri predicati statici/topologici usano il context congelato della
  transizione;
- il costo continuo di approccio, quando non avviene un ingresso, usa il
  `post_state`;
- le eccezioni specifiche dei semafori restano quelle della specifica: colore
  pre-action per il crossing e colore post-action per l'approccio.

Questa convenzione impedisce che l'uscita dell'altro attore nello stesso control
step renda retroattivamente legittimo un ingresso iniziato mentre la zona era
occupata.

### DEC-006 — Nessuna broad phase nella prima implementazione conforme

**Stato:** `ACCETTATA`

La prima versione itera tutti gli attori live e applica i filtri esatti di
classe, geometria e quota. La specifica consente esplicitamente questa scelta.

In questo modo:

- non serve inventare `v_max` per VRU e statici;
- il risultato normativo non dipende da un indice spaziale;
- la correttezza viene stabilita prima dell'ottimizzazione.

Una broad phase potrà essere aggiunta in seguito soltanto con test differenziali
che dimostrino equivalenza esatta rispetto all'iterazione completa.

### DEC-007 — Canonicalizzazione e hash delle geometrie

**Stato:** `ACCETTATA`

Per ID sintetici e confronti byte-equivalent si congela questa pipeline:

1. coordinate XY finite;
2. snap alla precision grid di `1e-3 m`;
3. rifiuto di geometrie vuote, collassate o invalide dopo lo snap;
4. normalizzazione canonica di anelli e componenti con la versione Shapely/GEOS
   congelata dal lockfile;
5. serializzazione WKB 2D big-endian, senza SRID;
6. envelope tipizzato serializzato in canonical JSON (`sort_keys=True`,
   separatori compatti, UTF-8);
7. digest SHA-256.

Il payload dell'ID sintetico include almeno:

```text
scenario_id, namespace, feature_type, canonical_wkb_hex
```

Per i `ConflictZoneRecord`, l'equivalenza della cache confronta la
rappresentazione canonica tipizzata del record, inclusi polygon, source key,
route entry/exit quantizzati e riferimento di elevazione. Il solo `object id`
Python o l'ordine di costruzione non partecipano mai all'identità.

### DEC-008 — Ownership completa di `RulebookMemory`

**Stato:** `ACCETTATA`

| Writer unico | Campi posseduti |
|---|---|
| collision | `previous_contact_ids` |
| dashed line | `active_dashed_boundary_id`, `dashed_line_timer_s` |
| signal | tutti i campi `active/previous/yellow/resolved_signal_*` |
| stop | tutti i campi `active/timer/previous/resolved_stop_*` |
| zone lifecycle | `preexisting_ego_occupancy_zone_ids` |
| crosswalk | `crosswalk_illegal_entries` |
| vehicle yield | `vehicle_yield_illegal_entries`, `frozen_actor_movement_keys` |
| progress | `previous_route_s_m` |

Un writer può produrre al massimo una scrittura completa per campo e
transizione. Il merge rifiuta campi sconosciuti, writer non proprietari e
scritture duplicate.

### DEC-009 — Refuso di configurazione

**Stato:** `ACCETTATA`

La chiave:

```yaml
movement_keyentity
```

verrà corretta in:

```yaml
movement_identity
```

senza alias di compatibilità, perché il config v2 non è ancora pubblico.

### DEC-010 — Osservazione semantica differita ma tracciata

**Stato:** `ACCETTATA`, `DEFERITA`

Il rulebook v2 può essere implementato e validato come monitor anche se la
policy non osserva ancora tutta la memoria procedurale. Dopo il freeze del
monitor sarà necessario aggiornare lo stato semantico per esporre almeno:

- timer e ID logico della dashed boundary attiva;
- stato procedurale del signal group pertinente;
- stato procedurale dello stop pertinente;
- task route canonica al posto della futura trajectory SDC dettagliata;
- eventuali indicatori compatti di conflict-zone commitment.

Con `dt=0.1 s`, l'attuale history di 5 step copre circa `0.5 s` e non ricostruisce
il timer dashed fino a `2 s`. Questo punto è registrato come `DEF-OBS-001` e non
deve essere dimenticato quando verrà revisionata l'osservazione.

### DEC-011 — Consolidamento rumoroso dei vertici 3D della route

**Stato:** `ACCETTATA`

Per punti consecutivi della centerline distanti al più `1e-3 m` in XY:

1. formare un unico cluster consecutivo;
2. usare la media aritmetica delle coordinate XY e la mediana delle quote;
3. usare predecessore e successore solo per validare la continuità, mai per
   modificare la quota del cluster;
4. se l'escursione delle quote nel cluster supera `z_tol = 3.0 m`, rifiutare
   il task route come non eleggibile.

La regola mitiga rumore di campionamento senza scegliere arbitrariamente una
quota da un livello stradale vicino. Riusa una tolleranza verticale già
congelata e non introduce un nuovo parametro.

### DEC-012 — Ordinamento delle componenti disgiunte di conflict zone

**Stato:** `ACCETTATA`

Le componenti polygonali disgiunte di una stessa conflict zone vengono
canonicalizzate secondo DEC-007 e ordinate in senso lessicografico crescente
sul loro WKB 2D big-endian. L'indice `k` dello schema di `zone_id` è l'indice
zero-based in questo ordine. L'ordinamento è unicamente identitario: non
esprime precedenza né prossimità lungo route.

---

## Architettura target

### Separazione dei livelli

```text
ScenarioDescription / MetaDrive live objects
            |
            v
StaticScenarioAdapter + LiveSnapshotter
            |
            v
EpisodeCache + EnvSnapshot pre/post
            |
            v
Event detection + ZoneLifecycleView
            |
            v
Pure component evaluators from v2 registry
            |
            v
RulebookMonitor aggregate + transactional validation
            |
            v
RulebookResult in info (reward scalare invariato)
```

### Layout previsto

```text
src/thesis_rl/rulebook/
├── v1/                         # compatibilità del rulebook attuale
└── v2/
    ├── __init__.py
    ├── config.py
    ├── errors.py
    ├── types.py
    ├── registry.py
    ├── monitor.py
    ├── aggregation.py
    ├── memory.py
    ├── cache.py
    ├── events.py
    ├── lifecycle.py
    ├── geometry/
    │   ├── canonical.py
    │   ├── route.py
    │   ├── lanes.py
    │   ├── vertical.py
    │   ├── controls.py
    │   ├── corridors.py
    │   ├── conflict_zones.py
    │   └── continuous_sat.py
    ├── context/
    │   ├── task_route.py
    │   ├── episode_builder.py
    │   ├── snapshotter.py
    │   ├── actor_catalog.py
    │   ├── pg_adapter.py
    │   └── waymo_adapter.py
    ├── components/
    │   ├── collision.py
    │   ├── interaction.py
    │   ├── road_geometry.py
    │   ├── controls.py
    │   ├── yield_rules.py
    │   └── progress.py
    └── validation/
        ├── offline.py
        ├── reset_smoke.py
        └── artifacts.py
```

L'organizzazione interna può essere accorpata se alcuni moduli restano piccoli,
ma devono rimanere separate le responsabilità di adapter, geometria, evaluator,
monitor e integrazione runtime.

### Integrazione wrapper

Il monitor v2 sarà installato vicino all'env base:

```text
base ScenarioEnv
  -> RulebookV2MonitorWrapper       # produce info, non cambia reward
  -> eventuale reward composition   # futuro, fuori scope
```

Lifecycle previsto:

```text
reset:
  load TaskRouteRecord
  build and validate EpisodeCache
  capture reset snapshot
  initialize RulebookMemory

step:
  capture pre_state
  clear onset buffer atomically
  env.step(action)
  capture post_state
  evaluate_transition(pre, post, memory, cache)
  validate and build next_cache
  commit cache and memory
  attach RulebookResult to info
  return original scalar reward
```

Se la valutazione fallisce, il wrapper non restituisce la transizione; il
planner non può quindi inserirla nel replay buffer.

---

## F0 — Allineamento normativo, scope e contratti

**Stato:** `COMPLETATA`

### Attività

- [x] Analizzare la specifica e confrontarla con registry, evaluator, wrapper,
      reward manager, catalogo, osservazione e MetaDrive vendorizzato.
- [x] Congelare DEC-001–DEC-010.
- [x] Creare questo tracker.
- [x] Riportare le decisioni normative rilevanti in una revisione della
      specifica v2.
- [x] Congelare i nomi pubblici di package, config e output `info`:
      `thesis_rl.rulebook.v2`, `conf/rulebook/v2.yaml`,
      `rulebook.version: v1|v2`, `info["rule_reward_vector"]`,
      `info["rule_components"]`, `info["rulebook"]`.
- [x] Aggiungere un config selector esplicito `rulebook.version: v1 | v2` senza
      cambiare il comportamento v1.
- [x] Registrare fixture e versioni Shapely/GEOS/MetaDrive usate dai test:
      container `dev`, Shapely `2.1.2`, GEOS `3.13.1`, MetaDrive senza
      attributo `__version__` esposto.

### Criteri di uscita

- specifica e tracker non si contraddicono;
- scope monitor-only esplicito;
- route task e ownership memoria espliciti;
- nessun cambiamento al percorso v1 senza test di regressione.

### Evidenze

- analisi architetturale: completata il 2026-07-15;
- decisioni utente: annotate il 2026-07-15;
- tracker: questo documento.

---

## F1 — Tipi canonici, configurazione e registry v2

**Stato:** `COMPLETATA`

### Attività

- [x] Implementare enum canonici di attori, map features, controlli, priorità e
      status.
- [x] Implementare dataclass frozen per snapshot, task route, map records,
      traffic controls, conflict zone, memory, delta, cache e risultati.
- [x] Implementare `RulebookEvaluationError` con scenario ID, step, componente
      e causa tipizzata.
- [x] Implementare config v2 immutabile e validazione dei parametri congelati
      già introdotti (execution, geometria e prediction); i parametri delle
      formule saranno aggiunti con i rispettivi evaluator nelle F6--F8.
- [x] Implementare `ComponentDefinition` e registry v2.
- [x] Collegare nel registry gli evaluator normativi reali e fornire dispatch
      tipizzato senza fallback; `zone_lifecycle` resta infrastrutturale.
- [x] Validare al bootstrap ordine macro, componenti richieste e ownership dei
      campi di memoria.
- [x] Testare immutabilità, finitezza, serializzazione diagnostica ed errori di
      configurazione.

### Criteri di uscita

- tutti i contratti della specifica sono rappresentabili senza `dict[str, Any]`
  nei confini normativi;
- registry completo e composizione fissa;
- nessuna collisione di ownership possibile al bootstrap.

---

## F2 — Primitive geometriche canoniche e 2.5D

**Stato:** `COMPLETATA`

### Attività

- [x] Implementare precision grid, canonical WKB e SHA-256 secondo DEC-007.
- [x] Implementare footprint OBB e validazione dimensioni.
- [x] Implementare elevation functions e compatibilità verticale 2.5D.
- [x] Implementare `RoutePolyline` e proiezione con continuità `previous_s`.
- [x] Implementare front/rear route coordinates e swept front bumper.
- [x] Implementare lane association e gap bumper-to-bumper.
- [x] Implementare superficie carrabile per livello verticale.
- [x] Implementare control line canonica.
- [x] Implementare decomposizione convessa deterministica e continuous SAT.
- [x] Implementare `MovementKey`, corridoi e conflict-zone construction base
      (vehicle/crosswalk, componenti, intervalli route e selezione), con test
      sintetici anche per merge e rotatorie; i record statici restano F3.
- [x] Aggiungere test sintetici per geometrie concave, hole, autointersezioni
      della route, cavalcavia, merge e rotatorie.

### Criteri di uscita

- primitive deterministiche su reset ripetuti;
- nessun conflitto 2D spurio fra livelli incompatibili;
- continuous SAT conforme ai casi `NO_INTERVAL`, finito e `OPEN_END`;
- canonical bytes e ID stabili nelle fixture congelate.

---

## F3 — Task route, adapter statici e validazione offline

**Stato:** `COMPLETATA`

### Attività

- [x] Definire e versionare `TaskRouteRecord`.
- [x] Implementare map-matching offline Waymo SDC -> lane ID sequence, senza
      conservare timing o future pose nel record.
- [x] Implementare estrazione diretta del task route PG tramite builder
      source-neutral di topologia lane.
- [x] Costruire adapter PG/Waymo verso lane, map feature, logical boundary,
      traffic control e crosswalk; priority/roundabout assenti nelle fixture
      sono rappresentati esplicitamente come cataloghi vuoti/NA.
- [x] Implementare conversione offline Waymo sulle fixture ScenarioNet
      vendorizzate: lane centerline/width, task route SDC topology-only,
      map features, crosswalk/road-line e signal control line; lane invalide
      sono escluse tipicamente.
- [x] Convertire `STOP_SIGN` con lane controllate e control line derivata;
      control point ambigui/incompatibili sono esclusi offline.
- [x] Rappresentare esplicitamente l'assenza di `MovementPriorityRecord` nelle
      fixture PG/Waymo: nessuna precedenza viene inferita dalla geometria.
- [x] Implementare conversione offline PG su `ScenarioDescription`: lane
      polygon/centerline, task route SDC topology-only, map features e signal
      controls tramite lo stesso contratto canonico.
- [x] Definire contratto source-neutral `StaticRecordSources`/`StaticRecordAdapter`
      per iniettare gli estrattori PG/Waymo senza branch runtime per sorgente.
- [x] Riutilizzare dove corretto l'estrazione topologica offline esistente,
      senza usarne le soglie euristiche come primitive runtime.
- [x] Implementare validazione source-neutral di route, quote, lane
      polygon/width, controlli, spawn overlap e configured speed caps.
- [x] Collegare validazione completa delle signal sequences all'artifact
      sorgente-specifico (lunghezza e stati `UNKNOWN` rifiutati offline).
- [x] Implementare validazione source-neutral di unicità lane/control group,
      quote finite, control line e record control completi.
- [x] Implementare artifact di eleggibilità separato dal generico
      `validation_status` del catalogo.
- [x] Indicizzare l'artifact per scenario UID, versione rulebook, versione
      adapter, hash config geometrica e hash calibrazione.
- [x] Aggiornare provider/runtime affinché il pool rulebook v2 includa soltanto
      record eleggibili.
- [x] Aggiungere filtro opzionale `eligible_scenario_uids` a Uniform/Fixed
      provider; il pool v2 può essere costruito direttamente dall'indice
      offline senza fallback.
- [x] Produrre report di esclusione per sorgente/adapter e causa.
- [x] Produrre report tipizzato di esclusione per adapter e causa a partire
      dall'indice offline.

### Criteri di uscita

- nessuna formula runtime contiene branch `if source == waymo/pg`;
- task route disponibile senza consultare online la track SDC;
- scenari invalidi esclusi con cause tipizzate;
- distribuzione delle esclusioni misurata e registrata.

---

## F4 — Snapshot live e contact-onset hook

**Stato:** `COMPLETATA`

### Attività

- [x] Implementare snapshotter live comune a PG e Waymo.
- [x] Introdurre `LiveSnapshotSources`/`LiveSnapshotAdapter`: l’estrazione
      source-specific passa soltanto da hook espliciti e manca di fallback su
      osservazione o traiettorie future.
- [x] Validare al bootstrap completezza e callable-ness dei provider live;
      mapping parziali o con chiavi sconosciute falliscono prima del reset.
- [x] Risolvere ID persistenti e classi canoniche degli oggetti MetaDrive tramite
      payload live obbligatorio (`actor_id` + `actor_class`), senza fallback.
- [x] Catturare pose 3D, velocità, heading, footprint OBB, lane ID e speed cap.
- [x] Implementare un buffer episodico thread-safe/control-step-safe per i
      contact onset.
- [x] Estendere localmente il callback collisioni vendorizzato preservando il
      callback MetaDrive originale e il suo return value.
- [x] Estrarre actor ID, class, contact point e normale unitaria orientata ego -> altro.
- [x] Deduplicare per actor ID nel control step senza perdere i contact point.
- [x] Rilevare contatti nati e terminati tra due control frame.
- [x] Verificare con test differenziale su ScenarioEnv/Waymo che l'hook non cambi
      la dinamica a parità di seed e azioni; il callback vendor resta invariato.

### Criteri di uscita

- snapshot immutable e completi;
- onset substep rilevati in modo riproducibile;
- nessun fallback a crash flag per la severità normativa;
- dinamica identica con hook attivo/disattivo a parità di seed e azioni.

---

## F5 — Eventi, memoria, cache e zone lifecycle

**Stato:** `COMPLETATA`

### Attività

- [x] Implementare detector puri di crossing e occupancy pre/post.
- [x] Implementare `ZoneLifecycleEvaluator` e `ZoneLifecycleView` secondo
      DEC-002.
- [x] Implementare init memoria al reset, inclusi contatti attivi, proiezione
      route e occupazioni preesistenti; prepassed controls saranno aggiunti con
      il catalogo statico F3.
- [x] Implementare merge dei `MemoryDelta` con ownership DEC-008.
- [x] Implementare overlay `cache + pending_delta`.
- [x] Implementare merge/apply dei `CacheDelta` con confronto canonico.
- [x] Implementare freeze/unfreeze delle `MovementKey` degli attori nel
      `MemoryDelta` owner `vehicle_yield`, con rimozione soltanto all'uscita.
- [x] Testare doppio writer, geometrie discordanti, lazy zone inside ego,
      eccezioni dopo proposta delta e commit unico.

### Criteri di uscita

- evaluator senza side effect;
- memoria e cache originali immutate dopo ogni errore;
- lifecycle condiviso con un solo writer;
- pending zones visibili nello stesso step senza commit anticipato.

---

## F6 — R1 collisione e R2 interazione dinamica

**Stato:** `COMPLETATA`

### Attività

- [x] Implementare collision onset e severità bounded da velocità pre-state.
- [x] Implementare floor numerico e normalizzazione per actor class.
- [x] Implementare RSS longitudinale sulla lane association canonica.
- [x] Implementare TTC generalizzato mediante continuous SAT.
- [x] Implementare clearance R2 su tutti gli attori live compatibili e conservare
      worst actor/diagnostics.
- [x] Aggregare R2 con massimo e conservare worst actor/diagnostics.
- [x] Implementare status applicabile/evaluabile per ogni componente.
- [x] Coprire i test disponibili delle sezioni 15.1–15.4, inclusi boundedness,
      NOT_APPLICABLE, onset-only, SAT e iterazione exhaustive degli attori.

### Criteri di uscita

- R1 e R2 finite e bounded;
- collisione emessa soltanto all'onset;
- nessuna dipendenza dal raggio dell'osservazione;
- nessun attore perso rispetto all'iterazione completa.

---

## F7 — R3 strada, controlli e precedenze

**Stato:** `IN_CORSO`

### Attività

- [x] Implementare off-road e wrong-way.
- [x] Implementare solid-line crossing/occupancy.
- [x] Implementare dashed-line timer con logical boundary ID.
- [x] Implementare catalogo e macchina a stati dei signal group.
- [x] Implementare stop zone, timer continuo/best e crossing.
- [x] Implementare crosswalk yield mediante zone lifecycle.
- [x] Implementare vehicle-yield scoped con i quattro predicati ammessi.
- [x] Applicare DEC-005 agli ingressi vehicle-yield: l'insieme degli attori
      entrati è dichiarato pre-state e l'alias esplicito rifiuta disaccordi.
- [x] Aggregare R3 con massimo conservando tutte le sottocomponenti.
- [x] Coprire i test disponibili delle sezioni 15.5–15.9 (suite v2: 74).

### Criteri di uscita

- componenti R3 conformi ai domini e ai fallback vietati;
- signal/stop selezionati lungo route, non per distanza euclidea;
- illegal-entry flags persistenti fino all'uscita completa;
- casi ambigui `NOT_APPLICABLE`, dati core mancanti fail-fast.

---

## F8 — R4 progresso, aggregazione e monitor transazionale

**Stato:** `IN_CORSO`

### Attività

- [x] Implementare progresso raw e normalizzato sulla task route canonica.
- [x] Verificare continuità fra `memory.previous_route_s_m` e pre-state.
- [x] Implementare aggregazione R1–R4 e macro status.
- [x] Implementare orchestrazione transazionale `evaluate_monitor_transition`.
- [x] Aggiungere dispatch `evaluate_registered_transition`: ogni componente
      normativo e `progress` devono fornire input canonici espliciti al registry;
      input mancanti o sconosciuti sono fail-fast.
- [x] Validare range, finitezza, complete evaluation, ownership e cache delta.
- [x] Garantire che qualunque errore non produca un risultato parziale.
- [x] Coprire i test delle sezioni 15.0, 15.10 e atomicità (72 test Rulebook
      v2 nel container; Ruff e `git diff --check` verdi).

### Criteri di uscita

- API restituisce sempre la tripla normativa o solleva;
- margini ordinati e range conformi;
- stesso input produce risultato, memoria e delta identici;
- nessun accesso a native longitude salvo diagnostics.

---

## F9 — Wrapper, wiring e output del rule vector

**Stato:** `COMPLETATA`

### Attività

- [x] Implementare `RulebookV2MonitorWrapper` monitor-only.
- [x] Integrare reset, snapshot pre/post e commit atomico memoria/cache tramite
      adapter iniettivi; l'onset buffer resta responsabilità dello snapshotter.
- [x] Collegare `rulebook.version=v2` nel runtime wiring tramite adapter esplicito
      (`env.rulebook_v2_adapter`), con fail-fast se mancante e nessun fallback v1.
- [x] Tipizzare il contratto adapter (`RulebookV2Adapter`) con snapshotter,
      evaluator transazionale e stato iniziale episodico; l'implementazione
      concreta MetaDrive/ScenarioNet resta dipendente dalle API live disponibili.
- [x] Preservare il percorso v1 e i suoi config esistenti.
- [x] Allegare output compatibile a `info` secondo DEC-003 (`rule_reward_vector`,
      `rule_components`, `rulebook`).
- [x] Aggiornare agent metrics, eval artifacts, video diagnostics e CSV per i
      quattro nomi macro v2.
- [x] Aggiornare rule criticality diagnostica del curriculum da v1 a v2 senza
      cambiare la usefulness primaria.
- [x] Assicurare supporto a env vectorized con memoria/cache per-env isolate.
- [x] Testare che il reward scalare nativo resti byte/float-equivalente in
      modalità monitor-only.

### Criteri di uscita

- training/evaluation possono leggere il rule vector senza scalarizzarlo;
- nessuna transizione invalida viene restituita al planner;
- istanze vectorized non condividono stato episodico;
- regressione v1 e native reward assente.

---

## F10 — Conformità, calibrazione, pilot PG/Waymo e freeze

**Stato:** `IN_CORSO`

### Attività

- [x] Implementare il protocollo di calibrazione di `b_e`.
- [x] Implementare runner automatico delle prove su pista rettilinea MetaDrive
      senza traffico, parametrizzato dalla configurazione ego congelata.
- [x] Produrre artifact in memoria con hash della configurazione ego.
- [ ] Eseguire tutti i test obbligatori della sezione 15.
- [x] Eseguire smoke deterministici su fixture PG e Waymo.
- [x] Eseguire pilot su campione stratificato per sorgente/topologia.
- [x] Misurare eleggibilità e cause di esclusione sul campione pilot.
- [x] Integrare il filtro statico Rulebook v2 prima della costruzione degli
      split ScenarioNet e delle runtime view.
- [ ] Misurare costo runtime del monitor per step.
- [x] Verificare assenza di future-track access nel monitor online.
- [x] Verificare equivalenza geometrica e ID fra reset ripetuti tramite test
      di canonicalizzazione e candidate rebuild deterministico.
- [x] Eseguire suite completa: 437 passati, senza skip; ISS-011 risolto.
- [x] Eseguire Ruff sui file del perimetro F10/v2.
- [x] Eseguire type checking mirato del modulo calibrazione; audit package completo tracciato in ISS-012.
- [x] Aggiornare documentazione, comandi `make` di validazione e metadata run.
- [ ] Congelare versione config, adapter, calibration e eligibility artifact.

Il file `data/scenarionet/rulebook_v2/ego_config.json` è stato predisposto con
i default fisici effettivi del repository (`vehicle_model=default`). Deve
essere sostituito prima della calibrazione se il run finale usa override fisici
del veicolo; il dataset split non contiene questi parametri.

### Criteri di uscita

- suite di conformità completamente verde;
- artifact `b_e` valido e corrispondente alla configurazione ego;
- nessun problema bloccante aperto;
- deviazioni assenti oppure esplicitamente approvate e riportate nella spec;
- report di eleggibilità disponibile;
- monitor pronto per essere consumato da scalarizer o learner futuri.

---

## Attività differite

### DEF-OBS-001 — Stato semantico rulebook-aware

**Stato:** `DEFERITA`  
**Quando riprenderla:** dopo F9, prima degli esperimenti finali con policy
semantic-state.

Aggiornare osservazione, `ObservationSpec`, encoder LQ e relativi test per
esporre memoria procedurale e task route senza future trajectory leakage.

### DEF-ALG-001 — Consumo del rule vector

**Stato:** `DEFERITA`

Definire in documenti separati:

- scalarizzazione baseline;
- learner distributional;
- learner lexicographic strict;
- learner lexicographic thresholded.

Nessuna di queste decisioni deve entrare nel core del monitor.

### DEF-PERF-001 — Broad phase

**Stato:** `DEFERITA`

Valutare un indice spaziale soltanto dopo il pilot. L'ottimizzazione viene
accettata solo con test differenziali exhaustive-vs-indexed sulle stesse
transizioni.

---

## Registro problemi e rischi

| ID | Stato | Severità | Problema/rischio | Azione proposta |
|---|---|---|---|---|
| ISS-001 | `APERTO` | alta | Pilot finale hash-validato su 10 PG + 10 Waymo: 13 eleggibili (65%) e 7 esclusi tipicamente, senza eccezioni; il filtro ora è integrato a monte degli split ma l'artifact sull'intero catalogo e il reset smoke restano da produrre | Eseguire `make rulebook-v2-filter-catalog` o l'intera pipeline e completare il reset smoke; il pilot campionato resta soltanto evidenza preliminare |
| ISS-002 | `RISOLTO` | alta | Le ambiguità PG erano campioni singoli sui confini canonici di lane consecutive; quattro Waymo hanno invece gap/non copertura reale della task route | Risoluzione deterministica soltanto per transizioni contigue confermate dai campioni adiacenti; route non univoche escluse con `task_route_lane_association_ambiguous_or_unavailable`, senza nearest-lane fallback |
| ISS-003 | `RISOLTO` | media | API Panda3D per callback di contatto e installazione hook verificati sul commit locale | Test differenziale reale ScenarioEnv/Waymo con seed/azioni identici: osservazioni, reward e terminazioni invariati |
| ISS-004 | `APERTO` | media | Qualità di polygon, width e quota differisce fra PG e Waymo; il pilot ha trovato due `ROAD_EDGE_BOUNDARY` Waymo a punto singolo nello stesso scenario | Mantenere esclusione tipizzata `invalid_map_feature_geometry`; misurare la frequenza sull'intero catalogo prima di valutare una regola di pertinenza più selettiva |
| ISS-005 | `APERTO` | media | `MovementKey` può essere ambigua prima della conflict zone | Restare `NOT_APPLICABLE`; misurare frequenza nel pilot |
| ISS-006 | `APERTO` | media | Costo del continuous SAT su tutti gli attori live non ancora misurato | Benchmark F10 prima di valutare DEF-PERF-001 |
| ISS-007 | `RISOLTO` | alta | Artifact di calibrazione `b_e` prodotto e validato contro l'hash della configurazione ego | `ego_min_brake_mps2=4.0`, hash `ce25f5a02c7be3974048ac8d3f664a43a73715c7a26882f226982d28b444cf00` |
| ISS-008 | `RISOLTO` | media | Shell di lavoro senza `pytest`, Ruff, Shapely e MetaDrive | Verifiche eseguite nel container `dev`: dipendenze disponibili, 23 test e Ruff verdi; versioni registrate in F0 |
| ISS-009 | `RISOLTO` | media | `RoutePolyline` deve unificare punti consecutivi entro 1 mm in XY, ma la specifica non definisce la quota risultante se tali punti hanno `z` differenti | DEC-011: cluster XY medio, mediana z, validazione con `z_tol=3 m`; approvata dall'utente il 2026-07-15 |
| ISS-010 | `RISOLTO` | media | La specifica assegna alle componenti disgiunte di conflict zone un indice `k` “dopo ordinamento canonico”, senza definire la chiave d'ordinamento | DEC-012: ordine lessicografico crescente del WKB DEC-007; approvata dall'utente il 2026-07-15 |
| ISS-011 | `RISOLTO` | media | Cataloghi smoke e mapping runtime erano artifact obsoleti: i file reali erano presenti sotto `database_7`, mentre gli artifact puntavano a `database_0`; la view ufficiale `scenario_catalog.parquet` + `runtime/train` era coerente | Suite completa finale dopo i fix pilot: 437 passati, nessuno skip; nessun dato sintetico, fallback o modifica al core v2 |
| ISS-013 | `APERTO` | alta | Gli export PG esistenti non persistono i lane ID della route pianificata; l'adapter legacy ricava ancora la task route via map-match offline della track SDC, mentre DEC-001 richiede estrazione diretta dalla route/topologia generata | Raccomandato: persistere il task route nel generatore/exporter PG e rigenerare gli artifact. Alternativa temporanea: documentare il map-match offline come deviazione per i dataset legacy; non usare inferenza nearest-lane o graph path arbitraria |
| ISS-012 | `APERTO` | bassa | Il codice v2 passa `mypy --ignore-missing-imports` senza errori in 40 file. Senza ignore restano 22 errori esclusivamente per import esterni non tipizzati: stub Shapely mancanti e `panda3d.core` senza `py.typed` | Installare/configurare stub compatibili per le versioni congelate; nessuna modifica semantica necessaria |

Quando un problema richiede una scelta non coperta dalla specifica:

1. aggiungere o aggiornare la riga nel registro;
2. descrivere almeno due opzioni realistiche;
3. indicare l'opzione consigliata e il motivo;
4. discutere la scelta prima di modificare la semantica;
5. registrare l'esito in **Decisioni congelate** oppure **Deviazioni**.

---

## Registro deviazioni dalla specifica

Al momento non esistono deviazioni approvate.

| ID | Stato | Sezione spec | Deviazione | Motivazione | Approvazione |
|---|---|---|---|---|---|
| — | — | — | Nessuna | — | — |

Una scorciatoia temporanea non viene nascosta come fallback. Se necessaria per
debug, deve essere protetta da un config non utilizzabile nei run finali e
registrata qui.

---

## Registro evidenze

Per ogni fase completata aggiungere:

- commit o diff rilevante;
- test eseguiti e risultato;
- fixture/dataset usati;
- artifact prodotti;
- benchmark o report;
- problemi chiusi o nuovi problemi aperti.

| Data | Fase | Evidenza | Risultato |
|---|---|---|---|
| 2026-07-15 | F0 | Analisi spec/architettura e decisioni DEC-001–DEC-010 | Piano pronto; implementazione non iniziata |
| 2026-07-15 | F0 | Contratti congelati riportati nella specifica; package/config/output pubblici definiti; `rulebook.version: v1` aggiunto al default | F0 resta in corso: fixture e versioni dipendenze richiedono l'environment di progetto (ISS-008) |
| 2026-07-15 | F1 | Aggiunti `thesis_rl.rulebook.v2` (types, errors, config, registry), `conf/rulebook/v2.yaml` e test contrattuali | Compilazione Python riuscita; test/Ruff non eseguibili nella shell corrente per ISS-008 |
| 2026-07-15 | F0/F1 | Container `dev`: `tests/test_rulebook_v2_contracts.py` + regressione `tests/test_rulebook_evaluator.py`, Ruff mirato | 19 test passati; F0/F1 completate; ISS-008 risolto |
| 2026-07-15 | F2 | DEC-007: canonicalizzazione Shapely (snap 1 mm, normalize, WKB 2D big-endian/no SRID, JSON canonico/SHA-256) e predicato verticale 2.5D | 23 test mirati passati e Ruff verde nel container; F2 resta in corso per le primitive restanti |
| 2026-07-15 | F2 | Aggiunti OBB canonico e `PolylineElevation` con interpolazione lineare e tie-break deterministico | 25 test mirati passati e Ruff verde nel container |
| 2026-07-15 | F2 | DEC-011 e `RoutePolyline`: consolidamento rumoroso, quota mediana, proiezione 3D con tie-break reset/`previous_s` | 28 test mirati passati e Ruff verde nel container |
| 2026-07-15 | F2 | Lane association route-only con tie-break/ambiguità e coordinate footprint/gap bumper-to-bumper | 11 test geometrici mirati passati e Ruff verde nel container |
| 2026-07-15 | F2 | Superficie carrabile per-step limitata al livello verticale ego; fallback polygon centerline/width | 12 test geometrici mirati passati e Ruff verde nel container |
| 2026-07-15 | F2 | Derivazione control line ortogonale, filtro verticale e signed distance con lato upstream positivo | 13 test geometrici mirati passati e Ruff verde nel container |
| 2026-07-15 | F2 | DEC-012: `MovementCorridor` e candidate conflict zone vehicle--vehicle pure, con componenti ordinate per WKB, filtro 2.5D e ID SHA-256 | 14 test geometrici mirati passati e Ruff verde nel container; selezione route/crosswalk/merge resta da completare |
| 2026-07-15 | F2 | Intervalli route entry/exit e selezione occupied-first/first-ahead; candidate crosswalk con namespace distinto | 16 test geometrici mirati passati e Ruff verde nel container |
| 2026-07-15 | F2 | Decomposizione convessa deterministica (inclusi concavità e hole) e continuous SAT CV con `NO_INTERVAL`, finito e `OPEN_END` simbolico | 18 test geometrici mirati passati e Ruff verde nel container |
| 2026-07-15 | F2 | Aggiunto swept front bumper canonico per i crossing events; regressione cumulativa v1/v2 | 38 test passati, Ruff verde e diff check pulito nel container |
| 2026-07-15 | F3 | `TaskRouteRecord` versionato, artifact `TaskRouteEligibility` e matcher SDC offline source-neutral; track samples non entrano nel record | 40 test cumulativi passati e Ruff verde nel container |
| 2026-07-15 | F3/F4 | Normalizzazione statica source-neutral di lane/map/control record; snapshot immutable e buffer contact-onset control-step-safe | 43 test cumulativi passati e Ruff verde nel container |
| 2026-07-15 | F5 | Merge fail-fast `MemoryDelta`/`CacheDelta`, apply immutabile della cache e `ZoneLifecycleEvaluator` unico writer DEC-002 | 47 test cumulativi passati e Ruff verde nel container |
| 2026-07-15 | F5 | Detector puro crossing/occupancy pre/post e `EpisodeCacheOverlay` per pending zones same-step | 49 test cumulativi passati e Ruff verde nel container |
| 2026-07-15 | F5 | Inizializzazione reset di `RulebookMemory` con contatti, route `s` e zone preesistenti | 50 test cumulativi passati e Ruff verde nel container |
| 2026-07-15 | F6 | Evaluator puro R1 collision onset/severità: pre-state closing speed, floor, cap per classe, worst actor e fail-fast | 53 test cumulativi passati e Ruff verde nel container |
| 2026-07-15 | F6 | Evaluator puro R2 clearance: soglie per classe, iterazione exhaustive live actor, filtro verticale e worst diagnostics | 55 test cumulativi passati e Ruff verde nel container |
| 2026-07-15 | F6 | Evaluator puro R2 TTC con moto relativo pre-state, soglie vehicle/VRU/static e continuous SAT | 57 test cumulativi passati e Ruff verde nel container |
| 2026-07-15 | F6 | Aggregatore R2 worst-case con massimo e diagnostica completa delle sottocomponenti | 59 test cumulativi passati e Ruff verde nel container |
| 2026-07-15 | F6 | RSS longitudinale con artifact `b_e` hash-validato, safe distance e deficit continuo | 61 test cumulativi passati e Ruff verde nel container; integrazione monitor/evaluator finale resta da verificare |
| 2026-07-15 | F7 | Evaluator puri off-road (area fraction con epsilon geometrica) e wrong-way (velocità longitudinale firmata su RoutePolyline, diagnostiche heading/segmento) | 24 test mirati cumulativi passati nel container; solid/dashed, signal/stop e precedenze restano da implementare |
| 2026-07-15 | F7 | Solid-line occupancy/crossing su boundary canoniche con buffer geometrico fisso e diagnostica degli ID attivi | 53 test Rulebook v2 passati cumulativamente e Ruff verde nel container; dashed, signal/stop e precedenze restano da implementare |
| 2026-07-15 | F7 | Timer dashed-line persistente con selezione boundary deterministica, reset su cambio ID e shaping quadratico 1–2 s | 54 test Rulebook v2 passati cumulativamente e Ruff verde nel container; signal/stop e precedenze restano da implementare |
| 2026-07-15 | F7 | Catalogo signal group: selezione per coordinata route, esclusione dei gruppi risolti e validazione fail-fast dello stato fisico concorde | 57 test Rulebook v2 passati cumulativamente e Ruff verde nel container; macchina completa di crossing/approach e stop/precedenze restano da implementare |
| 2026-07-15 | F7 | Stop zone: selezione route-ordered, timer continuo/best, crossing e risoluzione persistente con `MemoryDelta` owner `stop` | 63 test mirati cumulativi passati nel container e Ruff verde; signal approach/crossing e precedenze restano da implementare |
| 2026-07-15 | F7 | Signal transition: crossing giudicato sul colore pre-azione, approccio post-azione continuo, onset giallo congelato e risoluzione persistente | 66 test Rulebook v2 cumulativi passati e Ruff verde nel container; precedenze crosswalk/vehicle-yield restano da implementare |
| 2026-07-15 | F7 | Crosswalk yield: gap temporale, commitment gate, ingresso illegale e memoria persistente fino all'uscita completa | 69 test Rulebook v2 cumulativi passati e Ruff verde nel container; vehicle-yield resta da implementare |
| 2026-07-15 | F7 | Vehicle-yield scoped: gap/commitment, ingresso illegale persistente e cleanup all'uscita con memoria owner `vehicle_yield` | 67 test Rulebook v2 cumulativi passati e Ruff verde nel container; validazione esplicita dei quattro predicati DEC-005 e monitor aggregato restano da completare |
| 2026-07-15 | F8 | Aggregazione macro R1–R3, vettore ordinato `(m1,m2,m3,m4)` e orchestrazione transazionale con merge fail-fast di memoria/cache | 70 test Rulebook v2 cumulativi passati nel container, Ruff e diff check verdi; progresso canonico route e wiring completo restano da implementare |
| 2026-07-15 | F8 | Progresso canonico su `RoutePolyline`: delta raw, clipping normalizzato `m4`, continuità con `previous_route_s_m` e aggiornamento memory owner `progress` | 72 test Rulebook v2 cumulativi passati nel container, Ruff e diff check verdi |
| 2026-07-15 | F5/F7 | Freeze/unfreeze `MovementKey` per vehicle-yield, persistente fino all'uscita completa | 76 test Rulebook v2 nel container, Ruff e `git diff --check` verdi |
| 2026-07-15 | F1/F8 | Registry collegato agli evaluator normativi reali e dispatch puro per componente | 82 test Rulebook/runtime nel container, Ruff e `git diff --check` verdi |
| 2026-07-15 | F8 | Dispatch registry-driven della transizione completa, incluso evaluator progress, con rifiuto di input parziali | 84 test mirati Rulebook/runtime nel container, Ruff e `git diff --check` verdi |
| 2026-07-15 | F4/F9 | Validazione strict dei provider `LiveSnapshotSources` e adapter snapshot source-neutral | 86 test mirati Rulebook/runtime nel container, Ruff e `git diff --check` verdi |
| 2026-07-15 | F2 | Verifica finale primitive geometriche: concavità/hole, self-intersection route, cavalcavia, merge e rotatoria | 21 test geometrici passati nel container, Ruff e `git diff --check` verdi; F2 completata |
| 2026-07-15 | F3 | `TaskRouteEligibilityIndex` deterministico per UID con rifiuto degli scenari ineleggibili | 35 test F2/F3 mirati passati nel container, Ruff e `git diff --check` verdi |
| 2026-07-15 | F3 | Validazione metadata route/versione/hash e conferma matcher Waymo offline + builder PG topology-only | 34 test F2/F3 mirati passati nel container, Ruff e `git diff --check` verdi |
| 2026-07-15 | F3 | Validazione statica di quote, unicità lane/control group, control line e completezza record | 35 test F2/F3 mirati passati nel container, Ruff e `git diff --check` verdi |
| 2026-07-15 | F3 | Contratto `StaticRecordSources`/`StaticRecordAdapter` per provider PG/Waymo espliciti e normalizzazione comune | 36 test F2/F3 mirati passati nel container, Ruff e `git diff --check` verdi |
| 2026-07-15 | F3 | Report tipizzato di eleggibilità/esclusione per adapter e causa | 36 test F2/F3 mirati passati nel container, Ruff e `git diff --check` verdi |
| 2026-07-15 | F3 | Converter offline Waymo per fixture vendorizzata verso lane/task-route/map/control canonici | 87 test Rulebook v2 passati nel container, Ruff e `git diff --check` verdi |
| 2026-07-15 | F3 | Converter offline PG source-neutral e test di fail-fast; fixture PG non montata nel container di test | 3 test converter passati, 1 skip per fixture assente, Ruff verde |
| 2026-07-15 | F3 | Stop-sign statici Waymo/PG: lane controllate, control line e filtro dei punti non risolvibili | 3 test converter passati, 1 skip PG per fixture non montata, Ruff e diff check verdi |
| 2026-07-15 | F3 | Verifica fixture: nessun campo pairwise priority/junction/roundabout; adapter restituisce catalogo priority vuoto e vehicle-yield resta NOT_APPLICABLE | test Waymo converter passato, Ruff verde |
| 2026-07-15 | F3 | Validator reset offline per cap ego/veicoli, spawn overlap e signal `UNKNOWN` | 19 test F3 mirati passati, 1 skip PG per fixture non montata, Ruff verde |
| 2026-07-15 | F3 | Validazione sequenze semaforiche source-specifiche: lunghezza e `LANE_STATE_UNKNOWN` registrati come esclusioni | 19 test F3 mirati passati, 1 skip PG per fixture non montata, Ruff e diff check verdi |
| 2026-07-15 | F3 | Provider Uniform/Fixed filtrabili per UID eleggibili dell'artifact offline | 27 test F3/provider passati, Ruff e `git diff --check` verdi |
| 2026-07-15 | F3 | Chiusura fase: converter PG/Waymo, validazione reset/sequenze, artifact/index/report e filtro provider | 89 test Rulebook v2 passati, 1 skip PG per fixture non montata, Ruff e `git diff --check` verdi |
| 2026-07-15 | Regressione | Verifica suite completa repository dopo wiring registry | 385 passati, 8 falliti fuori dal perimetro v2; ISS-011 aperto e dettagliato sopra |
| 2026-07-15 | F8/F9 | Invarianti di aggregazione, merge transazionale, wrapper monitor-only, contratto `RulebookV2Adapter` e wiring runtime v2 esplicito | 79 test mirati Rulebook/runtime nel container, Ruff e `git diff --check` verdi |
| 2026-07-15 | F4 | Contratti live per actor snapshot/OBB, contact-manifold normalizzata e wrapper callback vendor-preserving; transizioni contatto born/terminated/persistent | 93 test Rulebook v2 passati, 1 skip PG per fixture non montata, Ruff e `git diff --check` verdi; resta il test differenziale con simulatore reale |
| 2026-07-15 | F4 | Hook installabile sul confine `setContactAddedCallback`, callback vendor e return value preservati; test di ordine e boundary fake | 94 test Rulebook v2 passati, 1 skip PG per fixture non montata, Ruff e `git diff --check` verdi; resta il test differenziale PG/ScenarioNet |
| 2026-07-15 | F4 | Regressione v1/reward/wiring dopo gli export e il collision hook | 40 test mirati passati (evaluator v1, reward manager/wrapper, runtime wiring, contratti v2); reward Gymnasium invariato |
| 2026-07-15 | F4/F3 | Corretto glob ricorsivo della fixture PG e irrigiditi gli ID actor/contact (nessuna coercizione silenziosa) | Suite live/snapshot/PG: 11 passati; suite Rulebook v2 cumulativa: 95 passati; Ruff e diff check verdi |
| 2026-07-15 | F4 | Test differenziale reale su `ScenarioEnv` Waymo: baseline vs callback hook, rollout sequenziale con seed/azioni identici | 1 test integration passato; osservazioni, reward, terminazioni e truncation identici; F4 completata |
| 2026-07-15 | F5 | Chiusura test ownership/atomicità: doppio writer, geometrie cache discordanti, lazy zone preesistente e conflitto dopo proposta `MemoryDelta` | 13 test F5 mirati passati, Ruff e `git diff --check` verdi; F5 completata |
| 2026-07-15 | F6 | Verifica boundedness/status R1-R2: onset simultanei/tangenziali/static/VRU e saturazione; RSS senza candidato; TTC parallelo/overlap/horizon; clearance per classe | 103 test Rulebook v2 passati, Ruff e `git diff --check` verdi; F6 completata |
| 2026-07-15 | F7 | Verifica finale R3 strada/controlli/precedenze, inclusi DEC-005 e lifecycle eventi | 20 test mirati F7 passati; suite v2 cumulativa a 103 test, Ruff e `git diff --check` verdi; F7 completata |
| 2026-07-15 | F8 | Verifica finale aggregazione/progresso/monitor transazionale e audit `current_sdc_route` | 31 test mirati passati; nessun accesso runtime a `current_sdc_route`; F8 completata |
| 2026-07-15 | F9 | Wrapper/output v2: metadata macro ordinato, agent metrics, artifact/trajectory diagnostics, criticality v2 e isolamento per-env | 13 test F9 mirati passati; reward nativo invariato, Ruff e `git diff --check` verdi; F9 completata |
| 2026-07-15 | F10 | Protocollo `b_e`: prove target 5/10/15/20, validità, quantile order-statistic `lower`, floor a 0.1 e cap 4.0; artifact hash-validato | 21 test calibrazione/RSS/contratti passati, Ruff e `git diff --check` verdi; pilot e freeze F10 restano aperti |
| 2026-07-15 | F10 | Persistenza/ricarica JSON dell’artifact `b_e` con schema, metadati protocollo e controllo config hash; smoke PG/Waymo | 5 test artifact passati; 7 smoke/converter test passati; Ruff e diff check verdi; suite obbligatoria/pilot/freeze restano aperti |
| 2026-07-15 | F10 | Verifica cumulativa dopo calibrazione e output F9; audit runtime su `current_sdc_route` | 108 test Rulebook v2 passati; Ruff e `git diff --check` verdi; l'unica occorrenza è documentale nel contratto wrapper |
| 2026-07-15 | F10 | Type-check mirato del nuovo modulo calibrazione | `mypy --ignore-missing-imports src/thesis_rl/rulebook/v2/calibration.py`: success; type-check package completo resta aperto come ISS-012 |
| 2026-07-15 | F10 | Audit type-check package v2 | `mypy --ignore-missing-imports src/thesis_rl/rulebook/v2`: 58 errori in 12 file; il controllo completo senza ignore segnala 80 errori includendo stubs Shapely/Panda3D; ISS-012 resta aperto |
| 2026-07-15 | Regressione/F10 | Verifica suite completa repository dopo calibrazione e smoke PG/Waymo | 420 test passati, 8 falliti fuori dal perimetro Rulebook v2 (ISS-011); nessun nuovo fallimento v2; il comando è stato eseguito nel container `dev` |
| 2026-07-15 | F10 | Chiusura tranche di verifica dopo correzione import Ruff nel driver Scenario ACL | 112 test mirati Rulebook v2/Scenario ACL passati; Ruff mirato passato; `git diff --check` pulito; suite completa confermata a 420 passati e 8 falliti esterni (ISS-011) |
| 2026-07-15 | F10 | Loader artifact: validazione completa dei target, cap normativo e valore positivo/finito; rifiuto artifact oltre cap | 109 test Rulebook v2 passati; Ruff e `git diff --check` verdi; artifact reale e pilot restano aperti (ISS-007) |
| 2026-07-15 | F10 | CLI `rulebook_v2_calibrate` e `rulebook_v2_pilot`; pilot offline stratificato su 2 PG per profilo (5 profili) + 10 Waymo | Report generato in `data/scenarionet/rulebook_v2/pilot/offline_pilot_20260715.json`: 8 conversioni con task-route eligibility differita (hash non forniti), 4 esclusioni per signal `UNKNOWN`, 8 eccezioni adapter; mean conversion 0.529 s, p95 lower 2.192 s; non è ancora costo monitor per-step né eligibility finale |
| 2026-07-15 | Regressione/F10 | Suite completa dopo le CLI F10 e il pilot offline | 423 test passati, 8 falliti fuori dal perimetro v2 (ISS-011); i test Rulebook v2/CLI restano verdi; Ruff e `git diff --check` verdi |
| 2026-07-15 | F10 | Target Makefile e documentazione operativa per calibrazione, validazione artifact, pilot preliminare/finale e check v2 | `make rulebook-v2-init`, `rulebook-v2-calibrate`, `rulebook-v2-validate-calibration`, `rulebook-v2-pilot`, `rulebook-v2-pilot-final`, `rulebook-v2-f10`; i target rifiutano input mancanti e non generano fallback sintetici |
| 2026-07-15 | F10 | Verifica reale del target Makefile `rulebook-v2-check` | 111 test Rulebook v2 passati; Ruff passato; `git diff --check` pulito |
| 2026-07-15 | F10 | Runner automatico `rulebook_v2_braking_trials` su pista `SSSSSSSSSS` senza traffico | Smoke reale 1 prova per target (4 prove) completato nel container; il runner misura il tratto 90%→10%, verifica lane/collisioni e produce il JSON trials; le 40 prove finali richiedono il file ego congelato dell'utente |
| 2026-07-15 | F10 | Verifica completa runner automatico su banco MetaDrive parametrico | 40/40 prove generate (10 per target) nel container con fixture di test; il risultato non viene usato come artifact finale perché la fixture vehicle config non è dichiarata identica al training ego |
| 2026-07-15 | Regressione/F10 | Suite completa dopo runner automatico calibrazione | 424 test passati, 8 falliti fuori dal perimetro v2 (ISS-011); nessun nuovo fallimento Rulebook v2; Ruff e `git diff --check` verdi |
| 2026-07-15 | F10 | Predisposizione configurazione ego per il target Make | Creato `data/scenarionet/rulebook_v2/ego_config.json` dai default MetaDrive correnti (`vehicle_model=default`); prove reali e sostituzione in caso di override fisici restano prerequisiti del freeze |
| 2026-07-15 | F10 | Raccolta reale con configurazione ego predisposta | 40/40 prove MetaDrive completate; dopo il taper di avvicinamento al target risultano 10/10 valide per 5, 10, 15 e 20 m/s |
| 2026-07-15 | F10 | Calibrazione e validazione artifact `b_e` | Artifact scritto e ricaricato con hash `ce25f5a02c7be3974048ac8d3f664a43a73715c7a26882f226982d28b444cf00`; `ego_min_brake_mps2=4.0`; target `make rulebook-v2-validate-calibration` passato |
| 2026-07-15 | F10 | Pilot offline preliminare dopo calibrazione | `make rulebook-v2-pilot` passato; report con 8 `adapter_exception`, 4 `adapter_excluded` e 8 `task_route_deferred_missing_hash`; non è ancora eligibility finale |
| 2026-07-15 | F10 | Regressione dopo correzione overshoot runner e calibrazione reale | `make rulebook-v2-check`: 112 test passati, Ruff passato, `git diff --check` pulito |
| 2026-07-16 | F10 | Pilot finale con hash ego/geometria forniti | `hashes_supplied=true`, 20 scenari campionati: 8 `task_route_eligible`, 4 `adapter_excluded` per `signal_state_unknown`, 8 `adapter_exception` (7 associazioni lane ambigue/non disponibili, 1 geometria GEOS degenerata); il pilot ha completato il contratto hash ma l'eligibility finale resta aperta |
| 2026-07-16 | Regressione/F10 | Check dopo aggiornamento ISS esterni | `make rulebook-v2-check`: 112 test passati, Ruff passato, `git diff --check` pulito |
| 2026-07-15 | F10/ISS-011 | Correzioni suite esterna: forced-rule v1 esplicito (`full` + scales), aspettativa preset Hydra allineata a `td3_sb3`, validazione ACL e firme compatibili (`chunk_id`, `rng`), smoke ScenarioNet con verifica preventiva catalogo/runtime reale | Test mirati: 37 passati, 2 skip per runtime/cataloghi reali incoerenti; Ruff sui file modificati passato; `git diff --check` passato; nessun accesso a `current_sdc_route` e nessuna modifica al core Rulebook v2 |
| 2026-07-15 | F10/ISS-011 | Suite completa finale e audit type-check | `pytest -q`: 430 passati, 2 skipped; `mypy --ignore-missing-imports src/thesis_rl/rulebook/v2`: 58 errori legacy/typing; `ruff check src tests`: 9 errori preesistenti fuori dai file modificati; `git diff --check` pulito |
| 2026-07-16 | F10/ISS-011/012 | Allineati gli smoke alla catalog/runtime ufficiale coerente e sanate le annotazioni v2 | `pytest -q`: 432 passati, 0 skipped; `ruff check src tests`: passato; `mypy --ignore-missing-imports src/thesis_rl/rulebook/v2`: success su 40 file; senza ignore restano 22 errori esclusivamente di stub Shapely/Panda3D; `git diff --check`: passato |
| 2026-07-16 | F10/ISS-001/002/004 | Correzione pilot reale: transizioni PG al boundary risolte solo con continuità canonica; route Waymo non univoche, geometrie degeneri e segnali pertinenti `UNKNOWN` diventano esclusioni tipizzate; segnali non raggiungibili non escludono | Pilot finale hash-validato: 13/20 eleggibili, 7 esclusi, 0 `adapter_exception`; cause: 4 route non map-matchabili, 1 scenario con due boundary a punto singolo, 2 scenari con signal `UNKNOWN` pertinente; mean 0.636 s, p95 lower 2.219 s |
| 2026-07-16 | Regressione/F10 | Verifica cumulativa dopo i fix adapter | 28 test mirati passati; `make rulebook-v2-check`: 117 passati, Ruff e diff check verdi; suite completa: 437 passati in 28.56 s |
| 2026-07-16 | F10 | Filtro Rulebook v2 integrato prima dello split ScenarioNet | Nuova CLI filtra il catalogo grezzo, salva `catalog_eligibility.json` con esiti/hash/cause e passa soltanto il catalogo eleggibile a split e runtime view; 4 test dedicati e 38 test pipeline/v2 mirati passati; esecuzione sull'intero catalogo resta da effettuare |
| 2026-07-16 | F10 | Verifica integrazione filtro a monte | `pytest -q` completo terminato con successo; Ruff sui file aggiunti/modificati passato; `git diff --check` pulito. L'esecuzione sul catalogo ScenarioNet completo resta un'attività dati separata e produrrà l'artifact di audit definitivo. |

---

## Definition of Done complessiva

Il rulebook v2 è implementato quando:

- [ ] tutti i test obbligatori della specifica passano;
- [ ] monitor, evaluator, memoria e cache rispettano la purezza transazionale;
- [ ] il vettore contiene esattamente quattro margini ordinati;
- [ ] quantità raw e sottocomponenti restano disponibili nei diagnostics;
- [ ] PG e Waymo usano gli stessi record e formule canoniche;
- [ ] nessuna future track viene consultata online;
- [ ] task route e eligibility artifact sono versionati e riproducibili;
- [ ] il contact hook non modifica la dinamica;
- [ ] `NOT_EVALUABLE` è sempre fail-fast nei run conformi;
- [ ] memoria/cache di env vectorized sono isolate;
- [ ] reward nativo resta invariato nella modalità monitor-only;
- [ ] calibrazione `b_e` e config ego hanno hash corrispondenti;
- [ ] report di eleggibilità ed esclusioni è disponibile;
- [ ] non esistono problemi bloccanti o deviazioni non approvate;
- [ ] documentazione e tracker riflettono lo stato reale.

---

## Prossimo passo consigliato

Completare F0 e procedere con F1–F4 in questo ordine di rischio:

1. contratti e registry;
2. spike task-route Waymo/PG su poche fixture reali;
3. primitive route/2.5D minime necessarie allo spike;
4. spike contact hook;
5. soltanto dopo, estendere sistematicamente tutte le formule.

Gli spike route e collisione vengono anticipati perché sono i due punti più
dipendenti dalle API e dai dati reali. Se falliscono, il problema viene discusso
prima di costruire il resto del monitor sopra assunzioni non verificate.
