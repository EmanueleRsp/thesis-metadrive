# Piano di implementazione e tracker — Rulebook v2

## Scopo

Questo documento traduce
[`rulebook_v2_spec.md`](rulebook_v2_spec.md) in un piano operativo e mantiene lo
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
**Fase corrente:** F2 — primitive geometriche canoniche e 2.5D
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
| F2 | Primitive geometriche canoniche e 2.5D | `IN_CORSO` | F1 |
| F3 | Task route, adapter statici e validazione offline | `NON_INIZIATA` | F1, F2 |
| F4 | Snapshot live e contact-onset hook | `NON_INIZIATA` | F1 |
| F5 | Eventi, memoria, cache e zone lifecycle | `NON_INIZIATA` | F2–F4 |
| F6 | R1 collisione e R2 interazione dinamica | `NON_INIZIATA` | F4, F5 |
| F7 | R3 strada, controlli e precedenze | `NON_INIZIATA` | F3, F5 |
| F8 | R4 progresso, aggregazione e monitor transazionale | `NON_INIZIATA` | F6, F7 |
| F9 | Wrapper, wiring e output del rule vector | `NON_INIZIATA` | F8 |
| F10 | Conformità, calibrazione, pilot PG/Waymo e freeze | `NON_INIZIATA` | F3–F9 |
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

**Stato:** `IN_CORSO`

### Attività

- [x] Implementare precision grid, canonical WKB e SHA-256 secondo DEC-007.
- [x] Implementare footprint OBB e validazione dimensioni.
- [x] Implementare elevation functions e compatibilità verticale 2.5D.
- [x] Implementare `RoutePolyline` e proiezione con continuità `previous_s`.
- [ ] Implementare front/rear route coordinates e swept front bumper.
- [x] Implementare lane association e gap bumper-to-bumper.
- [ ] Implementare superficie carrabile per livello verticale.
- [ ] Implementare control line canonica.
- [ ] Implementare decomposizione convessa deterministica e continuous SAT.
- [ ] Implementare `MovementKey`, corridoi e conflict-zone construction.
- [ ] Aggiungere test sintetici per geometrie concave, hole, autointersezioni
      della route, cavalcavia, merge e rotatorie.

### Criteri di uscita

- primitive deterministiche su reset ripetuti;
- nessun conflitto 2D spurio fra livelli incompatibili;
- continuous SAT conforme ai casi `NO_INTERVAL`, finito e `OPEN_END`;
- canonical bytes e ID stabili nelle fixture congelate.

---

## F3 — Task route, adapter statici e validazione offline

**Stato:** `NON_INIZIATA`

### Attività

- [ ] Definire e versionare `TaskRouteRecord`.
- [ ] Implementare map-matching offline Waymo SDC -> lane ID sequence, senza
      conservare timing o future pose nel record.
- [ ] Implementare estrazione diretta del task route PG.
- [ ] Costruire adapter PG/Waymo verso lane, map feature, logical boundary,
      traffic control, crosswalk, priority e roundabout records.
- [ ] Riutilizzare dove corretto l'estrazione topologica offline esistente,
      senza usarne le soglie euristiche come primitive runtime.
- [ ] Implementare validazione route, quote, lane polygon/width, controlli,
      signal sequences, spawn overlap e configured speed caps.
- [ ] Implementare artifact di eleggibilità separato dal generico
      `validation_status` del catalogo.
- [ ] Indicizzare l'artifact per scenario UID, versione rulebook, versione
      adapter, hash config geometrica e hash calibrazione.
- [ ] Aggiornare provider/runtime affinché il pool rulebook v2 includa soltanto
      record eleggibili.
- [ ] Produrre report di esclusione per sorgente e causa.

### Criteri di uscita

- nessuna formula runtime contiene branch `if source == waymo/pg`;
- task route disponibile senza consultare online la track SDC;
- scenari invalidi esclusi con cause tipizzate;
- distribuzione delle esclusioni misurata e registrata.

---

## F4 — Snapshot live e contact-onset hook

**Stato:** `NON_INIZIATA`

### Attività

- [ ] Implementare snapshotter live comune a PG e Waymo.
- [ ] Risolvere ID persistenti e classi canoniche degli oggetti MetaDrive.
- [ ] Catturare pose 3D, velocità, heading, footprint, lane ID e speed cap.
- [ ] Implementare un buffer episodico thread-safe/control-step-safe per i
      contact onset.
- [ ] Estendere localmente il callback collisioni vendorizzato preservando il
      callback MetaDrive originale.
- [ ] Estrarre actor ID, class, contact point e normale orientata ego -> altro.
- [ ] Deduplicare per actor ID nel control step senza perdere i contact point.
- [ ] Rilevare contatti nati e terminati tra due control frame.
- [ ] Verificare con test differenziale che l'hook non cambi la dinamica.

### Criteri di uscita

- snapshot immutable e completi;
- onset substep rilevati in modo riproducibile;
- nessun fallback a crash flag per la severità normativa;
- dinamica identica con hook attivo/disattivo a parità di seed e azioni.

---

## F5 — Eventi, memoria, cache e zone lifecycle

**Stato:** `NON_INIZIATA`

### Attività

- [ ] Implementare detector puri di crossing e occupancy pre/post.
- [ ] Implementare `ZoneLifecycleEvaluator` e `ZoneLifecycleView` secondo
      DEC-002.
- [ ] Implementare init memoria al reset, inclusi prepassed controls e
      occupazioni preesistenti.
- [ ] Implementare merge dei `MemoryDelta` con ownership DEC-008.
- [ ] Implementare overlay `cache + pending_delta`.
- [ ] Implementare merge/apply dei `CacheDelta` con confronto canonico.
- [ ] Implementare freeze/unfreeze delle `MovementKey` degli attori.
- [ ] Testare doppio writer, geometrie discordanti, lazy zone inside ego,
      eccezioni dopo proposta delta e commit unico.

### Criteri di uscita

- evaluator senza side effect;
- memoria e cache originali immutate dopo ogni errore;
- lifecycle condiviso con un solo writer;
- pending zones visibili nello stesso step senza commit anticipato.

---

## F6 — R1 collisione e R2 interazione dinamica

**Stato:** `NON_INIZIATA`

### Attività

- [ ] Implementare collision onset e severità bounded da velocità pre-state.
- [ ] Implementare floor numerico e normalizzazione per actor class.
- [ ] Implementare RSS longitudinale sulla lane association canonica.
- [ ] Implementare TTC generalized mediante continuous SAT.
- [ ] Implementare clearance per tutti gli attori live collidibili.
- [ ] Aggregare R2 con massimo e conservare worst actor/diagnostics.
- [ ] Implementare status applicabile/evaluabile per ogni componente.
- [ ] Coprire integralmente i test delle sezioni 15.1–15.4.

### Criteri di uscita

- R1 e R2 finite e bounded;
- collisione emessa soltanto all'onset;
- nessuna dipendenza dal raggio dell'osservazione;
- nessun attore perso rispetto all'iterazione completa.

---

## F7 — R3 strada, controlli e precedenze

**Stato:** `NON_INIZIATA`

### Attività

- [ ] Implementare off-road e wrong-way.
- [ ] Implementare solid-line crossing/occupancy.
- [ ] Implementare dashed-line timer con logical boundary ID.
- [ ] Implementare catalogo e macchina a stati dei signal group.
- [ ] Implementare stop zone, timer continuo/best e crossing.
- [ ] Implementare crosswalk yield mediante zone lifecycle.
- [ ] Implementare vehicle-yield scoped con i quattro predicati ammessi.
- [ ] Applicare DEC-005 agli ingressi vehicle-yield.
- [ ] Aggregare R3 con massimo conservando tutte le sottocomponenti.
- [ ] Coprire integralmente i test delle sezioni 15.5–15.9.

### Criteri di uscita

- componenti R3 conformi ai domini e ai fallback vietati;
- signal/stop selezionati lungo route, non per distanza euclidea;
- illegal-entry flags persistenti fino all'uscita completa;
- casi ambigui `NOT_APPLICABLE`, dati core mancanti fail-fast.

---

## F8 — R4 progresso, aggregazione e monitor transazionale

**Stato:** `NON_INIZIATA`

### Attività

- [ ] Implementare progresso raw e normalizzato sulla task route canonica.
- [ ] Verificare continuità fra `memory.previous_route_s_m` e pre-state.
- [ ] Implementare aggregazione R1–R4 e macro status.
- [ ] Implementare `RulebookMonitor.evaluate_transition`.
- [ ] Validare range, finitezza, complete evaluation, ownership e cache delta.
- [ ] Garantire che qualunque errore non produca un risultato parziale.
- [ ] Coprire i test delle sezioni 15.0, 15.10 e atomicità.

### Criteri di uscita

- API restituisce sempre la tripla normativa o solleva;
- margini ordinati e range conformi;
- stesso input produce risultato, memoria e delta identici;
- nessun accesso a native longitude salvo diagnostics.

---

## F9 — Wrapper, wiring e output del rule vector

**Stato:** `NON_INIZIATA`

### Attività

- [ ] Implementare `RulebookV2MonitorWrapper` monitor-only.
- [ ] Integrare reset, snapshot pre/post, onset buffer e commit atomico.
- [ ] Collegare `rulebook.version=v2` nel runtime wiring.
- [ ] Preservare il percorso v1 e i suoi config esistenti.
- [ ] Allegare output compatibile a `info` secondo DEC-003.
- [ ] Aggiornare agent metrics, eval artifacts, video diagnostics e CSV per i
      quattro nomi macro v2.
- [ ] Aggiornare rule criticality diagnostica del curriculum da v1 a v2 senza
      cambiare la usefulness primaria.
- [ ] Assicurare supporto a env vectorized con memoria/cache per-env isolate.
- [ ] Testare che il reward scalare nativo resti byte/float-equivalente in
      modalità monitor-only.

### Criteri di uscita

- training/evaluation possono leggere il rule vector senza scalarizzarlo;
- nessuna transizione invalida viene restituita al planner;
- istanze vectorized non condividono stato episodico;
- regressione v1 e native reward assente.

---

## F10 — Conformità, calibrazione, pilot PG/Waymo e freeze

**Stato:** `NON_INIZIATA`

### Attività

- [ ] Implementare il protocollo di calibrazione di `b_e`.
- [ ] Produrre artifact con hash della configurazione ego.
- [ ] Eseguire tutti i test obbligatori della sezione 15.
- [ ] Eseguire smoke deterministici su fixture PG e Waymo.
- [ ] Eseguire pilot su campione stratificato per sorgente/topologia.
- [ ] Misurare eleggibilità, cause di esclusione e costo runtime per step.
- [ ] Verificare assenza di future-track access nel monitor online.
- [ ] Verificare equivalenza geometrica e ID fra reset ripetuti.
- [ ] Eseguire suite completa, Ruff e type checking mirato.
- [ ] Aggiornare documentazione, comandi di validazione e metadata run.
- [ ] Congelare versione config, adapter, calibration e eligibility artifact.

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
| ISS-001 | `APERTO` | alta | Percentuale di scenari realmente eleggibili non ancora nota | Audit F3 su campione stratificato prima di completare tutte le regole |
| ISS-002 | `APERTO` | alta | Map-matching del task route può risultare ambiguo su junction/route parallele | Algoritmo deterministico + esclusione tipizzata; discutere solo se l'esclusione è eccessiva |
| ISS-003 | `APERTO` | media | API Panda3D per normale/contact manifold da verificare sul commit locale | Spike F4 con fixture frontale e normale invertita |
| ISS-004 | `APERTO` | media | Qualità di polygon, width e quota differisce fra PG e Waymo | Report validazione per campo e sorgente in F3 |
| ISS-005 | `APERTO` | media | `MovementKey` può essere ambigua prima della conflict zone | Restare `NOT_APPLICABLE`; misurare frequenza nel pilot |
| ISS-006 | `APERTO` | media | Costo del continuous SAT su tutti gli attori live non ancora misurato | Benchmark F10 prima di valutare DEF-PERF-001 |
| ISS-007 | `APERTO` | alta | Artifact di calibrazione `b_e` non ancora disponibile | Implementare protocollo e bloccare solo pilot/freeze finale, non unit test sintetici |
| ISS-008 | `RISOLTO` | media | Shell di lavoro senza `pytest`, Ruff, Shapely e MetaDrive | Verifiche eseguite nel container `dev`: dipendenze disponibili, 23 test e Ruff verdi; versioni registrate in F0 |
| ISS-009 | `RISOLTO` | media | `RoutePolyline` deve unificare punti consecutivi entro 1 mm in XY, ma la specifica non definisce la quota risultante se tali punti hanno `z` differenti | DEC-011: cluster XY medio, mediana z, validazione con `z_tol=3 m`; approvata dall'utente il 2026-07-15 |

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
