---
title: "Specificazione finale del rulebook per MetaDrive ScenarioEnv e ScenarioNet"
subtitle: "Regole, formule, contesto di applicazione, variabili, parametri, fonti e contratto implementativo"
author: "Report tecnico per la tesi"
date: "13 luglio 2026"
lang: it-IT
version: "4.4-final"
---

# Sintesi esecutiva

Il rulebook definitivo è costituito da quattro macro-regole in ordine totale:

$$
R_1^{\mathrm{collision}}
\succ
R_2^{\mathrm{interaction}}
\succ
R_3^{\mathrm{compliance}}
\succ
R_4^{\mathrm{progress}}.
$$

Il significato dell'ordine è:

1. minimizzare prima di tutto la severità delle collisioni realmente avvenute;
2. a parità di collisioni, minimizzare l'erosione preventiva dei margini di sicurezza;
3. a parità di sicurezza fisica, rispettare strada, direzione, lane markings, controlli di traffico e precedenze veicolari direttamente determinabili;
4. soltanto fra comportamenti equivalenti rispetto alle regole superiori, massimizzare l'avanzamento lungo la route.

La struttura deriva dal formalismo dei *Rulebook* di Censi et al. [1], nel quale ogni regola è una metrica di violazione e il preorder fra regole induce un ordine sui comportamenti. L'uso di una gerarchia compatta con sottocomponenti diagnostiche è inoltre coerente con ScenicRules [2].

Le decisioni centrali sono:

- le regole vengono calcolate **step-by-step sullo stato interno live di `ScenarioEnv`**, non sull'osservazione data alla policy;
- il learner ottimizza i ritorni scontati dei segnali per-step; il curriculum valuta separatamente la usefulness dell'intero episodio aggregando gli errori di apprendimento delle sue transizioni;
- è mantenuta solo una memoria causale minima fra step per collision onset, timer, eventi di crossing, stato del giallo e progresso;
- non si consultano future tracks ground-truth, futuri stati semaforici o intenzioni annotate;
- tutti gli scenari destinati al training rulebook-based devono superare una validazione preliminare del contratto dati; `NOT_EVALUABLE` non è una modalità ordinaria del learner ma un errore fail-fast del validatore, del wrapper o dello scenario; non richiede rollback del replay buffer;
- $R_1,R_2,R_3$ forniscono al learner costi bounded in $[0,1]$;
- $R_4$ conserva il progresso raw in metri per step nei diagnostics e usa per il learner una normalizzazione deterministica basata sul configured speed normalization cap dell'ego;
- le quantità fisiche raw restano nei diagnostics;
- sottoregole realmente discrete possono essere binarie; le violazioni dotate di una severità significativa sono continue;
- dentro una macro-regola si usa il **massimo**, non la somma: è una compressione worst-case deliberatamente lossy delle sottocomponenti diagnostiche, non un embedding fedele del loro product order;
- picco, integrale, durata e numero di eventi non alimentano learner o curriculum, ma vengono calcolati per la valutazione finale e il reporting;
- non si implementa nel core una regola giuridica generale di precedenza per ogni intersezione non controllata;
- si implementano semafori completi con costo continuo di avvicinamento e gestione della fase gialla, stop sign con control line derivata, crosswalk basato su conflict zone canoniche e una regola veicolo-veicolo limitata a quattro predicati deterministici di priorità: zona già occupata, ego soggetto a stop, ingresso in rotatoria e relazione di priorità esplicita;
- le conflict zone sono costruite da corridoi di movimento basati sui poligoni delle lane, ricevono un ID stabile e vengono mantenute in cache per l'intero episodio.

# 1. Ambito e assunzioni

## 1.1 Ambiente di esecuzione

Il rulebook è progettato per:

- `MetaDrive ScenarioEnv`;
- scenari caricati da `ScenarioDescription`;
- scenari Waymo convertiti tramite ScenarioNet;
- scenari procedurali esportati nello stesso formato;
- controllo closed-loop dell'ego tramite azioni esterne;
- stato degli altri attori letto dalla simulazione corrente.

`ScenarioEnv` materializza mappa, veicoli, oggetti e traffic lights come entità vive del simulatore e possiede già stati come collisioni, lane-line contacts, crosswalk contact, current lane, navigation longitude e route completion [7, 8]. Il monitor sfrutta tali oggetti live e le geometrie caricate nel motore.

## 1.2 Stato del simulatore e osservazione della policy

Le regole possono usare il ground truth corrente del simulatore per costruire reward e metriche. Questo non implica che le stesse variabili debbano essere passate alla policy.

Sono ammessi nel monitor:

- pose e velocità correnti;
- footprint e dimensioni;
- lane corrente e route ego;
- stato corrente degli attori attivi;
- stato corrente del semaforo;
- road-line contacts;
- geometria corrente di lane, road lines e crosswalk, oltre alle control line derivate e validate per semafori e stop sign;
- memoria causale del monitor.

Non sono ammessi:

- posizione futura registrata degli altri attori;
- route o intenzione futura ground-truth degli altri;
- futuri stati dei semafori;
- outcome futuro dello scenario;
- label del curriculum;
- difficoltà dello scenario.

## 1.3 Esclusioni esplicite

Non fanno parte della versione core:

- full RSS laterale e unstructured RSS;
- intent prediction multimodale;
- occlusion-aware reachable sets;
- safe corridor spaziotemporale completo;
- ricostruzione giuridica generale del diritto di precedenza per ogni intersezione non controllata;
- speed-limit compliance;
- comfort;
- lane centering;
- legalità rispetto a una specifica giurisdizione nazionale completa.

È invece inclusa una regola di yield **scoped** nei soli casi coperti da un predicato deterministico: conflict zone già occupata dall'altro attore, ego soggetto a stop mentre l'altro approccio non è soggetto a stop/yield, ingresso dell'ego in rotatoria con attore già sulla componente circolante, oppure relazione di priorità esplicitamente fornita e validata dalla mappa o dal wrapper. Merge senza relazione esplicita, intersezioni non controllate ambigue, entrambi gli approcci soggetti a stop e manovre dell'altro attore non univocamente ricostruibili sono `NOT_APPLICABLE`.

Queste esclusioni non impediscono di rilevare il rischio fisico: collisioni, RSS longitudinale, TTC e clearance restano attivi in ogni interazione osservabile.

# 2. Convenzioni comuni

## 2.1 Costo e margine

Per le prime tre macro-regole:

$$
c_k(t)\in[0,1],
\qquad
m_k(t)=-c_k(t),
\qquad
k\in\{1,2,3\}.
$$

Quindi:

- $c_k=0$, $m_k=0$: nessuna violazione valutabile;
- $c_k>0$, $m_k<0$: violazione;
- valore maggiore: violazione più grave.

Per il progresso si conserva la quantità fisica raw:

$$
\Delta s_{\mathrm{route}}(t)
=
s_{\mathrm{route}}(t)
-
s_{\mathrm{route}}(t-\Delta t),
$$

in metri per step. Il valore fornito al learner è:

$$
m_4(t)
=
\operatorname{clip}
\left(
\frac{
\Delta s_{\mathrm{route}}(t)
}{
v_{\max,e}\Delta t
},
-1,1
\right),
$$

dove $v_{\max,e}$ è il configured speed normalization cap dell'ego, ottenuto dalla configurazione `max_speed_m_s`: non è un parametro scelto o tarato e non viene interpretato come limite fisico invalicabile. Quindi:

- $m_4>0$: avanzamento;
- $m_4=0$: fermata;
- $m_4<0$: regressione o retromarcia.

Il valore raw $\Delta s_{\mathrm{route}}$ resta sempre nei diagnostics.

## 2.2 Perché i costi principali sono bounded

Censi et al. definiscono in modo generale regole a valori reali non negativi e non impongono il range $[0,1]$ [1]. ScenicRules usa molte metriche raw con unità fisiche differenti [2, 10]. Altri lavori di rule-based optimal control usano invece violation metrics bounded e gli algoritmi di lexicographic RL normalmente assumono reward bounded [11].

Per questa tesi si mantiene una doppia rappresentazione:

- `raw`: metri, secondi, $(m/s)^2$, timer e distanze;
- `cost`: valore dato al learner in $[0,1]$.

Questa scelta:

- rende stabile il critic;
- evita scale arbitrariamente diverse fra obiettivi;
- rende semanticamente sensato il massimo fra sottocomponenti;
- facilita la baseline scalarizzata;
- conserva l'interpretabilità attraverso i diagnostics raw.

Non è necessario che ogni sottoregola sia continua:

- eventi normativi discreti: binari;
- violazioni graduate: continue bounded.

## 2.3 Applicabilità, evaluabilità e stato

Ogni sottocomponente restituisce:

```text
cost
raw
applicable
evaluable
status
diagnostics
```

Gli stati sono:

- `NOT_APPLICABLE`: il controllo non esiste o non ricade nel dominio della formula nel contesto corrente;
- `NOT_EVALUABLE`: il controllo sarebbe applicabile, ma un'invariante del contratto dati o del wrapper non è soddisfatta;
- `SATISFIED`: applicabile, valutabile e non violato;
- `VIOLATED`: applicabile, valutabile e violato.

La distinzione operativa è:

- l'assenza contestuale di un controllo, per esempio nessun semaforo, nessun front vehicle o nessuna conflict zone pertinente, è `NOT_APPLICABLE` e produce contributo zero;
- un attore non attivo allo step corrente viene semplicemente escluso dall'insieme degli attori live e non rende la regola non valutabile;
- pose, velocità, footprint, route, lane graph e geometrie richieste dalla configurazione core devono essere disponibili dopo la validazione dello scenario;
- `NOT_EVALUABLE` non è una normale modalità di training e non viene mai convertito in costo zero.

Prima del training, ogni scenario deve superare una validazione offline e uno smoke test al reset. Il validatore può consultare l'intera descrizione dello scenario soltanto per verificarne l'eleggibilità; tali informazioni future non vengono mai esposte al monitor online o alla policy.

La validazione assegna:

```text
rulebook_eligible: bool
validation_errors: list[str]
```

Solo gli scenari con `rulebook_eligible=True` entrano nel pool rulebook-based. In particolare, per ogni semaforo pertinente alla route o a lane raggiungibili dall'ego devono essere disponibili:

- associazione valida fra segnale, lane/manovra e control point;
- sequenza temporale della lunghezza attesa;
- stati noti e validi per l'intero orizzonte dello scenario.

Uno scenario con semafori assenti resta eleggibile e la componente è `NOT_APPLICABLE`; uno scenario con un semaforo non pertinente invalido non viene escluso per questo solo motivo.

Dopo tale filtro, la comparsa di `NOT_EVALUABLE` durante il training è trattata come errore fail-fast:

1. viene sollevata `RulebookEvaluationError` con scenario ID, step, componente e causa;
2. il processo corrente viene interrotto e il replay buffer in memoria non viene salvato come stato valido;
3. lo scenario viene corretto o aggiunto alla blacklist e il run riparte da un checkpoint precedente;
4. non si implementano staging buffer episodici, rollback o cancellazioni retroattive delle transizioni.

La macro-regola aggrega soltanto componenti applicabili e valutabili e conserva la maschera delle componenti. Se nessuna componente è applicabile:

$$
c_k=0,
$$

con stato macro `NOT_APPLICABLE`.

## 2.4 Aggregazione interna

Dentro $R_2$ e $R_3$:

$$
c_k(t)
=
\max_{q\in\mathcal Q_k(t)}q(t),
$$

considerando le sole componenti applicabili e valutabili. Per tutte le massimizzazioni su insiemi di candidati si adotta la convenzione $\max\varnothing=0$.

Il massimo non implica che metri, secondi, area e violazioni normative siano fisicamente equivalenti. La semantica comune è invece:

$$
q=0
\Longleftrightarrow
\text{nessuna erosione del vincolo},
$$

$$
q\in(0,1)
\Longleftrightarrow
\text{frazione normalizzata del margine consumato},
$$

$$
q=1
\Longleftrightarrow
\text{margine esaurito o evento illegale completo}.
$$

Il massimo è preferito alla somma perché:

- non rende il reward dipendente dal numero di attori;
- non conta più volte lo stesso conflitto rilevato da RSS, TTC e clearance;
- non richiede pesi;
- rappresenta la peggiore manifestazione corrente della stessa macro-proprietà.

Questa aggregazione è una nuova macro-metrica worst-case. È deliberatamente non iniettiva: il peggioramento di una componente non massima può non modificare $c_k$. Non viene quindi presentata come aggregazione strettamente monotona capace di preservare integralmente il product order delle sottoregole di Censi et al. Le singole componenti e le rispettive quantità raw restano sempre loggate proprio per conservare l'informazione persa dalla compressione.

## 2.5 Semantica temporale e curriculum

L'addestramento usa i segnali per-step. Per ciascun obiettivo $k$, il learner ottimizza il ritorno scontato:

$$
G_k(t)
=
\sum_{j=0}^{T-t-1}
\gamma^j m_k(t+j).
$$

Questa è una realizzazione RL del rulebook mediante ordinamento lessicografico degli **expected discounted returns** delle macro-metriche per-step. Non è equivalente all'ordinamento diretto di Censi et al. sulle realizzazioni complete: a parità di severità istantanea, una violazione più lontana nel tempo pesa meno, mentre una violazione persistente accumula più costo di una breve.

Non è necessario trasformare il segnale primario in un peak reward o integral reward episodico:

- la collisione viene emessa soltanto all'onset e rappresenta un evento;
- off-road, wrong-way, clearance e altre condizioni persistenti restano attive finché persiste la violazione e ne rappresentano anche la durata.

Il curriculum è separato dalla semantica del rulebook. La usefulness di uno scenario è calcolata a posteriori aggregando sulle sue transizioni l'errore dell'oggetto di valore usato dall'algoritmo, per esempio Bellman residual assoluto medio per TD3/SAC o positive GAE per PPO. Il curriculum non usa picco o integrale dei costi del rulebook come definizione primaria della usefulness.

Per valutazione e reporting si calcolano comunque:

$$
V_k^{\mathrm{peak}}=\max_t c_k(t),
$$

$$
V_k^{\mathrm{sum}}=\sum_t c_k(t)\Delta t,
$$

oltre a durata e numero di eventi. Queste statistiche non alimentano learner o curriculum, ma sono richieste per distinguere frequenza, picco e persistenza delle violazioni e per verificare eventuali effetti del discounting.

## 2.6 Tolleranze geometriche e crossing

Le tolleranze usate per compensare discretizzazione, precisione floating-point e spessore delle polylines non sono parametri comportamentali e non fanno parte di uno sweep. Si adottano valori ingegneristici congelati, molto inferiori alle soglie semantiche delle regole:

| Quantità | Valore | Funzione |
|---|---:|---|
| griglia di precisione geometrica | $10^{-3}\,m$ | stabilizzazione delle operazioni geometriche [13] |
| $\varepsilon_A$ | $10^{-4}\,m^2$ | area esterna trascurabile per l'off-road |
| $\varepsilon_{\mathrm{geom}}$ | $10^{-2}\,m$ | buffer di $1\,cm$ per boundary rappresentate come polylines |
| $\varepsilon_{\delta}$ | $5\cdot10^{-2}\,m$ | deadband di $5\,cm$ sulle signed distances da control line e conflict-zone entry |

La griglia di precisione viene applicata senza modificare intenzionalmente la semantica della mappa; una geometria che collassa o diventa invalida dopo la normalizzazione non supera la validazione offline.

Un crossing di control line, linea continua o ingresso di conflict zone viene rilevato mediante il segmento percorso dal front bumper fra due control step, combinato con la signed distance e la deadband $\varepsilon_{\delta}$. In questo modo un salto fra frame non perde l'evento e il rumore vicino alla linea non genera crossing ripetuti.

I valori vengono verificati una volta con test unitari su PG e Waymo. Il test serve soltanto a confermare l'assenza di falsi crossing, oscillazioni numeriche e falsi off-road; non seleziona i valori in base alla performance del learner.

## 2.7 Orizzonte di previsione e insieme dei candidati

Le previsioni cinematiche locali di TTC, crosswalk e yield usano un unico orizzonte:

$$
T_{\mathrm{pred}}=3.0\,s.
$$

Tre secondi permettono di anticipare conflitti imminenti mantenendo limitato l'errore del modello a velocità e heading costanti. Non vengono consultate future tracks ground-truth.

L'insieme dei candidati del monitor è indipendente dal raggio e dalla configurazione dell'osservazione della policy. Se è necessaria una query spaziale broad-phase, il raggio viene derivato a ogni step da limiti conservativi che combinano configurazione e velocità realmente osservate:

$$
v_{e}^{\mathrm{bound}}(t)
=
\max\left\{
 v_{\max,e},
 \lVert\mathbf v_e(t)\rVert
\right\},
$$

$$
v_{i}^{\mathrm{bound}}(t)
=
\max\left\{
 v_{\max,i},
 \lVert\mathbf v_i(t)\rVert
\right\},
$$

$$
v_{\mathrm{rel}}^{\mathrm{bound}}(t)
=
v_{e}^{\mathrm{bound}}(t)
+
\max_i v_i^{\mathrm{bound}}(t).
$$

Il raggio broad-phase è:

$$
R_{\mathrm{monitor}}(t)
=
\max
\left\{
 d_{\mathrm{safe}}^{\mathrm{bound}}(t),
 v_{\mathrm{rel}}^{\mathrm{bound}}(t)T_{\mathrm{pred}},
 D_{\min}^{\max}
\right\}
+r_e^{\mathrm{circ}}+r_{\mathrm{actor}}^{\mathrm{circ}}
+\varepsilon_{\mathrm{geom}},
$$

dove:

- $d_{\mathrm{safe}}^{\mathrm{bound}}(t)$ è la distanza RSS ottenuta usando $v_e^{\mathrm{bound}}(t)$ e un front vehicle fermo;
- $D_{\min}^{\max}$ è la massima soglia di clearance;
- $r_e^{\mathrm{circ}}$ e $r_{\mathrm{actor}}^{\mathrm{circ}}$ sono raggi circoscritti conservativi dei footprint.

Il superamento temporaneo di `max_speed_m_s` non invalida lo scenario: viene incluso nei bound tramite le velocità osservate e registrato nei diagnostics. Il raggio resta una quantità derivata, non un nuovo iperparametro da tarare.

## 2.8 Corridoi di movimento e conflict zone canoniche

Questa sezione definisce un'unica costruzione geometrica condivisa da crosswalk e yield veicolo-veicolo.

### 2.8.1 Movimento e corridoio

Un movimento locale $m$ è una sequenza diretta e ordinata di lane:

$$
m=(\ell_0,\ell_1,\ldots,\ell_n).
$$

Il relativo corridoio è:

$$
C(m)=\bigcup_{\ell\in m}P_\ell,
$$

dove $P_\ell$ è il poligono live della lane in `ScenarioEnv`.

Per l'ego, $m_e$ è la sequenza di lane determinata dalla route corrente. Per un altro attore $i$, $m_i$ parte dalla lane corrente e segue soltanto successori univoci entro una lunghezza locale deterministica:

$$
D_i^{\mathrm{corr}}(t)
=
v_i^{\mathrm{bound}}(t)T_{\mathrm{pred}}+L_i,
$$

con:

$$
v_i^{\mathrm{bound}}(t)
=
\max\left\{
 v_{\max,i},
 \lVert\mathbf v_i(t)\rVert
\right\},
$$

dove $L_i$ è la lunghezza del footprint dell'attore. Dal punto corrente si aggiungono lane successive finché si verifica la prima delle condizioni seguenti:

- la lunghezza cumulativa coperta raggiunge $D_i^{\mathrm{corr}}(t)$;
- il lane graph termina;
- l'insieme dei successori non è singleton.

Se compare una diramazione prima che la conflict zone pertinente possa essere determinata, e nessuna relazione esplicita identifica il movimento, la componente è `NOT_APPLICABLE`; non si seleziona arbitrariamente una futura intenzione.

L'identificatore canonico di un movimento è la sequenza degli ID delle lane che lo compongono:

$$
\operatorname{movement\_id}(m)
=
(\operatorname{lane\_id}(\ell_0),\ldots,\operatorname{lane\_id}(\ell_n)).
$$

### 2.8.2 Intersezione fra movimenti

Per due movimenti:

$$
\widetilde Z_{e,i}=C(m_e)\cap C(m_i).
$$

Le componenti con area non superiore a $\varepsilon_A$ vengono scartate. Se l'intersezione contiene più componenti connesse, si seleziona quella raggiunta per prima dall'ego lungo la route:

$$
Z_{e,i}
=
\underset{Z\in\operatorname{CC}(\widetilde Z_{e,i})}{\arg\min}
\;s_e^{\mathrm{entry}}(Z),
$$

dove $s_e^{\mathrm{entry}}(Z)$ è la longitudine route del primo punto di ingresso dell'ego nella componente $Z$. Se non rimane alcuna componente valida, non esiste una conflict zone pertinente.

### 2.8.3 Merge e ingresso in rotatoria

Quando due movimenti confluiscono nella stessa lane ricevente, l'intersezione dei corridoi può estendersi lungo tutta la lane comune. In questo caso la conflict zone è la porzione iniziale della lane comune misurata dal punto di confluenza:

$$
Z_{e,i}=P_{\ell_c}[0,L_Z],
$$

con:

$$
L_Z=2L_{\max}^{\mathrm{veh}},
$$

dove $L_{\max}^{\mathrm{veh}}$ è la massima lunghezza dei veicoli supportati nello scenario. $L_Z$ è una dimensione geometrica derivata, non un parametro del reward da sottoporre a tuning.

### 2.8.4 Crosswalk

Per un crosswalk $W$ si calcolano le componenti connesse di:

$$
\widetilde Z_W=W\cap C(m_e).
$$

Ogni componente viene ordinata secondo la propria longitudine d'ingresso sulla route ego. Allo step corrente è pertinente la prima componente ancora davanti all'ego oppure quella attualmente occupata dall'ego:

$$
Z_W
=
\operatorname{firstAheadOrOccupiedComponent}
\left(
\operatorname{CC}(\widetilde Z_W)
\right).
$$

In questo modo porzioni indipendenti dello stesso grande poligono di crosswalk non vengono confuse fra loro.

### 2.8.5 ID stabile e cache

Ogni zona riceve un ID stabile:

$$
\operatorname{zone\_id}(Z)
=
(
\texttt{scenario\_id},
\texttt{zone\_type},
\operatorname{movement\_id}(m_e),
\operatorname{other\_movement\_id},
k
),
$$

dove `other_movement_id` è l'ID del movimento dell'altro veicolo, il crosswalk ID oppure `NONE`, e $k$ è l'indice della componente connessa ordinata lungo la route ego.

Le zone vengono costruite al reset quando tutti i movimenti sono già noti, oppure lazy alla prima richiesta; in entrambi i casi vengono inserite in una cache episodica. A parità di `zone_id`, la geometria non viene ricalcolata né modificata fino alla fine dell'episodio. I flag di ingresso illegittimo usano `zone_id` come chiave e vengono rimossi quando l'ego lascia la zona.

# 3. RulebookMonitor

## 3.1 Interfaccia

$$
(\mathbf m_t,h_t)
=
\operatorname{RulebookMonitor}
(\operatorname{EnvState}_t,h_{t-1}),
$$

dove:

- $\operatorname{EnvState}_t$ è lo stato live di `ScenarioEnv`;
- $h_{t-1}$ è la memoria minima del monitor;
- $\mathbf m_t=[m_1,m_2,m_3,m_4]$.

## 3.2 Memoria minima

```python
RulebookMemory:
    previous_contact_ids
    previous_actor_velocities

    active_dashed_boundary_id
    dashed_line_timer

    previous_signal_state
    yellow_must_stop
    previous_signal_stop_delta
    active_signal_id
    signal_resolved

    stop_continuous_timer
    stop_best_timer
    previous_stop_delta
    active_stop_id
    stop_resolved

    crosswalk_illegal_entry_active_by_zone
    vehicle_yield_illegal_entry_active_by_zone
    preexisting_ego_occupancy_by_zone

ConflictZoneCache:
    zone_geometry_by_stable_id
    zone_metadata_by_stable_id
```

`ConflictZoneCache` è stato statico episodico, non memoria comportamentale: conserva le geometrie canoniche definite nella Sezione 2.8 e garantisce che una stessa zona non cambi durante l'episodio.

Il monitor mantiene esplicitamente la memoria causale elencata in `RulebookMemory`. L'osservazione della policy deve contenere le stesse informazioni decisionali oppure una history sufficientemente lunga da permetterne la ricostruzione. La scelta fra timer espliciti, feature-history selettiva, frame stack completo o encoder ricorrente appartiene alla specifica dell'osservazione e non modifica le formule del rulebook.

### 3.2.1 Inizializzazione al reset

La memoria viene inizializzata dallo stato live subito dopo `env.reset()`, prima che la policy produca la prima azione:

```text
previous_contact_ids = ID dei contatti già attivi al reset
previous_actor_velocities = velocità iniziali degli attori live
previous_signal_state = stato corrente del segnale pertinente, se presente
previous_signal_stop_delta = signed distance corrente dalla relativa control line
previous_stop_delta = signed distance corrente dal prossimo stop pertinente
```

Timer, migliori dwell e flag `resolved` vengono inizializzati a zero o `False`, salvo quanto esplicitamente previsto per una fase gialla già attiva.

Se al reset il footprint ego occupa già una conflict zone di crosswalk o vehicle-yield, il relativo `zone_id` viene inserito in `preexisting_ego_occupancy_by_zone`. Tale occupazione non genera un evento di ingresso e viene ignorata fino alla prima uscita dell'ego dalla zona. Alla prima uscita, il flag viene rimosso; eventuali ingressi successivi nello stesso episodio vengono valutati normalmente. In questo modo il monitor non attribuisce alla policy una collisione o un ingresso avvenuti prima dell'inizio dell'episodio.

## 3.3 Ordine per-step

Dopo ogni `env.step(action)`:

1. leggere ego e attori live;
2. leggere contatti e geometria corrente;
3. calcolare le componenti;
4. produrre il vettore di margini;
5. aggiornare la memoria;
6. loggare diagnostics e masks.

## 3.4 Politica sui fallback

Sono vietati:

- semaforo vicino usato al posto di quello associato alla route;
- lane approssimata quando l'associazione è ambigua;
- stop/control line derivata senza poter verificare lane, posizione e orientamento;
- valore zero restituito dopo eccezione;
- future trajectory usata per compensare un'informazione mancante.

La pipeline deve:

1. validare preliminarmente gli scenari e costruire un pool `rulebook_eligible`;
2. fallire nei test quando un dato core non è esposto dal wrapper;
3. sollevare `RulebookEvaluationError` se emerge `NOT_EVALUABLE` a runtime;
4. non salvare come checkpoint valido il replay buffer del processo fallito.

Non sono previsti fallback silenziosi, rollback episodici o staging buffer delle transizioni.

# 4. Gerarchia

## 4.1 $R_1\succ R_2$

$R_1$ misura un impatto realmente avvenuto; $R_2$ misura un pericolo prima dell'impatto. Il segnale denso di $R_2$ aiuta il learning, ma una collisione resta il fallimento prioritario.

## 4.2 $R_2\succ R_3$

$R_1\succ R_3$ non basta. Due traiettorie possono essere entrambe collision-free ma avere margini di sicurezza molto diversi. Senza $R_2$, una quasi-collisione legalmente in-lane potrebbe essere preferita a una breve deviazione illegale che mantiene ampio margine.

La priorità esprime quindi:

> evitare un pericolo fisico grave può richiedere una deviazione normativa temporanea.

È coerente con gli esempi di Censi et al. e ScenicRules in cui evitare una collisione prevale sul lane keeping [1, 2].

## 4.3 $R_3\succ R_4$

Il progresso non giustifica rosso, stop ignorato, off-road, contromano o lane-marking violation. $R_4$ è l'ultimo tie-breaker.

## 4.4 Nota non normativa sull'uso thresholded

La definizione del rulebook e il monitor non introducono soglie di equivalenza fra obiettivi. Tali soglie appartengono all'algoritmo lessicografico impiegato successivamente e non modificano $c_k$ o $m_k$.

Come riferimento per una futura selezione thresholded fra un insieme discreto o campionato di candidate actions, indipendente dalla scala assoluta dei critic, sul set sopravvissuto $A_{k-1}(s)$ si può definire:

$$
\Delta Q_k(s)
=
\max_{a\in A_{k-1}(s)}Q_k(s,a)
-
\min_{a\in A_{k-1}(s)}Q_k(s,a),
$$

$$
A_k(s)
=
\left\{
a\in A_{k-1}(s):
Q_k^{\max}(s)-Q_k(s,a)
\le
\eta_k\Delta Q_k(s)
\right\}.
$$

Se $\Delta Q_k(s)\le10^{-8}$, tutte le azioni vengono mantenute perché il critic non le distingue numericamente. Configurazioni di riferimento:

| uso algoritmico futuro | $\eta_1$ | $\eta_2$ | $\eta_3$ |
|---|---:|---:|---:|
| strict | $0$ | $0$ | $0$ |
| near-strict consigliata | $0.001$ | $0.005$ | $0.005$ |
| eventuale ablation più permissiva | $0.0025$ | $0.01$ | $0.01$ |

Soltanto la prima configurazione è lessicografia stretta in senso matematico. Le altre mantengono un insieme di azioni quasi equivalenti e devono essere descritte come thresholded o near-strict. I valori sono riferimenti ingegneristici coerenti con gli slack task-dependent discussi nella letteratura lexicographic RL [11], non soglie universali. Questa sottosezione è informativa e non è richiesta per implementare il rulebook.

# 5. $R_1$ — Collision impact severity

## 5.1 Significato

$R_1$ valuta soltanto l'inizio di nuovi contatti fisici tra ego e:

- veicoli;
- pedoni;
- ciclisti;
- oggetti statici;
- edifici, guardrail, boundary fisici e sidewalk.

Non include TTC, clearance, off-road puramente geometrico o lane-line contact.

## 5.2 Collision onset

Sia $\mathcal C_t$ l'insieme degli ID con cui l'ego è in contatto nello step $t$:

$$
\mathcal C_t^{\mathrm{new}}
=
\mathcal C_t\setminus\mathcal C_{t-1}.
$$

La regola viene emessa soltanto per $\mathcal C_t^{\mathrm{new}}$. Questo evita di penalizzare ripetutamente lo stesso contatto per tutta la persistenza del manifold fisico.

## 5.3 Severità raw

Per un contatto nuovo con oggetto $i$, sia $\mathcal P_i$ l'insieme dei contact point restituiti dal manifold fisico per lo stesso actor ID. Per ciascun punto $p$:

$$
u_{i,p}(t)
=
\left[
(\mathbf v_e(t-\Delta t)-\mathbf v_i(t-\Delta t))^\top
\mathbf n_{i,p}
\right]_+.
$$

La velocità normale rappresentativa dell'oggetto è:

$$
u_i(t)=\max_{p\in\mathcal P_i}u_{i,p}(t).
$$

Dove:

- $\mathbf v_e(t-\Delta t)$: velocità ego al precedente control step;
- $\mathbf v_i(t-\Delta t)$: velocità dell'altro oggetto al precedente control step, nulla per un oggetto statico;
- $\mathbf n_{i,p}$: normale unitaria del contact point $p$, orientata esplicitamente dall'ego verso l'oggetto;
- $[x]_+=\max(x,0)$.

Si usa deliberatamente l'approssimazione al precedente **control step**, non la velocità esatta del substep fisico di collisione. Questa scelta mantiene il monitor coerente con la frequenza alla quale il learner osserva e controlla l'ambiente ed evita strumentazione substep non necessaria.

Il massimo fra i contact point evita che la scelta arbitraria del primo record o la media di normali discordanti riduca artificialmente la severità.

La severità raw è:

$$
s_{1,i}^{\mathrm{raw}}=u_i^2.
$$

Unità:

$$
(m/s)^2.
$$

La componente normale evita di equiparare uno sfioramento tangenziale a un impatto frontale.

## 5.4 Costo bounded

Il cap non è un iperparametro da tarare. Si definisce:

$$
u_{\mathrm{cap},i}
=
\begin{cases}
v_{\max,e}+v_{\max,i},
&i\text{ è un veicolo},\\[2mm]
v_{\max,e},
&i\text{ è un VRU o un oggetto statico}.
\end{cases}
$$

Dove:

- $v_{\max,e}$ è il configured speed normalization cap dell'ego, ottenuto da `max_speed_m_s`;
- $v_{\max,i}$ è il configured speed normalization cap del veicolo live;
- tali quantità normalizzano e saturano il costo, ma non sono assunte come limiti fisici invalicabili;
- per VRU e oggetti statici non si introduce un ulteriore cap semantico: l'eventuale contributo della loro velocità è già presente nel closing speed raw e il rapporto viene saturato.

Il costo è:

$$
q_{\mathrm{collision},i}
=
\max
\left\{
\varepsilon_{\mathrm{col}},
\left(
\frac{\min(u_i,u_{\mathrm{cap},i})}
{u_{\mathrm{cap},i}}
\right)^2
\right\}.
$$

Infine:

$$
c_1(t)
=
\begin{cases}
0,
&\mathcal C_t^{\mathrm{new}}=\varnothing,
\\[2mm]
\displaystyle
\max_{i\in\mathcal C_t^{\mathrm{new}}}
q_{\mathrm{collision},i},
&\text{altrimenti},
\end{cases}
$$

$$
m_1(t)=-c_1(t).
$$

I configured speed normalization cap derivano dalla configurazione del veicolo o da un catalogo validato e non vengono ottimizzati. Un veicolo live che non espone un valore valido rende lo scenario non eleggibile, perché manca il normalizzatore. Un superamento temporaneo del valore configurato non invalida invece lo scenario: il rapporto di collisione viene saturato a uno e il superamento viene conservato nei diagnostics.

## 5.5 Floor di collisione

$$
\varepsilon_{\mathrm{col}}=10^{-6}.
$$

Il floor ha esclusivamente funzione numerica: garantisce che ogni nuovo contatto abbia costo strettamente positivo anche quando il closing speed normale stimato al precedente control step è nullo o quasi nullo. Non rappresenta una soglia di danno e non viene tarato. ScenicRules usa lo stesso ordine di grandezza come floor per assicurare che ogni collisione abbia violazione non nulla [10].

La tolerance numerica del monitor resta $10^{-8}$ ed è distinta dalle eventuali soglie applicate in futuro ai ritorni o agli action values di un algoritmo thresholded. Una soglia algoritmica positiva può rendere quasi equivalenti stime di ritorno molto vicine, incluse quelle influenzate da collisioni minime; in tal caso l'algoritmo è near-strict e non modifica la definizione della metrica immediata. L'evento binario `new_collision` resta sempre nei diagnostics.

## 5.6 Origine della formula

- Censi et al.: richiedono una violation metric capace di distinguere la gravità del danno [1].
- ScenicRules: usa perdita di energia cinetica e un floor non nullo [2, 10].
- Formula adottata: **derivazione originale della tesi**, basata su squared normal closing speed.

Non vengono usate masse perché non sono disponibili in modo omogeneo per tutti gli attori ScenarioNet e una massa ridotta penalizzerebbe numericamente meno una collisione con un VRU leggero. Il tipo di vittima resta nel logging; la maggiore tutela preventiva dei VRU è affidata a TTC e clearance.

## 5.7 Variabili

| Variabile | Significato | Fonte |
|---|---|---|
| $\mathcal C_t$ | contatti correnti con actor ID | contact hook custom |
| $\mathcal C_{t-1}$ | contatti precedenti | memoria |
| $\mathbf v_e(t-\Delta t),\mathbf v_i(t-\Delta t)$ | velocità al precedente control step | memoria/stato precedente |
| $\mathcal P_i,\mathbf n_{i,p}$ | contact point e normali orientate ego $\rightarrow$ oggetto | contact manifold custom |
| $v_{\max,e},v_{\max,i}$ | configured speed normalization cap | configurazione/catalogo validato |
| actor type | vehicle, VRU, static, boundary | oggetto live |

## 5.8 Comportamento

- nessun nuovo contatto: $0$;
- contatto già attivo: $0$ nello step successivo;
- nuovo impatto a bassa closing speed: costo piccolo ma positivo;
- impatto a $u=u_{\mathrm{cap}}$: $1$;
- impatti multipli simultanei: massimo.

## 5.9 Fattibilità

MetaDrive espone crash flags e stato precedente del veicolo, ma il callback upstream non conserva nell'interfaccia standard tutti i dati richiesti per la severità, in particolare actor ID, normale orientata e deduplicazione dei contact point. Si implementa quindi un hook localizzato nel collision callback/manager che produca record del tipo:

```python
ContactRecord(
    other_id,
    other_type,
    normal_ego_to_other,
)
```

Il wrapper:

- raggruppa i contact point per actor ID senza scegliere arbitrariamente il primo;
- orienta ogni normale dall'ego verso la controparte;
- usa il massimo closing speed normale fra i contact point dello stesso oggetto;
- associa il record alle velocità memorizzate al precedente control step;
- confronta gli ID con `previous_contact_ids` per rilevare l'onset.

Questa modifica è confinata al monitor di collisione e non richiede intervenire sulla dinamica del simulatore o sui substep fisici. Non è previsto un fallback alla sola crash flag per calcolare la severità.

# 6. $R_2$ — Dynamic interaction safety

## 6.1 Significato

$R_2$ misura l'ingresso dell'ego nel safety envelope di altri attori prima della collisione.

Sottocomponenti:

1. RSS longitudinale;
2. TTC generalizzato;
3. clearance geometrica.

Sono complementari:

- RSS: car following;
- TTC: traiettorie convergenti;
- clearance: passaggi ravvicinati, inclusi paralleli.

## 6.2 RSS longitudinale

### 6.2.1 Contesto di applicazione

Si applica ai veicoli frontali nella stessa corrente di traffico.

Un attore è candidato se:

- è un veicolo;
- si trova davanti lungo la reference lane locale;
- ha heading concorde con la tangente locale;
- il footprint è compatibile con la stessa lane/corrente.

La selezione usa lane e coordinate locali live di MetaDrive. Se la lane association o il verso concorde non sono determinabili in modo affidabile per uno specifico attore, RSS è `NOT_APPLICABLE` per quell'attore; TTC e clearance restano valutabili. Questo caso non invalida lo scenario.

### 6.2.2 Distanza safe

$$
d_{\mathrm{safe},i}
=
\left[
v_e\rho
+
\frac12 a_{\max}^{\mathrm{acc}}\rho^2
+
\frac{(v_e+\rho a_{\max}^{\mathrm{acc}})^2}
{2b_e}
-
\frac{v_i^2}{2b_i}
\right]_+.
$$

Dove:

- $v_e=[\mathbf v_e^\top\mathbf t_{\mathrm{lane}}]_+$: velocità longitudinale positiva dell'ego nella corrente concorde;
- $v_i=[\mathbf v_i^\top\mathbf t_{\mathrm{lane}}]_+$: velocità longitudinale positiva del front vehicle nella stessa corrente;
- $d_i$: gap bumper-to-bumper;
- $\rho$: response time;
- $a_{\max}^{\mathrm{acc}}$: massima accelerazione durante la risposta;
- $b_e$: minima decelerazione garantita ego;
- $b_i$: massima frenata assunta per il front vehicle.

La formula di $d_{\mathrm{safe}}$ è quella longitudinale RSS [3].

### 6.2.3 Costo

RSS originale classifica la distanza come safe/dangerous e non definisce il nostro reward continuo. Si adotta quindi il deficit relativo:

$$
q_{\mathrm{RSS},i}
=
\begin{cases}
\left[
1-\dfrac{d_i}{d_{\mathrm{safe},i}}
\right]_+,
&d_{\mathrm{safe},i}>0,
\\[3mm]
0,
&d_{\mathrm{safe},i}=0.
\end{cases}
$$

Poiché $d_i\ge0$, il valore è già in $[0,1]$ senza upper clipping.

Diagnostics:

$$
\Delta d_{\mathrm{RSS},i}
=
[d_{\mathrm{safe},i}-d_i]_+.
$$

### 6.2.4 Parametri

Si adottano i valori iniziali conservativi proposti dalla documentazione Intel `ad-rss-lib` [4]:

| Parametro | Valore |
|---|---:|
| $\rho$ | $1.0\,s$ |
| $a_{\max}^{\mathrm{acc}}$ | $3.5\,m/s^2$ |
| $b_e$ | $4.0\,m/s^2$ |
| $b_i$ | $8.0\,m/s^2$ |

La fonte precisa che sono suggested starting values e non costanti universali. Per la tesi sono congelati come specifica del task e non sottoposti a tuning.

MetaDrive espone engine force e brake force, ma tali forze non sono direttamente decelerazioni in $m/s^2$; non vengono quindi sostituite nella formula [8]. Prima del training si esegue un test di frenata sul modello ego effettivamente usato. Se la decelerazione minima garantita è inferiore a $4.0\,m/s^2$, il parametro $b_e$ viene sostituito con il valore garantito misurato e lo stesso valore viene riutilizzato per semaforo, crosswalk e yield. Questa è una calibrazione del modello fisico, non uno sweep del reward.

## 6.3 TTC generalizzato

### 6.3.1 Formula

Per un attore dinamico o un ostacolo statico $i$, assumendo velocità e heading costanti sul breve orizzonte:

$$
TTC_i
=
\inf
\left\{
\tau\in[0,T_{\mathrm{pred}}]:
(P_e+\mathbf v_e\tau)
\cap
(P_i+\mathbf v_i\tau)
\neq\varnothing
\right\}.
$$

Per un ostacolo statico si pone:

$$
\mathbf v_i=\mathbf0.
$$

Se non esiste intersezione entro $T_{\mathrm{pred}}=3.0\,s$, si pone $TTC_i=\infty$. Il calcolo può essere implementato con continuous separating-axis theorem sui footprint convessi, come nel codice ScenicRules [10].

### 6.3.2 Costo

$$
q_{\mathrm{TTC},i}
=
\begin{cases}
\left[
1-\dfrac{TTC_i}{T_{\mathrm{TTC},\tau_i}}
\right]_+,
&TTC_i<\infty,
\\[3mm]
0,
&TTC_i=\infty.
\end{cases}
$$

Diagnostics:

$$
\Delta TTC_i
=
[T_{\mathrm{TTC},\tau_i}-TTC_i]_+.
$$

### 6.3.3 Parametri

| Tipo | Soglia | Fonte |
|---|---:|---|
| vehicle/static obstacle | $0.8\,s$ | ScenicRules per i veicoli; estensione conservativa agli statici |
| pedestrian/cyclist | $1.0\,s$ | ScenicRules, valore del codice |
| prediction horizon | $3.0\,s$ | scelta ingegneristica congelata |

Le soglie descrivono collisione imminente, non una distanza ordinaria di comfort. L'orizzonte più lungo serve soltanto a limitare la previsione e la candidate query; poiché le soglie TTC sono inferiori, un TTC maggiore produce comunque costo zero.

### 6.3.4 Limiti

- velocità costante;
- heading costante;
- nessuna intenzione futura;
- accuratezza solo su orizzonte breve;
- per gli statici non viene modellata una traiettoria: $\mathbf v_i=0$.

Una formula più sofisticata richiederebbe motion prediction e non rientra nel core.

## 6.4 Clearance

### 6.4.1 Formula

$$
d_i^{\mathrm{poly}}
=
d(P_e,P_i).
$$

$$
q_{\mathrm{clear},i}
=
\left[
1-\frac{d_i^{\mathrm{poly}}}
{D_{\min,\tau_i}}
\right]_+.
$$

Diagnostics:

$$
\Delta d_{\mathrm{clear},i}
=
[D_{\min,\tau_i}-d_i^{\mathrm{poly}}]_+.
$$

### 6.4.2 Parametri

| Tipo | $D_{\min}$ | Fonte |
|---|---:|---|
| vehicle | $0.8\,m$ | ScenicRules |
| VRU | $1.0\,m$ | ScenicRules |
| static object | $0.5\,m$ | scelta ingegneristica congelata |

### 6.4.3 Ruolo e insieme dei candidati

La clearance copre casi non catturati dal TTC, ad esempio ego e ciclista paralleli con distanza laterale insufficiente ma traiettorie non intersecanti.

Non viene calcolata indiscriminatamente su ogni oggetto materializzato nel mondo. Si definisce un insieme $\mathcal A_{\mathrm{clear}}(t)$ che include soltanto attori e oggetti:

- live e fisicamente collidibili;
- entro il raggio derivato $R_{\mathrm{monitor}}$, indipendente dall'osservazione della policy;
- appartenenti alle classi `vehicle`, `VRU` o `static obstacle` esplicitamente supportate;
- esclusi traffic lights, lane markings e altri elementi infrastrutturali non collidibili.

Il costo è calcolato per $i\in\mathcal A_{\mathrm{clear}}(t)$. L'eventuale sovrapposizione con RSS o TTC non produce doppio conteggio perché l'aggregatore è il massimo. Il monitor può ottenere tutti gli attori live e filtrarli oppure usare una query spaziale broad-phase con il raggio derivato in Sezione 2.7; non usa il raggio dell'osservazione come parametro del reward.

## 6.5 Aggregazione

$$
q_{\mathrm{RSS}}=\max_i q_{\mathrm{RSS},i},
$$

$$
q_{\mathrm{TTC}}=\max_i q_{\mathrm{TTC},i},
$$

$$
q_{\mathrm{clear}}=\max_i q_{\mathrm{clear},i},
$$

$$
c_2(t)
=
\max
\{
q_{\mathrm{RSS}},
q_{\mathrm{TTC}},
q_{\mathrm{clear}}
\},
$$

$$
m_2(t)=-c_2(t).
$$

## 6.6 Variabili

| Variabile | Significato |
|---|---|
| ego/actor pose | posizione e orientamento correnti |
| ego/actor velocity | moto corrente |
| ego/actor footprint | distanza, TTC |
| actor type | soglie |
| ego reference lane | front-vehicle selection |
| lane tangent | velocità longitudinali |
| local longitudinal coordinate | ordinamento davanti/dietro |
| polygon distance | clearance |
| $\Delta t$ | coerenza temporale |

# 7. $R_3$ — Road and traffic-rule compliance

## 7.1 Significato

$R_3$ aggrega:

- off-road;
- wrong-way;
- linea continua;
- permanenza eccessiva sulla linea tratteggiata;
- semaforo verde/giallo/rosso con margine continuo di arresto;
- stop sign;
- yield ai crosswalk mediante conflict zone;
- yield veicolo-veicolo nei soli contesti con priorità direttamente determinabile.

Off-road, wrong-way e lane-line compliance condividono geometrie e lane graph, ma restano componenti distinte:

- off-road valuta dove si trova il footprint;
- wrong-way valuta la direzione del moto;
- lane-line compliance valuta il crossing o l'occupazione di una boundary.

Fonderle in un'unica area cancellerebbe informazioni diagnostiche e non rileverebbe, per esempio, un veicolo interamente in lane ma in movimento contromano oppure un crossing illegale già completato.

## 7.2 Off-road

### 7.2.1 Regione carrabile

Sia $C_{\mathrm{drive}}(t)$ l'unione locale delle superfici carrabili materializzate nella mappa live.

Non incorpora:

- attori dinamici;
- direzione legale;
- linee continue;
- route futura registrata.

### 7.2.2 Formula

$$
q_{\mathrm{offroad}}(t)
=
\frac{
A(P_e(t)\setminus C_{\mathrm{drive}}(t))
}{
A(P_e(t))
}.
$$

Per costruzione:

$$
q_{\mathrm{offroad}}\in[0,1].
$$

Non serve clipping. Per evitare violazioni dovute a errori di discretizzazione geometrica, si pone $q_{\mathrm{offroad}}=0$ quando l'area esterna è inferiore a una tolleranza numerica $\varepsilon_A$ fissata dai test di geometria e non trattata come parametro semantico.

### 7.2.3 Motivazione

La forma deriva dalla drivable-area violation di ScenicRules, ma è adattata dividendo per l'area del footprint [2, 10]. Il termine di distanza quadratica di ScenicRules non viene usato: dopo che il veicolo è completamente fuori, il costo massimo $1$ è sufficiente e non serve introdurre una nuova scala.

### 7.2.4 Stato live

`ego.on_lane` e `_is_out_of_road` possono essere loggati come checks, ma la formula normativa è l'area fraction. Se la geometria carrabile non è interrogabile correttamente nello scenario caricato, la sottoregola è `NOT_EVALUABLE`; non si sostituisce silenziosamente con un flag differente.

## 7.3 Wrong-way

### 7.3.1 Velocità longitudinale firmata

Sia $\mathbf t_{\mathrm{ref}}(t)$ la tangente unitaria affidabile della route o della lane locale, orientata nel verso legale del movimento ego.

$$
v_{\parallel}(t)
=
\mathbf v_e(t)^\top
\mathbf t_{\mathrm{ref}}(t).
$$

### 7.3.2 Costo

$$
q_{\mathrm{wrongway}}(t)
=
\operatorname{clip}
\left(
\frac{
[-v_{\parallel}(t)]_+
}{
v_{\max,e}
},
0,1
\right).
$$

Comportamento:

- ego fermo: $0$;
- movimento nel verso corretto: $0$;
- movimento contrario a bassa velocità: costo piccolo;
- movimento contrario con modulo pari al configured speed normalization cap: $1$.

La formula penalizza il **moto contrario**, non il solo orientamento geometrico. Evita quindi di penalizzare un veicolo momentaneamente ruotato ma fermo e rileva anche una retromarcia nel verso opposto alla route.

### 7.3.3 Diagnostics

Si loggano:

- $v_{\parallel}$;
- heading ego;
- heading della tangente;
- $\Delta\psi$;
- sorgente della tangente, `route` o `lane`.

Il coseno fra heading ego e heading lane può rimanere una diagnostica, ma non definisce il costo principale.

### 7.3.4 Applicabilità

Richiede una direzione locale affidabile. Nei connector si usa in priorità la tangente della route; sulle lane ordinarie può essere usata la tangente della lane. Se nessuna tangente orientata è definibile in modo affidabile, lo scenario non supera la validazione offline; se l'anomalia emerge a runtime viene sollevata `RulebookEvaluationError`.

## 7.4 Linea continua

### 7.4.1 Formula

Per ogni boundary continua $L_b$ si usa la polyline originaria della mappa e il medesimo buffer numerico delle altre boundary:

$$
L_b^{\varepsilon}
=
\operatorname{buffer}(L_b,\varepsilon_{\mathrm{geom}}).
$$

Sia $S_{\mathrm{front}}(t-\Delta t,t)$ il segmento percorso dal front bumper fra i due control step. Il costo è:

$$
q_{\mathrm{solid}}(t)
=
\mathbf1
\left[
\exists b:
P_e(t)\cap L_b^{\varepsilon}\neq\varnothing
\;\lor\;
S_{\mathrm{front}}(t-\Delta t,t)\cap L_b\neq\varnothing
\right].
$$

Questa forma rileva sia l'occupazione corrente sia un crossing completato fra due frame. I flag live:

```text
ego.on_white_continuous_line
ego.on_yellow_continuous_line
```

restano diagnostics e controlli incrociati, ma non costituiscono la definizione normativa primaria.

### 7.4.2 Comportamento

- nessuna occupazione o crossing: $0$;
- footprint a contatto con marking continua: $1$;
- crossing completato fra due control frame: $1$ nello step dell'evento;
- permanenza per più step: costo $1$ ripetuto;
- dopo il completamento e in assenza di nuovo contatto/crossing: costo torna a $0$.

La lane raggiunta non viene resa permanentemente illegale: si penalizza l'occupazione o il crossing della marking.

### 7.4.3 Parametri

Nessun parametro semantico. Si usa soltanto $\varepsilon_{\mathrm{geom}}=10^{-2}\,m$ come tolleranza numerica già comune alle boundary.

L'equivalenza fra risultato geometrico e flag MetaDrive deve essere verificata con scenari unitari PG e ScenarioNet. Una discordanza sistematica invalida il wrapper o la geometria della sorgente e non attiva un fallback silenzioso.

## 7.5 Linea tratteggiata

### 7.5.1 Boundary geometrica continua

Il timer non usa direttamente il collider dei singoli trattini, perché durante un unico crossing il flag fisico può alternare fra vero e falso negli spazi vuoti della marking.

Per ogni boundary tratteggiata $L_b$ si usa la polyline continua originaria della mappa. Per sola tolleranza numerica è ammesso un piccolo buffer geometrico fisso:

$$
L_b^\epsilon
=
\operatorname{buffer}
(L_b,\epsilon_{\mathrm{geom}}).
$$

L'occupazione della boundary $b$ è:

$$
I_b(t)
=
\mathbf1
[
P_e(t)\cap L_b^\epsilon\neq\varnothing
].
$$

Si seleziona la boundary tratteggiata attiva con criterio deterministico: se la boundary attiva allo step precedente è ancora intersecata viene mantenuta; altrimenti si sceglie fra quelle intersecate la minima distanza dal centro del footprint, con actor/boundary ID come tie-breaker stabile. Se ne memorizza l'ID:

$$
I_{\mathrm{dash}}(t)
=
\max_b I_b(t).
$$

Il buffer serve soltanto a compensare precisione geometrica e spessore della polyline e viene fissato in base alla rappresentazione della mappa; non è un parametro semantico del comportamento.

### 7.5.2 Timer

Sia $b_t$ l'ID della boundary tratteggiata attiva:

$$
\tau_{\mathrm{dash}}(t)
=
\begin{cases}
\tau_{\mathrm{dash}}(t-\Delta t)+\Delta t,
&I_{\mathrm{dash}}(t)=1
\land
b_t=b_{t-\Delta t},
\\
\Delta t,
&I_{\mathrm{dash}}(t)=1
\land
b_t\neq b_{t-\Delta t},
\\
0,
&I_{\mathrm{dash}}(t)=0.
\end{cases}
$$

Il passaggio a una marking diversa resetta il timer. Gli spazi grafici fra i trattini della stessa boundary non lo interrompono, perché il test usa la polyline continua.

### 7.5.3 Funzione temporale

$$
h(\tau)
=
\begin{cases}
0,
&\tau\le T_0,
\\[2mm]
\left(
\dfrac{\tau-T_0}
{T_{\mathrm{cap}}-T_0}
\right)^2,
&T_0<\tau<T_{\mathrm{cap}},
\\[4mm]
1,
&\tau\ge T_{\mathrm{cap}}.
\end{cases}
$$

$$
q_{\mathrm{dash}}(t)
=
I_{\mathrm{dash}}(t)\,
h(\tau_{\mathrm{dash}}(t)).
$$

### 7.5.4 Parametri

$$
T_0=1.0\,s,
\qquad
T_{\mathrm{cap}}=2.0\,s.
$$

- fino a un secondo: attraversamento libero;
- da uno a due secondi: penalità quadratica;
- oltre due secondi: costo massimo.

L'esponente $2$ è incorporato nella formula e non viene trattato come parametro da ottimizzare.

I valori derivano dalla decisione progettuale concordata con il relatore: due secondi sono già più che sufficienti per attraversare la marking effettiva. Non derivano dalla durata complessiva di un lane change.

### 7.5.5 Stato della policy

Il monitor mantiene il timer. La policy deve ricevere il timer, una history selettiva della boundary attiva oppure un frame stack che copra almeno:

$$
H
\ge
\left\lceil
\frac{T_{\mathrm{cap}}}{\Delta t}
\right\rceil+1.
$$

La scelta concreta viene definita nella specifica dell'osservazione.

## 7.6 Semaforo completo

### 7.6.1 Segnale pertinente

Sia:

$$
L_e(t)
\in
\{
\mathrm{GREEN},
\mathrm{YELLOW},
\mathrm{RED},
\mathrm{FLASHING\_YELLOW},
\mathrm{UNKNOWN}
\}
$$

lo stato corrente del segnale che controlla la route/manovra ego.

La regola di Maierhofer et al. distingue il semaforo pertinente in funzione della direzione di svolta e proibisce il passaggio col rosso; col giallo proibisce il passaggio se è ancora possibile arrestarsi senza superare una soglia di decelerazione [6].

Il segnale deve essere associato alla lane/manovra tramite i dynamic map states e il relativo `stop_point`. Gli scenari in cui un segnale pertinente non ha associazione valida o contiene stati `UNKNOWN`/mancanti vengono esclusi offline dal pool rulebook-based. Se tale condizione ricompare a runtime non si restituisce un risultato invalido: viene sollevata `RulebookEvaluationError`.

### 7.6.2 Distanza signed dalla control line e semantica dello step

$$
\delta_{\mathrm{sig}}(t)
=
s_{\mathrm{control}}
-
s_{\mathrm{front}}(t).
$$

- positiva: front bumper prima della linea;
- zero o entro la deadband: sulla linea;
- minore di $-\varepsilon_{\delta}$: oltre la linea.

Evento di attraversamento:

$$
X_{\mathrm{sig}}(t)
=
\mathbf1
\left[
\delta_{\mathrm{sig}}(t-\Delta t)\ge-\varepsilon_{\delta}
\land
\delta_{\mathrm{sig}}(t)<-\varepsilon_{\delta}
\land
S_{\mathrm{front}}(t-\Delta t,t)\cap L_{\mathrm{control}}\neq\varnothing
\right].
$$

La regola usa due stati del segnale:

$$
L_e^{-}(t)=L_e(t-\Delta t),
\qquad
L_e^{+}(t)=L_e(t).
$$

Il crossing prodotto dall'azione appena eseguita è giudicato con $L_e^{-}(t)$, cioè con il colore disponibile prima dell'azione. $L_e^{+}(t)$ governa invece il costo di approccio e la decisione successiva. Questa convenzione elimina l'ambiguità quando crossing e cambio di fase avvengono nello stesso `env.step`.

### 7.6.3 Evento di inizio giallo

$$
X_{\mathrm{yellow}}(t)
=
\mathbf1[
L_e(t)=\mathrm{YELLOW}
\land
L_e(t-\Delta t)\neq\mathrm{YELLOW}
].
$$

Se l'episodio inizia con semaforo già giallo, la valutazione viene inizializzata al primo step.

### 7.6.4 Possibilità di arresto

Si usa la decelerazione minima garantita già fissata per RSS:

$$
b_{\mathrm{signal}}=b_e.
$$

Sia la velocità positiva di avvicinamento alla control line:

$$
v_e^{\mathrm{app}}(t)
=
[\mathbf v_e(t)^\top\mathbf t_{\mathrm{route}}(t)]_+.
$$

Con il control timestep reale $\Delta t$:

$$
d_{\mathrm{req}}(t)
=
v_e^{\mathrm{app}}(t)\Delta t
+
\frac{(v_e^{\mathrm{app}}(t))^2}
{2b_{\mathrm{signal}}}.
$$

All'onset del giallo corrente si assegna:

$$
Y_{\mathrm{must\_stop}}^{+}
=
\mathbf1[
\delta_{\mathrm{sig}}(t)
\ge
d_{\mathrm{req}}(t)
].
$$

Negli step successivi della stessa fase, $Y_{\mathrm{must\_stop}}^{+}$ indica il valore congelato in memoria all'onset. Non viene ricalcolato continuamente, altrimenti l'ego potrebbe rendere l'arresto impossibile accelerando dopo l'inizio del giallo.

Si distinguono l'obbligo pre-azione usato per il crossing e quello corrente usato per l'approccio:

$$
I_{\mathrm{cross\_must\_stop}}(t)
=
\mathbf1
\left[
L_e^{-}(t)=\mathrm{RED}
\lor
\left(
L_e^{-}(t)=\mathrm{YELLOW}
\land
Y_{\mathrm{must\_stop}}^{-}=1
\right)
\right],
$$

$$
I_{\mathrm{approach\_must\_stop}}(t)
=
\mathbf1
\left[
L_e^{+}(t)=\mathrm{RED}
\lor
\left(
L_e^{+}(t)=\mathrm{YELLOW}
\land
Y_{\mathrm{must\_stop}}^{+}=1
\right)
\right].
$$

### 7.6.5 Costo continuo di approccio

Prima della control line:

$$
q_{\mathrm{signal}}^{\mathrm{approach}}(t)
=
\begin{cases}
\operatorname{clip}
\left(
1-
\dfrac{
\delta_{\mathrm{sig}}(t)
}{
d_{\mathrm{req}}(t)
},
0,1
\right),
&
I_{\mathrm{approach\_must\_stop}}=1,\;
\delta_{\mathrm{sig}}>0,\;
d_{\mathrm{req}}>0,
\\[4mm]
0,
&\text{altrimenti}.
\end{cases}
$$

Interpretazione:

- costo zero se l'ego dispone di spazio almeno pari alla distanza di arresto;
- costo crescente quando il margine di arresto viene consumato;
- nessuna penalità per un veicolo fermo correttamente prima della linea, perché $d_{\mathrm{req}}=0$;
- il colore resta discreto, mentre la severità cinematica è continua.

### 7.6.6 Costo finale

$$
q_{\mathrm{signal}}(t)
=
\begin{cases}
1,
&X_{\mathrm{sig}}(t)=1
\land
I_{\mathrm{cross\_must\_stop}}(t)=1,
\\[1mm]
q_{\mathrm{signal}}^{\mathrm{approach}}(t),
&X_{\mathrm{sig}}(t)=0,
\\[1mm]
0,
&\text{crossing legittimo}.
\end{cases}
$$

Comportamento:

| Caso | Costo |
|---|---:|
| verde | 0 |
| giallo, arresto non sicuro all'inizio della fase | 0 |
| rosso/giallo con obbligo, margine sufficiente | 0 |
| rosso/giallo con obbligo, margine eroso | continuo in $(0,1)$ |
| crossing illegale secondo il colore pre-azione | 1 |
| rosso o giallo scatta nello stesso step dopo un crossing iniziato col verde | 0 |
| giallo lampeggiante | regola semaforica non applicabile; restano R2 e altri controlli |
| stato `UNKNOWN` | escluso dalla validazione offline; errore fail-fast se rilevato a runtime |

### 7.6.7 Macchina a stati

Per ogni segnale pertinente si mantiene un automa deterministico. Dopo `env.step(action)` l'ordine è:

1. leggere dalla memoria $L_e^{-}$ e il valore giallo congelato pre-azione;
2. rilevare il crossing e giudicarlo con lo stato pre-azione;
3. leggere $L_e^{+}$ e rilevare l'eventuale onset del giallo corrente;
4. congelare `yellow_must_stop` per la nuova fase gialla e calcolare il costo di approccio corrente;
5. se è avvenuto un crossing, impostare `signal_resolved=True` dopo aver emesso il costo e non rivalutare lo stesso segnale;
6. quando `relevant_signal_id` cambia, resettare `previous_signal_state`, `yellow_must_stop`, `previous_signal_stop_delta` e `signal_resolved`;
7. un nuovo segnale viene attivato soltanto quando è topologicamente pertinente e non risolto.

Al reset, lo stato corrente inizializza l'automa; se il primo stato è già giallo si calcola immediatamente `yellow_must_stop`.

### 7.6.8 Origine

- stop point, direzione pertinente e logica rosso/giallo: Maierhofer et al. [6];
- distanza di arresto con $b_e$: RSS e Maierhofer [4, 6];
- deficit relativo bounded durante l'approccio: adattamento originale della tesi, analogo nella struttura al deficit RSS;
- crossing illegale saturato a uno: evento normativo completo.

La regola è una specifica funzionale del task e non una dichiarazione di piena conformità legale per ogni giurisdizione.

## 7.7 Stop sign

### 7.7.1 Applicabilità

Si applica al prossimo stop sign non ancora risolto sulla route ego, quando non esiste un semaforo attivo che governa lo stesso movimento.

Richiede:

- stop sign pertinente;
- lane controllata;
- posizione del control point;
- control line derivata;
- signed distance;
- velocità ego.

### 7.7.2 Control line derivata

ScenarioDescription/Waymo conserva per lo stop sign la posizione e le lane associate, ma non garantisce una polyline dipinta della stop line già materializzata nel mondo fisico.

Al reset il context extractor:

1. legge posizione e lane associate dallo `ScenarioDescription` originale;
2. proietta il control point sulla lane pertinente;
3. calcola la tangente locale;
4. costruisce una linea ortogonale che attraversa la superficie della lane;
5. verifica che la linea sia coerente con la conflict zone e con il verso di percorrenza.

Questa geometria viene chiamata:

```text
derived stop control line
```

e non viene presentata come stop-line ground truth dipinta. Se lane, posizione o orientamento non permettono una derivazione verificabile, lo scenario non supera la validazione.

### 7.7.3 Stop zone

$$
0
\le
\delta_{\mathrm{stop}}(t)
\le
D_{\mathrm{stop}},
$$

con:

$$
D_{\mathrm{stop}}=1.0\,m.
$$

La distanza è dal front bumper. Il valore $1\,m$ coincide con il parametro di prossimità alla stop line usato da Maierhofer et al. [6].

### 7.7.4 Standstill

$$
S_{\mathrm{standstill}}(t)
=
\mathbf1[
v_e(t)\le v_{\mathrm{stop}}
],
$$

$$
v_{\mathrm{stop}}=0.1\,m/s.
$$

Il valore coincide con la tolleranza sperimentale di standstill usata da Maierhofer et al. [6].

### 7.7.5 Timer continuo

$$
T_{\mathrm{cont}}(t)
=
\begin{cases}
T_{\mathrm{cont}}(t-\Delta t)+\Delta t,
&
0\le\delta_{\mathrm{stop}}\le D_{\mathrm{stop}}
\land
v_e\le v_{\mathrm{stop}},
\\
0,
&\text{altrimenti}.
\end{cases}
$$

### 7.7.6 Migliore fermata continua

$$
T_{\mathrm{best}}(t)
=
\max
\{
T_{\mathrm{best}}(t-\Delta t),
T_{\mathrm{cont}}(t)
\}.
$$

Questo impedisce che due fermate separate da movimento vengano sommate.

### 7.7.7 Crossing

La deadband permette all'ego di fermarsi esattamente sulla control line senza considerare il semplice contatto come attraversamento. Si definisce:

$$
X_{\mathrm{stop}}(t)
=
\mathbf1
\left[
\delta_{\mathrm{stop}}(t-\Delta t)\ge-\varepsilon_{\delta}
\land
\delta_{\mathrm{stop}}(t)<-\varepsilon_{\delta}
\land
S_{\mathrm{front}}(t-\Delta t,t)\cap L_{\mathrm{stop}}\neq\varnothing
\right].
$$

La zona $0\le\delta_{\mathrm{stop}}\le D_{\mathrm{stop}}$ resta valida per il dwell; il crossing definitivo avviene soltanto dopo aver superato la linea di più di $\varepsilon_{\delta}=0.05\,m$.

### 7.7.8 Costo

$$
q_{\mathrm{stop}}(t)
=
X_{\mathrm{stop}}(t)
\left[
1-\frac{T_{\mathrm{best}}(t)}
{T_{\min}}
\right]_+,
$$

$$
T_{\min}=1.0\,s.
$$

Veer et al. usano esplicitamente una regola di stop di un secondo [5]. La formula del deficit lineare è un adattamento quantitativo per il reward vector.

### 7.7.9 Casi

| Comportamento | Costo |
|---|---:|
| nessuna fermata | 1 |
| rolling stop sopra $0.1\,m/s$ | 1 |
| fermata continua $0.4\,s$ | 0.6 |
| due fermate da $0.6\,s$ separate | 0.4 |
| fermata continua $\ge1\,s$ | 0 |
| fermata dopo la linea | 1 |
| fermata oltre $1\,m$ prima della linea | non valida |
| ritorno indietro dopo crossing | non annulla l'evento |

### 7.7.10 Macchina a stati

1. quando `relevant_stop_id` cambia, si azzerano `T_cont`, `T_best`, `previous_stop_delta` e `stop_resolved`;
2. dopo ogni step si aggiornano prima `T_cont` e `T_best` usando lo stato corrente se l'ego è ancora nella stop zone;
3. si rileva quindi il crossing con la deadband e si calcola il costo usando il `T_best` accumulato prima della risoluzione;
4. al crossing, il costo viene emesso una volta e `stop_resolved=True`;
5. lo stesso stop non viene rivalutato dopo il crossing, anche se l'ego arretra;
6. un nuovo stop viene attivato soltanto quando è il prossimo controllo pertinente sulla route.

## 7.8 Crosswalk yield

### 7.8.1 Scopo

La regola non inferisce l'intenzione di un VRU fermo lontano dal crosswalk e non usa la sua futura track ground-truth.

Rileva invece l'ingresso dell'ego in una porzione di crosswalk che:

- interseca il corridoio della route ego;
- si trova davanti all'ego;
- è occupata o raggiunta da un pedone/ciclista con un intervallo temporale incompatibile.

La sicurezza fisica generale resta inoltre coperta da TTC e clearance in $R_2$.

### 7.8.2 Conflict zone rilevante

Le conflict zone dei crosswalk vengono costruite secondo la Sezione 2.8.4. Per ogni crosswalk $W$, la zona pertinente $Z_W$ è la prima componente connessa dell'intersezione fra il poligono del crosswalk e il corridoio della route che sia ancora davanti all'ego, oppure la componente attualmente occupata dall'ego.

Se non esiste alcuna componente valida:

$$
Z_W=\varnothing,
$$

la regola è `NOT_APPLICABLE` per quel crosswalk. La geometria e il relativo `zone_id` provengono dalla cache episodica e non cambiano mentre l'ego si avvicina o attraversa la zona.

### 7.8.3 Intervalli di occupazione previsti

Usando esclusivamente footprint, posizione e velocità correnti e un modello locale a velocità costante, la funzione operativa:

```text
predict_occupancy_interval(actor, zone, horizon=3.0s)
```

restituisce un intervallo $[t_{\mathrm{in}},t_{\mathrm{out}}]$ oppure `NO_INTERVAL`.

I casi sono definiti esplicitamente:

- attore già nella zona: $t_{\mathrm{in}}=0$;
- attore fermo nella zona: uscita `OPEN_END`;
- attore fermo fuori dalla zona: `NO_INTERVAL`;
- nessun ingresso previsto entro $T_{\mathrm{pred}}=3.0\,s$: `NO_INTERVAL`;
- ingresso ed uscita entro l'orizzonte: estremi finiti;
- ingresso entro l'orizzonte ma nessuna uscita prevista entro l'orizzonte: uscita `OPEN_END`.

Un attore con `NO_INTERVAL` non è candidato. `OPEN_END` è rappresentato con uno stato dedicato e non mediante sottrazioni numeriche fra infiniti.

Per l'ego e il VRU $j$, con intervalli $I_e=[t_{e,\mathrm{in}},t_{e,\mathrm{out}}]$ e $I_j=[t_{j,\mathrm{in}},t_{j,\mathrm{out}}]$, il gap signed è:

$$
g_j^{\mathrm{sep}}
=
\begin{cases}
\max\{t_{e,\mathrm{in}}-t_{j,\mathrm{out}},\;t_{j,\mathrm{in}}-t_{e,\mathrm{out}}\},
&\text{entrambi gli estremi di uscita sono finiti},
\\[2mm]
t_{j,\mathrm{in}}-t_{e,\mathrm{out}},
&t_{e,\mathrm{out}}\text{ è finito e }t_{e,\mathrm{out}}\le t_{j,\mathrm{in}},
\\[2mm]
t_{e,\mathrm{in}}-t_{j,\mathrm{out}},
&t_{j,\mathrm{out}}\text{ è finito e }t_{j,\mathrm{out}}\le t_{e,\mathrm{in}},
\\[2mm]
-\infty,
&\text{altrimenti}.
\end{cases}
$$

Il valore $-\infty$ indica una sovrapposizione certa o open-ended e viene gestito direttamente come rischio massimo; non viene mai calcolato come $\infty-\infty$. Un valore positivo significa che uno dei due attraversamenti termina prima dell'inizio dell'altro; un valore negativo finito indica sovrapposizione di intervalli finiti.

### 7.8.4 Margine temporale

$$
r_{\mathrm{gap},j}
=
\operatorname{clip}
\left(
\frac{
T_{\mathrm{gap}}-g_j^{\mathrm{sep}}
}{
T_{\mathrm{gap}}
},
0,1
\right),
$$

con:

$$
T_{\mathrm{gap}}=1.0\,s.
$$

Con la convenzione $r_{\mathrm{gap},j}=1$ quando $g_j^{\mathrm{sep}}=-\infty$. Il valore è un buffer del task, coerente con il response time già usato nell'RSS; non viene presentato come norma legale universale.

### 7.8.5 Commitment dell'ego

Sia $d_e^{Z_W}$ la distanza del front bumper dall'ingresso della conflict zone e sia la velocità positiva di avvicinamento lungo la route:

$$
v_e^{\mathrm{app},W}
=
[\mathbf v_e^\top\mathbf t_{\mathrm{route}}]_+.
$$

$$
r_{\mathrm{commit}}^W
=
\begin{cases}
\operatorname{clip}
\left(
1-
\dfrac{
d_e^{Z_W}
}{
d_{\mathrm{stop}}(v_e^{\mathrm{app},W})
},
0,1
\right),
&d_{\mathrm{stop}}(v_e^{\mathrm{app},W})>0,
\\[3mm]
0,
&d_{\mathrm{stop}}(v_e^{\mathrm{app},W})=0,
\end{cases}
$$

dove:

$$
d_{\mathrm{stop}}(v_e^{\mathrm{app},W})
=
v_e^{\mathrm{app},W}\Delta t
+
\frac{(v_e^{\mathrm{app},W})^2}{2b_e}.
$$

Il costo continuo di approccio è definito soltanto finché l'ego si trova prima della zona e non deriva da un'occupazione già presente al reset. Si introduce quindi il gate:

$$
I_{\mathrm{before},W}(t)
=
\mathbf1
\left[
d_e^{Z_W}(t)>\varepsilon_{\delta}
\land
P_e(t)\cap Z_W=\varnothing
\land
\operatorname{zone\_id}(Z_W)\notin
\mathcal Z_{\mathrm{preexisting}}(t)
\right].
$$

Dove $\mathcal Z_{\mathrm{preexisting}}(t)$ contiene le conflict zone occupate dall'ego al reset e non ancora lasciate per la prima volta. Il gate impedisce al termine di approccio di rimanere attivo dopo un ingresso legittimo o durante un'occupazione preesistente.

### 7.8.6 Costo

Sia $X_W(t)$ l'evento di ingresso dell'ego in $Z_W$. Si mantiene inoltre il flag causale:

$$
I_{\mathrm{illegal},W}(t)=1
$$

se $X_W(t)=1$ mentre $r_{\mathrm{gap},j}>0$ per almeno un VRU candidato. Il flag resta attivo finché il footprint ego interseca $Z_W$ e viene resettato quando l'ego lascia la zona.

Il costo è:

$$
q_{\mathrm{crosswalk}}(t)
=
\max
\left\{
\max_{W,j}
\left(
I_{\mathrm{before},W}(t)\,
r_{\mathrm{gap},j}r_{\mathrm{commit}}^W
\right),
\max_W
\left(I_{\mathrm{illegal},W}(t)\,\mathbf1[P_e(t)\cap Z_W\neq\varnothing]\right)
\right\}.
$$

Quindi:

- prima della zona opera il costo continuo di approccio;
- un ingresso legittimo disattiva il termine di approccio durante l'attraversamento;
- un ingresso incompatibile e la successiva permanenza nella conflict zone producono costo $1$ fino all'uscita;
- un'occupazione preesistente al reset non produce né costo di approccio né evento di ingresso fino alla prima uscita.

Questa forma:

- è continua durante l'approccio prima della zona;
- satura a uno all'ingresso incompatibile;
- considera soltanto il punto/regione di conflitto davanti all'ego;
- evita di penalizzare ego e VRU presenti in parti indipendenti dello stesso grande poligono;
- non usa intenzioni annotate o futuro ground-truth.

### 7.8.7 Variabili

- crosswalk polygons;
- route movement e corridoio ego canonico;
- `zone_id` e geometria dalla cache episodica;
- ego/VRU footprints;
- ego/VRU posizione e velocità correnti;
- actor type;
- $b_e$ e $\Delta t$ già usati dalle altre regole;
- $T_{\mathrm{gap}}$ e $T_{\mathrm{pred}}=3\,s$;
- gate $I_{\mathrm{before},W}$ e insieme delle occupazioni preesistenti;
- flag di ingresso illegittimo per conflict zone.

## 7.9 Yield veicolo-veicolo scoped

### 7.9.1 Predicato deterministico di priorità

Non viene implementato un motore giuridico universale delle precedenze. Per una conflict zone $Z$ e un veicolo $i$, la priorità dell'altro veicolo rispetto all'ego è definita dal predicato:

$$
\Pi_{i\succ e}(Z,t)
=
O_i(Z,t)
\lor
S_{i\succ e}(Z)
\lor
R_{i\succ e}(Z)
\lor
M_{i\succ e}(Z).
$$

Le quattro condizioni sono:

$$
O_i(Z,t)
=
\mathbf 1[P_i(t)\cap Z\neq\varnothing],
$$

cioè il veicolo $i$ occupa già la conflict zone;

$$
S_{i\succ e}(Z)
=
\mathbf 1[
\text{approccio ego soggetto a STOP}
\land
\text{approccio di }i\text{ non soggetto a STOP/YIELD}
],
$$

cioè l'ego deve arrestarsi mentre l'altro flusso non è subordinato da un controllo equivalente;

$$
R_{i\succ e}(Z)
=
\mathbf 1[
\text{ego su lane di ingresso in rotatoria}
\land
\text{$i$ su lane della componente circolante}
],
$$

con la componente circolante identificata e validata una volta per mappa come ciclo diretto del lane graph;

$$
M_{i\succ e}(Z)
=
\mathbf 1[
\text{la mappa o il wrapper fornisce una relazione esplicita}
\;i\succ e
],
$$

che copre, per esempio, un `YIELD` associato alla manovra ego o un merge verso un flusso esplicitamente marcato come prioritario. La priorità di un merge non viene inferita dalla sola geometria.

La regola è `NOT_APPLICABLE` nei seguenti casi:

```text
intersezione non controllata senza priorità esplicita
entrambi gli approcci soggetti a STOP
merge senza relazione di priorità esplicita
più successori plausibili dell'attore prima della conflict zone
```

Restano comunque attivi collisione, TTC e clearance.

### 7.9.2 Conflict zone e candidati

La conflict zone $Z_{e,i}$ viene costruita dai movimenti ego e attore secondo le Sezioni 2.8.1--2.8.3. La geometria è recuperata dalla cache mediante `zone_id` stabile.

Per ogni zona $Z$, l'insieme dei candidati prioritari è:

$$
\mathcal A_Z(t)
=
\left\{
i:
\Pi_{i\succ e}(Z,t)=1
\land
I_i^Z(t)\neq\texttt{NO\_INTERVAL}
\right\},
$$

dove $I_i^Z(t)$ è l'intervallo di occupazione previsto del veicolo $i$ nella zona. Se $\mathcal A_Z(t)=\varnothing$, quella zona non contribuisce al costo nello step corrente.

### 7.9.3 Gap temporale

Gli intervalli ego e veicolo prioritario vengono calcolati con la stessa funzione operativa della Sezione 7.8.3 e con:

$$
T_{\mathrm{pred}}=3.0\,s.
$$

Un attore fermo fuori dalla conflict zone o che non vi entra entro l'orizzonte non è candidato; un attore già dentro e non previsto in uscita riceve un intervallo `OPEN_END`.

Per $I_e=[t_{e,\mathrm{in}},t_{e,\mathrm{out}}]$ e $I_i=[t_{i,\mathrm{in}},t_{i,\mathrm{out}}]$:

$$
g_i^{\mathrm{sep}}
=
\begin{cases}
\max\{t_{e,\mathrm{in}}-t_{i,\mathrm{out}},\;t_{i,\mathrm{in}}-t_{e,\mathrm{out}}\},
&\text{uscite entrambe finite},
\\[2mm]
t_{i,\mathrm{in}}-t_{e,\mathrm{out}},
&t_{e,\mathrm{out}}\text{ finito e }t_{e,\mathrm{out}}\le t_{i,\mathrm{in}},
\\[2mm]
t_{e,\mathrm{in}}-t_{i,\mathrm{out}},
&t_{i,\mathrm{out}}\text{ finito e }t_{i,\mathrm{out}}\le t_{e,\mathrm{in}},
\\[2mm]
-\infty,
&\text{altrimenti}.
\end{cases}
$$

$$
r_{\mathrm{gap},i}
=
\operatorname{clip}
\left(
\frac{T_{\mathrm{gap}}-g_i^{\mathrm{sep}}}{T_{\mathrm{gap}}},
0,1
\right),
$$

con la convenzione $r_{\mathrm{gap},i}=1$ quando $g_i^{\mathrm{sep}}=-\infty$.

Si usa:

$$
T_{\mathrm{gap}}=1.0\,s,
$$

condiviso con la regola crosswalk.

### 7.9.4 Commitment dell'ego

Sia $d_e^{Z_{e,i}}$ la distanza del front bumper dall'ingresso nella conflict zone e sia:

$$
v_e^{\mathrm{app},i}
=
[\mathbf v_e^\top\mathbf t_{\mathrm{route}}]_+.
$$

Allora:

$$
r_{\mathrm{commit},i}
=
\begin{cases}
\operatorname{clip}
\left(
1-
\dfrac{
d_e^{Z_{e,i}}
}{
d_{\mathrm{stop}}(v_e^{\mathrm{app},i})
},
0,1
\right),
&d_{\mathrm{stop}}(v_e^{\mathrm{app},i})>0,
\\[3mm]
0,
&d_{\mathrm{stop}}(v_e^{\mathrm{app},i})=0,
\end{cases}
$$

dove:

$$
d_{\mathrm{stop}}(v_e^{\mathrm{app},i})
=
v_e^{\mathrm{app},i}\Delta t
+
\frac{(v_e^{\mathrm{app},i})^2}{2b_e}.
$$

Anche per il vehicle-yield il costo continuo è attivo soltanto prima dell'ingresso nella zona e non durante un'occupazione preesistente al reset:

$$
I_{\mathrm{before},Z}(t)
=
\mathbf1
\left[
d_e^{Z}(t)>\varepsilon_{\delta}
\land
P_e(t)\cap Z=\varnothing
\land
\operatorname{zone\_id}(Z)\notin
\mathcal Z_{\mathrm{preexisting}}(t)
\right].
$$

Il gate è condiviso da tutti i candidati associati alla stessa conflict zone $Z$.

### 7.9.5 Costo

Sia $X_{e,i}(t)$ l'evento di ingresso dell'ego in $Z_{e,i}$. Si attiva:

$$
I_{\mathrm{illegal},e,i}(t)=1
$$

quando $X_{e,i}(t)=1$ e $r_{\mathrm{gap},i}>0$. Il flag resta attivo finché l'ego occupa $Z_{e,i}$ e viene resettato all'uscita.

Per ogni $Z$ e per ogni $i\in\mathcal A_Z(t)$:

$$
q_{\mathrm{yield},Z,i}^{\mathrm{approach}}
=
I_{\mathrm{before},Z}(t)\,
r_{\mathrm{gap},i}r_{\mathrm{commit},i}.
$$

Il costo complessivo include anche le conflict zone con flag illegale ancora attivo, anche se l'attore originario non è più un candidato live:

$$
q_{\mathrm{yield,vehicle}}
=
\max
\left\{
\max_Z\max_{i\in\mathcal A_Z(t)} q_{\mathrm{yield},Z,i}^{\mathrm{approach}},
\max_{Z_{e,i}\in\mathcal Z_{\mathrm{illegal}}}
I_{\mathrm{illegal},e,i}(t)\,\mathbf1[P_e(t)\cap Z_{e,i}\neq\varnothing]
\right\}.
$$

Prima della zona opera il costo continuo di approccio. Dopo un ingresso legittimo tale termine è disattivato; dopo un ingresso incompatibile, la permanenza nella conflict zone produce costo $1$ fino all'uscita. Le occupazioni preesistenti al reset restano ignorate fino alla prima uscita. La formula condivide le primitive geometriche e cinematiche con crosswalk e semaforo, è bounded e causale e non richiede future tracks o intent prediction.

### 7.9.6 Natura della formula

- relazione di priorità: predicato deterministico limitato ai quattro casi della Sezione 7.9.1;
- conflict zone: costruzione geometrica canonica e cached della Sezione 2.8;
- intervalli temporali: previsione locale a velocità costante;
- deficit temporale e commitment product: adattamento originale della tesi;
- non costituisce codifica completa del codice stradale.

## 7.10 Aggregazione

$$
c_3(t)
=
\max
\left\{
q_{\mathrm{offroad}},
q_{\mathrm{wrongway}},
q_{\mathrm{solid}},
q_{\mathrm{dash}},
q_{\mathrm{signal}},
q_{\mathrm{stop}},
q_{\mathrm{crosswalk}},
q_{\mathrm{yield,vehicle}}
\right\}_{\mathrm{applicable,evaluable}},
$$

$$
m_3(t)=-c_3(t).
$$

# 8. $R_4$ — Local route progress

## 8.1 Quantità raw

In `ScenarioEnv`, `TrajectoryNavigation` proietta l'ego sulla singola reference trajectory dello scenario e mantiene `current_longitude` e `last_longitude`. Si usa direttamente:

$$
\Delta s_{\mathrm{route}}(t)
=
\texttt{current\_longitude}
-
\texttt{last\_longitude}.
$$

- positivo: avanzamento;
- zero: fermata;
- negativo: regressione o retromarcia.

La quantità raw è espressa in metri per step e viene sempre conservata nei diagnostics. Non si costruisce una seconda coordinata arc-length wrapper-owned, salvo che i test sulle sorgenti concrete evidenzino salti sistematici della proiezione nativa.

Prima del clipping si verifica come invariante runtime soltanto che:

$$
\Delta s_{\mathrm{route}}\in\mathbb R
$$

e che il valore sia finito. Un valore `NaN` o infinito solleva `RulebookEvaluationError`.

La plausibilità cinematica del delta viene comunque monitorata nei diagnostics confrontandola con un bound preliminare derivato da velocità configurata, velocità osservate e control timestep. Il superamento di tale bound produce il flag diagnostico `route_projection_jump_suspected`, ma non interrompe il run: velocità intermedie fra due control step, curvatura della route e approssimazioni della proiezione possono rendere troppo rigido un bound costruito soltanto dagli stati agli estremi.

Salti sistematici o macroscopici devono essere rilevati negli unit test e negli smoke test su PG e Waymo e portano alla correzione del wrapper o all'esclusione dello scenario prima del training. Un'eventuale soglia fail-fast runtime potrà essere aggiunta soltanto dopo essere stata validata empiricamente sul branch MetaDrive e sulle sorgenti effettivamente impiegate; il clipping non deve essere usato per ignorare anomalie sistematiche.

## 8.2 Segnale per il learner

Per evitare una scala dipendente da control timestep e velocità configurata, senza introdurre un valore di riferimento arbitrario:

$$
m_4(t)
=
\operatorname{clip}
\left(
\frac{
\Delta s_{\mathrm{route}}(t)
}{
v_{\max,e}\Delta t
},
-1,1
\right).
$$

$v_{\max,e}$ è il configured speed normalization cap dell'ego già presente nella configurazione. Non è un parametro della letteratura, una velocità desiderata, un limite fisico invalicabile o una quantità da sottoporre a tuning.

## 8.3 Origine

`ScenarioEnv` calcola il dense driving reward usando:

```python
current_longitude - last_longitude
```

[7]. La quantità raw coincide con questa struttura, ma esclude lane-centering, heading e steering penalties. La normalizzazione è un adattamento deterministico della tesi per ottenere una frazione della scala di avanzamento configurata per control step.

## 8.4 Perché non serve un gate

Una fermata corretta a rosso, stop o davanti a un ostacolo produce:

$$
m_4=0,
$$

non una penalità. Poiché $R_4$ è subordinata, non può giustificare violazioni superiori.

## 8.5 Variabili e parametri

Variabili:

- `vehicle.navigation.current_longitude`;
- `vehicle.navigation.last_longitude`;
- route completion;
- reference trajectory;
- $v_{\max,e}$;
- $\Delta t$.

Parametri semantici:

```text
nessuno
```

La baseline scalarizzata usa lo stesso $m_4$ normalizzato; il progresso raw in metri resta disponibile per interpretazione e reporting.

# 9. Tabella consolidata dei parametri

| Regola | Parametro | Valore | Natura/provenienza |
|---|---|---:|---|
| common geometry | precision grid | $10^{-3}\,m$ | stabilizzazione geometrica congelata |
| common geometry | $\varepsilon_A$ | $10^{-4}\,m^2$ | tolleranza numerica area off-road |
| common geometry | $\varepsilon_{\mathrm{geom}}$ | $10^{-2}\,m$ | buffer numerico boundary |
| control/conflict crossing | $\varepsilon_{\delta}$ | $5\cdot10^{-2}\,m$ | deadband signed distance |
| local prediction | $T_{\mathrm{pred}}$ | $3.0\,s$ | orizzonte CV per TTC/conflict zone |
| R1 | $\varepsilon_{\mathrm{col}}$ | $10^{-6}$ | floor numerico, ruolo analogo a ScenicRules |
| R1 | comparison tolerance | $10^{-8}$ | sola tolerance numerica del monitor |
| R1 vehicle | $u_{\mathrm{cap},i}$ | $v_{\max,e}+v_{\max,i}$ | configured speed normalization cap |
| R1 VRU/static | $u_{\mathrm{cap},i}$ | $v_{\max,e}$ | configured speed normalization cap ego, nessun cap semantico aggiuntivo |
| RSS | $\rho$ | $1.0\,s$ | Intel ad-rss-lib suggested starting value |
| RSS | $a_{\max}^{acc}$ | $3.5\,m/s^2$ | Intel ad-rss-lib |
| RSS | $b_e$ | $4.0\,m/s^2$ iniziale | Intel ad-rss-lib; da verificare con brake test MetaDrive |
| RSS | $b_i$ | $8.0\,m/s^2$ | Intel ad-rss-lib |
| TTC vehicle/static | $T_{\mathrm{TTC}}$ | $0.8\,s$ | ScenicRules vehicle; estensione agli statici |
| TTC VRU | $T_{\mathrm{TTC}}$ | $1.0\,s$ | ScenicRules code |
| clearance vehicle | $D_{\min}$ | $0.8\,m$ | ScenicRules code |
| clearance VRU | $D_{\min}$ | $1.0\,m$ | ScenicRules code |
| clearance static | $D_{\min}$ | $0.5\,m$ | scelta ingegneristica congelata |
| dashed | $T_0$ | $1.0\,s$ | decisione di task |
| dashed | $T_{\mathrm{cap}}$ | $2.0\,s$ | decisione di task concordata |
| dashed | exponent | $2$ | shaping originale, incorporato |
| signal | $b_{\mathrm{signal}}$ | $b_e$ | parametro fisico condiviso con RSS |
| signal | reaction interval | $\Delta t$ | control timestep effettivo |
| stop | $D_{\mathrm{stop}}$ | $1.0\,m$ | Maierhofer parameter |
| stop | $v_{\mathrm{stop}}$ | $0.1\,m/s$ | Maierhofer parameter |
| stop | $T_{\min}$ | $1.0\,s$ | Veer et al. |
| crosswalk/yield | $T_{\mathrm{gap}}$ | $1.0\,s$ | buffer di task condiviso, coerente con $\rho$ |
| merge/roundabout conflict zone | $L_Z$ | $2L_{\max}^{\mathrm{veh}}$ | dimensione geometrica derivata, non parametro di reward |
| off-road | nessuno | — | area ratio |
| wrong-way | normalizzatore | $v_{\max,e}$ | configured speed normalization cap ego |
| solid | nessuno | — | geometric continuous-line contact/crossing |
| progress | normalizzatore | $v_{\max,e}\Delta t$ | scala di avanzamento configurata per step |

I valori fisici e geometrici della specifica non fanno parte dello sweep degli algoritmi RL. L'eventuale sostituzione di $b_e$ dopo brake test è una calibrazione una tantum della dinamica effettiva, non tuning del reward. Le soglie algoritmiche della Sezione 4.4 sono separate da questa tabella e non appartengono al monitor.

# 10. Matrice delle variabili

| Componente | Stato corrente | Memoria |
|---|---|---|
| collision | contatti, pose, velocity, configured speed normalization cap | previous contacts, previous actor velocities |
| RSS | front vehicle, lane, tangent, gap, speed | nessuna |
| TTC | polygons, velocities | nessuna |
| clearance | polygons | nessuna |
| off-road | ego polygon, drivable surfaces | nessuna |
| wrong-way | ego velocity, route/lane tangent, configured speed normalization cap | nessuna |
| solid | continuous-line contact/geometria | nessuna |
| dashed | dashed-boundary polylines, ego polygon | boundary ID, timer |
| signal | current signal, control line, speed, signed distance | previous color, must-stop, previous delta |
| stop | stop sign, lane, derived control line, speed, delta | continuous/best timer, previous delta |
| crosswalk | crosswalk, ego route movement, cached canonical zone, ego/VRU states | illegal-entry flag keyed by stable `zone_id` |
| vehicle yield | movement corridors, deterministic priority predicate, cached canonical zones, actor states | illegal-entry flag keyed by stable `zone_id` |
| progress | `current_longitude`, `last_longitude`, speed cap, timestep | nessuna memoria custom |

# 11. Contratto con MetaDrive/ScenarioEnv

## 11.1 Dati già presenti

Il codice corrente di MetaDrive espone nello stato del veicolo:

- `crash_vehicle`;
- `crash_human`;
- `crash_object`;
- `crash_sidewalk`;
- `red_light`;
- `yellow_light`;
- `on_yellow_continuous_line`;
- `on_white_continuous_line`;
- `on_broken_line`;
- `on_crosswalk`;
- `last_position`;
- `last_velocity`;
- `speed`;
- `lane`;
- `navigation.current_longitude`;
- `navigation.last_longitude`;
- `navigation.current_heading_theta_at_long`;
- `max_speed_m_s`.

`ScenarioEnv` usa già i flag delle linee, le collisioni, la route completion e il delta longitudinale [7, 8].

## 11.2 Dati da esporre o derivare nel wrapper

- contact actor ID e normale orientata ego $\rightarrow$ oggetto;
- lista degli attori live e velocità al precedente control step;
- valori `max_speed_m_s` usati come configured speed normalization cap, validati;
- footprint polygons;
- drivable lane surfaces locali;
- lane/route tangents orientate;
- road-line polylines originali, incluso ID e tipo della boundary;
- relevant signal ID/state/control line derivata dal `stop_point`;
- relevant stop sign e derived stop control line;
- crosswalk polygons;
- lane polygons e successor graph necessari a costruire i movement corridor;
- route ego come sequenza ordinata di lane;
- cache episodica delle conflict zone con `zone_id` stabile;
- classificazione validata della componente circolante delle rotatorie;
- relazioni di priorità esplicite per `YIELD` e merge, quando disponibili.

Il wrapper non deve modificare il motore fisico, salvo l'hook localizzato necessario a esportare i contact record. Le altre quantità vengono lette o derivate dallo stato live e dallo `ScenarioDescription` caricato.

## 11.3 ScenarioDescription

ScenarioDescription fornisce tracks, map features e dynamic map states in una rappresentazione unificata; la ricchezza effettiva dei campi dipende dalla sorgente [9, 12].

Le formule sono identiche per PG e Waymo. Cambia soltanto la capacità dello scenario di renderle valutabili.

# 12. Output del monitor

```python
@dataclass
class RuleComponentResult:
    name: str
    cost: float
    raw: dict
    applicable: bool
    evaluable: bool
    status: str
    diagnostics: dict

@dataclass
class RulebookResult:
    # m1..m3 bounded in [-1, 0], m4 bounded in [-1, 1]
    margins: tuple[float, float, float, float]
    costs: tuple[float, float, float]
    raw_progress_m: float
    components: dict[str, RuleComponentResult]
    complete_evaluation: bool
```

Nel training normale `complete_evaluation=True` per tutte le componenti core applicabili. Una violazione del contratto dati non viene rappresentata con un normale `RulebookResult`: il monitor solleva `RulebookEvaluationError`.

# 13. Pseudocodice

```python
def evaluate_rulebook(env, memory, context):
    ego = env.agent
    actors = get_live_actors(env)

    r1 = evaluate_collision(
        ego=ego,
        actors=actors,
        contact_records=context.contact_records,
        previous_contact_ids=memory.previous_contact_ids,
        previous_actor_velocities=memory.previous_actor_velocities,
    )

    r2_rss = evaluate_rss(ego, actors, context)
    r2_ttc = evaluate_ttc(
        ego, actors, prediction_horizon_s=context.prediction_horizon_s
    )
    r2_clear = evaluate_clearance(
        ego, actors, monitor_radius=context.derived_monitor_radius
    )
    c2 = max_evaluable(r2_rss, r2_ttc, r2_clear)

    r3_offroad = evaluate_offroad(ego, context.drivable_surface)
    r3_wrongway = evaluate_wrongway(
        ego,
        context.reference_tangent,
        context.ego_configured_speed_cap,
    )
    r3_solid = evaluate_solid_line(ego, context.road_lines)
    r3_dash = evaluate_dashed_line(
        ego,
        context.dashed_boundary_polylines,
        memory,
    )
    r3_signal = evaluate_signal(
        ego,
        context.relevant_signal,
        memory,
    )
    r3_stop = evaluate_stop(
        ego,
        context.relevant_stop,
        memory,
    )
    r3_crosswalk = evaluate_crosswalk_yield(
        ego=ego,
        actors=actors,
        context=context,
        memory=memory,
    )
    r3_vehicle_yield = evaluate_vehicle_yield(
        ego=ego,
        actors=actors,
        context=context,
        memory=memory,
    )

    c3 = max_evaluable(
        r3_offroad,
        r3_wrongway,
        r3_solid,
        r3_dash,
        r3_signal,
        r3_stop,
        r3_crosswalk,
        r3_vehicle_yield,
    )

    raw_progress = (
        ego.navigation.current_longitude
        - ego.navigation.last_longitude
    )
    validate_finite_route_delta(raw_progress)
    log_route_projection_diagnostics(
        raw_progress,
        configured_speed_cap=context.ego_configured_speed_cap,
        previous_speed=np.linalg.norm(ego.last_velocity),
        current_speed=np.linalg.norm(ego.velocity),
        timestep=context.control_timestep,
    )
    m4 = clip(
        raw_progress / (context.ego_configured_speed_cap * context.control_timestep),
        -1.0,
        1.0,
    )

    complete = all_required_components_evaluable(...)
    if not complete:
        raise RulebookEvaluationError(
            scenario_id=context.scenario_id,
            step=context.step,
            diagnostics=collect_non_evaluable_reasons(...),
        )

    update_memory(env, ego, actors, context, memory)

    return RulebookResult(
        margins=(-r1.cost, -c2.cost, -c3.cost, m4),
        costs=(r1.cost, c2.cost, c3.cost),
        raw_progress_m=raw_progress,
        components=...,
        complete_evaluation=True,
    )
```

Nel training loop non è richiesto alcun buffer episodico aggiuntivo:

```python
try:
    result = evaluate_rulebook(env, memory, context)
    store_transition_with_rulebook_margins(result.margins)
except RulebookEvaluationError as exc:
    log_rulebook_failure(exc)
    abort_run_without_saving_replay_checkpoint()
    raise
```

La funzione di preparazione del catalogo esegue invece:

```python
validation = validate_rulebook_scenario(scenario)
if validation.rulebook_eligible:
    add_to_rulebook_pool(scenario)
else:
    log_excluded_scenario(scenario.id, validation.errors)
```

# 14. Configurazione

```yaml
rulebook:
  version: 4.4-final

  order:
    - collision_impact
    - dynamic_interaction_safety
    - road_traffic_compliance
    - route_progress

  output:
    bounded_safety_costs: true
    normalized_progress_for_learner: true
    raw_diagnostics: true
    internal_aggregation: max
    learner_temporal_semantics: discounted_per_step_return
    episodic_peak_required_for_training: false
    episodic_integral_required_for_training: false
    episodic_diagnostics_required_for_evaluation: true
    not_evaluable_policy: fail_fast
    offline_scenario_validation: true
    replay_rollback_required: false
    silent_fallbacks: false
    reset_initialization:
      seed_previous_contacts_from_live_state: true
      seed_previous_actor_velocities_from_live_state: true
      seed_signal_and_stop_state_from_live_state: true
      ignore_preexisting_conflict_zone_occupancy_until_first_exit: true
    numerical_tolerances:
      geometry_precision_grid_m: 1.0e-3
      offroad_area_epsilon_m2: 1.0e-4
      polyline_buffer_epsilon_m: 1.0e-2
      signed_distance_epsilon_m: 5.0e-2

  prediction:
    constant_velocity_horizon_s: 3.0
    monitor_radius: derived_from_configured_caps_and_observed_speeds
    independent_from_policy_observation: true
    future_ground_truth_tracks: false

  collision:
    onset_only: true
    previous_control_step_velocity: true
    severity_raw: squared_max_contact_point_normal_closing_speed
    normal_orientation: ego_to_other
    contact_point_aggregation: max_normal_closing_speed_per_actor
    normalization:
      vehicle: ego_plus_other_configured_speed_cap
      vru_or_static: ego_configured_speed_cap
    epsilon: 1.0e-6
    comparison_tolerance: 1.0e-8
    actor_type_weighting: false
    require_custom_contact_records: true

  interaction:
    rss:
      response_time_s: 1.0
      max_accel_during_response_mps2: 3.5
      ego_min_brake_mps2: 4.0
      front_max_brake_mps2: 8.0
      validate_ego_braking: true
      use_positive_longitudinal_speed: true
    ttc:
      vehicle_s: 0.8
      static_s: 0.8
      vru_s: 1.0
      include_static_objects: true
      static_velocity_mps: 0.0
      horizon_s: 3.0
    clearance:
      vehicle_m: 0.8
      vru_m: 1.0
      static_m: 0.5
      candidate_policy: derived_monitor_radius_live_collidable

  compliance:
    offroad:
      definition: ego_area_fraction_outside_drivable_surface

    wrongway:
      definition: negative_signed_longitudinal_velocity
      normalization: ego_configured_speed_cap

    solid_line:
      definition: geometric_current_contact_or_swept_front_bumper_crossing
      live_flags_role: diagnostics_only

    dashed_line:
      definition: persistent_contact_with_continuous_boundary_polyline
      free_time_s: 1.0
      saturation_time_s: 2.0
      exponent: 2
      track_boundary_id: true

    signal:
      evaluate_green: true
      evaluate_yellow: true
      evaluate_red: true
      crossing_governed_by: pre_action_signal_state
      approach_governed_by: current_signal_state
      yellow_must_stop_frozen_at_onset: true
      braking_mps2: use_rss_ego_min_brake
      reaction_time: control_timestep
      approach_speed: positive_route_approach_speed
      crossing_deadband_m: 0.05
      offline_unknown_state_policy: exclude_scenario
      approach_cost: relative_stopping_distance_deficit
      illegal_crossing_cost: 1.0

    stop:
      control_line: derived_from_stop_position_and_lane
      zone_m: 1.0
      crossing_deadband_m: 0.05
      stopped_speed_mps: 0.1
      minimum_continuous_dwell_s: 1.0
      update_timer_before_crossing_resolution: true

    conflict_zones:
      movement_corridor: union_of_live_lane_polygons
      ego_movement: ordered_route_lane_sequence
      other_movement: current_lane_plus_unique_successor_chain
      ambiguous_successor_before_zone: not_applicable
      component_selection: first_entry_along_ego_route
      merge_or_roundabout_shared_lane_length: 2_times_max_supported_vehicle_length
      crosswalk_component_selection: first_ahead_or_currently_occupied
      cache_scope: episode
      stable_id_fields:
        - scenario_id
        - zone_type
        - ego_movement_id
        - other_movement_or_crosswalk_id
        - connected_component_index

    crosswalk:
      definition: canonical_route_conflict_zone_yield
      prediction: current_state_constant_velocity
      prediction_horizon_s: 3.0
      gap_s: 1.0
      temporal_separation: robust_symmetric_interval_gap
      explicit_no_interval_and_open_end_cases: true
      approach_gate: before_zone_and_not_preexisting_only
      disable_approach_cost_while_inside_zone: true
      persist_illegal_occupancy: true
      entry_violation_cost: 1.0

    vehicle_yield:
      enabled: true
      priority_predicates:
        - conflict_zone_already_occupied
        - ego_stop_actor_uncontrolled
        - roundabout_entry_actor_circulating
        - explicit_map_or_wrapper_priority
      explicit_priority_examples:
        - ego_yield_control
        - merge_to_explicit_priority_flow
      not_applicable_cases:
        - ambiguous_uncontrolled_intersection
        - both_approaches_stop_controlled
        - merge_without_explicit_priority
        - ambiguous_actor_successor_before_zone
      infer_merge_priority_from_geometry: false
      prediction: current_state_constant_velocity
      prediction_horizon_s: 3.0
      gap_s: 1.0
      temporal_separation: robust_symmetric_interval_gap
      explicit_no_interval_and_open_end_cases: true
      approach_gate: before_zone_and_not_preexisting_only
      disable_approach_cost_while_inside_zone: true
      persist_illegal_occupancy: true
      entry_violation_cost: 1.0

  progress:
    raw_definition: native_trajectory_navigation_longitudinal_delta_m
    custom_wrapper_arclength: false
    validate_finite_delta: true
    route_projection_plausibility_runtime_policy: diagnostic_only
    reject_projection_jump_before_clipping: false
    fail_fast_projection_threshold_requires_prior_empirical_validation: true
    learner_definition: normalized_route_longitudinal_delta
    normalization: ego_configured_speed_cap_times_control_timestep
    clip: [-1.0, 1.0]
```

La configurazione strict/thresholded dell'algoritmo lessicografico non è inclusa nel blocco `rulebook`, perché non modifica il monitor.

# 15. Test obbligatori

## 15.1 Collisione

- nuovo contatto frontale;
- normale restituita nel verso opposto e correttamente invertita;
- contatto tangenziale;
- contatto persistente;
- più contact point dello stesso oggetto e selezione del massimo closing speed normale;
- collisioni simultanee con oggetti differenti;
- oggetto statico;
- VRU;
- velocità al precedente control step;
- veicolo senza configured speed normalization cap valido;
- actor speed sopra il configured speed normalization cap: costo saturato e scenario non invalidato;
- verifica che il custom contact hook non modifichi la dinamica.

## 15.2 RSS

- front vehicle stessa lane;
- veicolo adiacente escluso;
- veicolo dietro escluso;
- gap esattamente safe;
- gap zero;
- lane association ambigua produce RSS `NOT_APPLICABLE` per l'attore;
- velocità longitudinale negativa o corrente non concorde esclusa da RSS;
- brake benchmark del modello ego;
- eventuale sostituzione documentata di $b_e$ se $4\,m/s^2$ non è garantito.

## 15.3 TTC

- traiettorie parallele;
- crossing;
- overlap corrente;
- veicolo;
- VRU;
- heading costante;
- ostacolo statico con $\mathbf v_i=0$;
- collisione prevista oltre $3\,s$ restituisce TTC infinito/costo zero;
- attore fuori dal raggio derivato di interesse.

## 15.4 Clearance

- veicoli paralleli;
- VRU laterale;
- static object;
- distanza uguale alla soglia;
- oggetto fuori dal raggio derivato escluso;
- modifica del raggio dell'osservazione non modifica i candidati del monitor;
- infrastruttura non collidibile esclusa.

## 15.5 Geometria stradale

- footprint parzialmente off-road;
- completamente off-road;
- griglia di precisione geometrica di $1\,mm$ preserva le geometrie valide;
- area esterna inferiore a $\varepsilon_A=10^{-4}\,m^2$ produce zero;
- buffer boundary di $1\,cm$ e deadband control-line di $5\,cm$ non generano oscillazioni;
- ego fermo orientato al contrario: wrong-way $=0$;
- moto contrario lento;
- moto contrario con modulo pari al configured speed normalization cap;
- retromarcia opposta alla route;
- linea continua rilevata geometricamente;
- crossing completato fra due frame rilevato dal segmento del front bumper;
- flag MetaDrive discordante resta diagnostico e non sostituisce la geometria;
- normale crossing tratteggiata sotto un secondo;
- permanenza fra uno e due secondi;
- permanenza oltre due secondi;
- nessun reset del timer negli spazi fra trattini della stessa boundary;
- reset passando a una boundary diversa.

## 15.6 Semaforo

- verde;
- rosso con margine sufficiente;
- rosso con margine eroso;
- arresto completo prima della linea senza costo residuo;
- crossing rosso;
- giallo con arresto possibile;
- giallo con arresto non possibile;
- accelerazione dopo yellow onset non modifica `must_stop`;
- giallo che diventa rosso prima del crossing;
- rosso dopo crossing legittimo;
- crossing e cambio verde→rosso nello stesso step: governa il verde pre-azione;
- crossing iniziato col rosso e colore successivo diverso: resta illegale;
- deadband di $5\,cm$ sulla control line;
- scenario con segnale pertinente `UNKNOWN` escluso dalla validazione offline;
- stato `UNKNOWN` inatteso a runtime solleva `RulebookEvaluationError`;
- segnale di un'altra lane;
- crossing fra due control frame;
- associazione lane/stop_point Waymo e PG;
- reset dell'automa quando cambia `active_signal_id`;
- nessuna rivalutazione dopo `signal_resolved=True`.

## 15.7 Stop

- derivazione della control line da lane e stop position;
- geometria non verificabile deve fallire la validazione;
- nessun arresto;
- rolling stop;
- fermata parziale;
- fermate separate;
- fermata completa;
- fermata dopo la linea;
- fermata troppo lontana;
- crossing fra due frame;
- arresto esattamente sulla linea non risolve il crossing finché non supera la deadband;
- aggiornamento di $T_{\mathrm{best}}$ prima della risoluzione del crossing;
- reset dei timer quando cambia `active_stop_id`;
- nessuna rivalutazione dopo `stop_resolved=True`.

## 15.8 Crosswalk

- crosswalk dietro l'ego escluso;
- crosswalk non intersecante la route escluso;
- più componenti connesse dello stesso crosswalk: selezione della prima ancora davanti o attualmente occupata;
- `zone_id` stabile e geometria invariata dalla cache per l'intero episodio;
- ego e VRU in porzioni indipendenti dello stesso crosswalk;
- VRU già nella conflict zone;
- VRU fermo nella conflict zone;
- VRU diretto lontano dalla conflict zone escluso;
- ego attraversa molto prima del VRU con gap sicuro: costo zero;
- VRU attraversa molto prima dell'ego con gap sicuro: costo zero;
- intervalli finiti sovrapposti: gap negativo e costo positivo;
- intervallo `OPEN_END` produce rischio massimo senza `NaN`;
- attore fermo fuori dalla zona produce `NO_INTERVAL`;
- ingresso oltre $3\,s$ escluso dai candidati;
- gap esattamente uguale alla soglia;
- commitment crescente durante l'approccio;
- termine di approccio attivo soltanto prima della zona;
- ingresso legittimo disattiva il costo di approccio durante l'attraversamento;
- occupazione preesistente al reset non produce costo di approccio fino alla prima uscita;
- ingresso con gap insufficiente produce costo $1$;
- costo resta $1$ finché l'ego occupa la conflict zone dopo un ingresso illegittimo;
- flag illegale si resetta all'uscita;
- crosswalk geometry Waymo e PG;
- nessuna future track consultata.

## 15.9 Yield veicolo-veicolo

- ingresso in rotatoria con attore sulla componente circolante validata;
- merge con relazione di priorità esplicita;
- merge senza relazione di priorità esplicita produce `NOT_APPLICABLE`;
- ego soggetto a stop e attore su approccio privo di stop/yield;
- entrambi gli approcci soggetti a stop producono `NOT_APPLICABLE`;
- conflict zone già occupata dall'attore attiva il predicato di priorità;
- priorità ambigua in intersezione uncontrolled produce `NOT_APPLICABLE`;
- attore con più successori prima della zona produce `NOT_APPLICABLE`;
- catena univoca troncata quando raggiunge $D_i^{\mathrm{corr}}(t)$;
- catena univoca troncata alla fine del lane graph;
- più componenti di intersezione: selezione della prima lungo la route ego;
- merge su lane comune: zona limitata a $2L_{\max}^{\mathrm{veh}}$ dal punto di confluenza;
- `zone_id` stabile e geometria invariata dalla cache per l'intero episodio;
- ego passa prima dell'attore prioritario con separazione sicura;
- ego passa dopo l'attore prioritario con separazione sicura;
- intervalli finiti sovrapposti producono gap negativo;
- intervallo `OPEN_END` gestito senza sottrazioni fra infiniti;
- attore senza ingresso entro $3\,s$ escluso;
- gap sicuro;
- gap insufficiente;
- termine di approccio attivo soltanto prima della zona;
- ingresso legittimo disattiva il costo di approccio durante l'attraversamento;
- occupazione preesistente al reset non produce costo di approccio fino alla prima uscita;
- ingresso illegittimo produce costo $1$;
- costo resta $1$ durante la permanenza illegittima nella conflict zone;
- flag illegale si resetta all'uscita;
- nessuna future track consultata.

## 15.10 Progress

- avanzamento raw e normalizzato;
- fermata;
- retromarcia;
- clipping a $\pm1$;
- passaggio fra segmenti route;
- nessun salto di proiezione;
- delta non finito solleva `RulebookEvaluationError`;
- superamento del bound cinematico preliminare produce `route_projection_jump_suspected` senza interrompere il run;
- salti sistematici negli unit/smoke test causano correzione del wrapper o esclusione dello scenario prima del training;
- il clipping non nasconde anomalie sistematiche nei diagnostics;
- configured speed normalization cap, velocità osservate e control timestep validi.

## 15.11 Validazione scenario e fail-fast

- contatto già attivo al reset non produce una nuova collisione al primo step;
- velocità iniziali degli attori popolano correttamente `previous_actor_velocities`;
- stato e signed distance di semaforo e stop vengono inizializzati dallo stato live al reset;
- ego già dentro una conflict zone al reset non genera ingresso illegittimo;
- dopo la prima uscita da una conflict zone preesistente, un nuovo ingresso viene valutato normalmente;
- componente non applicabile produce zero senza invalidare lo scenario;
- attore non live viene escluso senza generare `NOT_EVALUABLE`;
- route, lane graph, geometrie e control associations mancanti escludono lo scenario offline;
- semaforo pertinente con sequenza incompleta o stato `UNKNOWN` esclude lo scenario offline;
- semaforo non pertinente invalido non esclude automaticamente lo scenario;
- scenario senza semafori resta eleggibile e la componente è `NOT_APPLICABLE`;
- dato core mancante a runtime solleva `RulebookEvaluationError`;
- il run fallito non salva il replay buffer come checkpoint valido;
- non esistono staging buffer episodici né rollback delle transizioni;
- scenario ID, step, componente e causa sono sempre loggati.

# 16. Limiti dichiarati

1. La collision severity è una proxy cinematica al precedente control step, non un injury model e non usa l'esatto substep fisico dell'impatto.
2. RSS è solo longitudinale e richiede che il modello ego garantisca la decelerazione assunta.
3. TTC assume moto a velocità e heading costanti entro $T_{\mathrm{pred}}=3\,s$ e tratta gli ostacoli statici con velocità nulla.
4. Clearance non dipende dalla velocità.
5. Le conflict zone per crosswalk e yield sono costruite da corridoi di movimento e mantenute in cache episodica; la manovra di un altro attore deve essere topologicamente univoca oppure la componente è `NOT_APPLICABLE`. Le occupazioni sono previste per soli $3\,s$ senza intent prediction.
6. Il crosswalk rule non modella l'intenzione di un VRU fermo lontano dalla zona di conflitto.
7. Il vehicle-yield rule usa soltanto i quattro predicati deterministici della Sezione 7.9.1; merge senza priorità esplicita, doppio stop e intersezioni non controllate ambigue restano fuori dominio.
8. Off-road richiede geometria carrabile interrogabile.
9. Lane-line types, traffic-control associations, stop positions e crosswalk polygons devono essere verificati sulle sorgenti concrete Waymo e PG.
10. La derived stop control line è una costruzione operativa verificata, non necessariamente una marking dipinta ground-truth.
11. La policy deve osservare timer/stati procedurali oppure una history sufficiente a ricostruirli.
12. Il rulebook è reward/specification monitoring, non uno shield e non garantisce sicurezza durante il learning.
13. La legalità dipende dalla convenzione del task e non costituisce una completa codifica del codice stradale di una specifica nazione.
14. Il massimo interno è una compressione worst-case lossy: configurazioni diagnostiche differenti possono avere lo stesso costo macro.
15. Il learner ottimizza expected discounted returns; non garantisce assenza di violazioni in ogni rollout e può pesare meno eventi più lontani nel tempo.
16. Il filtro sui semafori può modificare la distribuzione degli scenari; numero, sorgente e categoria degli scenari esclusi devono essere riportati.

# 17. Stato finale

La specifica semantica del rulebook è congelata e pronta per l'implementazione. Il costo di approccio di crosswalk e vehicle-yield è attivo esclusivamente prima della conflict zone, mentre il controllo runtime del progresso è fail-fast soltanto per valori non finiti e usa i salti plausibilmente anomali come diagnostics. Le soglie thresholded riportate in Sezione 4.4 sono un riferimento algoritmico non normativo e non condizionano il monitor.

Restano verifiche tecniche da eseguire durante l'implementazione, non decisioni aperte sulla definizione delle regole:

1. confermare sul branch MetaDrive usato l'accesso stabile a actor ID e contact normal e implementare il contact hook;
2. confermare e validare i valori `max_speed_m_s` usati come configured speed normalization cap dei veicoli live e loggare eventuali superamenti senza invalidare lo scenario;
3. misurare con brake benchmark la decelerazione garantita dell'ego e confermare o aggiornare $b_e$;
4. verificare per Waymo e PG continuous/broken-line types, signal association, stop-sign position e crosswalk geometry;
5. testare su intersection, merge, roundabout e crosswalk la costruzione canonica, la selezione della prima componente, gli ID stabili e la cache episodica delle conflict zone;
6. verificare che il wrapper esponga esclusivamente le relazioni di priorità esplicite usate da $M_{i\succ e}$ e che non inferisca la priorità dei merge dalla sola geometria;
7. definire nella specifica dell'osservazione come rendere disponibili timer, yellow state e altre dipendenze temporali;
8. confermare con unit test PG e Waymo le tolleranze geometriche congelate e l'assenza di `NaN` negli intervalli `OPEN_END`.

Se una verifica fallisce, lo scenario o il wrapper deve essere corretto prima del training. Gli scenari non eleggibili vengono esclusi dal pool rulebook-based e contabilizzati per sorgente e categoria. Non sono previsti fallback silenziosi; un `NOT_EVALUABLE` inatteso a runtime interrompe il run in modalità fail-fast, senza staging buffer o rollback del replay.

# 18. Riferimenti

[1] A. Censi et al., “Liability, Ethics, and Culture-Aware Behavior Specification using Rulebooks,” ICRA 2019.  
https://arxiv.org/abs/1902.09355

[2] K. K.-C. Chang et al., “ScenicRules: An Autonomous Driving Benchmark with Multi-Objective Specifications and Abstract Scenarios,” 2026.  
https://arxiv.org/abs/2602.16073

[3] S. Shalev-Shwartz, S. Shammah, A. Shashua, “On a Formal Model of Safe and Scalable Self-driving Cars,” 2017.  
https://arxiv.org/abs/1708.06374

[4] Intel, `ad-rss-lib`, “Parameter Discussion.” Suggested starting values: $\rho_{ego}=1s$, $a_{accel,max}=3.5m/s^2$, $a_{brake,min}=4m/s^2$, $a_{brake,max}=8m/s^2$.  
https://intel.github.io/ad-rss-lib/ad_rss/Appendix-ParameterDiscussion/

[5] S. Veer et al., “Receding Horizon Planning with Rule Hierarchies for Autonomous Vehicles,” 2023.  
https://arxiv.org/abs/2212.03323

[6] S. Maierhofer, A.-K. Rettinger, E. C. Mayer, M. Althoff, “Formalization of Intersection Traffic Rules in Temporal Logic,” IV 2022.  
https://mediatum.ub.tum.de/doc/1664592/uw2i3i5kwjh3w4ezek0qov5og.Maierhofer-2022-IV.pdf

[7] MetaDrive, current `ScenarioEnv` source, consulted 12 July 2026.  
https://raw.githubusercontent.com/metadriverse/metadrive/main/metadrive/envs/scenario_env.py

[8] MetaDrive, current `BaseVehicle` source, consulted 12 July 2026.  
https://raw.githubusercontent.com/metadriverse/metadrive/main/metadrive/component/vehicle/base_vehicle.py

[9] MetaDrive documentation, “Scenario Description,” consulted 12 July 2026.  
https://metadrive-simulator.readthedocs.io/en/latest/scenario_description.html

[10] Berkeley Learn Verify, ScenicRules `rule_functions.py` and `utils.py`, consulted 12 July 2026.  
https://github.com/BerkeleyLearnVerify/ScenicRules

[11] J. Skalse et al., “Lexicographic Multi-Objective Reinforcement Learning,” 2022.  
https://arxiv.org/abs/2212.13769

[12] Q. Li et al., “ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling,” NeurIPS 2023.  
https://arxiv.org/abs/2306.12241

[13] Shapely documentation, `set_precision`, consulted 13 July 2026.  
https://shapely.readthedocs.io/en/2.1.2/reference/shapely.set_precision.html
