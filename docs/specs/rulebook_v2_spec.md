---
title: "Specificazione finale del rulebook per MetaDrive ScenarioEnv e ScenarioNet"
subtitle: "Regole, formule, contesto di applicazione, variabili, parametri, fonti e contratto implementativo"
author: "Report tecnico per la tesi"
date: "15 luglio 2026"
lang: it-IT
version: "4.6-final-implementation-complete"
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
- è mantenuta solo una memoria causale minima fra step per collision onset, timer, eventi di crossing e stato del giallo; il progresso usa una coordinata curvilinea wrapper-owned deterministica;
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
- le conflict zone sono costruite da corridoi di movimento basati sui poligoni delle lane, ricevono un ID stabile basato su una `MovementKey` topologica e sono proposte tramite un `CacheDelta` episodico; una geometria già committata non viene mai modificata;
- il monitor valuta una transizione esplicita `pre_state -> post_state` mediante evaluator puri e restituisce atomicamente risultato, nuova memoria e delta della cache;
- memoria e cache vengono committate insieme soltanto dopo il completamento valido di tutte le componenti;
- tutte le operazioni spaziali sono 2.5D: le geometrie planari vengono considerate soltanto quando risultano compatibili in quota entro una tolleranza verticale congelata;
- lane association, gap longitudinale, superficie carrabile, traffic controls, control line, occupancy interval e route projection sono primitive canoniche con un solo algoritmo ammesso.

## Contratti implementativi congelati

Le seguenti precisazioni implementative fanno parte della versione
`4.6-final-implementation-complete`:

- il task route è un input statico: ScenarioNet può derivarlo offline dalla
  track SDC soltanto come sequenza topologica di lane; il runtime non legge
  `current_sdc_route`, timing, velocità o pose future;
- `ZoneLifecycleEvaluator` è l'unico writer di
  `preexisting_ego_occupancy_zone_ids`;
- il monitor restituisce soltanto il vettore ordinato e diagnostiche:
  scalarizzazione e learner lessicografico/distribuzionale restano esterni;
- il registry può essere modulare, ma ordine macro, componenti normative e
  ownership della memoria sono fissi;
- gli eventi di ingresso e l'occupazione preesistente dell'altro attore usano
  il pre-state; i costi continui di approccio usano il post-state;
- la prima implementazione valuta tutti gli attori live, senza broad phase;
- geometrie e ID sintetici usano snap a `1e-3 m`, WKB 2D big-endian senza
  SRID, envelope JSON canonico e SHA-256;
- l'osservazione semantic-state resta differita: il monitor non modifica
  l'osservazione della policy in questa versione iniziale.

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

È invece inclusa una regola di yield **scoped** nei soli casi coperti da un predicato deterministico: conflict zone già occupata dall'altro attore, ego soggetto a stop mentre l'altro approccio non presenta alcun controllo core pertinente, ingresso dell'ego in rotatoria con attore già sulla componente circolante, oppure relazione pairwise di priorità esplicitamente fornita e validata dalla mappa o dal wrapper. Merge senza relazione esplicita, yield sign non modellati nel core, intersezioni non controllate ambigue, entrambi gli approcci soggetti a stop e manovre dell'altro attore non univocamente ricostruibili sono `NOT_APPLICABLE`.

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
| $\varepsilon_{\psi}$ | $10^{-6}\,rad$ | equivalenza numerica nel disallineamento angolare della lane association |
| $\varepsilon_{\mathrm{lat}}$ | $10^{-3}\,m$ | equivalenza numerica nella coordinata laterale della lane association |
| $z_{\mathrm{tol}}$ | $3.0\,m$ | compatibilità verticale 2.5D fra attore, lane, zona e geometrie sovrapposte |

La griglia di precisione viene applicata senza modificare intenzionalmente la semantica della mappa; una geometria che collassa o diventa invalida dopo la normalizzazione non supera la validazione offline.

Per l'ego oriented bounding box, sia $B_{\mathrm{front}}(x)$ il segmento che unisce i due vertici anteriori nello stato $x$. La regione spazzata dal front bumper nella transizione è:

$$
S_{\mathrm{front}}(x_t,x_{t+1})
=
\operatorname{conv}
\left(
B_{\mathrm{front}}(x_t)\cup B_{\mathrm{front}}(x_{t+1})
\right).
$$

Un crossing di control line o linea continua richiede contemporaneamente il cambio di signed side previsto dalla relativa formula e l'intersezione fra $S_{\mathrm{front}}$ e la linea. Per una conflict zone con ingresso $s_Z^{\mathrm{entry}}$, l'evento è:

$$
X_Z(t)=
\mathbf1
\left[
P_e^{-}\cap Z=\varnothing
\land
s_{e,\mathrm{front}}^{-}<s_Z^{\mathrm{entry}}-\varepsilon_\delta
\land
s_{e,\mathrm{front}}^{+}\ge s_Z^{\mathrm{entry}}-\varepsilon_\delta
\land
S_{\mathrm{front}}(x_t,x_{t+1})\cap Z\neq\varnothing
\right].
$$

Questa definizione rileva anche il salto completo dentro o oltre la zona fra due frame e impedisce crossing ripetuti dovuti alla deadband.

I valori vengono verificati una volta con test unitari su PG e Waymo. Il test serve soltanto a confermare l'assenza di falsi crossing, oscillazioni numeriche e falsi off-road; non seleziona i valori in base alla performance del learner.

## 2.7 Orizzonte di previsione e insieme dei candidati

Le previsioni cinematiche locali di TTC, crosswalk e yield usano un unico orizzonte:

$$
T_{\mathrm{pred}}=3.0\,s.
$$

Tre secondi permettono di anticipare conflitti imminenti mantenendo limitato l'errore del modello a velocità e heading costanti. Non vengono consultate future tracks ground-truth.

L'insieme normativo dei candidati è l'insieme di tutti gli attori live della classe richiesta che superano i filtri semantici e verticali della regola. La broad phase è esclusivamente un'ottimizzazione: ometterla e iterare tutti gli attori live deve produrre esattamente lo stesso risultato.

Se viene usata una query spaziale, si definiscono:

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
\right\}.
$$

Se non esistono attori live candidati, per convenzione:

$$
\max_{i\in\varnothing}v_i^{\mathrm{bound}}(t)=0.
$$

Quindi:

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
- $r_e^{\mathrm{circ}}$ è il raggio circoscritto del footprint ego corrente;
- $r_{\mathrm{actor}}^{\mathrm{circ}}$ è il massimo raggio circoscritto configurato fra le classi di attori supportate; se nessuna classe è abilitata vale zero.

Il superamento temporaneo di `max_speed_m_s` non invalida lo scenario: viene incluso nei bound tramite le velocità osservate e registrato nei diagnostics. Il raggio resta una quantità derivata, non un nuovo iperparametro da tarare. Dopo la broad phase si applicano sempre i filtri esatti di classe, geometria e compatibilità verticale; un attore escluso da tali filtri non è candidato anche se ricade nel raggio.

## 2.8 Corridoi di movimento e conflict zone canoniche

Questa sezione definisce l'unica costruzione geometrica ammessa per crosswalk e yield veicolo-veicolo. Tutte le geometrie vengono normalizzate sulla griglia di precisione della Sezione 2.6 prima delle operazioni booleane e sono filtrate mediante la compatibilità verticale 2.5D della Sezione 2.9.1-bis.

### 2.8.1 `MovementKey` stabile e corridoio locale

Si distinguono due oggetti che non devono essere confusi:

- `MovementKey`: identità topologica stabile della manovra;
- `MovementCorridor`: geometria topologica stabile usata per costruire la conflict zone.

La chiave normativa è:

```python
@dataclass(frozen=True, order=True)
class MovementKey:
    approach_lane_id: str
    conflict_node_id: str
    exit_lane_id: str
```

Dove:

- `approach_lane_id` è l'ultima lane prima della regione topologica di conflitto;
- `conflict_node_id` è l'ID persistente del junction, merge node o roundabout component validato dal lane graph;
- `exit_lane_id` è la prima lane univoca dopo la regione di conflitto.

La chiave non dipende da velocità, orizzonte di previsione o numero di lane incluse nel corridoio. Se `conflict_node_id` o `exit_lane_id` non sono determinabili univocamente prima della zona e non esiste un'associazione esplicita validata, la componente vehicle-yield è `NOT_APPLICABLE` per quella coppia. Una volta che un attore entra nella zona, la sua `MovementKey` viene congelata fino all'uscita completa.

Il corridoio canonico di un movimento è la sequenza topologica minima e completa:

$$
m=(\ell_{\mathrm{approach}},\ell_1,\ldots,\ell_{\mathrm{exit}}),
$$

che parte da `approach_lane_id`, attraversa il `conflict_node_id` e include per intero `exit_lane_id`. La geometria è:

$$
C(m)=\bigcup_{\ell\in m}P_\ell.
$$

Il corridoio non dipende da velocità, posizione corrente dell'attore o $T_{\mathrm{pred}}$. Il lane graph e la `MovementKey` devono determinare un'unica sequenza; in caso contrario la componente è `NOT_APPLICABLE`. L'orizzonte di $3\,s$ limita esclusivamente la previsione degli intervalli di occupazione, non la geometria della conflict zone. In questo modo la stessa `MovementKey` produce la stessa geometria canonica e lo stesso `zone_id` in run differenti.

### 2.8.2 Intersezione fra movimenti e selezione della componente

Per due corridoi:

$$
\widetilde Z_{e,i}=C(m_e)\cap C(m_i).
$$

Una componente planare viene conservata soltanto se i due movimenti risultano verticalmente compatibili su di essa secondo `vertical_overlap_compatible` della Sezione 2.9.1-bis. Le componenti con area $\le\varepsilon_A$ vengono scartate.

Per ogni componente connessa $Z_j$ si interseca la centerline della `RoutePolyline` con $Z_j$ bufferizzata di $\varepsilon_{\mathrm{geom}}$. Gli estremi curvilinei delle porzioni di route interne alla componente definiscono:

$$
s_j^{\mathrm{entry}}
=
\min\{s:\mathbf r(s)\in Z_j^{\varepsilon}\},
\qquad
s_j^{\mathrm{exit}}
=
\max\{s:\mathbf r(s)\in Z_j^{\varepsilon}\}.
$$

Se la route non interseca la componente, essa viene scartata. La componente pertinente viene scelta con `first_ahead_or_occupied`:

1. fra le componenti occupate dall'ego, scegliere quella con area di overlap $A(P_e\cap Z_j)$ massima;
2. in parità, scegliere il minore $s_j^{\mathrm{entry}}$;
3. se nessuna è occupata, considerare soltanto le componenti con
   $$
   s_j^{\mathrm{exit}}\ge s_e^{\mathrm{front}}-\varepsilon_\delta;
   $$
4. scegliere quella con minore $s_j^{\mathrm{entry}}$;
5. in ulteriore parità, scegliere l'indice canonico della componente più piccolo.

Se non rimane alcuna componente valida, non esiste una conflict zone pertinente.

### 2.8.3 Merge e ingresso in rotatoria

Quando i due movimenti confluiscono nella stessa lane ricevente $\ell_c$, la conflict zone non è l'intera parte comune. Sia $s_{\mathrm{merge}}$ l'ascissa curvilinea sulla centerline di $\ell_c$ del punto di confluenza validato. Si definisce:

$$
Z_{e,i}
=
P_{\ell_c}
\cap
\left\{
\mathbf x:
0\le
s_{\ell_c}(\mathbf x)-s_{\mathrm{merge}}
\le L_Z
\right\},
$$

con:

$$
L_Z=2L_{\max}^{\mathrm{veh}}.
$$

Operativamente, $P_{\ell_c}$ viene tagliato mediante due rette ortogonali alla centerline in $s_{\mathrm{merge}}$ e $s_{\mathrm{merge}}+L_Z$. Si conserva la componente connessa che interseca la centerline fra le due sezioni. Se il clipping non produce una geometria valida, la relazione non supera la validazione offline.

Per una rotatoria la componente circolante deve essere classificata offline come ciclo diretto del lane graph; le entry lane sono le lane con successore nella componente circolante ma non appartenenti al ciclo.

### 2.8.4 Crosswalk

Per un crosswalk $W$:

$$
\widetilde Z_W=W\cap C(m_e).
$$

Il crosswalk deve possedere una quota canonica o derivabile secondo la Sezione 2.9.1-bis. Sono conservate soltanto le componenti verticalmente compatibili con il corridoio ego. Le componenti con area $\le\varepsilon_A$ vengono scartate e ordinate con la stessa procedura della Sezione 2.8.2. La zona pertinente è la componente occupata dall'ego, altrimenti la prima ancora davanti. Porzioni indipendenti dello stesso poligono non vengono aggregate.

### 2.8.5 ID stabile, `CacheDelta` e lifecycle

Ogni conflict zone riceve un ID stabile e tipizzato.

Per una zona veicolo-veicolo:

$$
\operatorname{zone\_id}^{\mathrm{vehicle}}(Z)
=
(
\texttt{scenario\_id},
\texttt{vehicle\_yield},
\operatorname{MovementKey}(m_e),
\operatorname{MovementKey}(m_i),
k
),
$$

dove $\operatorname{MovementKey}(m_i)$ identifica il movimento dell'altro
veicolo e $k$ è l'indice della componente dopo ordinamento canonico.

Per una zona di crosswalk:

$$
\operatorname{zone\_id}^{\mathrm{crosswalk}}(Z_W)
=
(
\texttt{scenario\_id},
\texttt{crosswalk},
\operatorname{MovementKey}(m_e),
\operatorname{crosswalk\_id}(W),
k
),
$$

dove $\operatorname{crosswalk\_id}(W)$ è l'ID persistente della feature
crosswalk e $k$ è l'indice della componente dopo ordinamento canonico.

I due schemi di ID appartengono a namespace distinti. Un ID viene costruito
esclusivamente con lo schema corrispondente al tipo della zona; per una zona
crosswalk non esiste né viene usata una `other_movement_key`.

La cache committata è episodica e append-only, ma nessun evaluator può modificarla direttamente. Le zone statiche di crosswalk, semafori e stop vengono costruite al reset. Le zone vehicle-yield possono essere proposte lazy mediante il `CacheDelta` definito nella Sezione 3.2.

Regole normative:

- una geometria associata a un `zone_id` già committato non viene mai modificata;
- la stessa `MovementKey` e la stessa componente recuperano lo stesso `zone_id` e la stessa geometria canonica;
- una nuova `MovementKey` può produrre un nuovo `zone_id`;
- i flag comportamentali sono indicizzati da `(actor_id, zone_id)`;
- un flag di ingresso illegale già attivo resta legato alla zona realmente attraversata finché l'ego non la lascia, anche se l'attore cambia movimento o scompare;
- se una zona lazy viene creata quando l'ego la occupa già, il `MemoryDelta` della stessa valutazione deve aggiungerla a `preexisting_ego_occupancy_zone_ids` prima di qualsiasi giudizio di ingresso;
- `CacheDelta` e `RulebookMemory` vengono committati insieme soltanto dopo la validazione completa del risultato.

Se il movimento diventa ambiguo prima dell'ingresso, vehicle-yield diventa `NOT_APPLICABLE` per quella coppia; collisione, TTC e clearance restano attivi.

## 2.9 Primitive canoniche del `SceneContext`

Le primitive di questa sezione sono normative. Un'implementazione conforme non può sostituirle con euristiche equivalenti, flag approssimati o fallback dipendenti dalla sorgente.

### 2.9.1 Tassonomia e ID persistenti

Ogni feature e attore viene convertito in uno dei tipi:

```python
class ActorClass(Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    STATIC_COLLIDABLE = "static_collidable"
    INFRASTRUCTURE_NON_COLLIDABLE = "infrastructure_non_collidable"

class MapFeatureClass(Enum):
    DRIVABLE_LANE = "drivable_lane"
    SIDEWALK = "sidewalk"
    CROSSWALK = "crosswalk"
    ROAD_BOUNDARY = "road_boundary"
    LANE_MARKING_SOLID = "lane_marking_solid"
    LANE_MARKING_DASHED = "lane_marking_dashed"
    OTHER_NON_DRIVABLE = "other_non_drivable"
```

L'ID stabile viene scelto in quest'ordine:

1. ID persistente di `ScenarioDescription`;
2. ID persistente della map feature o dell'oggetto MetaDrive;
3. hash deterministico di `scenario_id`, tipo e geometria canonica arrotondata a $1\,mm$.

I segmenti grafici appartenenti alla stessa linea tratteggiata devono condividere un unico `logical_boundary_id`, derivato dalla feature originaria, non dal singolo collider.

Il footprint canonico di veicoli, pedoni e ciclisti è l'oriented bounding box costruito da centro, heading, lunghezza e larghezza correnti. Le dimensioni devono essere finite e strettamente positive. Per `STATIC_COLLIDABLE` si usa il polygon collisionale live normalizzato; se è assente o invalido, lo scenario non è eleggibile. Tutte le formule geometriche del rulebook usano questi footprint canonici.

### 2.9.1-bis Compatibilità verticale 2.5D

Le operazioni geometriche vengono eseguite in pianta soltanto dopo un filtro verticale. Si adotta:

$$
z_{\mathrm{tol}}=3.0\,m.
$$

Ogni snapshot di attore contiene `position_z`. Ogni lane canonica contiene una funzione di quota piecewise-linear $z_\ell(s)$ derivata dalla sua centerline 3D. Se la sorgente fornisce soltanto una centerline 2D, l'adapter deve associare una quota valida dalla geometria stradale della sorgente; se non può farlo senza ambiguità in una mappa multilivello, lo scenario non è eleggibile.

Un attore $a$ è verticalmente compatibile con una lane $\ell$ nel punto proiettato $s_a$ se:

$$
|z_a-z_\ell(s_a)|\le z_{\mathrm{tol}}.
$$

Per una geometria planare $G$ derivata da lane o crosswalk, si conserva una funzione `elevation_at_xy`. Per confrontare due geometrie $G_1,G_2$ su una componente di overlap $Z$, si usa il set canonico:

$$
Q(Z)=\{\operatorname{representative\_point}(Z)\}\cup\operatorname{vertices}(Z).
$$

La componente è verticalmente compatibile se:

$$
\max_{\mathbf q\in Q(Z)}
|z_{G_1}(\mathbf q)-z_{G_2}(\mathbf q)|
\le z_{\mathrm{tol}}.
$$

Per un crosswalk privo di quota esplicita, la quota viene derivata dalle lane drivable che lo intersecano: al suo `representative_point` tutte le lane candidate devono concordare entro $z_{\mathrm{tol}}$; altrimenti il crosswalk non supera la validazione.

Una conflict zone conserva l'elevazione del corridoio ego tramite `elevation_at_xy`. Un attore è candidato per TTC zonale, crosswalk o vehicle-yield soltanto se il suo centro è verticalmente compatibile con la zona nel punto planare corrente o proiettato. Cavalcavia e sottopassi che si intersecano soltanto in proiezione 2D non generano associazioni, collision candidates o conflict zone.

### 2.9.2 Route polyline e proiezione curvilinea

Al reset si costruisce una `RoutePolyline` wrapper-owned concatenando le centerline 3D delle lane della route nell'ordine topologico. Le operazioni curvilinee usano la proiezione XY, mentre la quota interpolata resta disponibile come `z_at_s`. I punti consecutivi a distanza planare $\le10^{-3}\,m$ vengono unificati. Ogni segmento conserva il proprio intervallo di ascissa cumulativa.

Per proiettare un punto $\mathbf p$:

1. calcolare la distanza planare da ogni segmento candidato mediante spatial index;
2. scartare i segmenti non verticalmente compatibili con `position_z`, quando disponibile;
3. porre $d_{\min}$ uguale alla distanza minima;
4. mantenere le proiezioni con distanza $\le d_{\min}+\varepsilon_{\mathrm{geom}}$;
5. al reset scegliere il minore valore $s$;
6. negli step successivi scegliere il candidato che minimizza $|s-s_{\mathrm{previous}}|$;
7. in parità scegliere l'indice di segmento minore.

La funzione restituisce sempre `(s, tangent_xy, z_at_s, lateral_distance, segment_index)`. Un risultato non finito o l'assenza di segmenti validi causa `RulebookEvaluationError`.

Per un footprint $P$, si proietta prima il centro ottenendo $s_P$; ciascun vertice viene quindi proiettato usando `previous_s=s_P`, così tutti i vertici restano sullo stesso ramo locale della route. Si definiscono:

$$
s^{\mathrm{front}}(P)=\max_{\mathbf x\in\operatorname{vertices}(P)}s_{\mathrm{route}}(\mathbf x;s_P),
$$

$$
s^{\mathrm{rear}}(P)=\min_{\mathbf x\in\operatorname{vertices}(P)}s_{\mathrm{route}}(\mathbf x;s_P).
$$

$B_{\mathrm{front}}(P)$ è il segmento fra i due vertici con coordinata longitudinale body-frame massima.

### 2.9.3 Lane association e gap RSS

Per un attore $i$ si considerano esclusivamente le lane della route ego. Una lane è candidata se:

1. il centro del footprint appartiene al polygon della lane bufferizzato di $\varepsilon_{\mathrm{geom}}$;
2. la lane è verticalmente compatibile con l'attore secondo la Sezione 2.9.1-bis.

Le candidate vengono ordinate per:

1. minore disallineamento angolare assoluto con la tangente legale della lane;
2. minore valore assoluto della coordinata laterale;
3. `lane_id` lessicograficamente minore.

Se le prime due candidate differiscono di non più di $\varepsilon_{\psi}=10^{-6}\,rad$ nel disallineamento e di non più di $\varepsilon_{\mathrm{lat}}=10^{-3}\,m$ nella coordinata laterale, l'associazione è ambigua e RSS è `NOT_APPLICABLE` per quell'attore.

Il verso è concorde se:

$$
\mathbf h_i^\top\mathbf t_{\ell_i}>0.
$$

Il gap bumper-to-bumper è:

$$
d_i=\max\left(0,s_i^{\mathrm{rear}}-s_e^{\mathrm{front}}\right).
$$

L'attore è davanti se:

$$
s_i^{\mathrm{rear}}\ge s_e^{\mathrm{front}}-\varepsilon_\delta.
$$

### 2.9.4 Superficie carrabile 2.5D

Al reset non si collassano tutte le lane in un unico polygon globale privo di quota. Si costruisce invece:

```python
@dataclass(frozen=True)
class DrivableLaneRecord:
    lane_id: str
    polygon_xy: Polygon
    elevation_at_xy: Callable
```

A ogni step, per l'ego si selezionano tutte e sole le lane `DRIVABLE_LANE` verticalmente compatibili con il centro ego e con almeno una parte del footprint entro la query planare. La superficie normativa dello step è:

$$
C_{\mathrm{drive}}(t)
=
\bigcup_{\ell\in\mathcal L_{\mathrm{vehicle}}^{e}(t)}P_\ell.
$$

Per ogni lane:

- se esiste un polygon valido, viene usato dopo normalizzazione;
- altrimenti viene costruito bufferizzando la centerline di metà `lane_width` con cap piatto e join mitre;
- se manca anche `lane_width`, lo scenario non è eleggibile.

Sidewalk, shoulder non autorizzate, crosswalk e marking non vengono aggiunti separatamente. Gli attori dinamici non sottraggono area. Uno spatial index può accelerare la selezione, ma il risultato deve essere equivalente all'unione di tutte le lane verticalmente compatibili.

### 2.9.5 Catalogo dei traffic controls e selezione attiva

Al reset viene costruito:

```python
@dataclass(frozen=True)
class TrafficControlRecord:
    control_group_id: str
    control_type: str              # SIGNAL | STOP
    controlled_lane_ids: tuple[str, ...]
    movement_key: MovementKey
    control_line: LineString
    route_s: float
    elevation_m: float
    physical_control_ids: tuple[str, ...]
```

Più semafori fisici che governano la stessa `MovementKey` e la stessa control line formano un solo `control_group_id`. Se i loro stati correnti sono discordanti, lo scenario non è eleggibile.

Per ogni control line, `route_s` è il minimo valore curvilineo dei punti di intersezione fra la linea e la `RoutePolyline` bufferizzata di $\varepsilon_{\mathrm{geom}}$, limitato alle intersezioni verticalmente compatibili. Se non esiste intersezione oppure esistono intersezioni appartenenti a movimenti differenti non separabili dal record di lane association, il controllo non supera la validazione.

Un controllo è candidato se la sua `MovementKey` coincide con il movimento ego pertinente, è verticalmente compatibile e non è già risolto. Il controllo attivo è:

$$
c^\star
=
\arg\min_c(s_c-s_e^{\mathrm{front}}),
\qquad
s_c-s_e^{\mathrm{front}}\ge-\varepsilon_\delta.
$$

Tie-break: `route_s`, poi `control_group_id`. Al reset i controlli già oltre il front bumper di più di $\varepsilon_\delta$ vengono raccolti negli insiemi `prepassed_signal_ids` e `prepassed_stop_ids`. Un segnale attivo sopprime lo stop associato alla stessa `MovementKey`.

Non si seleziona mai un controllo mediante sola distanza euclidea.

### 2.9.6 Control line canonica

Una control line esplicita, associata e validata prevale. Altrimenti si usa esclusivamente:

```python
derive_control_line(control_point, controlled_lane) -> LineString
```

Algoritmo:

1. proiettare il control point sulla centerline della lane ottenendo $s_c$, punto $\mathbf q_c$, quota e tangente $\mathbf t_c$;
2. verificare la compatibilità verticale fra controllo e lane;
3. costruire la retta passante per $\mathbf q_c$ con direzione normale $\mathbf n_c=(-t_{c,y},t_{c,x})$;
4. intersecarla con il polygon della lane;
5. conservare la componente che contiene $\mathbf q_c$, oppure quella a distanza minima da $\mathbf q_c$ entro $\varepsilon_{\mathrm{geom}}$;
6. orientare la signed distance positiva nel lato precedente alla linea e negativa nel lato successivo;
7. se esiste una conflict zone immediatamente successiva, verificare
   $$
   s_c<s_{\mathrm{entry}}(Z)-\varepsilon_\delta.
   $$

Intersezione vuota, geometria multipla non risolvibile o verso incoerente rendono lo scenario non eleggibile. Non è ammesso spostare la linea alla fine della lane o al controllo più vicino.

### 2.9.7 Intervallo di occupazione mediante continuous SAT

La funzione normativa è:

```python
predict_occupancy_interval(actor_footprint, actor_velocity, zone, horizon_s)
```

Durante $[0,T_{\mathrm{pred}}]$ il footprint trasla a velocità costante e non ruota:

$$
P_i(\tau)=P_i(0)+\mathbf v_i\tau.
$$

Prima del solver si verifica la compatibilità verticale attore-zona. Se non è soddisfatta, il risultato è `NO_INTERVAL`.

La decomposizione convessa è deterministica:

1. normalizzare l'orientamento dell'anello esterno in senso antiorario e degli hole in senso orario;
2. ordinare gli hole per il loro vertice lessicograficamente minimo;
3. collegare ogni hole all'anello esterno mediante la coppia di vertici visibili con distanza minima, con tie-break lessicografico;
4. applicare ear clipping; a ogni iterazione scegliere fra le ear valide quella il cui vertice centrale ha la tripla `(x, y, original_index)` lessicograficamente minima;
5. scartare triangoli con area $\le\varepsilon_A$;
6. ordinare i triangoli risultanti per `(centroid_x, centroid_y, area)`.

Si applica il continuous Separating Axis Theorem a ogni coppia di triangoli usando le normali agli spigoli di entrambe. Su ogni asse unitario $\mathbf n$, l'intervallo mobile dell'attore è:

$$
[a_{\min}+w\tau,\;a_{\max}+w\tau],
\qquad
w=\mathbf v_i^\top\mathbf n,
$$

e quello statico della zona è $[z_{\min},z_{\max}]$. Le disuguaglianze di overlap vengono convertite in un intervallo temporale e intersecate con $[0,T_{\mathrm{pred}}]$.

Gli intervalli di tutte le coppie vengono uniti; intervalli separati da non più di $\varepsilon_t=10^{-6}\,s$ vengono fusi. Si restituisce:

1. l'intervallo che contiene $0$, se presente;
2. altrimenti l'intervallo con ingresso minimo;
3. `NO_INTERVAL` se l'unione è vuota;
4. `OPEN_END` se l'estremo destro raggiunge $T_{\mathrm{pred}}-\varepsilon_t$ e i polygon risultano ancora sovrapposti a $T_{\mathrm{pred}}$.

Un attore fermo dentro produce `[0, OPEN_END]`; un attore fermo fuori produce `NO_INTERVAL`. Non si rappresenta `OPEN_END` con infinito floating-point.

### 2.9.8 Tipi di controllo e relazioni di priorità

Il context extractor core usa soltanto:

```python
class ApproachControl(Enum):
    NONE = "none"
    STOP = "stop"
    SIGNAL = "signal"
    UNKNOWN = "unknown"

class MovementPriority(Enum):
    OTHER_HAS_PRIORITY = "other_has_priority"
    EGO_HAS_PRIORITY = "ego_has_priority"
    UNDEFINED = "undefined"

@dataclass(frozen=True)
class MovementPriorityRecord:
    ego_movement_key: MovementKey
    other_movement_key: MovementKey
    relation: MovementPriority
```

La priorità esplicita è esclusivamente pairwise e non è un valore di `ApproachControl`. La sola geometria non genera un `MovementPriorityRecord`. `UNKNOWN` non viene interpretato come `NONE`.

`ApproachControl` viene assegnato con questa precedenza: `SIGNAL` se esiste un signal group pertinente; altrimenti `STOP`; altrimenti `NONE`. Un controllo presente ma non interpretabile produce `UNKNOWN` e rende non applicabile il predicato che richiederebbe `NONE`. Yield sign e altre categorie non mappate nel core vengono dichiarate fuori dominio o producono `UNKNOWN`; non vengono trasformate in priorità implicita.

Record pairwise contraddittori rendono lo scenario non eleggibile. La coesistenza signal+stop sulla stessa `MovementKey` è ammessa perché il segnale sopprime operativamente lo stop.

### 2.9.9 Esiti delle primitive

Ogni primitiva ha esattamente tre esiti:

```text
valore valido
None / NOT_APPLICABLE
RulebookEvaluationError / NOT_EVALUABLE
```

`None` è ammesso soltanto quando il contesto non appartiene al dominio della regola. Un dato obbligatorio mancante o una geometria invalida dopo che la regola è risultata applicabile causa errore. Non esistono fallback silenziosi.

# 3. RulebookMonitor

## 3.1 Interfaccia transazionale di transizione

Il monitor valuta una transizione completa, non un singolo snapshot:

$$
(\operatorname{result}_t,h_t,\Delta\mathcal K_t)
=
\operatorname{RulebookMonitor}
(x_t,x_{t+1},h_{t-1},\mathcal K_{t-1}).
$$

Dove:

- $x_t$ è il `pre_state` catturato immediatamente prima di `env.step(action)`;
- $x_{t+1}$ è il `post_state` catturato immediatamente dopo;
- $h_{t-1}$ è la memoria comportamentale immutabile durante la valutazione;
- $\mathcal K_{t-1}$ è la cache episodica committata e immutabile durante la valutazione;
- $\Delta\mathcal K_t$ contiene soltanto nuove conflict zone;
- risultato, memoria e cache delta vengono committati insieme soltanto se tutte le componenti terminano senza errore.

Interfaccia normativa:

```python
result, next_memory, cache_delta = monitor.evaluate_transition(
    pre_state=pre_state,
    post_state=post_state,
    memory=memory,
    cache=episode_cache,
)
```

Gli evaluator sono funzioni pure e restituiscono:

```python
component_result, memory_delta, cache_delta
```

Nessun evaluator modifica lateralmente `memory` o `cache`.

## 3.2 Stato episodico e delta

```python
@dataclass(frozen=True)
class RulebookMemory:
    previous_contact_ids: frozenset[str]

    active_dashed_boundary_id: str | None
    dashed_line_timer_s: float

    active_signal_group_id: str | None
    previous_signal_state: str | None
    yellow_must_stop: bool
    previous_signal_delta_m: float | None
    resolved_signal_group_ids: frozenset[str]

    active_stop_group_id: str | None
    stop_continuous_timer_s: float
    stop_best_timer_s: float
    previous_stop_delta_m: float | None
    resolved_stop_group_ids: frozenset[str]

    crosswalk_illegal_entries: frozenset[tuple[str, str]]
    vehicle_yield_illegal_entries: frozenset[tuple[str, str]]
    preexisting_ego_occupancy_zone_ids: frozenset[str]
    frozen_actor_movement_keys: tuple[tuple[str, MovementKey], ...]

    previous_route_s_m: float

@dataclass(frozen=True)
class MemoryDelta:
    writes: tuple[tuple[str, object], ...] = ()

@dataclass(frozen=True)
class CacheDelta:
    new_conflict_zones: tuple[ConflictZoneRecord, ...] = ()

@dataclass(frozen=True)
class EpisodeCache:
    scenario_id: str
    route_polyline: RoutePolyline
    drivable_lane_records: tuple[DrivableLaneRecord, ...]
    map_feature_catalog: Mapping[str, MapFeatureRecord]
    logical_boundaries: Mapping[str, BoundaryRecord]
    traffic_control_catalog: tuple[TrafficControlRecord, ...]
    movement_priority_records: tuple[MovementPriorityRecord, ...]
    roundabout_components: tuple[RoundaboutComponent, ...]
    conflict_zones: Mapping[str, ConflictZoneRecord]
```

La cache contiene geometrie e cataloghi, non timer o decisioni della policy.

`merge_memory_deltas(memory, deltas)` è l'unica funzione che costruisce `next_memory`. Ogni campo della memoria possiede un solo writer normativo per transizione. Due delta che scrivono lo stesso campo producono `RulebookEvaluationError`; non si applica un ordine arbitrario di sovrascrittura.

`merge_cache_deltas(cache, deltas)` verifica che:

- non esistano due geometrie differenti per lo stesso `zone_id`;
- una zona già presente sia byte-equivalente nella rappresentazione canonica;
- i nuovi ID siano unici.

La funzione restituisce un unico `CacheDelta`, ma non modifica la cache.

### 3.2.1 Inizializzazione al reset

Subito dopo `env.reset()`:

1. costruire e validare `EpisodeCache` statico;
2. catturare `reset_state`;
3. inizializzare `previous_contact_ids` con i contatti già attivi;
4. proiettare l'ego sulla `RoutePolyline` e salvare `previous_route_s_m`;
5. selezionare il primo signal/stop pertinente;
6. inizializzare colore, signed distance e automi dai controlli live;
7. se il segnale è già giallo, calcolare subito `yellow_must_stop` dallo stato di reset;
8. inizializzare timer a zero;
9. inizializzare
   ```python
   resolved_signal_group_ids = frozenset(prepassed_signal_ids)
   resolved_stop_group_ids = frozenset(prepassed_stop_ids)
   ```
10. inserire in `preexisting_ego_occupancy_zone_ids` tutte le conflict zone statiche già occupate dall'ego;
11. inizializzare `frozen_actor_movement_keys` vuoto.

Un contatto o ingresso già presente al reset non viene attribuito alla policy. Dopo la prima uscita da una zona preesistente, il relativo ID viene rimosso e gli ingressi successivi vengono valutati normalmente. Se una zona vehicle-yield viene creata lazy mentre l'ego la occupa già, viene aggiunta allo stesso insieme attraverso il `MemoryDelta` della transizione di creazione.

## 3.3 Ordine normativo per-step

Per ogni azione:

1. catturare `pre_state` completo;
2. eseguire `env.step(action)`;
3. catturare `post_state` completo;
4. risolvere ID, attori live, controlli, `MovementKey` e corridoi senza modificare memoria o cache;
5. proporre eventuali zone lazy in `CacheDelta`; tali zone sono visibili agli evaluator della stessa transizione attraverso una vista immutabile `cache + pending_delta`;
6. rilevare sul segmento pre→post collision onset, crossing di marking/control line e ingresso/uscita dalle conflict zone;
7. calcolare le componenti state-based sul `post_state`;
8. valutare ogni componente temporale, ottenendo `RuleComponentResult`, `MemoryDelta` e `CacheDelta`;
9. usare il colore del `pre_state` per giudicare il crossing del semaforo e lo stato del `post_state` per il costo di approccio;
10. fondere i delta, costruire `next_memory` e il `cache_delta` complessivo;
11. calcolare macro-costi, margini e diagnostics;
12. verificare evaluabilità, finitezza, unicità dei writer e consistenza della cache;
13. restituire `(result, next_memory, cache_delta)`;
14. il training loop applica atomicamente `cache_delta`, assegna `memory = next_memory` e soltanto dopo memorizza la transizione.

I timer post-update vengono usati per il costo della transizione corrente, come specificato nelle relative regole. Se una componente solleva `RulebookEvaluationError`, nessuno dei tre oggetti viene committato.

## 3.4 Politica sui fallback

Sono vietati:

- semaforo o stop scelto per sola prossimità euclidea;
- lane approssimata quando l'associazione è ambigua;
- intersezioni planari trattate come conflitti senza verifica verticale;
- control line spostata a fine lane dopo una derivazione fallita;
- route longitude nativa usata al posto della proiezione canonica;
- sampling temporale discreto usato al posto del continuous SAT;
- valore zero restituito dopo eccezione;
- future trajectory usata per compensare dati mancanti;
- modifica diretta di memoria o cache da parte di un evaluator.

La pipeline deve validare preliminarmente gli scenari, fallire quando manca un dato core e non salvare come valido lo stato del processo fallito. Non sono previsti rollback episodici o staging buffer: il commit transazionale impedisce l'inserimento della transizione invalida.

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

Il collision hook mantiene l'insieme dei contatti attivi a ogni physics substep e produce un `ContactOnsetRecord` quando un ID stabile passa da non attivo ad attivo. Al reset l'insieme viene inizializzato con i contatti già presenti, che non generano onset.

Durante un singolo control step, gli onset dello stesso attore vengono deduplicati per actor ID; i relativi contact point restano tutti disponibili per il massimo di severità. Si definisce:

$$
\mathcal C_t^{\mathrm{new}}
=
\left\{
i:\exists\,\texttt{ContactOnsetRecord}(i)
\text{ durante }x_t\rightarrow x_{t+1}
\right\}.
$$

Il post-state conserva anche l'insieme dei contatti ancora attivi, che diventa `previous_contact_ids` nella nuova memoria. Un contatto iniziato e terminato interamente fra due control frame viene comunque rilevato. Un contatto persistente non genera nuovi onset finché non si verifica almeno un physics substep senza contatto e un successivo nuovo inizio.

## 5.3 Severità raw

Per un contatto nuovo con oggetto $i$, sia $\mathcal P_i$ l'insieme dei contact point restituiti dal manifold fisico per lo stesso ID stabile. Per ciascun punto $p$:

$$
u_{i,p}(t)
=
\left[
(\mathbf v_e^{-}-\mathbf v_i^{-})^\top
\mathbf n_{i,p}
\right]_+,
$$

dove le velocità $\mathbf v^{-}$ provengono dal `pre_state` della transizione. Per un oggetto classificato `STATIC_COLLIDABLE` si pone $\mathbf v_i^{-}=\mathbf0$. Se un attore dinamico non esiste nel `pre_state` ma compare già in contatto nel `post_state`, il monitor solleva `RulebookEvaluationError`: non viene inventata una velocità pre-impatto. La validazione offline deve inoltre escludere spawn con footprint inizialmente sovrapposto all'ego.

La velocità normale rappresentativa dell'oggetto è:

$$
u_i(t)=\max_{p\in\mathcal P_i}u_{i,p}(t).
$$

Dove:

- $\mathbf v_e^{-}$: velocità ego nel `pre_state`;
- $\mathbf v_i^{-}$: velocità dell'altro oggetto nel `pre_state`, nulla soltanto per `STATIC_COLLIDABLE`;
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
| $\mathcal C_t^{\mathrm{new}}$ | onset deduplicati durante la transizione | contact hook custom |
| $\mathcal C_{t+1}^{\mathrm{active}}$ | contatti attivi nel post-state | contact hook custom |
| $\mathbf v_e^{-},\mathbf v_i^{-}$ | velocità nel `pre_state` | transition snapshot |
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
- associa il record alle velocità del `pre_state`;
- rileva l'onset a ogni physics substep, deduplica per control step e restituisce anche gli ID ancora attivi nel post-state.

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

Si applica ai veicoli frontali associati alla stessa sequenza di lane della route ego mediante la primitiva della Sezione 2.9.3.

Per ciascun veicolo live:

1. eseguire `associate_actor_to_ego_route`;
2. richiedere heading concorde $\mathbf h_i^\top\mathbf t_{\ell_i}>0$;
3. calcolare $s_i^{\mathrm{rear}}$ e $s_e^{\mathrm{front}}$ sulla `RoutePolyline`;
4. considerarlo frontale se $s_i^{\mathrm{rear}}\ge s_e^{\mathrm{front}}-\varepsilon_\delta$;
5. calcolare il gap canonico:
   $$
   d_i=\max(0,s_i^{\mathrm{rear}}-s_e^{\mathrm{front}}).
   $$

Un'associazione ambigua produce `NOT_APPLICABLE` per quello specifico attore; TTC e clearance restano attivi. Un errore nella route o nei polygon richiesti dopo che lo scenario è stato validato produce `RulebookEvaluationError`.

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

### 6.2.4 Parametri e calibrazione di $b_e$

Valori congelati:

| Parametro | Valore |
|---|---:|
| $\rho$ | $1.0\,s$ |
| $a_{\max}^{\mathrm{acc}}$ | $3.5\,m/s^2$ |
| $b_i$ | $8.0\,m/s^2$ |
| $b_e$ | valore prodotto dal protocollo seguente |

$\rho$, $a_{\max}^{\mathrm{acc}}$ e $b_i$ seguono i suggested starting values di Intel `ad-rss-lib` [4]. $b_e$ è una costante del task generata una volta per il modello ego effettivo.

Protocollo normativo:

1. strada rettilinea, piana, senza traffico e con la stessa configurazione fisica del training;
2. verificare preliminarmente che la configurazione ego raggiunga stabilmente almeno $20\,m/s$; in caso contrario la configurazione non può usare questo protocollo e il run finale non può iniziare;
3. velocità target $\{5,10,15,20\}\,m/s$;
4. dieci prove valide per ciascuna velocità;
5. una prova è valida se raggiunge la velocità target entro $0.2\,m/s$, non collide e non esce dalla lane;
6. applicare il massimo comando di frenata fino all'arresto;
7. misurare la decelerazione media positiva fra il primo campione sotto il $90\%$ e il primo campione sotto il $10\%$ della velocità iniziale;
8. se non sono disponibili dieci prove valide per ogni velocità, la calibrazione fallisce;
9. calcolare il quinto percentile con metodo order-statistic `lower`:
   $$
   b_{\mathrm{meas}}=Q_{0.05}^{\mathrm{lower}}(\{b_j\});
   $$
10. fissare:
   $$
   b_e=
   \min\left(4.0,\frac{\lfloor10b_{\mathrm{meas}}\rfloor}{10}\right)\,m/s^2.
   $$

Un valore non positivo o non finito invalida la configurazione. Il file prodotto dalla calibrazione, comprensivo di hash della configurazione del veicolo, deve essere caricato dal validatore. Lo stesso $b_e$ viene usato da RSS, semaforo, crosswalk e vehicle-yield e non cambia fra algoritmi, seed o sorgenti di scenario.

## 6.3 TTC generalizzato

### 6.3.1 Formula

Per un attore dinamico o un ostacolo statico $i$ verticalmente compatibile con l'ego secondo la Sezione 2.9.1-bis, assumendo velocità e heading costanti sul breve orizzonte:

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

Se non esiste intersezione entro $T_{\mathrm{pred}}=3.0\,s$, si pone $TTC_i=\infty$. Il calcolo usa obbligatoriamente il continuous SAT della Sezione 2.9.7 applicato al moto relativo dei due footprint; per gli statici si pone $\mathbf v_i=\mathbf0$.

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
- verticalmente compatibili con l'ego secondo la Sezione 2.9.1-bis;
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
| canonical route curvilinear coordinate | ordinamento davanti/dietro |
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

La regione normativa dello step è la superficie 2.5D definita nella Sezione 2.9.4:

$$
C_{\mathrm{drive}}(t)
=
\bigcup_{\ell\in\mathcal L_{\mathrm{vehicle}}^{e}(t)}P_\ell,
$$

dove $\mathcal L_{\mathrm{vehicle}}^{e}(t)$ contiene tutte e sole le lane `DRIVABLE_LANE` verticalmente compatibili con l'ego. I `DrivableLaneRecord` e le relative quote sono costruiti al reset; l'unione della superficie pertinente viene invece determinata a ogni step. Non dipende dalla route futura, dagli attori dinamici, dalla direzione legale o dalle lane markings. Una query spaziale può limitare le lane planari candidate, ma deve poi applicare il filtro verticale e produrre lo stesso risultato dell'unione di tutte le lane compatibili.

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

`ego.on_lane` e `_is_out_of_road` possono essere loggati come checks, ma la formula normativa è l'area fraction sulla superficie verticale pertinente. Se polygon o quota carrabile non sono interrogabili correttamente, la sottoregola è `NOT_EVALUABLE`; non si sostituisce silenziosamente con un flag differente.

## 7.3 Wrong-way

### 7.3.1 Velocità longitudinale firmata

Sia $\mathbf t_{\mathrm{ref}}(t)$ la tangente unitaria restituita dalla proiezione canonica sulla `RoutePolyline`, orientata secondo l'ordine della route ego.

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
- indice del segmento della `RoutePolyline` che ha fornito la tangente.

Il coseno fra heading ego e heading lane può rimanere una diagnostica, ma non definisce il costo principale.

### 7.3.4 Applicabilità

La tangente normativa è quella restituita dalla proiezione canonica del centro ego sulla `RoutePolyline`. Non si usa una lane vicina come fallback. Se la proiezione è finita, la regola è applicabile; se route o proiezione non sono valide, viene sollevata `RulebookEvaluationError`. La tangente della lane può essere loggata esclusivamente come controllo diagnostico.

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

Si seleziona la boundary tratteggiata attiva con criterio deterministico: se la boundary attiva allo step precedente è ancora intersecata viene mantenuta; altrimenti si sceglie fra quelle intersecate la minima distanza dal centro del footprint, con `logical_boundary_id` lessicograficamente minore come tie-breaker stabile. Se ne memorizza l'ID:

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
\{\mathrm{GREEN},\mathrm{YELLOW},\mathrm{RED},\mathrm{FLASHING\_YELLOW},\mathrm{UNKNOWN}\}
$$

lo stato del `control_group_id` attivo selezionato esclusivamente mediante il catalogo e l'algoritmo della Sezione 2.9.5.

Più corpi semaforici sullo stesso movimento sono un unico gruppo. Devono avere stato concorde a ogni step; una discordanza è `NOT_EVALUABLE`. Gli scenari con stato `UNKNOWN` o sequenza incompleta per un gruppo pertinente vengono esclusi offline. Un gruppo non pertinente non influisce sull'eleggibilità.

Quando non esiste un signal group non risolto davanti all'ego, la componente è `NOT_APPLICABLE`. Non si usa mai il semaforo euclideamente più vicino.

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

La memoria usa `active_signal_group_id` e l'insieme immutabile `resolved_signal_group_ids`.

Per la transizione corrente:

1. il gruppo attivo è quello selezionato nel `pre_state` e non ancora risolto;
2. colore, `yellow_must_stop` e signed distance pre-azione provengono dalla memoria;
3. il crossing viene giudicato con tali valori pre-azione;
4. se non avviene crossing, si legge il colore dello stesso gruppo nel `post_state`, si rileva l'onset del giallo e si aggiorna il costo di approccio;
5. se avviene crossing, il gruppo viene aggiunto a `resolved_signal_group_ids` dopo il calcolo del costo e non produce costo di approccio nel `post_state`;
6. usando il `post_state`, si seleziona il prossimo gruppo non risolto; se cambia ID, il nuovo automa viene inizializzato dal suo colore e dalla sua signed distance correnti;
7. se il nuovo gruppo è già giallo, `yellow_must_stop` viene calcolato immediatamente;
8. `FLASHING_YELLOW` rende la componente semaforica `NOT_APPLICABLE` per quel gruppo; `UNKNOWN` è errore.

Il gruppo risolto non viene rivalutato anche se l'ego arretra.

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

### 7.7.2 Control line canonica

La stop control line viene recuperata dal `TrafficControlRecord` attivo. Una linea esplicita e validata prevale; altrimenti viene costruita una sola volta al reset mediante `derive_control_line` della Sezione 2.9.6.

La derivazione usa il control point, la lane controllata, la sezione ortogonale del polygon e il verso route. Se la geometria non è unica o non è collocata prima della conflict zone associata, lo scenario non è eleggibile. La linea derivata è una costruzione operativa e non viene descritta come marking dipinta ground-truth.

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

La memoria usa `active_stop_group_id` e `resolved_stop_group_ids`.

1. selezionare nel `pre_state` il primo stop non risolto, salvo soppressione da parte di un signal group sullo stesso movimento;
2. calcolare nel `post_state` il nuovo `T_cont`: valore precedente più $\Delta t$ se l'ego è nella stop zone ed è in standstill, altrimenti zero;
3. calcolare `T_best_next=max(T_best_previous,T_cont_next)`;
4. rilevare il crossing pre→post;
5. se avviene crossing, usare `T_best_next` nel costo, aggiungere il gruppo a `resolved_stop_group_ids` e non rivalutarlo più;
6. selezionare dal `post_state` il prossimo stop non risolto; se cambia ID, inizializzare i suoi timer a zero e la signed distance al valore corrente;
7. un arretramento dopo il crossing non rimuove il gruppo dall'insieme `resolved`.

L'aggiornamento di timer e risoluzione avviene nella copia `next_memory`; la memoria originale resta invariata fino al commit.

## 7.8 Crosswalk yield

### 7.8.1 Scopo

La regola non inferisce l'intenzione di un VRU fermo lontano dal crosswalk e non usa la sua futura track ground-truth. Rileva invece un'incompatibilità temporale fra ego e pedone/ciclista in una porzione di crosswalk che interseca il corridoio ego ed è verticalmente compatibile. La sicurezza fisica generale resta inoltre coperta da TTC e clearance in $R_2$.

### 7.8.2 Conflict zone rilevante

Le conflict zone dei crosswalk vengono costruite secondo la Sezione 2.8.4. Per ogni crosswalk $W$, la zona pertinente $Z_W$ è la componente occupata dall'ego, altrimenti la prima ancora davanti. Se non esiste alcuna componente valida o verticalmente compatibile, la regola è `NOT_APPLICABLE` per quel crosswalk.

La distanza route del front bumper dall'ingresso è definita esplicitamente come:

$$
d_e^{Z_W}(t)
=
s_{Z_W}^{\mathrm{entry}}-s_e^{\mathrm{front}}(t).
$$

### 7.8.3 Insieme dei candidati e intervalli

Si calcola prima l'intervallo ego:

$$
I_e^{Z_W}(t)=\operatorname{predict\_occupancy\_interval}(e,Z_W).
$$

Se:

$$
I_e^{Z_W}(t)=\texttt{NO\_INTERVAL},
$$

la zona non contribuisce nello step corrente: il suo costo è zero e non viene calcolato alcun gap.

Altrimenti l'insieme normativo dei VRU candidati è:

$$
\mathcal A_W(t)
=
\left\{
j:
\begin{array}{l}
j\text{ è live e }\operatorname{class}(j)\in\{\texttt{PEDESTRIAN},\texttt{CYCLIST}\},\\
j\text{ è verticalmente compatibile con }Z_W,\\
I_j^{Z_W}(t)\neq\texttt{NO\_INTERVAL}
\end{array}
\right\}.
$$

Per $I_e=[t_{e,\mathrm{in}},t_{e,\mathrm{out}}]$ e $I_j=[t_{j,\mathrm{in}},t_{j,\mathrm{out}}]$:

$$
g_j^{\mathrm{sep}}
=
\begin{cases}
\max\{t_{e,\mathrm{in}}-t_{j,\mathrm{out}},\;t_{j,\mathrm{in}}-t_{e,\mathrm{out}}\},
&\text{uscite entrambe finite},
\\[2mm]
t_{j,\mathrm{in}}-t_{e,\mathrm{out}},
&t_{e,\mathrm{out}}\text{ finito e }t_{e,\mathrm{out}}\le t_{j,\mathrm{in}},
\\[2mm]
t_{e,\mathrm{in}}-t_{j,\mathrm{out}},
&t_{j,\mathrm{out}}\text{ finito e }t_{j,\mathrm{out}}\le t_{e,\mathrm{in}},
\\[2mm]
-\infty,
&\text{altrimenti}.
\end{cases}
$$

`OPEN_END` è simbolico; non si eseguono operazioni floating-point con infinito.

### 7.8.4 Margine temporale

$$
r_{\mathrm{gap},j}
=
\operatorname{clip}
\left(
\frac{T_{\mathrm{gap}}-g_j^{\mathrm{sep}}}{T_{\mathrm{gap}}},
0,1
\right),
\qquad
T_{\mathrm{gap}}=1.0\,s,
$$

con $r_{\mathrm{gap},j}=1$ quando $g_j^{\mathrm{sep}}=-\infty$.

### 7.8.5 Commitment dell'ego

Sia:

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
1-\dfrac{d_e^{Z_W}}{d_{\mathrm{stop}}(v_e^{\mathrm{app},W})},
0,1
\right),
&d_{\mathrm{stop}}(v_e^{\mathrm{app},W})>0,
\\[3mm]
0,&d_{\mathrm{stop}}(v_e^{\mathrm{app},W})=0,
\end{cases}
$$

con:

$$
d_{\mathrm{stop}}(v)
=
v\Delta t+\frac{v^2}{2b_e}.
$$

Il gate è:

$$
I_{\mathrm{before},W}(t)
=
\mathbf1
\left[
d_e^{Z_W}(t)>\varepsilon_\delta
\land P_e(t)\cap Z_W=\varnothing
\land \operatorname{zone\_id}(Z_W)\notin\mathcal Z_{\mathrm{preexisting}}(t)
\right].
$$

### 7.8.6 Evento, memoria e costo

Sia $X_W(t)$ l'evento di ingresso dell'ego in $Z_W$. Per ogni $j\in\mathcal A_W(t)$ con $r_{\mathrm{gap},j}>0$, l'evaluator propone nel proprio `MemoryDelta` la chiave:

$$
(\operatorname{actor\_id}(j),\operatorname{zone\_id}(Z_W))
$$

in `crosswalk_illegal_entries` se $X_W(t)=1$. La chiave resta attiva finché l'ego interseca $Z_W$ e viene rimossa alla prima uscita completa.

Il costo è:

$$
q_{\mathrm{crosswalk}}(t)
=
\max
\left\{
\max_W\max_{j\in\mathcal A_W(t)}
I_{\mathrm{before},W}(t)r_{\mathrm{gap},j}r_{\mathrm{commit}}^W,
\max_W I_{\mathrm{illegal},W}(t)\mathbf1[P_e(t)\cap Z_W\neq\varnothing]
\right\},
$$

con massimo sull'insieme vuoto uguale a zero. Prima della zona opera il costo continuo; dopo un ingresso legittimo si disattiva; dopo un ingresso incompatibile la permanenza nella zona vale uno fino all'uscita.

### 7.8.7 Variabili

- crosswalk polygon e quota canonica;
- `MovementKey` ego e conflict zone cached o pending;
- stato 3D ego e VRU live;
- intervalli ego/VRU;
- $d_e^{Z_W}$, $T_{\mathrm{gap}}$, $T_{\mathrm{pred}}$, $b_e$, $\Delta t$;
- `crosswalk_illegal_entries` e `preexisting_ego_occupancy_zone_ids`.

## 7.9 Yield veicolo-veicolo scoped

### 7.9.1 Predicato deterministico di priorità

Non viene implementato un motore giuridico universale. Per una conflict zone $Z$ e un veicolo $i$:

$$
\Pi_{i\succ e}(Z,t)
=
O_i(Z,t)\lor S_{i\succ e}(Z)\lor R_{i\succ e}(Z)\lor M_{i\succ e}(Z).
$$

**Zona già occupata**

$$
O_i(Z,t)=\mathbf1[P_i(t)\cap Z\neq\varnothing],
$$

purché l'attore sia verticalmente compatibile con la zona.

**Ego soggetto a stop**

$$
S_{i\succ e}(Z)=1
$$

se e solo se:

```text
ego_approach_control == STOP
other_approach_control == NONE
```

`SIGNAL` e `UNKNOWN` rendono il predicato falso. Yield sign non modellati non vengono convertiti in `NONE`.

**Ingresso in rotatoria**

$$
R_{i\succ e}(Z)=1
$$

se e solo se la lane ego appartiene alle entry lane validate e la lane dell'attore appartiene al ciclo diretto validato della stessa componente circolante.

**Relazione esplicita pairwise**

$$
M_{i\succ e}(Z)=1
$$

se e solo se esiste un `MovementPriorityRecord` con le esatte `MovementKey` della coppia e relazione `OTHER_HAS_PRIORITY`.

La sola geometria non crea priorità. La regola è `NOT_APPLICABLE` quando nessun predicato è vero, nei doppi stop, nei merge senza record esplicito, nei controlli `UNKNOWN`, nelle intersezioni uncontrolled ambigue o quando la `MovementKey` dell'attore non è univoca.

### 7.9.2 Conflict zone e candidati

La conflict zone $Z_{e,i}$ viene costruita mediante le Sezioni 2.8.1--2.8.3 dal corridoio topologico stabile identificato dalle `MovementKey`. Una nuova zona lazy è restituita tramite `CacheDelta`.

La distanza ego è:

$$
d_e^Z(t)=s_Z^{\mathrm{entry}}-s_e^{\mathrm{front}}(t).
$$

Si calcola l'intervallo ego $I_e^Z(t)$. Se è `NO_INTERVAL`, la zona non contribuisce nello step corrente.

Altrimenti:

$$
\mathcal A_Z(t)
=
\left\{
i:
\begin{array}{l}
i\text{ è un veicolo live e verticalmente compatibile con }Z,\\
\Pi_{i\succ e}(Z,t)=1,\\
I_i^Z(t)\neq\texttt{NO\_INTERVAL}
\end{array}
\right\}.
$$

Se $\mathcal A_Z(t)=\varnothing$, la zona non produce costo di approccio.

### 7.9.3 Gap temporale

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
-\infty,&\text{altrimenti}.
\end{cases}
$$

$$
r_{\mathrm{gap},i}
=
\operatorname{clip}
\left(
\frac{T_{\mathrm{gap}}-g_i^{\mathrm{sep}}}{T_{\mathrm{gap}}},0,1
\right),
\qquad T_{\mathrm{gap}}=1.0\,s,
$$

con $r_{\mathrm{gap},i}=1$ per $g_i^{\mathrm{sep}}=-\infty$.

### 7.9.4 Commitment dell'ego

$$
v_e^{\mathrm{app},i}=[\mathbf v_e^\top\mathbf t_{\mathrm{route}}]_+,
$$

$$
r_{\mathrm{commit},i}
=
\begin{cases}
\operatorname{clip}\left(1-\dfrac{d_e^Z}{d_{\mathrm{stop}}(v_e^{\mathrm{app},i})},0,1\right),
&d_{\mathrm{stop}}(v_e^{\mathrm{app},i})>0,
\\[2mm]
0,&\text{altrimenti}.
\end{cases}
$$

Il gate è:

$$
I_{\mathrm{before},Z}(t)
=
\mathbf1\left[
d_e^Z(t)>\varepsilon_\delta
\land P_e(t)\cap Z=\varnothing
\land \operatorname{zone\_id}(Z)\notin\mathcal Z_{\mathrm{preexisting}}(t)
\right].
$$

### 7.9.5 Evento, memoria e costo

Sia $X_{e,i}(t)$ l'evento di ingresso. Se $X_{e,i}(t)=1$ e $r_{\mathrm{gap},i}>0$, l'evaluator propone nel proprio `MemoryDelta` la chiave `(actor_id, zone_id)` in `vehicle_yield_illegal_entries`. Il flag resta attivo fino all'uscita completa.

Una `MovementKey` viene congelata in `frozen_actor_movement_keys` dall'ingresso nella zona fino all'uscita. Se una zona lazy viene creata quando l'ego la occupa già, essa viene prima marcata preesistente e non genera ingresso illegale nello stesso step.

Per ogni $i\in\mathcal A_Z(t)$:

$$
q_{\mathrm{yield},Z,i}^{\mathrm{approach}}
=
I_{\mathrm{before},Z}(t)r_{\mathrm{gap},i}r_{\mathrm{commit},i}.
$$

Il costo è:

$$
q_{\mathrm{yield,vehicle}}
=
\max\left\{
\max_Z\max_{i\in\mathcal A_Z(t)}q_{\mathrm{yield},Z,i}^{\mathrm{approach}},
\max_{Z\in\mathcal Z_{\mathrm{illegal}}}I_{\mathrm{illegal},Z}(t)\mathbf1[P_e(t)\cap Z\neq\varnothing]
\right\},
$$

con massimo sull'insieme vuoto uguale a zero. Il secondo termine resta attivo anche se l'attore originario scompare.

### 7.9.6 Natura della formula

- priorità: quattro predicati deterministici e limitati;
- identità del movimento: `MovementKey` topologica stabile;
- conflict zone: geometria 2.5D canonica e cache transazionale;
- intervalli: previsione locale a velocità costante;
- nessuna inferenza universale del codice stradale.

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

Il progresso normativo usa esclusivamente la `RoutePolyline` wrapper-owned della Sezione 2.9.2. Siano $s_t$ e $s_{t+1}$ le proiezioni canoniche del centro ego nel `pre_state` e nel `post_state`:

$$
\Delta s_{\mathrm{route}}(t)=s_{t+1}-s_t.
$$

- positivo: avanzamento;
- zero: fermata;
- negativo: regressione o retromarcia.

La posizione ego del `pre_state` viene proiettata usando `memory.previous_route_s_m`; il risultato deve differire dalla memoria di non più di $\varepsilon_{\mathrm{geom}}$, altrimenti viene sollevata `RulebookEvaluationError` per discontinuità dello snapshot. La proiezione del `post_state` usa $s_t$ come riferimento di continuità. Il valore deve essere finito; altrimenti viene sollevata `RulebookEvaluationError`. `TrajectoryNavigation.current_longitude` e `last_longitude` vengono loggati soltanto come confronto diagnostico e non possono sostituire la coordinata canonica.

La quantità raw in metri per step viene sempre conservata. Poiché la route polyline e i tie-break sono deterministici, non esiste un'opzione runtime per passare alla longitudine nativa.

## 8.2 Segnale per il learner

$$
m_4(t)
=
\operatorname{clip}
\left(
\frac{\Delta s_{\mathrm{route}}(t)}{v_{\max,e}\Delta t},
-1,1
\right).
$$

$v_{\max,e}$ è il configured speed normalization cap validato dell'ego. Il clipping limita il segnale del learner ma non modifica il valore raw nei diagnostics.

## 8.3 Origine

MetaDrive usa un delta longitudinale analogo nel dense driving reward [7]. La specifica conserva l'idea di avanzamento lungo route, ma sostituisce la proiezione nativa con una coordinata curvilinea esplicita e condivisa fra PG e Waymo.

## 8.4 Perché non serve un gate

Una fermata corretta produce $m_4=0$, non una penalità. Essendo $R_4$ subordinata, il progresso non può compensare una violazione superiore.

## 8.5 Variabili e parametri

Variabili:

- `EpisodeCache.route_polyline`;
- proiezioni $s_t,s_{t+1}$;
- $v_{\max,e}$;
- $\Delta t$;
- longitudine MetaDrive solo diagnostica.

Parametri semantici: nessuno.

# 9. Tabella consolidata dei parametri

| Regola | Parametro | Valore | Natura/provenienza |
|---|---|---:|---|
| common geometry | precision grid | $10^{-3}\,m$ | stabilizzazione geometrica congelata |
| common geometry | $\varepsilon_A$ | $10^{-4}\,m^2$ | tolleranza numerica area off-road |
| common geometry | $\varepsilon_{\mathrm{geom}}$ | $10^{-2}\,m$ | buffer numerico boundary |
| control/conflict crossing | $\varepsilon_{\delta}$ | $5\cdot10^{-2}\,m$ | deadband signed distance |
| lane association | $\varepsilon_{\psi}$ | $10^{-6}\,rad$ | equivalenza angolare numerica |
| lane association | $\varepsilon_{\mathrm{lat}}$ | $10^{-3}\,m$ | equivalenza laterale numerica |
| common 2.5D geometry | $z_{\mathrm{tol}}$ | $3.0\,m$ | filtro verticale congelato per lane, attori e conflict zone |
| local prediction | $T_{\mathrm{pred}}$ | $3.0\,s$ | orizzonte CV per TTC/conflict zone |
| occupancy interval | $\varepsilon_t$ | $10^{-6}\,s$ | tolleranza numerica di fusione intervalli |
| R1 | $\varepsilon_{\mathrm{col}}$ | $10^{-6}$ | floor numerico, ruolo analogo a ScenicRules |
| R1 | comparison tolerance | $10^{-8}$ | sola tolerance numerica del monitor |
| R1 vehicle | $u_{\mathrm{cap},i}$ | $v_{\max,e}+v_{\max,i}$ | configured speed normalization cap |
| R1 VRU/static | $u_{\mathrm{cap},i}$ | $v_{\max,e}$ | configured speed normalization cap ego, nessun cap semantico aggiuntivo |
| RSS | $\rho$ | $1.0\,s$ | Intel ad-rss-lib suggested starting value |
| RSS | $a_{\max}^{acc}$ | $3.5\,m/s^2$ | Intel ad-rss-lib |
| RSS | $b_e$ | `min(4.0, floor(10 Q_0.05^lower)/10)` $m/s^2$ | calibrazione una tantum obbligatoria |
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
| progress geometry | route polyline | wrapper-owned, fixed at reset | coordinata canonica condivisa PG/Waymo |

I valori fisici e geometrici della specifica non fanno parte dello sweep degli algoritmi RL. $b_e$ deve essere disponibile in un artifact di calibrazione valido prima di avviare un run finale; la calibrazione è una tantum e non è tuning del reward. Le soglie algoritmiche della Sezione 4.4 sono separate da questa tabella e non appartengono al monitor.

# 10. Matrice delle variabili

| Componente | Stato corrente | Memoria |
|---|---|---|
| collision | contatti e velocità pre/post, configured speed normalization cap | previous contact IDs |
| RSS | canonical lane association, route front/rear $s$, gap, speed | nessuna |
| TTC | polygons, velocities | nessuna |
| clearance | polygons | nessuna |
| off-road | ego polygon/quote, cached `DrivableLaneRecord`, current vertical-layer surface | nessuna |
| wrong-way | ego velocity, route/lane tangent, configured speed normalization cap | nessuna |
| solid | continuous-line contact/geometria | nessuna |
| dashed | dashed-boundary polylines, ego polygon | boundary ID, timer |
| signal | current signal, control line, speed, signed distance | previous color, must-stop, previous delta |
| stop | stop sign, lane, derived control line, speed, delta | continuous/best timer, previous delta |
| crosswalk | crosswalk, ego route movement, cached or pending canonical 2.5D zone, ego/VRU states | illegal-entry flag keyed by `(actor_id, zone_id)` |
| vehicle yield | `MovementKey`, movement corridors, typed pairwise priority records, cached or pending 2.5D zones, actor states | illegal-entry flag keyed by `(actor_id, zone_id)` |
| progress | canonical route polyline, pre/post projection, speed cap, timestep | previous route $s$ |

# 11. Contratto con MetaDrive/ScenarioEnv

## 11.1 Snapshot di transizione richiesti

Il wrapper deve produrre snapshot immutabili pre/post contenenti almeno:

```python
@dataclass(frozen=True)
class ActorSnapshot:
    actor_id: str
    actor_class: ActorClass
    position_xy: tuple[float, float]
    position_z: float
    heading_rad: float
    velocity_xy: tuple[float, float]
    footprint: Polygon
    live_lane_id: str | None
    configured_speed_cap_mps: float | None

@dataclass(frozen=True)
class EnvSnapshot:
    scenario_id: str
    step_index: int
    sim_time_s: float
    ego: ActorSnapshot
    actors: tuple[ActorSnapshot, ...]
    contact_onset_records: tuple[ContactOnsetRecord, ...]
    active_contact_ids: frozenset[str]
    signal_states_by_physical_id: Mapping[str, str]
```

`position_z` deve essere finito. Gli snapshot sono catturati ai control step; future tracks e futuri signal states non sono inclusi.

Prima di `env.step` il wrapper svuota il buffer degli onset del control step; i physics substep lo popolano; `capture_snapshot` del post-state lo legge senza perdere eventi e acquisisce gli ID ancora attivi. Lettura e svuotamento sono atomici.

## 11.2 Dati e primitive da esporre nel wrapper

Il wrapper deve fornire o derivare una volta al reset:

- contact actor ID stabile e normali orientate ego→oggetto;
- footprint polygon validi e quota corrente degli attori;
- catalogo tipizzato di attori e map features;
- lane centerline 3D, lane width/polygon, funzione di quota, orientamento legale e successor graph;
- route ego come sequenza ordinata di lane e `RoutePolyline` 3D canonica;
- `DrivableLaneRecord` 2.5D, non un'unione globale priva di quota;
- logical boundary ID per linee solide e tratteggiate;
- traffic-control catalog con gruppi, `MovementKey`, route $s$, quota e control line;
- crosswalk polygon con quota esplicita o derivata senza ambiguità;
- componenti circolanti delle rotatorie validate;
- `MovementPriorityRecord` pairwise espliciti;
- artifact di calibrazione di $b_e$ con hash della configurazione ego;
- cache episodica immutabile e applicazione atomica dei `CacheDelta`.

L'unica modifica al motore fisico ammessa è l'hook localizzato che esporta i contact record. Tutte le altre quantità sono lette o derivate senza alterare la dinamica.

## 11.3 Normalizzazione fra sorgenti

ScenarioDescription fornisce tracks, map features e dynamic map states. Un adapter per ciascuna sorgente deve convertirli nella tassonomia e nei record canonici.

Le formule non contengono branch `if source == waymo` o `if source == pg`. Una differenza di disponibilità, inclusa la quota, viene risolta nel validatore: lo scenario è eleggibile oppure viene escluso con una causa tipizzata. Una mappa effettivamente monolivello può usare una quota costante validata; una mappa multilivello senza quota affidabile non è eleggibile.

# 12. Output del monitor

```python
@dataclass(frozen=True)
class RuleComponentResult:
    name: str
    cost: float
    raw: dict
    applicable: bool
    evaluable: bool
    status: str
    diagnostics: dict

@dataclass(frozen=True)
class RulebookResult:
    margins: tuple[float, float, float, float]
    costs: tuple[float, float, float]
    raw_progress_m: float
    components: dict[str, RuleComponentResult]
    complete_evaluation: bool
```

L'API restituisce:

```python
tuple[RulebookResult, RulebookMemory, CacheDelta]
```

Prima del ritorno vengono verificate le invarianti:

- tutti i costi sono finiti;
- $c_1,c_2,c_3\in[0,1]$ entro tolerance $10^{-8}$;
- $m_4\in[-1,1]$;
- ogni componente applicabile è evaluabile;
- ogni campo di memoria ha al massimo un writer;
- ogni nuovo `zone_id` identifica una sola geometria canonica;
- `complete_evaluation=True`.

Una violazione del contratto non produce output parziali: solleva `RulebookEvaluationError`.

# 13. Pseudocodice

```python
def evaluate_transition(pre_state, post_state, memory, cache, cfg):
    actors_post = get_live_collidable_actors(post_state)
    active_signal = select_active_signal(pre_state, memory, cache)
    active_stop = select_active_stop(pre_state, memory, cache, active_signal)

    movement_context, zone_cache_delta = resolve_movements_and_pending_zones(
        pre_state, post_state, memory, cache, cfg
    )
    cache_view = overlay_cache(cache, zone_cache_delta)

    events = detect_transition_events(
        pre_state, post_state, memory, cache_view, movement_context
    )

    component_outputs = []

    component_outputs.append(evaluate_collision_onset(
        pre_state, post_state, memory, events, cfg.collision
    ))

    component_outputs.append(evaluate_rss(post_state, cache_view, cfg.interaction.rss))
    component_outputs.append(evaluate_ttc(post_state, actors_post, cache_view, cfg.prediction))
    component_outputs.append(evaluate_clearance(post_state, actors_post, cache_view, cfg.interaction.clearance))

    ego_drivable_surface = build_ego_drivable_surface(post_state.ego, cache_view)
    component_outputs.append(evaluate_offroad(post_state.ego, ego_drivable_surface))
    component_outputs.append(evaluate_wrongway(post_state.ego, cache_view.route_polyline))
    component_outputs.append(evaluate_solid_line(events, post_state, cache_view))
    component_outputs.append(evaluate_dashed_line(pre_state, post_state, memory, events, cache_view, cfg))
    component_outputs.append(evaluate_signal_transition(
        pre_state, post_state, memory, active_signal, events, cache_view, cfg
    ))
    component_outputs.append(evaluate_stop_transition(
        pre_state, post_state, memory, active_stop, events, cache_view, cfg
    ))
    component_outputs.append(evaluate_crosswalk_yield(
        pre_state, post_state, memory, events, cache_view, cfg
    ))
    component_outputs.append(evaluate_vehicle_yield(
        pre_state, post_state, memory, events, movement_context, cache_view, cfg
    ))

    results = [o.result for o in component_outputs]
    memory_deltas = [o.memory_delta for o in component_outputs]
    cache_deltas = [zone_cache_delta] + [o.cache_delta for o in component_outputs]

    progress_result, progress_delta = evaluate_progress(
        pre_state, post_state, memory, cache_view.route_polyline, cfg
    )
    results.append(progress_result)
    memory_deltas.append(progress_delta)

    next_memory = merge_memory_deltas(memory, memory_deltas)
    cache_delta = merge_cache_deltas(cache, cache_deltas)

    result = aggregate_and_validate_rulebook_result(results, cfg)
    validate_transaction(result, next_memory, cache, cache_delta)
    return result, next_memory, cache_delta
```

Training loop:

```python
pre_state = capture_snapshot(env)
obs_next, _, terminated, truncated, info = env.step(action)
post_state = capture_snapshot(env)

try:
    result, next_memory, cache_delta = monitor.evaluate_transition(
        pre_state, post_state, memory, episode_cache
    )
except RulebookEvaluationError as exc:
    log_rulebook_failure(exc)
    abort_run_without_saving_replay_checkpoint()
    raise

next_cache = apply_cache_delta(episode_cache, cache_delta)

# Commit logico unico: nessuna operazione fallibile deve rimanere fra i tre assegnamenti.
episode_cache = next_cache
memory = next_memory
store_transition_with_rulebook_margins(result.margins)
```

`store_transition_with_rulebook_margins` deve essere chiamata solo dopo che `apply_cache_delta` ha completato le verifiche senza errore. Il replay checkpoint viene salvato soltanto da stati già committati.

Validazione catalogo:

```python
validation = validate_rulebook_scenario(scenario, adapter, calibration_artifact)
if validation.rulebook_eligible:
    add_to_rulebook_pool(scenario)
else:
    log_excluded_scenario(scenario.id, validation.errors)
```

# 14. Configurazione

```yaml
rulebook:
  version: 4.6-final-implementation-complete

  order:
    - collision_impact
    - dynamic_interaction_safety
    - road_traffic_compliance
    - route_progress

  execution:
    api: evaluate_transition_returns_result_memory_cache_delta
    evaluator_side_effects: forbidden
    memory_and_cache_commit: atomic_after_success
    transition_store: after_memory_and_cache_commit
    duplicate_memory_writer_policy: fail_fast
    not_evaluable_policy: fail_fast
    silent_fallbacks: false
    future_ground_truth_tracks: false

  geometry:
    precision_grid_m: 1.0e-3
    offroad_area_epsilon_m2: 1.0e-4
    polyline_buffer_epsilon_m: 1.0e-2
    signed_distance_epsilon_m: 5.0e-2
    lane_angle_equivalence_epsilon_rad: 1.0e-6
    lane_lateral_equivalence_epsilon_m: 1.0e-3
    vertical_compatibility_tolerance_m: 3.0
    interval_time_epsilon_s: 1.0e-6
    stable_id_geometry_rounding_m: 1.0e-3
    convex_decomposition: deterministic_hole_bridging_and_ear_clipping

  context:
    actor_and_map_taxonomy: canonical_enums
    route_projection: wrapper_owned_3d_route_polyline_with_xy_arc_length
    projection_tie_break: vertical_filter_then_min_distance_then_previous_s_then_segment_index
    drivable_surface: per_step_union_of_vertically_compatible_drivable_lanes
    lane_polygon_fallback: centerline_buffer_by_half_lane_width
    missing_lane_width_policy: exclude_scenario
    missing_elevation_multilevel_policy: exclude_scenario
    rss_lane_selection: route_lanes_polygon_and_vertical_membership
    rss_lane_tie_break: angle_then_lateral_distance_then_lane_id
    traffic_control_selection: first_unresolved_ahead_along_route
    traffic_control_grouping: same_movement_key_and_control_line
    control_line: explicit_validated_else_canonical_orthogonal_section
    movement_keyentity: stable_movement_key
    movement_corridor: stable_topological_approach_conflict_exit_geometry

  prediction:
    horizon_s: 3.0
    motion_model: constant_velocity_constant_heading_no_rotation
    occupancy_solver: continuous_sat_after_deterministic_convex_decomposition
    occupancy_interval_selection: containing_zero_else_earliest
    open_end_representation: symbolic
    broad_phase_role: optimization_only
    empty_candidate_max_speed_mps: 0.0

  collision:
    onset_only: true
    velocity_source: pre_state
    static_velocity_mps: 0.0
    dynamic_spawn_in_contact_policy: fail_fast
    severity_raw: squared_max_contact_point_normal_closing_speed
    normal_orientation: ego_to_other
    contact_point_aggregation: max_per_stable_actor_id
    normalization:
      vehicle: ego_plus_other_configured_speed_cap
      vru_or_static: ego_configured_speed_cap
    epsilon: 1.0e-6
    require_custom_contact_records: true

  interaction:
    rss:
      response_time_s: 1.0
      max_accel_during_response_mps2: 3.5
      ego_min_brake_mps2: required_from_calibration_artifact
      calibration_requires_ego_reaches_mps: 20.0
      calibration_max_reference_mps2: 4.0
      calibration_quantile: lower_0.05
      calibration_rounding_mps2: floor_to_0.1
      front_max_brake_mps2: 8.0
      gap: route_projected_bumper_to_bumper
    ttc:
      vehicle_s: 0.8
      static_s: 0.8
      vru_s: 1.0
      solver: continuous_sat
    clearance:
      vehicle_m: 0.8
      vru_m: 1.0
      static_m: 0.5

  compliance:
    offroad:
      definition: ego_area_fraction_outside_current_vertical_layer_drivable_surface
    wrongway:
      tangent_source: canonical_route_projection
      normalization: ego_configured_speed_cap
    solid_line:
      definition: current_overlap_or_swept_front_bumper_crossing
      boundary_id: stable_logical_boundary_id
    dashed_line:
      boundary_id: stable_logical_boundary_id
      free_time_s: 1.0
      saturation_time_s: 2.0
      exponent: 2
    signal:
      group_selection: canonical_catalog
      crossing_governed_by: pre_action_group_state
      approach_governed_by: post_action_group_state
      yellow_must_stop_frozen_at_onset: true
      braking_mps2: use_calibrated_rss_ego_brake
      crossing_deadband_m: 0.05
      unknown_or_discordant_state_policy: fail_validation_or_runtime
    stop:
      group_selection: canonical_catalog
      suppress_when_signal_controls_same_movement_key: true
      zone_m: 1.0
      stopped_speed_mps: 0.1
      minimum_continuous_dwell_s: 1.0
      timer_value_used_for_current_transition: post_update
    conflict_zones:
      component_selection: occupied_else_first_ahead
      vertical_filter: required
      merge_clip_length: 2_times_max_supported_vehicle_length
      cache: immutable_plus_cache_delta
      movement_change: new_zone_only_for_new_movement_key
      lazy_zone_inside_ego: mark_preexisting_before_entry_evaluation
      behavior_flag_key: actor_id_and_zone_id
    crosswalk:
      candidate_classes: [pedestrian, cyclist]
      require_ego_interval: true
      require_actor_interval: true
      prediction_solver: canonical_occupancy_interval
      gap_s: 1.0
      approach_gate: before_zone_and_not_preexisting
      illegal_entry_cost: 1.0
    vehicle_yield:
      approach_control_enum: [none, stop, signal, unknown]
      explicit_priority_is_pairwise_only: true
      priority_record_type: MovementPriorityRecord
      priority_predicates:
        - zone_already_occupied
        - ego_stop_other_none
        - ego_roundabout_entry_other_circulating
        - explicit_other_has_priority_record
      infer_priority_from_geometry: false
      yield_sign_core_support: false
      ambiguous_case: not_applicable
      require_ego_interval: true
      require_actor_interval: true
      gap_s: 1.0
      illegal_entry_cost: 1.0

  progress:
    raw_definition: canonical_route_s_post_minus_pre
    native_metadrive_longitude_role: diagnostics_only
    normalization: ego_configured_speed_cap_times_control_timestep
    clip: [-1.0, 1.0]
```

La configurazione dell'algoritmo lessicografico resta esterna al monitor.

# 15. Test obbligatori

## 15.0 Primitive canoniche, 2.5D e atomicità

- stessa route e stesso punto producono lo stesso $s$ su PG e Waymo adapter;
- route autointersecante: tie-break mediante continuità con `previous_s`;
- cavalcavia e strada sottostante non producono lane association, off-road surface condivisa o conflict zone;
- attore e lane entro $z_{\mathrm{tol}}$ sono compatibili; oltre soglia sono esclusi;
- crosswalk senza quota derivabile in mappa multilivello esclude lo scenario;
- lane association univoca e verticalmente compatibile;
- lane association ambigua entro $\varepsilon_\psi$ e $\varepsilon_{\mathrm{lat}}$ produce RSS `NOT_APPLICABLE`;
- gap bumper-to-bumper con veicoli separati, tangenti e sovrapposti;
- lane senza polygon ma con width produce polygon canonico;
- lane senza polygon o width esclude lo scenario;
- superficie carrabile cambia correttamente fra due livelli sovrapposti;
- più semafori dello stesso gruppo con stessa `MovementKey`, control line compatibile e stato concorde;
- stati discordanti nello stesso signal group escludono lo scenario o falliscono a runtime;
- due controlli davanti: selezione del minore `route_s`;
- controllo euclideamente vicino ma su altro livello o fuori route non viene selezionato;
- derivazione control line ortogonale e verticalmente valida;
- continuous SAT: nessun intervallo, intervallo finito, attore già dentro e `OPEN_END`;
- decomposizione con hole e polygon concavo produce triangoli interni, non degeneri e ordinati deterministicamente;
- stessa `MovementKey` con velocità differenti mantiene lo stesso `zone_id`;
- nuova `MovementKey` crea un nuovo `zone_id` senza modificare la geometria precedente;
- zona lazy creata con ego già dentro viene marcata preesistente e non genera ingresso illegale;
- evaluator non modifica direttamente memoria o cache;
- due `MemoryDelta` che scrivono lo stesso campo causano fail-fast;
- due `CacheDelta` con geometrie diverse per lo stesso `zone_id` causano fail-fast;
- eccezione dopo la proposta di una zona non modifica né cache né memoria e non inserisce la transizione;
- valutazione riuscita committa cache, memoria e transizione una sola volta.

## 15.1 Collisione

- nuovo contatto frontale;
- normale restituita nel verso opposto e correttamente invertita;
- contatto tangenziale;
- contatto persistente senza nuovo onset;
- contatto che inizia e termina fra due control frame viene rilevato;
- separazione per almeno un physics substep e ricontatto produce un nuovo onset;
- più contact point dello stesso oggetto e selezione del massimo closing speed normale;
- collisioni simultanee con oggetti differenti;
- oggetto statico;
- attore dinamico spawnato già in contatto produce `RulebookEvaluationError`;
- ID sintetico stabile uguale fra reset ripetuti dello stesso scenario;
- VRU;
- velocità pre-state del control step;
- veicolo senza configured speed normalization cap valido;
- actor speed sopra il configured speed normalization cap: costo saturato e scenario non invalidato;
- verifica che il custom contact hook non modifichi la dinamica.

## 15.2 RSS

- selezione front vehicle mediante lane association canonica;
- veicolo in lane adiacente escluso;
- veicolo dietro escluso;
- heading opposto escluso;
- lane association entro $\varepsilon_{\psi}$ e $\varepsilon_{\mathrm{lat}}$ produce `NOT_APPLICABLE`;
- gap bumper-to-bumper positivo, esattamente zero e footprint longitudinalmente sovrapposti;
- gap esattamente pari a $d_{\mathrm{safe}}$ produce costo zero;
- velocità longitudinale negativa esclusa da RSS;
- artifact di calibrazione mancante, non finito o con hash diverso invalida la configurazione;
- protocollo con meno di dieci prove valide per velocità fallisce;
- quantile `lower`, arrotondamento per difetto e cap a $4.0\,m/s^2$ producono il valore atteso.

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
- inizializzazione del nuovo automa quando cambia `active_signal_group_id`;
- nessuna rivalutazione per un ID presente in `resolved_signal_group_ids`.

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
- reset dei timer quando cambia `active_stop_group_id`;
- nessuna rivalutazione per un ID presente in `resolved_stop_group_ids`.

## 15.8 Crosswalk

- ego con `NO_INTERVAL` produce contributo zero senza calcolare gap;
- insieme candidati contiene soltanto pedestrian/cyclist live, verticalmente compatibili e con intervallo non vuoto;
- VRU su cavalcavia/sottopasso sovrapposto in XY viene escluso;
- distanza $d_e^Z=s_Z^{entry}-s_e^{front}$ verificata prima, dentro e dopo la zona;

- crosswalk dietro l'ego escluso;
- crosswalk non intersecante la route escluso;
- più componenti connesse dello stesso crosswalk: selezione della prima ancora davanti o attualmente occupata;
- stessa `MovementKey` produce la stessa geometria e lo stesso `zone_id`; una nuova `MovementKey` crea una nuova chiave senza modificare geometrie precedenti;
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

- la priorità esplicita pairwise non esiste come valore di `ApproachControl`;
- ego STOP e altro NONE attiva il predicato; altro SIGNAL o UNKNOWN non lo attiva;
- record pairwise `EGO_HAS_PRIORITY` non viene interpretato come priorità dell'altro;
- yield sign non supportato non viene convertito in NONE;
- ego con `NO_INTERVAL` produce contributo zero;
- attore su livello verticale differente viene escluso;
- `MovementKey` resta congelata dall'ingresso all'uscita;

- ingresso in rotatoria con attore sulla componente circolante validata;
- merge con relazione di priorità esplicita;
- merge senza relazione di priorità esplicita produce `NOT_APPLICABLE`;
- ego soggetto a stop e attore con `ApproachControl.NONE`;
- entrambi gli approcci soggetti a stop producono `NOT_APPLICABLE`;
- conflict zone già occupata dall'attore attiva il predicato di priorità;
- priorità ambigua in intersezione uncontrolled produce `NOT_APPLICABLE`;
- attore con più successori prima della zona produce `NOT_APPLICABLE`;
- sequenza topologica completa da approach lane a exit lane; lane graph ambiguo produce `NOT_APPLICABLE`;
- più componenti di intersezione: zona occupata, altrimenti prima ancora davanti;
- merge su lane comune: zona limitata a $2L_{\max}^{\mathrm{veh}}$ dal punto di confluenza;
- stessa `MovementKey` produce la stessa geometria e lo stesso `zone_id`; una nuova `MovementKey` crea una nuova chiave senza modificare geometrie precedenti;
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
- clipping a $\pm1$ senza modifica del raw;
- passaggio fra segmenti consecutivi;
- route autointersecante con tie-break di continuità;
- proiezione iniziale con tie-break sul minore $s$;
- proiezione non finita solleva `RulebookEvaluationError`;
- stessa traiettoria produce delta uguale a parità di geometria canonica su PG e Waymo;
- `current_longitude` MetaDrive discordante resta soltanto diagnostico e non cambia il reward.

## 15.11 Validazione scenario e fail-fast

- controlli già superati al reset inizializzano direttamente gli insiemi `resolved`;
- arretrare dopo il reset non riattiva un controllo prepassed;
- mappa multilivello priva di quote affidabili viene esclusa;
- configurazione ego incapace di raggiungere $20\,m/s$ fallisce il protocollo di calibrazione prima delle prove;

- contatto già attivo al reset non produce una nuova collisione al primo step;
- stato e signed distance di semaforo e stop vengono inizializzati dallo stato live al reset;
- ego già dentro una conflict zone al reset non genera ingresso illegittimo;
- dopo la prima uscita da una conflict zone preesistente, un nuovo ingresso viene valutato normalmente;
- componente non applicabile produce zero senza invalidare lo scenario;
- attore non live viene escluso senza generare `NOT_EVALUABLE`;
- route, lane graph, lane width/polygon, geometrie e control associations mancanti escludono lo scenario offline;
- semaforo pertinente con sequenza incompleta o stato `UNKNOWN` esclude lo scenario offline;
- semaforo non pertinente invalido non esclude automaticamente lo scenario;
- scenario senza semafori resta eleggibile e la componente è `NOT_APPLICABLE`;
- dato core mancante a runtime solleva `RulebookEvaluationError`;
- il run fallito non committa memoria né cache, non inserisce la transizione e non salva il replay buffer come checkpoint valido;
- non esistono staging buffer episodici né rollback: il commit transazionale di memoria e cache impedisce l'inserimento della transizione invalida;
- scenario ID, step, componente e causa sono sempre loggati.

# 16. Limiti dichiarati

1. La collision severity è una proxy cinematica al precedente control step, non un injury model e non usa l'esatto substep fisico dell'impatto.
2. RSS è solo longitudinale e richiede che il modello ego garantisca la decelerazione assunta.
3. TTC assume moto a velocità e heading costanti entro $T_{\mathrm{pred}}=3\,s$ e tratta gli ostacoli statici con velocità nulla.
4. Clearance non dipende dalla velocità.
5. Le conflict zone per crosswalk e yield sono costruite da corridoi 2.5D e identificate da `MovementKey` topologiche stabili. La cache committata è append-only e viene aggiornata soltanto tramite `CacheDelta` atomico. La manovra deve essere topologicamente univoca oppure la componente è `NOT_APPLICABLE`. Le occupazioni sono previste per soli $3\,s$ senza intent prediction.
6. Il crosswalk rule non modella l'intenzione di un VRU fermo lontano dalla zona di conflitto.
7. Il vehicle-yield rule usa soltanto i quattro predicati deterministici della Sezione 7.9.1; yield sign non modellati, merge senza priorità esplicita, doppio stop e intersezioni non controllate ambigue restano fuori dominio.
8. Off-road richiede geometria carrabile e quota interrogabili; la tolleranza verticale 2.5D evita di unire livelli stradali sovrapposti ma non costituisce un modello volumetrico completo.
9. Lane-line types, traffic-control associations, stop positions e crosswalk polygons devono essere verificati sulle sorgenti concrete Waymo e PG.
10. Le control line derivate sono sezioni ortogonali canoniche e verificate, non necessariamente marking dipinte ground-truth.
11. La policy deve osservare timer/stati procedurali oppure una history sufficiente a ricostruirli.
12. Il rulebook è reward/specification monitoring, non uno shield e non garantisce sicurezza durante il learning.
13. La legalità dipende dalla convenzione del task e non costituisce una completa codifica del codice stradale di una specifica nazione.
14. Il massimo interno è una compressione worst-case lossy: configurazioni diagnostiche differenti possono avere lo stesso costo macro.
15. Il learner ottimizza expected discounted returns; non garantisce assenza di violazioni in ogni rollout e può pesare meno eventi più lontani nel tempo.
16. Il filtro sui semafori può modificare la distribuzione degli scenari; numero, sorgente e categoria degli scenari esclusi devono essere riportati.

# 17. Stato finale

La versione `4.6-final-implementation-complete` congela la semantica delle regole e tutte le primitive necessarie a calcolarle. Non restano scelte affidate all'implementatore riguardo a:

- tassonomia, ID e quote canoniche;
- filtro verticale 2.5D;
- route projection;
- lane association e gap RSS;
- superficie carrabile per livello verticale;
- selezione di signal e stop e gestione dei controlli prepassed;
- derivazione delle control line;
- `MovementKey` e corridoio topologico stabili;
- costruzione, selezione e lifecycle delle conflict zone;
- decomposizione convessa e intervalli di occupazione;
- insiemi dei candidati crosswalk e vehicle-yield;
- predicati di priorità pairwise;
- evaluator puri, `MemoryDelta`, `CacheDelta` e commit transazionale;
- progresso lungo route.

Il perimetro resta deliberatamente limitato: il documento non pretende di codificare l'intero codice stradale, ma ogni caso incluso possiede dominio, algoritmo, tie-break, stato e comportamento d'errore determinati.

Le attività residue sono verifiche di conformità del codice, non decisioni sulla specifica:

1. implementare gli adapter PG e Waymo verso i record canonici 2.5D;
2. implementare e testare il contact hook;
3. eseguire il protocollo obbligatorio di calibrazione e generare l'artifact di $b_e$;
4. eseguire tutti i test della Sezione 15;
5. escludere e contabilizzare gli scenari che non superano la validazione.

Il training definitivo può iniziare soltanto quando la suite di conformità passa e l'artifact di calibrazione corrisponde alla configurazione ego. Un fallimento richiede la correzione dell'adapter, del wrapper o dello scenario; non autorizza fallback non presenti nella specifica.

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
