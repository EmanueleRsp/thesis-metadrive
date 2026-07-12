# Conversione Waymo per ScenarioNet

La conversione Waymo `training_20s` usa un ambiente separato dal runtime RL.
L'immagine principale del progetto non installa TensorFlow né le dipendenze
specifiche del dataset, così training ed evaluation restano riproducibili e
leggeri.

## Prerequisiti

- Docker Compose disponibile;
- dataset Waymo autorizzato dall'utente;
- directory raw contenente file con nome `training_20s.tfrecord*`;
- spazio disco sufficiente per i file convertiti.

L'accettazione della licenza e l'autenticazione Google non sono automatizzate;
il download dei file viene invece automatizzato dalla pipeline dopo il login.

## Autenticazione Google Cloud (una sola volta)

`gcloud` è il programma da riga di comando per autenticarsi a Google Cloud e
leggere il bucket che contiene i TFRecord. Non è una dipendenza Python del
progetto e non richiede di inserire password, token o file JSON in `.env`.

Installa il [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), poi
esegui una sola volta:

Su Ubuntu/WSL puoi installarlo con questi comandi, **uno alla volta**. Non
separare il carattere `|` dalla riga e non eseguire `\` da solo:

```bash
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
sudo apt-get update
sudo apt-get install -y google-cloud-cli
gcloud --version
```

Su Ubuntu/WSL il repository include anche un installer assistito:

```bash
make install-gcloud
```

L'installer installa solo il client locale. Non esegue il login al posto
dell'utente: l'autorizzazione Waymo richiede una conferma OAuth interattiva.

Se disponi già di un service account autorizzato al dataset, puoi usare
un'automazione non interattiva indicando in `.env` soltanto il percorso di un
file JSON conservato fuori dal repository:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS=/percorso/privato/waymo-service-account.json
```

La pipeline attiva quel service account con `gcloud` prima del download. Il
file JSON non deve essere committato né copiato dentro `.env`.

## Numero di scenari e spazio disco

Il pool convertito contiene tutti gli scenari presenti nei TFRecord. La
configurazione finale seleziona verso i target della specifica:

```text
Waymo: 1000 train + 250 validation + 500 test
PG:    1000 train + 250 validation + 500 test
```

La selezione Waymo avviene per gruppi interi; i conteggi effettivi possono
quindi essere leggermente inferiori e vengono registrati nel manifest. Gli
scenari non selezionati restano nel database convertito, ma non entrano nelle
runtime view.

Nel nostro smoke un TFRecord è risultato di circa 82 MB e la conversione del
relativo shard di circa 77 MB. Una stima realistica per 1000 shard è quindi
circa 82 GB di raw più 78 GB di database convertito. Aggiungendo PG, runtime,
immagini Docker e temporanei, è prudente avere almeno 200 GB liberi; 250 GB
offrono un margine più sicuro.

Durante il secondo `apt-get update` devi vedere una riga riferita a
`packages.cloud.google.com`. Se `google-cloud-cli` non viene trovato, il
repository non è stato registrato correttamente.

Quando compare il prompt `>` significa che la shell sta aspettando il seguito
di un comando multilinea: premi `Ctrl+C` e riparti dalla riga completa.

```bash
make waymo-auth
```

Il comando avvia `gcloud init`, che apre il flusso di login, configura l'account
e il progetto Google Cloud. L'account deve avere accesso al Waymo Open Dataset
e devi aver accettato le relative condizioni d'uso. Per la conversione locale
non serve creare API key: il download usa le credenziali OAuth del CLI.

Su WSL, se il browser non viene aperto automaticamente, usa:

```bash
make waymo-auth GCLOUD_INIT_FLAGS=--console-only
```

Il comando mostrerà un URL da copiare nel browser Windows.

## Pipeline automatica

Le impostazioni personali e di macchina sono in `.env` e partono da
`.env.example`. I default scientifici della pipeline sono invece versionati in
[`conf/scenarios/pipeline_v1.yaml`](../../conf/scenarios/pipeline_v1.yaml) e
possono essere sovrascritti dal `.env` senza modificarli:

- `WAYMO_GCS_URI` e `WAYMO_GCS_OBJECT_PATTERN`: sorgente Google Cloud;
- `WAYMO_RAW_DATA_PATH`: directory raw sul host;
- `WAYMO_NUM_FILES=1`: smoke test iniziale; vuoto per tutti i file già scaricati;
- `WAYMO_NUM_WORKERS` e `WAYMO_OVERWRITE`: opzioni del converter.

In particolare, target PG, seed, politica di split e numero di worker dei check
provengono dal file YAML quando le relative variabili `.env` sono vuote. Il
`.env` resta quindi dedicato soprattutto a path, macchina, download e override
locali.

Con l'autenticazione già configurata, il comando unico è:

```bash
make waymo-pipeline
```

La pipeline controlla l'account attivo, scarica i file mancanti, verifica la
presenza di `training_20s.tfrecord*`, costruisce il container dedicato e avvia
la conversione. Il default scarica un solo shard e lo converte per lo smoke
test, evitando un download completo accidentale.

Per il dataset completo, dopo aver verificato lo smoke test, imposta in `.env`:

```dotenv
WAYMO_NUM_FILES=
WAYMO_GCS_OBJECT_PATTERN=training_20s.tfrecord-*
WAYMO_SKIP_DOWNLOAD_IF_PRESENT=false
```

e rilancia `make waymo-pipeline`. I file già presenti non vengono riscaricati.

## Build manuale dell'immagine dedicata

```bash
make build-waymo
```

L'immagine installa TensorFlow 2.11 e il converter dei commit locali di
ScenarioNet/MetaDrive. Il servizio è esposto dal profilo Compose `waymo` e non
viene avviato da `make up` o dal runtime RL standard.

## Conversione

In alternativa, senza scaricare, si può convertire una directory raw già
disponibile:

```bash
make waymo-convert \
  WAYMO_RAW_DATA_PATH=/percorso/assoluto/waymo_raw \
  NUM_FILES=1 \
  OVERWRITE=1
```

`WAYMO_RAW_DATA_PATH` è un path del **host**: Compose lo monta in sola lettura
come `/workspace/waymo_raw` nel container, quindi non è necessario che il raw
dataset si trovi sotto `./data`.

Senza `NUM_FILES` viene processata l'intera directory. L'output predefinito è
`${SCENARIONET_DATA_ROOT}/waymo/database` (normalmente
`./data/scenarionet/waymo/database` sul host). È possibile cambiare la data
root tramite `SCENARIONET_DATA_ROOT` e il mount host tramite `HOST_DATA_DIR`.

Il wrapper passa sempre al converter:

- `dataset_name=waymo`;
- `version=training_20s`;
- il path raw e il database assoluti;
- il numero di worker configurato (8 di default).

Prima dell'avvio vengono rifiutate directory inesistenti o varianti Waymo
diverse da `training_20s`. Dopo la conversione, la pipeline F5 carica i pickle,
estrae il catalogo e assegna gli split interni senza rompere i gruppi logici.

## Verifiche successive

La conversione reale non è inclusa nei test ordinari perché richiede dati
licenziati. Una volta disponibile il database, eseguire nell'ambiente del
progetto i comandi di validazione e costruzione delle viste runtime descritti
nel tracker [`scenarionet_integration_implementation_plan.md`](../specs/scenarionet_integration_implementation_plan.md).

## Pipeline completa con un solo comando

Dopo aver autenticato `gcloud`, il comando consigliato è:

```bash
make scenarionet-pipeline
```

Il comando esegue, nell'ordine:

1. download e conversione Waymo tramite `make waymo-pipeline`;
2. generazione PG offline;
3. costruzione del catalogo unificato;
4. split train/validation/test senza leakage;
5. calcolo delle soglie train-only e assegnazione degli arms;
6. costruzione delle viste `runtime/train`, `runtime/validation` e
   `runtime/test`;
7. validazione applicativa e check ufficiali existence, simulation e overlap.

Per impostazione predefinita il pipeline usa i target baseline della specifica
(1000/250/500 per sorgente) e assegna gruppi interi, registrando i conteggi
effettivi nel manifest. In questo modo gruppi Waymo da 61 scenari, ad esempio,
non richiedono conteggi manuali impossibili.

Se vuoi imporre conteggi esatti, disabilita l'assegnazione automatica e compila
in `.env` i sei conteggi:

```dotenv
SCENARIONET_AUTO_SPLIT=false
SCENARIONET_WAYMO_TRAIN_COUNT=...
SCENARIONET_WAYMO_VALIDATION_COUNT=...
SCENARIONET_WAYMO_TEST_COUNT=...
SCENARIONET_PG_TRAIN_COUNT=...
SCENARIONET_PG_VALIDATION_COUNT=...
SCENARIONET_PG_TEST_COUNT=...
```

Con `SCENARIONET_AUTO_SPLIT=true` i gruppi Waymo possono produrre conteggi
leggermente diversi dai target; la riduzione è esplicita nel
`split_manifest.json`. Con `false`, il pipeline fallisce se i gruppi rendono i
conteggi incompatibili. I path catalogo, manifest, soglie e runtime possono
essere personalizzati in `.env`; i default sono sotto
`${SCENARIONET_DATA_ROOT}`.

In caso di errore TensorFlow/protobuf, conservare l'output del container nel
tracker: non installare automaticamente queste dipendenze nell'immagine RL.
