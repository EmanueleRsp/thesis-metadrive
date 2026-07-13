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

Initial Google/Waymo authorization and license acceptance require user
confirmation; file downloads are automated by the pipeline after login.

## Autenticazione Google Cloud (una sola volta)

`gcloud` è il programma da riga di comando per autenticarsi a Google Cloud e
leggere il bucket che contiene i TFRecord. Non è una dipendenza Python del
project. In the normal workflow, no credential needs to be placed in `.env`:
`gcloud` stores the OAuth login locally.

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
The JSON file must not be committed or pasted into `.env`: only its path goes
in `.env`, for example `/home/me/secrets/waymo-sa.json`, with `chmod 600`
permissions. Leave this variable empty when using the normal OAuth login.

There is intentionally no `GOOGLE_API_KEY` field: an API key identifies a
project and quota for some APIs, but does not grant the IAM authorization
required to read this private Cloud Storage bucket. This workflow requires
OAuth, ADC/service-account credentials, or a federated identity.

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
[`conf/scenarios/pipeline_v1.yaml`](../../conf/scenarios/pipeline_v1.yaml):
this YAML is the single source of truth for counts, seeds, split targets, and
checks. Edit that file directly to change them; do not duplicate them in `.env`.

- `WAYMO_GCS_URI` e `WAYMO_GCS_OBJECT_PATTERN`: sorgente Google Cloud;
- `WAYMO_RAW_DATA_PATH`: directory raw sul host;
- `WAYMO_NUM_FILES=1`: initial smoke test; with a wildcard pattern this limits
  both download and conversion; leave empty for all shards;
- `WAYMO_NUM_WORKERS` e `WAYMO_OVERWRITE`: opzioni del converter.

The resolver reads the YAML at startup and passes those values to the pipeline
inside the container, making executions reproducible and independent of hidden
scientific environment variables.

Con l'autenticazione già configurata, il comando unico è:

```bash
make waymo-pipeline
```

La pipeline controlla l'account attivo, scarica i file mancanti, verifica la
presenza di `training_20s.tfrecord*`, costruisce il container dedicato e avvia
la conversione. Il default scarica un solo shard e lo converte per lo smoke
test, evitando un download completo accidentale.

To convert all shards after verifying the smoke test, set the following in
`.env`:

```dotenv
WAYMO_NUM_FILES=
WAYMO_GCS_OBJECT_PATTERN=training_20s.tfrecord-*
WAYMO_SKIP_DOWNLOAD_IF_PRESENT=false
```

e rilancia `make waymo-pipeline`. I file già presenti non vengono riscaricati.

Downloading all shards is not required. The one-command dataset pipeline now
counts only Waymo scenarios whose signal reliability is `complete` or
`not_applicable`, then downloads and converts unseen shards in bounded batches
until the configured target of 1,750 eligible scenarios is reached:

```bash
make scenarionet-pipeline
```

The batch size, worker count and safety cap are versioned under `waymo` in
`conf/scenarios/pipeline_v1.yaml`. Each batch is converted into a separate
database below `waymo/database/batches`, so the existing pool is never
overwritten. Successful batch TFRecords are removed by default; rejected
`partial`/`missing` scenarios stay in the converted audit pool but cannot enter
the final split or runtime database. Acquisition state is written under
`data/scenarionet/waymo/acquisition`. The status is cached against a
fingerprint of converted paths, sizes and modification times; changing the
database, eligibility policy or policy version triggers a full rescan.

`make waymo-pipeline` remains the manual fixed-shard conversion command and
continues to honor `WAYMO_NUM_FILES`; it is useful for smoke tests, not for
filling the final eligible target automatically.

To inspect the number of available shards and their size without downloading
them, run:

```bash
make waymo-inventory
```

`.env.example` already uses the full wildcard; `WAYMO_NUM_FILES=1` still limits
the smoke test to the first shard.

The command queries Cloud Storage, reports the number of objects matching the
pattern, and prints the total size in bytes. It only requires an authenticated
`gcloud` session.

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

The raw files and converted database only need to coexist during conversion:
the converter reads the TFRecords and writes the ScenarioNet database used by
the rest of the project. After a successful conversion, avoid keeping both by
setting:

```dotenv
WAYMO_CLEANUP_RAW_AFTER_CONVERSION=true
```

The pipeline verifies that the database contains files before deleting
anything, and removes only `training_20s.tfrecord*` from the raw directory. If
you need to reconvert later, the TFRecords must be downloaded again; therefore
the default remains `false`.

## Verifiche successive

La conversione reale non è inclusa nei test ordinari perché richiede dati
licenziati. Una volta disponibile il database, eseguire nell'ambiente del
progetto i comandi di validazione e costruzione delle viste runtime descritti
nel tracker [`scenarionet_integration_implementation_plan.md`](../specs/scenarionet_integration_implementation_plan.md).

## Pipeline completa con un solo comando

Durante l'esecuzione il comando mostra gli stadi numerati, i parametri risolti
e il tempo totale. Le CLI usano Rich per visualizzare una progress bar durante
la generazione PG e la validazione degli scenari, oltre a spinner e riepiloghi
per catalogo, split, soglie, runtime e check ufficiali.

L'output umano viene scritto su `stderr`, mentre i report JSON su `stdout`
restano disponibili per script e log machine-readable. In terminali non
interattivi o CI Rich riduce automaticamente l'output a righe leggibili.

Dopo aver autenticato `gcloud`, il comando consigliato è:

```bash
make scenarionet-pipeline
```

Se Waymo e PG sono già preparati e occorre soltanto ricostruire catalogo,
split, soglie e runtime (per esempio dopo una modifica alla classificazione),
usare:

```bash
make scenarionet-recatalog
```

Questo target non scarica, non riconverte e non rigenera scenari; sovrascrive
soltanto gli artifact derivati e riesegue i controlli finali.

La pipeline usa il servizio Compose `dataset-pipeline`, costruito dal target
CPU-only condiviso con l'immagine principale: non installa PyTorch né le
librerie CUDA. Il solo stadio che usa una dipendenza pesante separata è la
conversione Waymo, eseguita da `waymo-converter` con TensorFlow 2.11. Il
container `dev` resta riservato a training ed evaluation RL.

La pipeline resta in un unico terminale: la conversione Waymo cattura l'output
dei worker e lo riduce a un'unica dashboard Rich aggiornata in-place. Anche la
generazione PG usa processi MetaDrive isolati (`pg.workers`, 4 nel profilo
baseline) e una sola barra aggregata; i riepiloghi stampati da ogni scenario
non vengono propagati al terminale. Non viene creata alcuna sessione o pane
`tmux`.

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

To enforce exact counts, set `split.auto` to `false` in the YAML and keep the
six values in `split.targets`; in this mode the targets are interpreted as exact
counts and the command fails when Waymo groups make them impossible:

```yaml
split:
  auto: false
  targets:
    waymo: {train: 1000, validation: 250, test: 500}
    pg: {train: 1000, validation: 250, test: 500}
```

With `split.auto: true`, Waymo groups may produce counts slightly different
from the targets; the reduction is explicit in `split_manifest.json`. With
`false`, the pipeline fails when groups make the counts incompatible. Catalog,
manifest, threshold, and runtime paths can be customized through their
dedicated path variables; defaults are under
`${SCENARIONET_DATA_ROOT}`.

In caso di errore TensorFlow/protobuf, conservare l'output del container nel
tracker: non installare automaticamente queste dipendenze nell'immagine RL.
