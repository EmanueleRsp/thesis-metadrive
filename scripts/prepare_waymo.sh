#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -f .env ]]; then
  # .env contains project configuration, never credentials. It is ignored by git.
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

raw_dir="${WAYMO_RAW_DATA_PATH:-./data/waymo_raw}"
gcs_uri="${WAYMO_GCS_URI:-gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s}"
object_pattern="${WAYMO_GCS_OBJECT_PATTERN:-training_20s.tfrecord-00000-of-01000}"
skip_download="${WAYMO_SKIP_DOWNLOAD_IF_PRESENT:-true}"
num_files="${WAYMO_NUM_FILES-1}"
num_workers="${WAYMO_NUM_WORKERS:-8}"
overwrite="${WAYMO_OVERWRITE:-false}"

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

die() {
  echo "waymo-pipeline: $*" >&2
  exit 2
}

command -v gcloud >/dev/null 2>&1 || die \
  "gcloud CLI non trovato. Esegui 'make waymo-auth' dopo aver installato il Google Cloud CLI."

credentials_file="${GOOGLE_APPLICATION_CREDENTIALS:-}"
if [[ -n "$credentials_file" ]]; then
  [[ -f "$credentials_file" ]] || die \
    "GOOGLE_APPLICATION_CREDENTIALS non punta a un file esistente: $credentials_file"
  echo "Attivazione service account dal file indicato in GOOGLE_APPLICATION_CREDENTIALS"
  gcloud auth activate-service-account --key-file="$credentials_file" --quiet
fi

active_account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -n 1 || true)"
[[ -n "$active_account" ]] || die \
  "nessun account Google Cloud attivo. Esegui una volta 'make waymo-auth'."

mkdir -p "$raw_dir"

has_raw=false
if find "$raw_dir" -maxdepth 1 -type f -name 'training_20s.tfrecord*' -print -quit | grep -q .; then
  has_raw=true
fi

if [[ "$has_raw" == true ]] && is_true "$skip_download"; then
  echo "Waymo raw già presente in $raw_dir"
else
  source_uri="${gcs_uri%/}/${object_pattern}"
  echo "Download Waymo da $source_uri"
  gcloud storage cp "$source_uri" "$raw_dir/"
fi

find "$raw_dir" -maxdepth 1 -type f -name 'training_20s.tfrecord*' -print -quit | grep -q . || die \
  "download completato ma nessun training_20s.tfrecord* trovato in $raw_dir"

echo "Account Google Cloud: $active_account"
echo "Raw Waymo: $raw_dir"
echo "Build del container di conversione..."
make build-waymo

convert_args=("WAYMO_RAW_DATA_PATH=$raw_dir" "NUM_WORKERS=$num_workers")
if [[ -n "$num_files" ]]; then
  convert_args+=("NUM_FILES=$num_files")
fi
if is_true "$overwrite"; then
  convert_args+=("OVERWRITE=1")
fi

make waymo-convert "${convert_args[@]}"

echo "Waymo pipeline completata. Database: ${SCENARIONET_DATA_ROOT:-/workspace/data/scenarionet}/waymo/database"
