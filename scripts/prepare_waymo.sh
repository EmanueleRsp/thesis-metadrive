#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -f .env ]]; then
  # .env contains project configuration and, optionally, a path to credentials;
  # it never contains credential contents. It is ignored by git.
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

raw_dir="${WAYMO_RAW_DATA_PATH:-./data/waymo_raw}"
gcs_uri="${WAYMO_GCS_URI:-gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s}"
object_pattern="${WAYMO_GCS_OBJECT_PATTERN:-training_20s.tfrecord-*}"
skip_download="${WAYMO_SKIP_DOWNLOAD_IF_PRESENT:-true}"
num_files="${WAYMO_NUM_FILES-1}"
num_workers="${WAYMO_NUM_WORKERS:-8}"
overwrite="${WAYMO_OVERWRITE:-false}"
cleanup_raw="${WAYMO_CLEANUP_RAW_AFTER_CONVERSION:-false}"
host_data_dir="${HOST_DATA_DIR:-./data}"
database_dir="${host_data_dir%/}/scenarionet/waymo/database"
temporary_files=()

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

cleanup_temporary_files() {
  local status=$?
  if ((${#temporary_files[@]} > 0)); then
    rm -f "${temporary_files[@]}"
  fi
  return "$status"
}
trap cleanup_temporary_files EXIT

command -v gcloud >/dev/null 2>&1 || die \
  "gcloud CLI not found. Run 'make waymo-auth' after installing the Google Cloud CLI."

credentials_file="${GOOGLE_APPLICATION_CREDENTIALS:-}"
if [[ -n "$credentials_file" ]]; then
  [[ -f "$credentials_file" ]] || die \
    "GOOGLE_APPLICATION_CREDENTIALS does not point to an existing file: $credentials_file"
  echo "Activating the service account from GOOGLE_APPLICATION_CREDENTIALS"
  gcloud auth activate-service-account --key-file="$credentials_file" --quiet
fi

active_account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -n 1 || true)"
[[ -n "$active_account" ]] || die \
  "no active Google Cloud account. Run 'make waymo-auth' once."

mkdir -p "$raw_dir"

has_raw=false
if find "$raw_dir" -maxdepth 1 -type f -name 'training_20s.tfrecord*' -print -quit | grep -q .; then
  has_raw=true
fi

if [[ "$has_raw" == true ]] && is_true "$skip_download"; then
  echo "Waymo raw data already present in $raw_dir"
else
  source_uri="${gcs_uri%/}/${object_pattern}"
  echo "Download Waymo da $source_uri"
  if [[ -n "$num_files" && "$object_pattern" == *'*'* ]]; then
    [[ "$num_files" =~ ^[1-9][0-9]*$ ]] || die \
      "WAYMO_NUM_FILES must be empty or a positive integer: $num_files"
    object_list="$(mktemp)"
    selected_list="$(mktemp)"
    sorted_list="$(mktemp)"
    temporary_files=("$object_list" "$selected_list" "$sorted_list")
    gcloud storage ls "$source_uri" | awk '/^gs:\/\// {print}' > "$object_list"
    available_count="$(awk 'NF {n++} END {print n+0}' "$object_list")"
    (( available_count >= num_files )) || die \
      "requested $num_files shards, but the pattern exposes only $available_count"
    sort "$object_list" > "$sorted_list"
    head -n "$num_files" "$sorted_list" > "$selected_list"
    echo "Selected $num_files shards out of $available_count available"
    gcloud storage cp --read-paths-from-stdin "$raw_dir/" < "$selected_list"
    rm -f "${temporary_files[@]}"
    temporary_files=()
  else
    gcloud storage cp "$source_uri" "$raw_dir/"
  fi
fi

find "$raw_dir" -maxdepth 1 -type f -name 'training_20s.tfrecord*' -print -quit | grep -q . || die \
  "download completed, but no training_20s.tfrecord* found in $raw_dir"

echo "Google Cloud account: $active_account"
echo "Waymo raw data: $raw_dir"
echo "Building the conversion container..."
make build-waymo

convert_args=("WAYMO_RAW_DATA_PATH=$raw_dir" "NUM_WORKERS=$num_workers")
if [[ -n "$num_files" ]]; then
  convert_args+=("NUM_FILES=$num_files")
fi
if is_true "$overwrite"; then
  convert_args+=("OVERWRITE=1")
fi

make waymo-convert "${convert_args[@]}"

if is_true "$cleanup_raw"; then
  case "$raw_dir" in
    ""|"/"|"."|"./")
      die "WAYMO_CLEANUP_RAW_AFTER_CONVERSION=true requires an explicit safe raw directory: $raw_dir"
      ;;
  esac
  find "$database_dir" -type f -print -quit 2>/dev/null | grep -q . || die \
    "conversion completed, but the Waymo database contains no files; raw data was not deleted"
  echo "Conversion verified: removing raw TFRecords from $raw_dir"
  find "$raw_dir" -maxdepth 1 -type f -name 'training_20s.tfrecord*' -delete
fi

echo "Waymo pipeline completed. Database: ${SCENARIONET_DATA_ROOT:-/workspace/data/scenarionet}/waymo/database"
