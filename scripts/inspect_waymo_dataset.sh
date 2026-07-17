#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

local_gcloud="${repo_root}/.gcloud-sdk/google-cloud-sdk/bin/gcloud"
if ! command -v gcloud >/dev/null 2>&1 && [[ -x "$local_gcloud" ]]; then
  PATH="${local_gcloud%/gcloud}:${PATH}"
  export PATH
fi
CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-${repo_root}/.gcloud-sdk/config}"
export CLOUDSDK_CONFIG
mkdir -p "$CLOUDSDK_CONFIG"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

die() {
  echo "waymo-inventory: $*" >&2
  exit 2
}

command -v gcloud >/dev/null 2>&1 || die \
  "gcloud CLI not found. Run 'make install-gcloud'."

gcs_uri="${WAYMO_GCS_URI:-gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s}"
object_pattern="${WAYMO_GCS_OBJECT_PATTERN:-training_20s.tfrecord-*}"
source_uri="${gcs_uri%/}/${object_pattern}"

active_account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -n 1 || true)"
[[ -n "$active_account" ]] || die \
  "no active Google Cloud account. Run 'make waymo-auth'."

echo "Account: $active_account"
echo "Pattern: $source_uri"
echo
echo "Available objects:"
object_count="$(gcloud storage ls "$source_uri" | awk '/^gs:\/\// {n++} END {print n+0}')"
echo "  shards: $object_count"
echo
echo "Raw storage in the bucket:"
gcloud storage du --summarize --readable-sizes "$source_uri"
echo
echo "Note: this command queries the Cloud Storage catalog and does not download files."
