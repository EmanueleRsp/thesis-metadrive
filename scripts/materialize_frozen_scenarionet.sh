#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

local_gcloud="${repo_root}/.gcloud-sdk/google-cloud-sdk/bin/gcloud"
if ! command -v gcloud >/dev/null 2>&1 && [[ -x "$local_gcloud" ]]; then
  PATH="${local_gcloud%/gcloud}:${PATH}"
  export PATH
fi
local_cloudsdk_config="${repo_root}/.gcloud-sdk/config"
if [[ -z "${CLOUDSDK_CONFIG:-}" && -d "$local_cloudsdk_config" ]] \
  && CLOUDSDK_CONFIG="$local_cloudsdk_config" gcloud auth list \
    --filter=status:ACTIVE --format='value(account)' 2>/dev/null | grep -q .; then
  CLOUDSDK_CONFIG="$local_cloudsdk_config"
  export CLOUDSDK_CONFIG
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

host_data_dir="${HOST_DATA_DIR:-./data}"
case "$host_data_dir" in
  /*) ;;
  *) host_data_dir="${repo_root}/${host_data_dir}" ;;
esac
host_root="${host_data_dir%/}/scenarionet"
container_data_dir="${CONTAINER_DATA_DIR:-/workspace/data}"
container_root="${SCENARIONET_DATA_ROOT:-${container_data_dir%/}/scenarionet}"
index_path="${SCENARIONET_FROZEN_INDEX:-${repo_root}/data/scenarionet/frozen/scenario_selection_index.json}"
case "$index_path" in
  /*) ;;
  *) index_path="${repo_root}/${index_path}" ;;
esac
raw_root="${WAYMO_RAW_DATA_PATH:-${host_data_dir%/}/waymo_raw}"
case "$raw_root" in
  /*) ;;
  *) raw_root="${repo_root}/${raw_root}" ;;
esac
pipeline_service="${SCENARIONET_PIPELINE_SERVICE:-dataset-pipeline}"
waymo_workers="${WAYMO_NUM_WORKERS:-16}"
keep_raw="${WAYMO_KEEP_RAW_BATCHES:-false}"
pg_workers="${SCENARIONET_PG_WORKERS:-16}"
overwrite="${FROZEN_OVERWRITE:-false}"
temporary_files=()

cleanup() {
  if ((${#temporary_files[@]} > 0)); then
    rm -f "${temporary_files[@]}"
  fi
}
trap cleanup EXIT

die() {
  echo "scenarionet-materialize-frozen: $*" >&2
  exit 2
}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

[[ -f "$index_path" ]] || die "frozen index is missing: $index_path"
command -v gcloud >/dev/null 2>&1 || die "gcloud CLI not found; run 'make install-gcloud' first"
active_account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -n 1 || true)"
[[ -n "$active_account" ]] || die "no active Google Cloud account; run 'make waymo-auth' once"

# Keep the canonical repo index available through the normal data mount for
# the container-side PG and replay commands. A caller-provided data-root index
# is used in place and is never replaced.
data_index_path="${host_root}/frozen/scenario_selection_index.json"
if [[ "$index_path" != "$data_index_path" ]]; then
  mkdir -p "$(dirname "$data_index_path")"
  cp "$index_path" "$data_index_path"
  index_path="$data_index_path"
fi

container_index_path="${container_root}/frozen/scenario_selection_index.json"

shard_list="$(mktemp)"
temporary_files+=("$shard_list")
python3 - "$index_path" "$shard_list" <<'PY'
import json
import sys
from pathlib import Path

index_path, shard_path = map(Path, sys.argv[1:])
payload = json.loads(index_path.read_text(encoding="utf-8"))
shards = sorted(payload["source_inventory"]["waymo"]["selected_shards"])
if not shards:
    raise SystemExit("frozen index contains no selected Waymo shards")
shard_path.write_text("".join(f"{shard}\n" for shard in shards), encoding="utf-8")
PY

batch_size="$(python3 - "$index_path" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["split_manifest"]["waymo_acquisition"]["batch_size_shards"])
PY
)"
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || die \
  "frozen index contains an invalid Waymo batch size: $batch_size"
shard_count="$(awk 'NF {count++} END {print count+0}' "$shard_list")"
(( shard_count > 0 )) || die "frozen index contains no selected Waymo shards"
echo "scenarionet-materialize-frozen: materializing ${shard_count} frozen Waymo shards as ${active_account}"

WAYMO_FROZEN_SHARDS_FILE="$shard_list" \
  WAYMO_FROZEN_INDEX="$container_index_path" \
  WAYMO_BATCH_SHARDS="$batch_size" \
  WAYMO_MAX_NEW_SHARDS="$shard_count" \
  WAYMO_REQUIRED_ELIGIBLE=0 \
  WAYMO_NUM_WORKERS="$waymo_workers" \
  WAYMO_KEEP_RAW_BATCHES="$keep_raw" \
  WAYMO_SKIP_INITIAL_STATUS=true \
  bash scripts/expand_waymo_pool.sh

pg_args=(
  --index "$container_index_path"
  --data-root "$container_root"
  --repo-root /workspace/thesis-metadrive
  --workers "$pg_workers"
)
if is_true "$overwrite"; then
  pg_args+=(--overwrite)
fi
docker compose run --rm "$pipeline_service" uv run --no-sync python \
  -m thesis_rl.cli.scenarios.generate_pg_from_frozen "${pg_args[@]}"

replay_args=(
  --index "$container_index_path"
  --data-root "$container_root"
)
if is_true "$overwrite"; then
  replay_args+=(--overwrite)
fi
docker compose run --rm "$pipeline_service" uv run --no-sync python \
  -m thesis_rl.cli.scenarios.replay_frozen_dataset "${replay_args[@]}"
echo "scenarionet-materialize-frozen: frozen dataset materialization completed"
