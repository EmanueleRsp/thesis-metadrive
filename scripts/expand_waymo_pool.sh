#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

required="${WAYMO_REQUIRED_ELIGIBLE:?WAYMO_REQUIRED_ELIGIBLE is required}"
batch_size="${WAYMO_BATCH_SHARDS:-8}"
max_new_shards="${WAYMO_MAX_NEW_SHARDS:-128}"
num_workers="${WAYMO_NUM_WORKERS:-8}"
keep_raw="${WAYMO_KEEP_RAW_BATCHES:-false}"
gcs_uri="${WAYMO_GCS_URI:-gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/training_20s}"
object_pattern="${WAYMO_GCS_OBJECT_PATTERN:-training_20s.tfrecord-*}"
host_data_dir="${HOST_DATA_DIR:-./data}"
container_data_dir="${CONTAINER_DATA_DIR:-/workspace/data}"
host_root="${host_data_dir%/}/scenarionet"
container_root="${SCENARIONET_DATA_ROOT:-${container_data_dir%/}/scenarionet}"
host_database="${host_root}/waymo/database"
container_database="${container_root}/waymo/database"
state_dir="${host_root}/waymo/acquisition"
raw_root="${WAYMO_RAW_DATA_PATH:-${host_data_dir%/}/waymo_raw}"
pipeline_service="${SCENARIONET_PIPELINE_SERVICE:-dataset-pipeline}"
temporary_files=()
converter_container=""

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] waymo-expand: %s\n' -1 "$*"
}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

die() {
  echo "waymo-expand: $*" >&2
  exit 2
}

[[ "$required" =~ ^[0-9]+$ ]] || die "required eligible count must be non-negative: $required"
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || die "WAYMO_BATCH_SHARDS must be positive: $batch_size"
[[ "$max_new_shards" =~ ^[1-9][0-9]*$ ]] || die "WAYMO_MAX_NEW_SHARDS must be positive: $max_new_shards"
mkdir -p "$host_database" "$state_dir" "$raw_root/batches"

cleanup() {
  local status=$?
  if [[ -n "$converter_container" ]]; then
    docker rm -f "$converter_container" >/dev/null 2>&1 || true
  fi
  if ((${#temporary_files[@]} > 0)); then
    rm -f "${temporary_files[@]}"
  fi
  return "$status"
}
trap cleanup EXIT

refresh_status() {
  local output key value
  local arm_args=()
  if [[ -n "${WAYMO_REQUIRED_ARM_A4_VRU:-}" ]]; then
    arm_args+=(--required-arm "A4_vru=${WAYMO_REQUIRED_ARM_A4_VRU}")
  fi
  log "refreshing converted-pool status from $container_database"
  output="$(docker compose run --rm -T "$pipeline_service" uv run --no-sync python \
    -m thesis_rl.cli.scenarios.waymo_pool_status \
    --database "$container_database" \
    --data-root "$container_root" \
    --required "$required" \
    --allowed-signal-reliability complete \
    --allowed-signal-reliability not_applicable \
    --report "${container_root}/waymo/acquisition/pool_status.json" \
    --shards-output "${container_root}/waymo/acquisition/converted_shards.txt" \
    --reuse-report-if-current \
    "${arm_args[@]}" \
    --format env)"
  while IFS=$'\t' read -r key value; do
    [[ "$key" == SCENARIONET_WAYMO_* ]] || continue
    printf -v "$key" '%s' "$value"
  done <<< "$output"
  log "status refreshed: eligible=${SCENARIONET_WAYMO_ELIGIBLE_COUNT}/${required}, deficit=${SCENARIONET_WAYMO_ELIGIBLE_DEFICIT}"
}

refresh_status
echo "Waymo eligible pool: ${SCENARIONET_WAYMO_ELIGIBLE_COUNT}/${required} "\
"(converted: ${SCENARIONET_WAYMO_POOL_TOTAL}, deficit: ${SCENARIONET_WAYMO_ELIGIBLE_DEFICIT})"
if [[ -n "${WAYMO_REQUIRED_ARM_A4_VRU:-}" ]]; then
  echo "Waymo A4_vru target: ${SCENARIONET_WAYMO_ELIGIBLE_A4_VRU:-0}/${WAYMO_REQUIRED_ARM_A4_VRU} "\
"(deficit: ${SCENARIONET_WAYMO_DEFICIT_A4_VRU:-0})"
fi
if is_true "$SCENARIONET_WAYMO_POOL_COMPLETE"; then
  echo "Waymo target already satisfied; no download or conversion is needed."
  exit 0
fi

command -v gcloud >/dev/null 2>&1 || die \
  "gcloud CLI not found. Run 'make waymo-auth' once."
active_account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -n 1 || true)"
[[ -n "$active_account" ]] || die "no active Google Cloud account. Run 'make waymo-auth' once."

remote_list="$(mktemp)"
candidate_list="$(mktemp)"
batch_list="$(mktemp)"
temporary_files=("$remote_list" "$candidate_list" "$batch_list")
gcloud storage ls "${gcs_uri%/}/${object_pattern}" | awk '/^gs:\/\// {print}' | sort > "$remote_list"
log "loaded remote shard inventory from ${gcs_uri%/}/${object_pattern}"
awk '
  NR == FNR { used[$1] = 1; next }
  {
    name = $0
    sub(/^.*\//, "", name)
    if (!(name in used)) print $0
  }
' "$state_dir/converted_shards.txt" "$remote_list" > "$candidate_list"

available_new="$(awk 'NF {count++} END {print count+0}' "$candidate_list")"
(( available_new > 0 )) || die "all remote shards are already converted, but the eligible target is unmet"
echo "Automatic expansion: batch=${batch_size} shards, safety cap=${max_new_shards} new shards"
echo "Google Cloud account: $active_account"
log "building/verifying the Waymo conversion image"
make build-waymo
log "Waymo conversion image is ready"
converter_container="thesis_waymo_expand_$$"
log "starting persistent Waymo converter container: $converter_container"
WAYMO_RAW_DATA_PATH="$raw_root" \
  docker compose -f compose.yaml -f compose.waymo.yaml --profile waymo run -d \
    --name "$converter_container" \
    --entrypoint sh \
    waymo-converter \
    -c 'trap "exit 0" TERM INT; while :; do sleep 3600; done' >/dev/null
log "persistent Waymo converter container is ready"

new_shards=0
batch_number=0
while ! is_true "$SCENARIONET_WAYMO_POOL_COMPLETE"; do
  (( new_shards < max_new_shards )) || die \
    "safety cap reached with ${SCENARIONET_WAYMO_ELIGIBLE_DEFICIT} eligible scenarios still missing"
  remaining_cap=$((max_new_shards - new_shards))
  current_batch_size="$batch_size"
  (( current_batch_size <= remaining_cap )) || current_batch_size="$remaining_cap"
  awk -v start="$((new_shards + 1))" -v count="$current_batch_size" \
    'NR >= start && NR < start + count' "$candidate_list" > "$batch_list"
  selected="$(awk 'NF {count++} END {print count+0}' "$batch_list")"
  (( selected > 0 )) || die \
    "no unseen remote shards remain; eligible deficit is ${SCENARIONET_WAYMO_ELIGIBLE_DEFICIT}"

  first_name="$(awk 'NR == 1 {print; exit}' "$batch_list")"
  first_name="${first_name##*/}"
  last_name="$(awk 'NF {last = $0} END {print last}' "$batch_list")"
  last_name="${last_name##*/}"
  first_index="$(sed -E 's/.*-([0-9]{5})-of-.*/\1/' <<< "$first_name")"
  last_index="$(sed -E 's/.*-([0-9]{5})-of-.*/\1/' <<< "$last_name")"
  batch_id="batch_${first_index}_${last_index}"
  raw_batch="${raw_root%/}/batches/$batch_id"
  final_database="${host_database%/}/batches/$batch_id"
  container_staging="${container_root%/}/waymo/staging/$batch_id"
  [[ ! -e "$final_database" ]] || die \
    "batch destination already exists without being registered: $batch_id"
  mkdir -p "$raw_batch"

  batch_number=$((batch_number + 1))
  log "batch ${batch_number}: downloading ${selected} unseen shards (${first_index}-${last_index}) to $raw_batch"
  gcloud storage cp --read-paths-from-stdin "$raw_batch/" < "$batch_list"
  log "batch ${batch_number}: download complete; converting inside persistent container with ${num_workers} workers"
  docker exec "$converter_container" python -m thesis_rl.cli.scenarios.convert_waymo \
    --raw-data-path "/workspace/waymo_raw/batches/$batch_id" \
    --database-path "$container_staging" \
    --num-workers "$num_workers" \
    --num-files "$selected" \
    --overwrite
  log "batch ${batch_number}: conversion finished; moving staging database into final batch directory"
  docker exec "$converter_container" sh -c \
    'mkdir -p "$1" && mv "$2" "$3"' sh \
    "${container_database%/}/batches" "$container_staging" \
    "${container_database%/}/batches/$batch_id"
  log "batch ${batch_number}: final database registered at ${container_database%/}/batches/$batch_id"
  if ! is_true "$keep_raw"; then
    log "batch ${batch_number}: removing raw TFRecords from $raw_batch"
    find "$raw_batch" -maxdepth 1 -type f -name 'training_20s.tfrecord*' -delete
    rmdir "$raw_batch" 2>/dev/null || true
  fi

  new_shards=$((new_shards + selected))
  refresh_status
  log "batch ${batch_number} complete: eligible ${SCENARIONET_WAYMO_ELIGIBLE_COUNT}/${required}; remaining deficit ${SCENARIONET_WAYMO_ELIGIBLE_DEFICIT}"
  if [[ -n "${WAYMO_REQUIRED_ARM_A4_VRU:-}" ]]; then
    log "batch ${batch_number} A4_vru: ${SCENARIONET_WAYMO_ELIGIBLE_A4_VRU:-0}/${WAYMO_REQUIRED_ARM_A4_VRU}; remaining A4 deficit ${SCENARIONET_WAYMO_DEFICIT_A4_VRU:-0}"
  fi
done

echo "Waymo eligible target reached after ${new_shards} new shards."
