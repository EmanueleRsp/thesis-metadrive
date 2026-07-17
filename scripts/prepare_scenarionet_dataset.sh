#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

started_at="$(date +%s)"
current_stage="startup"

on_exit() {
  status=$?
  elapsed=$(( $(date +%s) - started_at ))
  if [[ "$status" -eq 0 ]]; then
    echo
    echo "ScenarioNet pipeline completed successfully in ${elapsed}s."
  else
    echo >&2
    echo "ScenarioNet pipeline stopped in stage '${current_stage}' after ${elapsed}s (exit ${status})." >&2
  fi
}
trap on_exit EXIT

stage() {
  current_stage="$1"
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "$current_stage"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

die() {
  echo "scenarionet-pipeline: $*" >&2
  exit 2
}

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

next_pg_replenishment_seed() {
  local max_seed block_index
  max_seed="$(find "${host_data_root}/pg/database" -type f -name 'sd_*.pkl' -printf '%f\n' 2>/dev/null \
    | sed -n 's/.*PGMap-\([0-9][0-9]*\)\.pkl/\1/p' \
    | sort -n | tail -n 1)"
  if [[ -z "$max_seed" ]]; then
    echo 5920000
    return
  fi
  block_index=$((max_seed / 1000000 + 1))
  echo $((block_index * 1000000 + 920000))
}

pipeline_config="${SCENARIONET_PIPELINE_CONFIG:-/workspace/thesis-metadrive/conf/scenarios/pipeline_v1.yaml}"
pipeline_service="${SCENARIONET_PIPELINE_SERVICE:-dataset-pipeline}"
echo "Resolving pipeline configuration (single YAML source): $pipeline_config"
stage "[0/9] Preparing the CPU-only dataset pipeline container"
docker compose --progress quiet build "$pipeline_service"
while IFS=$'\t' read -r key value; do
  [[ -n "$key" ]] || continue
  printf -v "$key" '%s' "$value"
  # shellcheck disable=SC2163
  # Export the dynamically named configuration key.
  export "$key"
done < <(
  docker compose run --rm -T "$pipeline_service" uv run --no-sync python \
    -m thesis_rl.cli.scenarios.pipeline_config --config "$pipeline_config"
)

data_root="${SCENARIONET_DATA_ROOT:-/workspace/data/scenarionet}"
host_data_dir="${HOST_DATA_DIR:-${repo_root}/data}"
case "$host_data_dir" in
  /*) ;;
  *) host_data_dir="${repo_root}/${host_data_dir}" ;;
esac
host_data_root="${SCENARIONET_HOST_DATA_ROOT:-${host_data_dir%/}/scenarionet}"
case "$host_data_root" in
  /*) ;;
  *) host_data_root="${repo_root}/${host_data_root}" ;;
esac
catalog_raw="${SCENARIONET_RAW_CATALOG_PATH:-${data_root}/catalog/scenario_catalog_raw.parquet}"
catalog_rulebook="${SCENARIONET_RULEBOOK_V2_CATALOG_PATH:-${data_root}/catalog/scenario_catalog_rulebook_v2.parquet}"
catalog_split="${SCENARIONET_SPLIT_CATALOG_PATH:-${data_root}/catalog/scenario_catalog_split.parquet}"
catalog_final="${SCENARIONET_FINAL_CATALOG_PATH:-${data_root}/catalog/scenario_catalog.parquet}"
rulebook_eligibility="${SCENARIONET_RULEBOOK_V2_ELIGIBILITY_PATH:-${data_root}/rulebook_v2/catalog_eligibility.json}"
groups_path="${SCENARIONET_GROUPS_PATH:-${data_root}/splits/scenario_groups.json}"
split_manifest="${SCENARIONET_SPLIT_MANIFEST_PATH:-${data_root}/splits/split_manifest.yaml}"
thresholds_path="${SCENARIONET_THRESHOLDS_PATH:-${data_root}/splits/arm_thresholds.json}"
pg_count="${SCENARIONET_PG_COUNT:?pipeline YAML must define pg.count_per_profile}"
pg_seed_start="${SCENARIONET_PG_SEED_START:?pipeline YAML must define pg.seed_start}"
pg_workers="${SCENARIONET_PG_WORKERS:?pipeline YAML must define pg.workers}"
pg_max_composition_replenishments="${SCENARIONET_PG_MAX_COMPOSITION_REPLENISHMENT_BLOCKS:?pipeline YAML must define pg.max_composition_replenishment_blocks}"
pg_replenishment_candidate_budget="${SCENARIONET_PG_REPLENISHMENT_CANDIDATE_BUDGET:?pipeline YAML must define pg.replenishment_candidate_budget}"
split_seed="${SCENARIONET_SPLIT_SEED:?pipeline YAML must define split.seed}"
overwrite="${SCENARIONET_OVERWRITE:-false}"
auto_split="${SCENARIONET_AUTO_SPLIT:?pipeline YAML must define split.auto}"
rulebook_v2_enabled="${SCENARIONET_RULEBOOK_V2_ENABLED:?pipeline YAML must define rulebook_v2.enabled}"
rulebook_v2_workers="${SCENARIONET_RULEBOOK_V2_WORKERS:?pipeline YAML must define rulebook_v2.workers}"

waymo_train_target="${SCENARIONET_WAYMO_TRAIN_TARGET:?pipeline YAML must define waymo train target}"
waymo_validation_target="${SCENARIONET_WAYMO_VALIDATION_TARGET:?pipeline YAML must define waymo validation target}"
waymo_test_target="${SCENARIONET_WAYMO_TEST_TARGET:?pipeline YAML must define waymo test target}"
pg_train_target="${SCENARIONET_PG_TRAIN_TARGET:?pipeline YAML must define pg train target}"
pg_validation_target="${SCENARIONET_PG_VALIDATION_TARGET:?pipeline YAML must define pg validation target}"
pg_test_target="${SCENARIONET_PG_TEST_TARGET:?pipeline YAML must define pg test target}"
check_workers="${SCENARIONET_CHECK_WORKERS:?pipeline YAML must define checks.workers}"
run_simulation_check="${SCENARIONET_RUN_SIMULATION_CHECK:?pipeline YAML must define checks.simulation}"
waymo_auto_expand="${SCENARIONET_WAYMO_AUTO_EXPAND:?pipeline YAML must define waymo.auto_expand}"
waymo_batch_shards="${SCENARIONET_WAYMO_BATCH_SHARDS:?pipeline YAML must define waymo.batch_shards}"
waymo_max_new_shards="${SCENARIONET_WAYMO_MAX_NEW_SHARDS:?pipeline YAML must define waymo.max_new_shards}"
waymo_workers="${SCENARIONET_WAYMO_WORKERS:?pipeline YAML must define waymo.workers}"
waymo_keep_raw_batches="${SCENARIONET_WAYMO_KEEP_RAW_BATCHES:?pipeline YAML must define waymo.keep_raw_batches}"
waymo_required_a4_vru="${SCENARIONET_WAYMO_REQUIRED_A4_VRU:?pipeline YAML must define waymo.required_arms.A4_vru}"

echo "Resolved pipeline parameters:"
echo "  PG: ${pg_count} scenarios/profile, seed=${pg_seed_start}"
echo "  PG workers: ${pg_workers}"
echo "  PG composition replenishment blocks: ${pg_max_composition_replenishments}"
echo "  PG targeted replenishment budget: ${pg_replenishment_candidate_budget} candidates/cycle"
echo "  Waymo targets: train=${waymo_train_target}, validation=${waymo_validation_target}, test=${waymo_test_target}"
echo "  PG targets: train=${pg_train_target}, validation=${pg_validation_target}, test=${pg_test_target}"
echo "  Waymo required A4_vru: ${waymo_required_a4_vru}"
echo "  Split mode: auto=${auto_split}, seed=${split_seed}"
echo "  Rulebook v2 eligibility before split: ${rulebook_v2_enabled}"
echo "  Rulebook v2 filter workers: ${rulebook_v2_workers}"
echo "  Official simulation check: ${run_simulation_check} (workers=${check_workers})"

if ! is_true "${SCENARIONET_SKIP_WAYMO:-false}"; then
  if is_true "$waymo_auto_expand"; then
    stage "[1/9] Deferring Waymo expansion until post-Rulebook feasibility"
  else
    stage "[1/9] Downloading/converting Waymo training_20s"
    make waymo-pipeline
  fi
else
  stage "[1/9] Waymo skipped: SCENARIONET_SKIP_WAYMO=true"
fi

if ! is_true "${SCENARIONET_SKIP_PG:-false}"; then
  pg_database="${host_data_root}/pg/database"
  pg_report="${host_data_root}/pg/pilot/pg_pilot_report.json"
  if [[ -d "$pg_database" && -f "$pg_report" ]] && ! is_true "${SCENARIONET_PG_OVERWRITE:-false}"; then
    stage "[2/9] Reusing existing PG seeds (set SCENARIONET_PG_OVERWRITE=true to regenerate)"
  else
    stage "[2/9] Generating PG (${pg_count} scenarios per profile)"
    pg_overwrite=()
    # Reuse existing deterministic PG seeds across feasibility cycles.  A full
    # PG regeneration is explicit because it invalidates incremental catalog and
    # Rulebook caches; set SCENARIONET_PG_OVERWRITE=true when intentionally
    # replacing the generated scenarios.
    if is_true "${SCENARIONET_PG_OVERWRITE:-false}"; then
      pg_overwrite+=(--overwrite)
    fi
    docker compose run --rm "$pipeline_service" uv run --no-sync python \
      -m thesis_rl.cli.scenarios.generate_pg_dataset \
      --data-root "$data_root" \
      --repo-root /workspace/thesis-metadrive \
      --count "$pg_count" \
      --seed-start "$pg_seed_start" \
      --workers "$pg_workers" \
      "${pg_overwrite[@]}"
  fi
else
  stage "[2/9] PG generation skipped: SCENARIONET_SKIP_PG=true"
fi

catalog_overwrite=()
if is_true "$overwrite"; then
  catalog_overwrite+=(--overwrite)
fi

if ! is_true "$rulebook_v2_enabled"; then
  die "ScenarioNet v1.1 requires rulebook_v2.enabled=true"
fi

split_args=(
  --output "$catalog_split"
  --groups "$groups_path"
  --split-manifest "$split_manifest"
  --pg-replenishment-report "${data_root}/pg/replenishment_report.json"
  --split-seed "$split_seed"
  --waymo-ordering-seed "$split_seed"
  --waymo-batch-shards "$waymo_batch_shards"
  --waymo-max-new-shards "$waymo_max_new_shards"
  --arm-minimums-config "$pipeline_config"
)
if is_true "$auto_split"; then
  split_args+=(
    --auto-targets
    --waymo-target-train "$waymo_train_target"
    --waymo-target-validation "$waymo_validation_target"
    --waymo-target-test "$waymo_test_target"
    --pg-target-train "$pg_train_target"
    --pg-target-validation "$pg_validation_target"
    --pg-target-test "$pg_test_target"
  )
else
  split_args+=(
    --waymo-train "$waymo_train_target"
    --waymo-validation "$waymo_validation_target"
    --waymo-test "$waymo_test_target"
    --pg-train "$pg_train_target"
    --pg-validation "$pg_validation_target"
    --pg-test "$pg_test_target"
  )
fi

max_batches=$(( (waymo_max_new_shards + waymo_batch_shards - 1) / waymo_batch_shards ))
[[ "$pg_max_composition_replenishments" =~ ^[0-9]+$ ]] || die \
  "PG composition replenishment block limit must be a non-negative integer: ${pg_max_composition_replenishments}"
[[ "$pg_replenishment_candidate_budget" =~ ^[1-9][0-9]*$ ]] || die \
  "PG replenishment candidate budget must be a positive integer: ${pg_replenishment_candidate_budget}"
pg_composition_replenishments=0
catalog_args=()
if is_true "$waymo_auto_expand"; then
  catalog_args+=(--allow-empty-waymo)
fi
for ((cycle=0; ; cycle++)); do
  cycle_overwrite=("${catalog_overwrite[@]}")
  if (( cycle > 0 )); then
    cycle_overwrite=(--overwrite)
  fi
  stage "[3/9] Building catalog and groups (feasibility cycle $((cycle + 1)))"
  docker compose run --rm "$pipeline_service" uv run --no-sync python \
    -m thesis_rl.cli.scenarios.build_catalog \
    --data-root "$data_root" \
    --pg-seed-start "$pg_seed_start" \
    --pg-count "$pg_count" \
    --pg-include-all \
    --waymo-workers "$waymo_workers" \
    --pg-workers "$pg_workers" \
    --output "$catalog_raw" \
    --groups-output "$groups_path" \
    "${catalog_args[@]}" \
    "${cycle_overwrite[@]}"

  stage "[4/9] Filtering the catalog with Rulebook v2 static eligibility"
  docker compose run --rm "$pipeline_service" uv run --no-sync python \
    -m thesis_rl.cli.scenarios.filter_rulebook_v2_catalog \
    --catalog "$catalog_raw" \
    --data-root "$data_root" \
    --output-catalog "$catalog_rulebook" \
    --eligibility-output "$rulebook_eligibility" \
    --ego-config "${data_root}/rulebook_v2/ego_config.json" \
    --calibration "${data_root}/rulebook_v2/calibration_b_e.json" \
    --workers "$rulebook_v2_workers" \
    "${cycle_overwrite[@]}"

  stage "[5/9] Building leakage-free train/validation/test splits"
  if docker compose run --rm "$pipeline_service" uv run --no-sync python \
    -m thesis_rl.cli.scenarios.build_splits \
    --catalog "$catalog_rulebook" \
    "${split_args[@]}" \
    "${cycle_overwrite[@]}"; then
    break
  fi
  if ! is_true "$waymo_auto_expand" || is_true "${SCENARIONET_SKIP_WAYMO:-false}"; then
    die "split targets are infeasible after Rulebook filtering; automatic Waymo expansion is disabled"
  fi
  pg_report_host_path="${host_data_root}/pg/replenishment_report.json"
  pg_report_container_path="${data_root}/pg/replenishment_report.json"
  if [[ -f "$pg_report_host_path" ]]; then
    pg_report_values="$(docker compose run --rm -T "$pipeline_service" uv run --no-sync python -c \
      'import json,sys; p=sys.argv[1]; d=json.load(open(p, encoding="utf-8")); print("{}\t{}".format(int(d.get("hard_count_shortfall", 0)), d.get("selection_error") or ""))' \
      "$pg_report_container_path")"
    IFS=$'\t' read -r pg_shortfall pg_selection_error <<<"$pg_report_values"
    if [[ "$pg_shortfall" =~ ^[1-9][0-9]*$ ]]; then
      die "PG replenishment required before Waymo expansion: ${pg_shortfall} additional runtime-eligible records"
    fi
    if [[ "$pg_selection_error" == runtime\ split\ target\ mismatch\ for\ pg/* ]] \
      && (( pg_composition_replenishments < pg_max_composition_replenishments )); then
      pg_profile_counts="$(docker compose run --rm -T "$pipeline_service" uv run --no-sync python \
        -m thesis_rl.cli.scenarios.plan_pg_replenishment \
        --report "$pg_report_container_path" \
        --budget "$pg_replenishment_candidate_budget")"
      if [[ "$pg_profile_counts" == "{}" ]]; then
        echo "PG composition report has no arm-level deficit suitable for targeted replenishment"
      else
        stage "[2/9] Replenishing PG after compositional split infeasibility"
        replenishment_seed_start="$(next_pg_replenishment_seed)"
        pg_composition_replenishments=$((pg_composition_replenishments + 1))
        echo "PG targeted compositional replenishment ${pg_composition_replenishments}/${pg_max_composition_replenishments}: budget=${pg_replenishment_candidate_budget}, profiles=${pg_profile_counts}, seed=${replenishment_seed_start}"
        SCENARIONET_PG_REPLENISH_COUNT="$pg_count" \
          SCENARIONET_PG_REPLENISH_SEED_START="$replenishment_seed_start" \
          SCENARIONET_PG_PROFILE_COUNTS="$pg_profile_counts" \
          make scenarionet-pg-replenish
        continue
      fi
    fi
  fi
  if (( cycle >= max_batches )); then
    die "Waymo cap of ${waymo_max_new_shards} new shards reached before post-Rulebook split feasibility"
  fi
  status_output="$(docker compose run --rm -T "$pipeline_service" uv run --no-sync python \
    -m thesis_rl.cli.scenarios.waymo_pool_status \
    --catalog "$catalog_rulebook" \
    --data-root "$data_root" \
    --required "$((waymo_train_target + waymo_validation_target + waymo_test_target))" \
    --required-arm "A4_vru=${waymo_required_a4_vru}" \
    --allowed-signal-reliability complete \
    --allowed-signal-reliability not_applicable \
    --require-rulebook-eligible \
    --format env)"
  if ! grep -q $'^SCENARIONET_WAYMO_POOL_COMPLETE\tfalse$' <<<"$status_output"; then
    die "split targets remain infeasible although Waymo post-Rulebook coverage is sufficient; inspect split_report.json for PG/group constraints"
  fi
  remaining_cap=$((waymo_max_new_shards - cycle * waymo_batch_shards))
  stage "[1/9] Acquiring one additional Waymo batch after post-Rulebook deficit"
  WAYMO_REQUIRED_ELIGIBLE="$((waymo_train_target + waymo_validation_target + waymo_test_target))" \
    WAYMO_REQUIRED_ARM_A4_VRU="$waymo_required_a4_vru" \
    WAYMO_BATCH_SHARDS="$waymo_batch_shards" \
    WAYMO_MAX_NEW_SHARDS="$remaining_cap" \
    WAYMO_NUM_WORKERS="$waymo_workers" \
    WAYMO_KEEP_RAW_BATCHES="$waymo_keep_raw_batches" \
  WAYMO_FORCE_ONE_BATCH=true \
  WAYMO_SKIP_INITIAL_STATUS=true \
    make waymo-expand
done

stage "[6/9] Computing train-only thresholds and assigning arms"
docker compose run --rm "$pipeline_service" uv run --no-sync python \
  -m thesis_rl.cli.scenarios.compute_arm_thresholds \
  --catalog "$catalog_split" \
  --output-catalog "$catalog_final" \
  --thresholds "$thresholds_path" \
  --balance-seed "$split_seed" \
  "${catalog_overwrite[@]}"

stage "[7/9] Split arm balance already frozen by balanced_arm_source"

stage "[8/9] Building train/validation/test runtime views"
docker compose run --rm "$pipeline_service" uv run --no-sync python \
  -m thesis_rl.cli.scenarios.build_runtime_databases \
  --catalog "$catalog_final" \
  --data-root "$data_root" \
  --runtime-root "${data_root}/runtime" \
  --output-catalog "$catalog_final" \
  "${catalog_overwrite[@]}"

stage "[9/9] Validating mappings and running official ScenarioNet checks"
for split in train validation test; do
  runtime_path="${data_root}/runtime/${split}"
  docker compose run --rm "$pipeline_service" uv run --no-sync python \
    -m thesis_rl.cli.scenarios.validate_database "$runtime_path" \
    --data-root "$data_root" \
    --split "$split" \
    --catalog "$catalog_final"
  docker compose run --rm "$pipeline_service" uv run --no-sync python \
    -m thesis_rl.cli.scenarios.check_database existence "$runtime_path" \
    --error-file-path "${data_root}/validation/${split}" \
    --num-workers "$check_workers" \
    --overwrite
  if is_true "$run_simulation_check"; then
    docker compose run --rm "$pipeline_service" uv run --no-sync python \
      -m thesis_rl.cli.scenarios.check_database simulation "$runtime_path" \
      --error-file-path "${data_root}/validation/${split}_simulation" \
      --num-workers "$check_workers" \
      --overwrite
  fi
done

for pair in \
  "train validation" "train test" "validation test"; do
  read -r left right <<<"$pair"
  docker compose run --rm "$pipeline_service" uv run --no-sync python \
    -m thesis_rl.cli.scenarios.check_database overlap \
    "${data_root}/runtime/${left}" \
    --other-database-path "${data_root}/runtime/${right}"
done

echo "ScenarioNet dataset pipeline completed."
echo "Catalogo finale: $catalog_final"
echo "Runtime root: ${data_root}/runtime"
