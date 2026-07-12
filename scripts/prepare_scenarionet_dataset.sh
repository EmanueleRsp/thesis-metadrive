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

pipeline_config="${SCENARIONET_PIPELINE_CONFIG:-/workspace/thesis-metadrive/conf/scenarios/pipeline_v1.yaml}"
pipeline_service="${SCENARIONET_PIPELINE_SERVICE:-dataset-pipeline}"
echo "Resolving pipeline configuration (single YAML source): $pipeline_config"
stage "[0/7] Preparing the CPU-only dataset pipeline container"
docker compose --progress plain build "$pipeline_service"
while IFS=$'\t' read -r key value; do
  [[ -n "$key" ]] || continue
  printf -v "$key" '%s' "$value"
  export "$key"
done < <(
  docker compose run --rm -T "$pipeline_service" uv run --no-sync python \
    -m thesis_rl.cli.scenarios.pipeline_config --config "$pipeline_config"
)

data_root="${SCENARIONET_DATA_ROOT:-/workspace/data/scenarionet}"
catalog_raw="${SCENARIONET_RAW_CATALOG_PATH:-${data_root}/catalog/scenario_catalog_raw.parquet}"
catalog_split="${SCENARIONET_SPLIT_CATALOG_PATH:-${data_root}/catalog/scenario_catalog_split.parquet}"
catalog_final="${SCENARIONET_FINAL_CATALOG_PATH:-${data_root}/catalog/scenario_catalog.parquet}"
groups_path="${SCENARIONET_GROUPS_PATH:-${data_root}/splits/scenario_groups.json}"
split_manifest="${SCENARIONET_SPLIT_MANIFEST_PATH:-${data_root}/splits/split_manifest.json}"
thresholds_path="${SCENARIONET_THRESHOLDS_PATH:-${data_root}/splits/arm_thresholds.json}"
pg_count="${SCENARIONET_PG_COUNT:?pipeline YAML must define pg.count_per_profile}"
pg_seed_start="${SCENARIONET_PG_SEED_START:?pipeline YAML must define pg.seed_start}"
pg_workers="${SCENARIONET_PG_WORKERS:?pipeline YAML must define pg.workers}"
split_seed="${SCENARIONET_SPLIT_SEED:?pipeline YAML must define split.seed}"
overwrite="${SCENARIONET_OVERWRITE:-false}"
auto_split="${SCENARIONET_AUTO_SPLIT:?pipeline YAML must define split.auto}"

waymo_train_target="${SCENARIONET_WAYMO_TRAIN_TARGET:?pipeline YAML must define waymo train target}"
waymo_validation_target="${SCENARIONET_WAYMO_VALIDATION_TARGET:?pipeline YAML must define waymo validation target}"
waymo_test_target="${SCENARIONET_WAYMO_TEST_TARGET:?pipeline YAML must define waymo test target}"
pg_train_target="${SCENARIONET_PG_TRAIN_TARGET:?pipeline YAML must define pg train target}"
pg_validation_target="${SCENARIONET_PG_VALIDATION_TARGET:?pipeline YAML must define pg validation target}"
pg_test_target="${SCENARIONET_PG_TEST_TARGET:?pipeline YAML must define pg test target}"
check_workers="${SCENARIONET_CHECK_WORKERS:?pipeline YAML must define checks.workers}"
run_simulation_check="${SCENARIONET_RUN_SIMULATION_CHECK:?pipeline YAML must define checks.simulation}"

echo "Resolved pipeline parameters:"
echo "  PG: ${pg_count} scenarios/profile, seed=${pg_seed_start}"
echo "  PG workers: ${pg_workers}"
echo "  Waymo targets: train=${waymo_train_target}, validation=${waymo_validation_target}, test=${waymo_test_target}"
echo "  PG targets: train=${pg_train_target}, validation=${pg_validation_target}, test=${pg_test_target}"
echo "  Split mode: auto=${auto_split}, seed=${split_seed}"
echo "  Official simulation check: ${run_simulation_check} (workers=${check_workers})"

if ! is_true "${SCENARIONET_SKIP_WAYMO:-false}"; then
  stage "[1/7] Downloading/converting Waymo training_20s"
  make waymo-pipeline
else
  stage "[1/7] Waymo skipped: SCENARIONET_SKIP_WAYMO=true"
fi

stage "[2/7] Generating PG (${pg_count} scenarios per profile)"
pg_overwrite=()
if is_true "$overwrite"; then
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

catalog_overwrite=()
if is_true "$overwrite"; then
  catalog_overwrite+=(--overwrite)
fi

stage "[3/7] Building catalog and groups"
docker compose run --rm "$pipeline_service" uv run --no-sync python \
  -m thesis_rl.cli.scenarios.build_catalog \
  --data-root "$data_root" \
  --pg-seed-start "$pg_seed_start" \
  --pg-count "$pg_count" \
  --output "$catalog_raw" \
  --groups-output "$groups_path" \
  "${catalog_overwrite[@]}"

stage "[4/7] Building leakage-free train/validation/test splits"
split_args=(
  --catalog "$catalog_raw"
  --output "$catalog_split"
  --groups "$groups_path"
  --split-manifest "$split_manifest"
  --split-seed "$split_seed"
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
docker compose run --rm "$pipeline_service" uv run --no-sync python \
  -m thesis_rl.cli.scenarios.build_splits "${split_args[@]}" "${catalog_overwrite[@]}"

stage "[5/7] Computing train-only thresholds and assigning arms"
docker compose run --rm "$pipeline_service" uv run --no-sync python \
  -m thesis_rl.cli.scenarios.compute_arm_thresholds \
  --catalog "$catalog_split" \
  --output-catalog "$catalog_final" \
  --thresholds "$thresholds_path" \
  --balance-seed "$split_seed" \
  "${catalog_overwrite[@]}"

stage "[6/7] Building train/validation/test runtime views"
docker compose run --rm "$pipeline_service" uv run --no-sync python \
  -m thesis_rl.cli.scenarios.build_runtime_databases \
  --catalog "$catalog_final" \
  --data-root "$data_root" \
  --runtime-root "${data_root}/runtime" \
  --output-catalog "$catalog_final" \
  "${catalog_overwrite[@]}"

stage "[7/7] Validating mappings and running official ScenarioNet checks"
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
