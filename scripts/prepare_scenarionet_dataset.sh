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
if ! is_true "${SCENARIONET_SKIP_CONFIG_RESOLVE:-false}"; then
  echo "Risoluzione configurazione pipeline: $pipeline_config"
  docker compose build dev >/dev/null
  while IFS=$'\t' read -r key value; do
    [[ -n "$key" ]] || continue
    printf -v "$key" '%s' "$value"
    export "$key"
  done < <(
    docker compose run --rm -T dev uv run --no-sync python \
      -m thesis_rl.cli.scenarios.pipeline_config --config "$pipeline_config"
  )
fi

data_root="${SCENARIONET_DATA_ROOT:-/workspace/data/scenarionet}"
catalog_raw="${SCENARIONET_RAW_CATALOG_PATH:-${data_root}/catalog/scenario_catalog_raw.parquet}"
catalog_split="${SCENARIONET_SPLIT_CATALOG_PATH:-${data_root}/catalog/scenario_catalog_split.parquet}"
catalog_final="${SCENARIONET_FINAL_CATALOG_PATH:-${data_root}/catalog/scenario_catalog.parquet}"
groups_path="${SCENARIONET_GROUPS_PATH:-${data_root}/splits/scenario_groups.json}"
split_manifest="${SCENARIONET_SPLIT_MANIFEST_PATH:-${data_root}/splits/split_manifest.json}"
thresholds_path="${SCENARIONET_THRESHOLDS_PATH:-${data_root}/splits/arm_thresholds.json}"
pg_count="${SCENARIONET_PG_COUNT:-20}"
pg_seed_start="${SCENARIONET_PG_SEED_START:-920000}"
split_seed="${SCENARIONET_SPLIT_SEED:-0}"
overwrite="${SCENARIONET_OVERWRITE:-false}"
auto_split="${SCENARIONET_AUTO_SPLIT:-true}"

require_count() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die \
    "${name} non impostato. Inserisci i conteggi train/validation/test in .env."
}

if ! is_true "$auto_split"; then
  for variable in \
    SCENARIONET_WAYMO_TRAIN_COUNT SCENARIONET_WAYMO_VALIDATION_COUNT \
    SCENARIONET_WAYMO_TEST_COUNT SCENARIONET_PG_TRAIN_COUNT \
    SCENARIONET_PG_VALIDATION_COUNT SCENARIONET_PG_TEST_COUNT; do
    require_count "$variable"
  done
fi

waymo_train_target="${SCENARIONET_WAYMO_TRAIN_TARGET:-1000}"
waymo_validation_target="${SCENARIONET_WAYMO_VALIDATION_TARGET:-250}"
waymo_test_target="${SCENARIONET_WAYMO_TEST_TARGET:-500}"
pg_train_target="${SCENARIONET_PG_TRAIN_TARGET:-1000}"
pg_validation_target="${SCENARIONET_PG_VALIDATION_TARGET:-250}"
pg_test_target="${SCENARIONET_PG_TEST_TARGET:-500}"

if ! is_true "${SCENARIONET_SKIP_WAYMO:-false}"; then
  echo "[1/7] Download/conversione Waymo..."
  make waymo-pipeline
else
  echo "[1/7] Waymo saltato: SCENARIONET_SKIP_WAYMO=true"
fi

echo "[2/7] Generazione PG (${pg_count} scenari per profilo)..."
pg_overwrite=()
if is_true "$overwrite"; then
  pg_overwrite+=(--overwrite)
fi
docker compose run --rm dev uv run --no-sync python \
  -m thesis_rl.cli.scenarios.generate_pg_dataset \
  --data-root "$data_root" \
  --repo-root /workspace/thesis-metadrive \
  --count "$pg_count" \
  --seed-start "$pg_seed_start" \
  "${pg_overwrite[@]}"

catalog_overwrite=()
if is_true "$overwrite"; then
  catalog_overwrite+=(--overwrite)
fi

echo "[3/7] Costruzione catalogo e gruppi..."
docker compose run --rm dev uv run --no-sync python \
  -m thesis_rl.cli.scenarios.build_catalog \
  --data-root "$data_root" \
  --pg-seed-start "$pg_seed_start" \
  --pg-count "$pg_count" \
  --output "$catalog_raw" \
  --groups-output "$groups_path" \
  "${catalog_overwrite[@]}"

echo "[4/7] Costruzione split..."
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
    --waymo-train "$SCENARIONET_WAYMO_TRAIN_COUNT"
    --waymo-validation "$SCENARIONET_WAYMO_VALIDATION_COUNT"
    --waymo-test "$SCENARIONET_WAYMO_TEST_COUNT"
    --pg-train "$SCENARIONET_PG_TRAIN_COUNT"
    --pg-validation "$SCENARIONET_PG_VALIDATION_COUNT"
    --pg-test "$SCENARIONET_PG_TEST_COUNT"
  )
fi
docker compose run --rm dev uv run --no-sync python \
  -m thesis_rl.cli.scenarios.build_splits "${split_args[@]}" "${catalog_overwrite[@]}"

echo "[5/7] Calcolo soglie e arms..."
docker compose run --rm dev uv run --no-sync python \
  -m thesis_rl.cli.scenarios.compute_arm_thresholds \
  --catalog "$catalog_split" \
  --output-catalog "$catalog_final" \
  --thresholds "$thresholds_path" \
  "${catalog_overwrite[@]}"

echo "[6/7] Costruzione runtime/train, validation e test..."
docker compose run --rm dev uv run --no-sync python \
  -m thesis_rl.cli.scenarios.build_runtime_databases \
  --catalog "$catalog_final" \
  --data-root "$data_root" \
  --runtime-root "${data_root}/runtime" \
  --output-catalog "$catalog_final" \
  "${catalog_overwrite[@]}"

echo "[7/7] Validazione mapping e controlli ufficiali..."
for split in train validation test; do
  runtime_path="${data_root}/runtime/${split}"
  docker compose run --rm dev uv run --no-sync python \
    -m thesis_rl.cli.scenarios.validate_database "$runtime_path" \
    --data-root "$data_root" \
    --catalog "$catalog_final"
  docker compose run --rm dev uv run --no-sync python \
    -m thesis_rl.cli.scenarios.check_database existence "$runtime_path" \
    --error-file-path "${data_root}/validation/${split}" \
    --num-workers "${SCENARIONET_CHECK_WORKERS:-8}" \
    --overwrite
  if is_true "${SCENARIONET_RUN_SIMULATION_CHECK:-true}"; then
    docker compose run --rm dev uv run --no-sync python \
      -m thesis_rl.cli.scenarios.check_database simulation "$runtime_path" \
      --error-file-path "${data_root}/validation/${split}_simulation" \
      --num-workers "${SCENARIONET_CHECK_WORKERS:-8}" \
      --overwrite
  fi
done

for pair in \
  "train validation" "train test" "validation test"; do
  read -r left right <<<"$pair"
  docker compose run --rm dev uv run --no-sync python \
    -m thesis_rl.cli.scenarios.check_database overlap \
    "${data_root}/runtime/${left}" \
    --other-database-path "${data_root}/runtime/${right}"
done

echo "ScenarioNet dataset pipeline completata."
echo "Catalogo finale: $catalog_final"
echo "Runtime root: ${data_root}/runtime"
