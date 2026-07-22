.PHONY: setup verify verify-gpu build build-gpu build-waymo install-gcloud waymo-auth waymo-inventory waymo-convert waymo-pipeline waymo-expand scenarionet-pipeline scenarionet-rebuild-existing scenarionet-recatalog scenarionet-pg-replenish scenarionet-freeze scenarionet-from-frozen scenarionet-materialize-frozen up up-gpu shell test lint format format-check gpu-check smoke smoke-gpu run run-train run-golden-rulebook config config-gpu rulebook-v2-init rulebook-v2-prepare rulebook-v2-collect-trials rulebook-v2-calibrate rulebook-v2-validate-calibration rulebook-v2-filter-catalog rulebook-v2-pilot rulebook-v2-pilot-final rulebook-v2-check rulebook-v2-f10

PYTHON_QUALITY_PATHS ?= src tests scripts

# The final integration run is intentionally explicit about the learner. All
# values are Hydra config names under conf/agent/planner/algorithm/.
ALGORITHM ?=
ENCODER ?= lq_v3
RUN_PROFILE ?= smoke
RUN_NAME ?= run
RUN_OVERRIDES ?=
NUM_ENVS ?= 20
SEED ?= 42
RUN_ALGORITHM_CONFIGS := ppo ppo_sb3 sac sac_sb3 td3 td3_sb3
RUN_PROFILE_CONFIGS := default fast medium long thesis tune smoke
GOLD_MANIFEST ?= docs/audits/scalar_pipeline_audit_2026-07-19/golden_suite_content_validated/golden_suite_manifest.json
GOLD_MANIFEST_CONTAINER ?= /workspace/thesis-metadrive/$(GOLD_MANIFEST)

RULEBOOK_V2_DATA_ROOT ?= data/scenarionet
RULEBOOK_V2_CONTAINER_DATA_ROOT ?= /workspace/data/scenarionet
RULEBOOK_V2_EGO_CONFIG ?= $(RULEBOOK_V2_DATA_ROOT)/rulebook_v2/ego_config.json
RULEBOOK_V2_EGO_CONFIG_SOURCE ?= conf/rulebook_v2/ego_calibration.json
RULEBOOK_V2_TRIALS ?= $(RULEBOOK_V2_DATA_ROOT)/rulebook_v2/braking_trials.json
RULEBOOK_V2_CALIBRATION ?= $(RULEBOOK_V2_DATA_ROOT)/rulebook_v2/calibration_b_e.json
RULEBOOK_V2_PILOT_REPORT ?= $(RULEBOOK_V2_DATA_ROOT)/rulebook_v2/pilot_final.json
RULEBOOK_V2_PILOT_PRELIMINARY ?= $(RULEBOOK_V2_DATA_ROOT)/rulebook_v2/pilot_offline.json
RULEBOOK_V2_RAW_CATALOG ?= $(RULEBOOK_V2_DATA_ROOT)/catalog/scenario_catalog_raw.parquet
RULEBOOK_V2_FILTERED_CATALOG ?= $(RULEBOOK_V2_DATA_ROOT)/catalog/scenario_catalog_rulebook_v2.parquet
RULEBOOK_V2_ELIGIBILITY ?= $(RULEBOOK_V2_DATA_ROOT)/rulebook_v2/catalog_eligibility.json
RULEBOOK_V2_WORKERS ?= 16
SCENARIONET_PG_REPLENISH_COUNT ?= 350
SCENARIONET_PG_REPLENISH_SEED_START ?= 5920000
SCENARIONET_PG_PROFILE_COUNTS ?=
RULEBOOK_V2_EGO_CONFIG_CONTAINER ?= $(RULEBOOK_V2_CONTAINER_DATA_ROOT)/rulebook_v2/ego_config.json
RULEBOOK_V2_TRIALS_CONTAINER ?= $(RULEBOOK_V2_CONTAINER_DATA_ROOT)/rulebook_v2/braking_trials.json
RULEBOOK_V2_CALIBRATION_CONTAINER ?= $(RULEBOOK_V2_CONTAINER_DATA_ROOT)/rulebook_v2/calibration_b_e.json
RULEBOOK_V2_PILOT_REPORT_CONTAINER ?= $(RULEBOOK_V2_CONTAINER_DATA_ROOT)/rulebook_v2/pilot_final.json
RULEBOOK_V2_PILOT_PRELIMINARY_CONTAINER ?= $(RULEBOOK_V2_CONTAINER_DATA_ROOT)/rulebook_v2/pilot_offline.json
RULEBOOK_V2_RAW_CATALOG_CONTAINER ?= $(RULEBOOK_V2_CONTAINER_DATA_ROOT)/catalog/scenario_catalog_raw.parquet
RULEBOOK_V2_FILTERED_CATALOG_CONTAINER ?= $(RULEBOOK_V2_CONTAINER_DATA_ROOT)/catalog/scenario_catalog_rulebook_v2.parquet
RULEBOOK_V2_ELIGIBILITY_CONTAINER ?= $(RULEBOOK_V2_CONTAINER_DATA_ROOT)/rulebook_v2/catalog_eligibility.json

# Optional override for the existing-source ScenarioNet rebuild. If empty, the
# filtering CLI default applies. Example: make scenarionet-rebuild-existing SCENARIONET_REBUILD_WORKERS=64
SCENARIONET_REBUILD_WORKERS ?=

rulebook-v2-init:
	mkdir -p "$(RULEBOOK_V2_DATA_ROOT)/rulebook_v2"
	@test -f "$(RULEBOOK_V2_EGO_CONFIG_SOURCE)" || (echo "Missing canonical ego config source: $(RULEBOOK_V2_EGO_CONFIG_SOURCE)" >&2; exit 2)
	@if [ ! -f "$(RULEBOOK_V2_EGO_CONFIG)" ]; then \
		cp "$(RULEBOOK_V2_EGO_CONFIG_SOURCE)" "$(RULEBOOK_V2_EGO_CONFIG)"; \
		echo "Installed canonical Rulebook v2 ego config: $(RULEBOOK_V2_EGO_CONFIG)"; \
	fi

rulebook-v2-prepare: rulebook-v2-init
	@if [ ! -f "$(RULEBOOK_V2_CALIBRATION)" ]; then \
		$(MAKE) rulebook-v2-collect-trials; \
		$(MAKE) rulebook-v2-calibrate; \
	fi
	@$(MAKE) rulebook-v2-validate-calibration

rulebook-v2-collect-trials: rulebook-v2-init
	@test -f "$(RULEBOOK_V2_EGO_CONFIG)" || (echo "Missing $(RULEBOOK_V2_EGO_CONFIG): provide the frozen ego MetaDrive JSON config first" >&2; exit 2)
	docker compose run --rm dataset-pipeline uv run --no-sync python -m thesis_rl.cli.rulebook_v2_braking_trials \
		--config "$(RULEBOOK_V2_EGO_CONFIG_CONTAINER)" \
		--trials-per-target 10 \
		--out "$(RULEBOOK_V2_TRIALS_CONTAINER)"

rulebook-v2-calibrate:
	@test -f "$(RULEBOOK_V2_EGO_CONFIG)" || (echo "Missing $(RULEBOOK_V2_EGO_CONFIG): freeze the ego config as canonical JSON first" >&2; exit 2)
	@test -f "$(RULEBOOK_V2_TRIALS)" || (echo "Missing $(RULEBOOK_V2_TRIALS): collect the 40 real braking trials first" >&2; exit 2)
	@ego_hash=$$(docker compose run --rm -T dataset-pipeline uv run --no-sync python -c 'import hashlib,json; from pathlib import Path; p=Path("$(RULEBOOK_V2_EGO_CONFIG_CONTAINER)"); print(hashlib.sha256(json.dumps(json.loads(p.read_text()),sort_keys=True,separators=(",",":")).encode()).hexdigest())'); \
	docker compose run --rm -T dataset-pipeline uv run --no-sync python -m thesis_rl.cli.rulebook_v2_calibrate \
		--trials "$(RULEBOOK_V2_TRIALS_CONTAINER)" \
		--config-hash "$$ego_hash" \
		--out "$(RULEBOOK_V2_CALIBRATION_CONTAINER)"

rulebook-v2-validate-calibration:
	@test -f "$(RULEBOOK_V2_EGO_CONFIG)" || (echo "Missing $(RULEBOOK_V2_EGO_CONFIG)" >&2; exit 2)
	@test -f "$(RULEBOOK_V2_CALIBRATION)" || (echo "Missing $(RULEBOOK_V2_CALIBRATION): run make rulebook-v2-calibrate" >&2; exit 2)
	@ego_hash=$$(docker compose run --rm -T dataset-pipeline uv run --no-sync python -c 'import hashlib,json; from pathlib import Path; p=Path("$(RULEBOOK_V2_EGO_CONFIG_CONTAINER)"); print(hashlib.sha256(json.dumps(json.loads(p.read_text()),sort_keys=True,separators=(",",":")).encode()).hexdigest())'); \
	docker compose run --rm -T dataset-pipeline uv run --no-sync python -c 'from thesis_rl.rulebook.v2.calibration import load_calibration_artifact; a=load_calibration_artifact("$(RULEBOOK_V2_CALIBRATION_CONTAINER)", expected_config_hash="'"$$ego_hash"'"); print(a)'

rulebook-v2-filter-catalog:
	@test -f "$(RULEBOOK_V2_RAW_CATALOG)" || (echo "Missing $(RULEBOOK_V2_RAW_CATALOG): build the raw ScenarioNet catalog first" >&2; exit 2)
	@test -f "$(RULEBOOK_V2_EGO_CONFIG)" || (echo "Missing $(RULEBOOK_V2_EGO_CONFIG)" >&2; exit 2)
	@test -f "$(RULEBOOK_V2_CALIBRATION)" || (echo "Missing $(RULEBOOK_V2_CALIBRATION): run make rulebook-v2-calibrate" >&2; exit 2)
	docker compose run --rm dataset-pipeline uv run --no-sync python -m thesis_rl.cli.scenarios.filter_rulebook_v2_catalog \
		--catalog "$(RULEBOOK_V2_RAW_CATALOG_CONTAINER)" \
		--data-root "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)" \
		--output-catalog "$(RULEBOOK_V2_FILTERED_CATALOG_CONTAINER)" \
		--eligibility-output "$(RULEBOOK_V2_ELIGIBILITY_CONTAINER)" \
		--ego-config "$(RULEBOOK_V2_EGO_CONFIG_CONTAINER)" \
		--calibration "$(RULEBOOK_V2_CALIBRATION_CONTAINER)" \
		--workers "$(RULEBOOK_V2_WORKERS)" \
		--overwrite

rulebook-v2-pilot:
	docker compose run --rm dev uv run --no-sync python -m thesis_rl.cli.rulebook_v2_pilot \
		--data-root "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)" \
		--pg-per-profile 2 --waymo-count 10 \
		--out "$(RULEBOOK_V2_PILOT_PRELIMINARY_CONTAINER)"

rulebook-v2-pilot-final:
	@test -n "$(GEOMETRY_CONFIG_HASH)" || (echo "GEOMETRY_CONFIG_HASH is required for the final pilot" >&2; exit 2)
	@test -f "$(RULEBOOK_V2_EGO_CONFIG)" || (echo "Missing $(RULEBOOK_V2_EGO_CONFIG)" >&2; exit 2)
	@test -f "$(RULEBOOK_V2_CALIBRATION)" || (echo "Missing $(RULEBOOK_V2_CALIBRATION): run make rulebook-v2-calibrate" >&2; exit 2)
	@ego_hash=$$(docker compose run --rm dev uv run --no-sync python -c 'import hashlib,json; from pathlib import Path; p=Path("$(RULEBOOK_V2_EGO_CONFIG_CONTAINER)"); print(hashlib.sha256(json.dumps(json.loads(p.read_text()),sort_keys=True,separators=(",",":")).encode()).hexdigest())'); \
	docker compose run --rm dev uv run --no-sync python -m thesis_rl.cli.rulebook_v2_pilot \
		--data-root "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)" \
		--pg-per-profile 2 --waymo-count 10 \
		--geometry-config-hash "$(GEOMETRY_CONFIG_HASH)" \
		--calibration-hash "$$ego_hash" \
		--out "$(RULEBOOK_V2_PILOT_REPORT_CONTAINER)"

rulebook-v2-check:
	docker compose run --rm dev uv run --no-sync python -m pytest -q tests/test_rulebook_v2_*.py
	docker compose run --rm dev uv run --no-sync ruff check src/thesis_rl/rulebook/v2 src/thesis_rl/cli/rulebook_v2_calibrate.py src/thesis_rl/cli/rulebook_v2_pilot.py src/thesis_rl/cli/rulebook_v2_braking_trials.py
	git diff --check

rulebook-v2-f10: rulebook-v2-collect-trials rulebook-v2-calibrate rulebook-v2-validate-calibration rulebook-v2-pilot-final rulebook-v2-check

setup:
	./setup.sh

verify:
	./setup.sh --verify

verify-gpu:
	./setup.sh --verify --gpu

config:
	docker compose config --quiet

config-gpu:
	docker compose -f compose.yaml -f compose.gpu.yaml config --quiet

build:
	docker compose build

build-gpu:
	docker compose -f compose.yaml -f compose.gpu.yaml build

build-waymo:
	docker compose --progress quiet -f compose.yaml -f compose.waymo.yaml --profile waymo build waymo-converter

install-gcloud:
	bash scripts/install_gcloud.sh

waymo-auth:
	GCLOUD_LOGIN_FLAGS="$(GCLOUD_INIT_FLAGS)" bash scripts/waymo_auth.sh

waymo-inventory:
	bash scripts/inspect_waymo_dataset.sh

waymo-convert:
	@test -n "$(WAYMO_RAW_DATA_PATH)" || (echo "WAYMO_RAW_DATA_PATH is required" >&2; exit 2)
	@echo "Starting Waymo converter container: raw=$(WAYMO_RAW_DATA_PATH) workers=$(NUM_WORKERS) files=$(NUM_FILES) database=$(DATABASE_PATH)"
	WAYMO_RAW_DATA_PATH="$(WAYMO_RAW_DATA_PATH)" docker compose -f compose.yaml -f compose.waymo.yaml --profile waymo run --rm waymo-converter \
		--raw-data-path /workspace/waymo_raw \
		$(if $(DATABASE_PATH),--database-path "$(DATABASE_PATH)",) \
		$(if $(NUM_WORKERS),--num-workers "$(NUM_WORKERS)",) \
		$(if $(NUM_FILES),--num-files "$(NUM_FILES)",) \
		$(if $(OVERWRITE),--overwrite,)

waymo-pipeline:
	bash scripts/prepare_waymo.sh

waymo-expand:
	bash scripts/expand_waymo_pool.sh

scenarionet-pipeline:
	bash scripts/prepare_scenarionet_dataset.sh

# Revalidate and rebuild only from already materialized source databases.
# Docker TTY allocation is intentionally retained so Rich progress remains
# visible. No discovery, acquisition, or source generation is performed.
scenarionet-rebuild-existing:
	@test -f "$(RULEBOOK_V2_RAW_CATALOG_CONTAINER)" || (echo "Missing raw catalog: $(RULEBOOK_V2_RAW_CATALOG_CONTAINER)" >&2; exit 2)
	@test -f "$(RULEBOOK_V2_EGO_CONFIG_CONTAINER)" || (echo "Missing frozen ego config: $(RULEBOOK_V2_EGO_CONFIG_CONTAINER)" >&2; exit 2)
	@test -f "$(RULEBOOK_V2_CALIBRATION_CONTAINER)" || (echo "Missing Rulebook calibration: $(RULEBOOK_V2_CALIBRATION_CONTAINER)" >&2; exit 2)
	docker compose run --rm dataset-pipeline uv run --no-sync python \
		-m thesis_rl.cli.scenarios.filter_rulebook_v2_catalog \
		--catalog "$(RULEBOOK_V2_RAW_CATALOG_CONTAINER)" \
		--data-root "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)" \
		--output-catalog "$(RULEBOOK_V2_FILTERED_CATALOG_CONTAINER)" \
		--eligibility-output "$(RULEBOOK_V2_ELIGIBILITY_CONTAINER)" \
		--ego-config "$(RULEBOOK_V2_EGO_CONFIG_CONTAINER)" \
		--calibration "$(RULEBOOK_V2_CALIBRATION_CONTAINER)" \
		$(if $(strip $(SCENARIONET_REBUILD_WORKERS)),--workers "$(SCENARIONET_REBUILD_WORKERS)",) \
		--no-incremental \
		--overwrite
	docker compose run --rm dataset-pipeline uv run --no-sync python \
		-m thesis_rl.cli.scenarios.build_splits \
		--catalog "$(RULEBOOK_V2_FILTERED_CATALOG_CONTAINER)" \
		--output "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)/catalog/scenario_catalog_split.parquet" \
		--groups "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)/splits/scenario_groups.json" \
		--split-manifest "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)/splits/split_manifest.yaml" \
		--pg-replenishment-report "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)/pg/replenishment_report.json" \
		--split-seed 0 \
		--waymo-ordering-seed 0 \
		--waymo-batch-shards 128 \
		--waymo-max-new-shards 1000 \
		--arm-minimums-config /workspace/thesis-metadrive/conf/scenarios/pipeline_v1.yaml \
		--auto-targets \
		--waymo-target-train 1000 \
		--waymo-target-validation 250 \
		--waymo-target-test 500 \
		--pg-target-train 1000 \
		--pg-target-validation 250 \
		--pg-target-test 500 \
		--overwrite
	docker compose run --rm dataset-pipeline uv run --no-sync python \
		-m thesis_rl.cli.scenarios.compute_arm_thresholds \
		--catalog "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)/catalog/scenario_catalog_split.parquet" \
		--output-catalog "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)/catalog/scenario_catalog.parquet" \
		--thresholds "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)/splits/arm_thresholds.json" \
		--balance-seed 0 \
		--overwrite
	docker compose run --rm dataset-pipeline uv run --no-sync python \
		-m thesis_rl.cli.scenarios.build_runtime_databases \
		--catalog "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)/catalog/scenario_catalog.parquet" \
		--data-root "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)" \
		--runtime-root "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)/runtime" \
		--output-catalog "$(RULEBOOK_V2_CONTAINER_DATA_ROOT)/catalog/scenario_catalog.parquet" \
		--overwrite
	$(MAKE) scenarionet-freeze OVERWRITE=1

scenarionet-pg-replenish:
	docker compose run --rm dataset-pipeline uv run --no-sync python -m thesis_rl.cli.scenarios.generate_pg_dataset \
		--data-root /workspace/data/scenarionet \
		--repo-root /workspace/thesis-metadrive \
		--count "$(SCENARIONET_PG_REPLENISH_COUNT)" \
		--seed-start "$(SCENARIONET_PG_REPLENISH_SEED_START)" \
		--workers "$(SCENARIONET_PG_WORKERS)" \
		--report-output "/workspace/data/scenarionet/pg/replenishment/pg_pilot_report_$(SCENARIONET_PG_REPLENISH_SEED_START).json" \
		$(if $(strip $(SCENARIONET_PG_PROFILE_COUNTS)),--profile-counts-json '$(SCENARIONET_PG_PROFILE_COUNTS)',) \
		--overwrite

scenarionet-recatalog:
	SCENARIONET_SKIP_PG=true SCENARIONET_SKIP_WAYMO=true bash scripts/prepare_scenarionet_dataset.sh

scenarionet-freeze:
	docker compose run --rm dataset-pipeline uv run --no-sync python -m thesis_rl.cli.scenarios.freeze_dataset \
		--data-root /workspace/data/scenarionet \
		$(if $(filter 1 true yes on,$(OVERWRITE)),--overwrite,)

scenarionet-from-frozen:
	docker compose run --rm dataset-pipeline uv run --no-sync python -m thesis_rl.cli.scenarios.replay_frozen_dataset \
		--index /workspace/data/scenarionet/frozen/scenario_selection_index.json \
		--data-root /workspace/data/scenarionet \
		$(if $(filter 1 true yes on,$(OVERWRITE)),--overwrite,)

scenarionet-materialize-frozen:
	bash scripts/materialize_frozen_scenarionet.sh

up:
	docker compose up -d

up-gpu:
	docker compose -f compose.yaml -f compose.gpu.yaml up -d

shell:
	docker compose exec dev bash

test:
	docker compose run --rm dev uv run --no-sync python -m pytest -q

lint:
	docker compose run --rm dev uv run --no-sync ruff check $(PYTHON_QUALITY_PATHS)

format:
	docker compose run --rm dev uv run --no-sync ruff format $(PYTHON_QUALITY_PATHS)

format-check:
	docker compose run --rm dev uv run --no-sync ruff format --check $(PYTHON_QUALITY_PATHS)

gpu-check:
	docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(); x = torch.tensor([1.0], device="cuda"); assert (x * 2).item() == 2.0; print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")'

smoke:
	docker compose run --rm dev uv run --no-sync python -m thesis_rl.cli.train --config-name presets/test/smoke_train

smoke-gpu:
	docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev uv run --no-sync python -m thesis_rl.cli.train --config-name presets/test/smoke_train

# Canonical final-pipeline entry point. The dev service supplies the repository
# and outputs mounts and mounts the dataset read-only; the GPU overlay is
# mandatory for this target.
run-train:
	@if [ -z "$(strip $(ALGORITHM))" ]; then \
		echo "ALGORITHM is required. Valid Hydra configs: $(RUN_ALGORITHM_CONFIGS)" >&2; \
		exit 2; \
	fi
	@if ! printf '%s\n' "$(NUM_ENVS)" | grep -Eq '^[1-9][0-9]*$$'; then \
		echo "NUM_ENVS must be an integer greater than or equal to 1; got '$(NUM_ENVS)'" >&2; \
		exit 2; \
	fi
	@if ! printf '%s\n' "$(SEED)" | grep -Eq '^[0-9]+$$'; then \
		echo "SEED must be a non-negative integer; got '$(SEED)'" >&2; \
		exit 2; \
	fi
	@if ! printf '%s\n' "$(RUN_ALGORITHM_CONFIGS)" | tr ' ' '\n' | grep -Fxq "$(ALGORITHM)"; then \
		echo "Unsupported ALGORITHM='$(ALGORITHM)'. Valid Hydra configs: $(RUN_ALGORITHM_CONFIGS)" >&2; \
		exit 2; \
	fi
	@if ! printf '%s\n' "$(RUN_PROFILE_CONFIGS)" | tr ' ' '\n' | grep -Fxq "$(RUN_PROFILE)"; then \
		echo "Unsupported RUN_PROFILE='$(RUN_PROFILE)'. Valid profiles: $(RUN_PROFILE_CONFIGS)" >&2; \
		exit 2; \
	fi
	docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev \
		uv run --no-sync python -m thesis_rl.cli.train \
		env=scenarionet \
		obs=semantic_v3 \
		agent/planner/encoder=$(ENCODER) \
		agent/planner/decoder=mlp_encoded \
		reward=scalar_reward \
		scalarization=default \
		curriculum=scenario_acl_scenarionet \
		rulebook.version=4.7-final-implementation-complete \
		env.provider.strict=true \
		env.provider.allow_fallback=false \
		env.provider.source_probability.waymo=0.5 \
		env.provider.source_probability.pg=0.5 \
		env.config.num_scenarios=-1 \
		env.vectorized.enabled=$(if $(filter 1,$(NUM_ENVS)),false,true) \
		env.vectorized.num_envs=$(NUM_ENVS) \
		seed=$(SEED) \
		agent/planner/algorithm=$(ALGORITHM) \
		run_profile=$(RUN_PROFILE) \
		experiment.name=$(RUN_NAME) \
		$(RUN_OVERRIDES) \
		video.enabled=true

run: run-train

run-golden-rulebook:
	@test -f "$(GOLD_MANIFEST)" || (echo "Missing golden-suite manifest: $(GOLD_MANIFEST)" >&2; exit 2)
	docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev \
		uv run --no-sync python -m thesis_rl.cli.scenarios.golden_rulebook_trace \
		gold_suite.manifest_path=$(GOLD_MANIFEST_CONTAINER) \
		env.provider.kind=fixed_sequence \
		env.provider.repeat=false \
		env.provider.scenario_uids_file=$(GOLD_MANIFEST_CONTAINER) \
		env.split=train \
		curriculum=disabled \
		experiment.name=gold_rulebook_trace \
		video.enabled=true \
		video.save_manifest=true \
		video.save_trajectory_log=true
