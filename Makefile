.PHONY: setup verify verify-gpu build build-gpu build-waymo install-gcloud waymo-auth waymo-inventory waymo-convert waymo-pipeline scenarionet-pipeline up up-gpu shell test gpu-check smoke smoke-gpu config config-gpu

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
	docker compose -f compose.yaml -f compose.waymo.yaml --profile waymo build waymo-converter

install-gcloud:
	bash scripts/install_gcloud.sh

waymo-auth:
	@command -v gcloud >/dev/null 2>&1 || (echo "gcloud CLI is required; install it from https://cloud.google.com/sdk/docs/install" >&2; exit 2)
	gcloud init $(GCLOUD_INIT_FLAGS)

waymo-inventory:
	bash scripts/inspect_waymo_dataset.sh

waymo-convert:
	@test -n "$(WAYMO_RAW_DATA_PATH)" || (echo "WAYMO_RAW_DATA_PATH is required" >&2; exit 2)
	WAYMO_RAW_DATA_PATH="$(WAYMO_RAW_DATA_PATH)" docker compose -f compose.yaml -f compose.waymo.yaml --profile waymo run --rm waymo-converter \
		--raw-data-path /workspace/waymo_raw \
		$(if $(DATABASE_PATH),--database-path "$(DATABASE_PATH)",) \
		$(if $(NUM_WORKERS),--num-workers "$(NUM_WORKERS)",) \
		$(if $(NUM_FILES),--num-files "$(NUM_FILES)",) \
		$(if $(OVERWRITE),--overwrite,)

waymo-pipeline:
	bash scripts/prepare_waymo.sh

scenarionet-pipeline:
	bash scripts/prepare_scenarionet_dataset.sh

up:
	docker compose up -d

up-gpu:
	docker compose -f compose.yaml -f compose.gpu.yaml up -d

shell:
	docker compose exec dev bash

test:
	docker compose run --rm dev uv run --no-sync python -m pytest -q

gpu-check:
	docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(); x = torch.tensor([1.0], device="cuda"); assert (x * 2).item() == 2.0; print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")'

smoke:
	docker compose run --rm dev uv run --no-sync python -m thesis_rl.cli.train --config-name presets/test/smoke_train

smoke-gpu:
	docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev uv run --no-sync python -m thesis_rl.cli.train --config-name presets/test/smoke_train
