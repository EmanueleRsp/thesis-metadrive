.PHONY: setup verify verify-gpu build build-gpu up up-gpu shell test gpu-check smoke smoke-gpu config config-gpu

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
