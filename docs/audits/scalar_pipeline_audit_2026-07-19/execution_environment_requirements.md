# Execution Environment Requirements For Live Validation

## Observed Audit Host

- Architecture: `aarch64`.
- Kernel identity: `Linux gh200-1 6.2.0-1015-nvidia-64k`.
- Docker GPU container: `torch 2.9.1+cu128`, CUDA available, NVIDIA GH200 visible.
- Current capacity: `nvidia-smi` reported approximately 2 GiB free. A direct `torch.cuda.mem_get_info()` allocation probe raised CUDA out-of-memory. This is a transient shared-GPU capacity limit, not a repository defect.
- Consequence: containerized source/content checks and focused tests are usable; do not start learner smoke or training until sufficient free GPU memory is available. Do not alter pinned Torch/CUDA dependencies merely to accommodate this audit host.

## Intended Execution Environment

The portable intended execution target remains an `x86_64` Linux host with a compatible NVIDIA GPU, NVIDIA driver, Docker Engine with Compose v2, and NVIDIA Container Toolkit. This ARM64 GH200 host is also usable only with the repository's already provisioned compatible container image. The repository documents `TORCH_BACKEND=cu128` for modern NVIDIA/RTX 50xx hardware and `cu126` for legacy NVIDIA hardware. The protected ScenarioNet root must be mounted at `/workspace/data/scenarionet`; outputs must use a separate writable host directory.

Repository-independent provisioning requirements are: an x86-64 host; a supported NVIDIA driver for the selected PyTorch CUDA backend; working GPU container runtime; Docker-daemon access for the user; sufficient disk for the immutable dataset plus separate outputs; and the exact frozen dataset root available without renaming, copying, or modifying its source files.

## Commands To Run On The Intended Host

```bash
uname -m
# Expected: x86_64

# In .env, set host-local paths without changing repository source:
TORCH_BACKEND=cu128
HOST_DATA_DIR=/absolute/path/containing/scenarionet
HOST_OUTPUTS_DIR=/absolute/path/for/separate-outputs

make config-gpu
docker compose -f compose.yaml -f compose.gpu.yaml build
make gpu-check

# Read-only frozen-source existence and catalog reconstruction check.
docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev \
  uv run --no-sync python -c '
from thesis_rl.scenarios.frozen import load_frozen_index, verify_frozen_sources
p = load_frozen_index("data/scenarionet/frozen/scenario_selection_index.json")
print(len(verify_frozen_sources(p, "/workspace/data/scenarionet")))
'

# Focused currently authored ACL and transition-replay checks.
docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev \
  uv run --no-sync python -m pytest -q \
  tests/test_transition_replay_config.py tests/test_transition_replay_per.py \
  tests/test_transition_boundary.py tests/test_transition_replay_persistence.py \
  tests/test_scenario_acl_buffer.py tests/test_scenario_acl_config.py \
  tests/test_scenario_acl_mab.py tests/test_scenario_acl_usefulness.py

# Rulebook and representative baseline checks.
make rulebook-v2-check
make smoke-gpu
```

The final 48-reference manifest has now passed read-only content checks. The raw zero-policy S0 path has passed once for PG and once for Waymo, but the full scalar S0 runner/configuration remains to be registered and exercised. Do not substitute a generic training command.
