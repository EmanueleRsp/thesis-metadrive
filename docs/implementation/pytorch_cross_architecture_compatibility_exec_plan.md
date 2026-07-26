# PyTorch Cross-Architecture Compatibility ExecPlan

## 1. Metadata

- Plan ID: `PLAN-ENV-TORCH-001`
- Feature: reproducible PyTorch CUDA installation across supported Linux CPU
  architectures
- Authority: user request on 2026-07-19; no dedicated scientific specification
  applies
- Status: `IN_PROGRESS`
- Created: 2026-07-19
- Last updated: 2026-07-19
- Related ADRs: `ADR-013-pytorch-cross-architecture-validation.md`
- Owner: Codex

## 2. Objective And Scope

Enable the Docker development image to install one pinned, CUDA-enabled PyTorch
release on both Linux `x86_64` and Linux `aarch64` hosts when that release has a
matching official wheel. Preserve explicit backend selection (`cpu`, `cu126`,
or `cu128`), the platform-neutral `uv.lock`, the existing GPU smoke test, and
the current x86_64/RTX 50xx workflow.

This plan does not claim support for every NVIDIA GPU or driver. A valid wheel,
a driver compatible with the selected CUDA wheel, an NVIDIA runtime, and a GPU
architecture supported by that wheel remain mandatory. It does not add a
runtime fallback, change experiment configurations, migrate checkpoints, or
change a dataset or scientific contract.

## 3. Authoritative Requirements

| ID | Requirement | Source |
|---|---|---|
| `REQ-ENV-TORCH-001` | A supported Linux ARM64 GPU host must not fail solely because the selected official PyTorch wheel lacks an ARM64 tag. | User request |
| `REQ-ENV-TORCH-002` | The existing x86_64/RTX 50xx CUDA workflow must remain supported. | User request and current public workflow |
| `REQ-ENV-TORCH-003` | Dependency selection must remain pinned and documented for reproducibility. | `AGENTS.md` reproducibility requirements |
| `REQ-ENV-TORCH-004` | Setup must report incompatible architecture/backend/version combinations before or at the build boundary with an actionable diagnostic. | User request and existing `setup.sh` contract |

## 4. Current Repository Analysis

- `VERIFIED`: `Dockerfile` installs `torch==${TORCH_VERSION}` from the selected
  `TORCH_BACKEND` index after `uv sync`; PyTorch is excluded from `uv.lock`.
- `VERIFIED`: `compose.yaml` forwards `TORCH_VERSION` and `TORCH_BACKEND` from
  `.env`; its defaults are `2.8.0` and `cu128`.
- `VERIFIED`: `.env.example`, `README.md`, and
  `docs/setup/environment_setup.md` document `cu126` and `cu128` manual
  selection.
- `VERIFIED`: on the current Linux `aarch64` host, the build fails because the
  official `torch 2.8.0+cu128` index has no matching ARM64 Python 3.10 wheel.
- `VERIFIED`: the official PyTorch wheel indexes list Python 3.10 Linux ARM64
  and x86_64 wheels for `torch 2.9.1` with both `cu126` and `cu128`.
- `VERIFIED`: the bundled Stable-Baselines3 requires `torch>=2.8,<3.0`; 2.9.1
  satisfies that declared constraint.
- `VERIFIED`: MetaDrive has no direct Torch dependency; ScenarioNet's selected
  base profile has no Torch dependency. Their editable installations completed
  in the candidate image build.
- `VERIFIED`: both candidate ARM64 builds (`2.9.1+cu126` and `2.9.1+cu128`)
  installed Torch and completed `import torch`, but the existing `uv pip check`
  then failed on `nvidia-cusparselt-cu12==0.7.1` with "built for a different
  platform".
- `VERIFIED`: NVIDIA publishes that exact cuSPARSELt version as a Linux ARM64
  wheel and declares Linux Arm64 plus SM 9.0 support. The observed failure is
  therefore a `uv pip check` platform-validation issue or metadata
  interpretation issue, not evidence of an x86_64 CUDA package installation.
- `INFERRED`: the current project imports and test suite will be compatible
  with 2.9.1. This must be established by the mandatory test matrix, not
  assumed.

## 5. Assumptions And Invariants

- Python remains `>=3.10,<3.11` and the image Python remains 3.10.20.
- The selected PyTorch version and backend are build inputs, never silently
  changed at runtime.
- CPU, CUDA 12.6, and CUDA 12.8 remain explicit, inspectable build profiles.
- The container CUDA smoke test must perform a real tensor operation.
- Experiment provenance must continue to expose the installed Torch version
  through the existing environment/import logs; no numerical equivalence across
  different architectures is implied.

## 6. Decisions And Approval Gates

| ID | Category | Issue | Alternatives | Recommendation | Impact | Status |
|---|---|---|---|---|---|---|
| `DEC-ENV-TORCH-001` | specification clarification | Select the shared pinned Torch release. | A: retain 2.8.0 and reject ARM64; B: pin 2.9.1 for all supported hosts; C: architecture-specific versions. | B: `2.9.1` on both architectures and existing backends. | Dependency version, reproducibility baseline, validation. | Approved by ADR-013 |
| `DEC-ENV-TORCH-002` | implementation detail | Determine whether backend selection should become automatic. | A: keep explicit `TORCH_BACKEND`; B: add auto-detection based on GPU/driver. | A: retain explicit selection and strengthen validation. | Public `.env` interface and failure behavior. | Approved by ADR-013 |

The requested automatic selection is deliberately not implemented: compute
capability alone cannot determine a compatible CUDA wheel,
because host architecture, Python ABI, wheel availability, and driver version
also constrain the result. Explicit profiles are reproducible and can be
validated safely.

## 7. Proposed Design

After `DEC-ENV-TORCH-001` approval, update the single shared Torch pin from
2.8.0 to 2.9.1 in Docker/Compose/example configuration and documentation.
Keep `TORCH_BACKEND` explicit. Extend `setup.sh` so architecture/backend/version
preflight identifies an unavailable official-wheel combination with the exact
remediation, rather than presenting ARM64 as an x86_64-only warning.

The initial compatibility matrix is:

| Host image architecture | CPU | `cu126` | `cu128` |
|---|---|---|---|
| Linux `x86_64` | `torch 2.9.1` official wheel | `torch 2.9.1` official wheel | `torch 2.9.1` official wheel |
| Linux `aarch64` | Validate during implementation | `torch 2.9.1` official wheel | `torch 2.9.1` official wheel |

An unsupported architecture or profile must fail explicitly, without a
best-effort fallback. GPU compute capability remains a guard for Blackwell:
`sm_120` requires `cu128`.

## 8. Traceability

| Requirement | Acceptance criteria | Implementation | Tests | Status |
|---|---|---|---|---|
| `REQ-ENV-TORCH-001` | `AC-ENV-TORCH-001`, ARM64 `cu128` build resolves a matching wheel | `Dockerfile`, `compose.yaml` | Docker ARM64 build and GPU smoke | Planned |
| `REQ-ENV-TORCH-002` | `AC-ENV-TORCH-002`, x86_64 profile remains selectable and validated | `setup.sh`, docs | shell test/preflight and x86_64 build or CI evidence | Planned |
| `REQ-ENV-TORCH-003` | `AC-ENV-TORCH-003`, one documented fixed version is used | `.env.example`, docs | static assertions and config expansion | Planned |
| `REQ-ENV-TORCH-004` | `AC-ENV-TORCH-004`, unsupported profile has actionable failure | `setup.sh` | focused shell tests/manual fixture | Planned |

## 9. Test Strategy Defined Before Implementation

| ID | Level | Behavior | Fixture/input | Expected result | Requirement |
|---|---|---|---|---|---|
| `TEST-ENV-TORCH-001` | Static | Supported profile values | default compose expansion | pin/backend are passed to the dev build | `REQ-ENV-TORCH-003` |
| `TEST-ENV-TORCH-002` | Build | ARM64 CUDA wheel resolution | current `aarch64`, `cu128` | build installs the pinned official ARM64 wheel | `REQ-ENV-TORCH-001` |
| `TEST-ENV-TORCH-003` | Smoke | CUDA execution | current ARM64 NVIDIA container | `torch.cuda.is_available()` and tensor operation pass | `REQ-ENV-TORCH-001` |
| `TEST-ENV-TORCH-004` | Regression | Blackwell protection | deterministic `compute_cap=12.0`, `cu126` fixture | actionable failure requiring `cu128` | `REQ-ENV-TORCH-002` |
| `TEST-ENV-TORCH-005` | Integration | Existing package compatibility | container imports and pytest | imports, dependency check, and applicable tests pass | `REQ-ENV-TORCH-002` |

Commands: `make config`, `make config-gpu`, `bash -n setup.sh`,
`./setup.sh --verify --gpu`, `git diff --check`, and, on the x86_64 machine,
`./setup.sh --verify --gpu`. No repository command currently provides
architecture-emulated Docker testing; cross-architecture evidence needs the
two real hosts or CI runners.

## 10. Milestones

- [x] Record the failing platform tag and inspect official wheel availability.
- [x] Inspect MetaDrive, ScenarioNet, and Stable-Baselines3 declared Torch
  constraints and run candidate ARM64 dependency builds.
- [x] Obtain approval for `DEC-ENV-TORCH-001` and `DEC-ENV-TORCH-002` through ADR-013.
- [x] Investigate and approve a non-weakening resolution for the ARM64
  `uv pip check` false-positive/metadata-validation failure through ADR-013.
- [x] Add acceptance/regression coverage before production edits.
- [x] Update Docker, Compose, setup validation, and user documentation.
- [x] Validate CUDA execution on the current ARM64 GPU host after GPU memory
  is released. Re-run 2026-07-26: host GPU memory is now free (58,549 MiB
  free of 97,871 MiB, versus 2,095 MiB free of 480 GiB previously). Ran the
  real CUDA tensor smoke directly (`torch.zeros(1000, 1000, device="cuda")`,
  matrix multiply, `torch.cuda.synchronize()`) via
  `docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev` —
  passed: `torch 2.9.1+cu128`, `cuda available: True`, tensor op completed.
  Also re-ran `scripts/validate_torch_install.py --torch-version 2.9.1
  --require-cusparselt` (PASS, ARM64 ELF-validated) and
  `tests/test_validate_torch_install.py` (`3 passed`). The x86_64/RTX 50xx
  host validation remains open — no such host is available in this
  session; must be run separately when that hardware is accessible.
- [ ] Reconcile requirements, update the index, and review the final diff.
  Blocked only on the still-open x86_64/RTX 50xx host validation above.

## 11. Progress And Findings Log

- 2026-07-19: `make verify-gpu` on Linux `aarch64` failed while resolving
  `torch==2.8.0` with `cu128`; the error reported available platform tags only
  for `manylinux_2_28_x86_64` and `win_amd64`.
- 2026-07-19: official index inspection found `torch 2.9.1` Python 3.10 wheels
  for Linux `aarch64` and `x86_64` under both `cu126` and `cu128`.
- 2026-07-19: candidate Docker builds with `torch 2.9.1+cu128` and
  `torch 2.9.1+cu126` on ARM64 each installed Torch, all declared project
  packages, and imported Torch successfully. Both failed only at `uv pip check`
  because it labelled NVIDIA's ARM64 `nvidia-cusparselt-cu12==0.7.1` as built
  for a different platform. NVIDIA's published wheel metadata documents an
  ARM64 wheel and SM 9.0 support, so this must be resolved before the profile
  can be accepted.
- 2026-07-19: the first full ARM64 validation built the image and passed the
  build-time Torch/ELF validation. Its runtime import smoke exposed a shell
  interpreter-path defect in the new validator script; the script invoked the
  system Python rather than the project `uv` environment. The script was
  corrected to use `uv run --no-sync python` and requires regression validation.
- 2026-07-19: the first CUDA tensor smoke failed with `cudaErrorMemoryAllocation`.
  This is an external GPU-capacity condition and must be retried after checking
  host GPU allocation. The full pytest run had seven pre-existing unrelated
  failures in Hydra preset expectations and ScenarioNet configuration reset;
  the new focused validator tests passed.
- 2026-07-19: after the import-script correction, an incremental ARM64 image
  build, project import smoke, Torch version check, and cuSPARSELt ARM64 ELF
  validation all passed. `nvidia-smi` reported only 2,095 MiB free of 480 GiB;
  the real CUDA tensor smoke again failed with `cudaErrorMemoryAllocation`.
  This is an external allocation blocker, not a dependency-resolution failure.
- 2026-07-19: review found that the native cuSPARSELt payload is not installed
  by the CPU Torch profile. The validator was amended to require this payload
  only for CUDA profiles, preserving CPU/CI compatibility.
- 2026-07-19: the final ARM64 CUDA image rebuild passed with the backend-aware
  validator arguments, confirming that the completed Dockerfile incorporates
  the approved validation path.

## 12. Deviations

| ID | Original contract | Actual change | Reason | Approval | Affected tests/docs |
|---|---|---|---|---|---|
| `DEV-ENV-TORCH-001` | `uv pip check` must pass after Torch installation. | On ARM64 only, accept its single documented cuSPARSELt platform false positive after validating the installed ELF payload and Torch version. | NVIDIA publishes a matching ARM64 wheel; broad suppression would hide unrelated conflicts. | ADR-013 | `scripts/validate_torch_*`, Docker build, import smoke |

## 13. Files

| Path | Action | Purpose |
|---|---|---|
| `Dockerfile` | Planned modification | Pin the approved cross-architecture Torch release and improve profile validation. |
| `compose.yaml` | Planned modification | Forward the approved default pin. |
| `.env.example` | Planned modification | Document the approved reproducible default. |
| `setup.sh` | Planned modification | Provide platform/profile diagnostics. |
| `README.md` | Planned modification | Update user-facing environment matrix. |
| `docs/setup/environment_setup.md` | Planned modification | Document supported architectures and explicit backend choice. |
| `scripts/validate_torch_install.py` | Added | Validate the installed Torch version and cuSPARSELt native architecture. |
| `scripts/validate_torch_environment.sh` | Added | Preserve `uv pip check` while tightly handling the documented ARM64 false positive. |
| `tests/test_validate_torch_install.py` | Added | Deterministic regression coverage for native-payload validation. |
| `docs/decisions/ADR-013-pytorch-cross-architecture-validation.md` | Added | Approved environment compatibility decision. |
| `docs/implementation/pytorch_cross_architecture_compatibility_exec_plan.md` | Added | Decision, traceability, and validation record. |

## 14. Validation Results

| Command | Result | Date | Notes and evidence |
|---|---|---|---|
| `make verify-gpu` | FAIL | 2026-07-19 | Expected regression evidence: `torch 2.8.0+cu128` has no Linux ARM64 wheel. |
| Official PyTorch wheel index inspection | PASS | 2026-07-19 | Confirmed Linux ARM64 and x86_64 Python 3.10 wheels for 2.9.1 `cu126`/`cu128`. |
| Candidate ARM64 build, `TORCH_VERSION=2.9.1`, `TORCH_BACKEND=cu128` | FAIL | 2026-07-19 | Torch installation and import passed; existing `uv pip check` rejected the published ARM64 cuSPARSELt wheel. |
| Candidate ARM64 build, `TORCH_VERSION=2.9.1`, `TORCH_BACKEND=cu126` | FAIL | 2026-07-19 | Same post-install `uv pip check` failure as `cu128`; not a backend-selection solution. |
| ARM64 `./setup.sh --verify --gpu`, first run | PARTIAL | 2026-07-19 | Build passed and accepted the ARM64 ELF-validated cuSPARSELt exception. Import smoke exposed a fixed script-path defect; CUDA smoke hit external GPU OOM; pytest had 7 unrelated existing failures (605 passed, 1 skipped). |
| `tests/test_validate_torch_install.py` | PASS | 2026-07-19 | 3 deterministic validator regression tests passed. |
| Focused Ruff, shell syntax, Compose, and whitespace checks | PASS | 2026-07-19 | New Python and shell validators pass lint/format and Compose expansion. |
| ARM64 incremental Docker build and import smoke | PASS | 2026-07-19 | `thesis_rl`, MetaDrive, SB3, and `torch 2.9.1+cu128` imported; ARM64 cuSPARSELt payload validated. |
| Final ARM64 Docker build | PASS | 2026-07-19 | The completed Dockerfile passed `torch 2.9.1+cu128` installation and backend-aware ARM64 ELF validation. |
| ARM64 CUDA tensor smoke | BLOCKED | 2026-07-19 | GH200 had 2,095 MiB free of 480 GiB; `cudaErrorMemoryAllocation`. Retry after GPU memory is released. |
| ARM64 CUDA tensor smoke, retry | PASS | 2026-07-26 | GPU memory now free (58,549 MiB free of 97,871 MiB). `torch.zeros(1000,1000,device="cuda")`, matrix multiply, `torch.cuda.synchronize()` all completed via `docker compose -f compose.yaml -f compose.gpu.yaml run --rm dev`. |
| `scripts/validate_torch_install.py --torch-version 2.9.1 --require-cusparselt`, retry | PASS | 2026-07-26 | `torch=2.9.1+cu128 cuSPARSELt=.../libcusparseLt.so.0 architecture=aarch64` |
| `tests/test_validate_torch_install.py`, retry | PASS | 2026-07-26 | `3 passed` |

## 15. Final Reconciliation

`REQ-ENV-TORCH-001` through `REQ-ENV-TORCH-004` are implemented. Build,
import, and CUDA tensor-op validation now all pass on the ARM64 host
(2026-07-26 retry, after the previously blocking external GPU memory
pressure cleared). The x86_64/RTX 50xx host validation remains the only
open item — no such host was available in this session; it must be run
there before this plan can be marked `VERIFIED`. The repository-wide
pytest result contains seven unrelated existing failures (as of the
2026-07-19 full run); the focused regression suite passes. The change is
ready for dependency/import use on ARM64, pending only the x86_64 host
check.
