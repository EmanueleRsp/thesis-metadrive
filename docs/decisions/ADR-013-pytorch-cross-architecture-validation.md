# ADR-013: Pinned Cross-Architecture PyTorch And ARM64 CUDA Validation

- Status: `Approved`
- Date: `2026-07-19`
- Decision owner: thesis repository maintainer
- Approval date: `2026-07-19`
- Supersedes: `NONE`
- Affected specifications: none; environment compatibility only
- Affected ExecPlans:
  `docs/implementation/pytorch_cross_architecture_compatibility_exec_plan.md`

## Context

The pinned `torch==2.8.0` CUDA wheels used by the Docker development image do
not include Linux ARM64 builds, blocking the current NVIDIA ARM64 host before
the project can be imported or tested. The same repository must retain the
Linux x86_64 CUDA path needed by an RTX 50xx host.

Official PyTorch 2.9.1 CUDA 12.6 and CUDA 12.8 indexes provide Python 3.10
wheels for both Linux ARM64 and Linux x86_64. The bundled Stable-Baselines3
declares `torch>=2.8,<3.0`; MetaDrive and the selected ScenarioNet base profile
have no direct Torch dependency.

On ARM64, the existing `uv pip check` rejects the installed
`nvidia-cusparselt-cu12==0.7.1` package as built for a different platform even
though NVIDIA publishes its Linux ARM64 wheel and declares ARM64 support. This
is a known, narrowly evidenced validator/metadata interpretation failure, not
evidence that the installed payload is x86_64.

## Decision

1. Pin the Docker development image to `torch==2.9.1` for every supported
   architecture. Keep `TORCH_BACKEND` explicit: `cu128` is required for
   Blackwell/RTX 50xx, while `cu126` remains available for other compatible
   NVIDIA hosts and `cpu` remains available without NVIDIA.
2. Do not introduce automatic backend selection. GPU compute capability alone
   cannot select a valid wheel because architecture, Python ABI, available
   wheel tags, and driver compatibility also constrain the result.
3. Continue to run `uv pip check`. On Linux ARM64 only, accept precisely its
   one documented cuSPARSELt platform false positive after an additional
   validator proves that the installed native cuSPARSELt library has the ARM64
   ELF machine code and that the installed Torch version equals the selected
   pin. Any other `uv pip check` result remains a build failure.
4. Retain the existing runtime CUDA tensor-operation smoke check. It verifies
   that the selected CUDA runtime is usable on the actual GPU; it is not
   replaced by metadata inspection.

## Alternatives Considered

| Alternative | Reason not selected |
|---|---|
| Retain Torch 2.8.0 and reject ARM64 | It excludes a supported current host solely through an unavailable wheel. |
| Use architecture-specific Torch versions | It fragments experimental provenance without a compatibility need because 2.9.1 has official wheels for both hosts. |
| Ignore `uv pip check` on ARM64 | It could hide unrelated dependency conflicts. |
| Use only `pip check` on ARM64 | It would silently weaken the existing validation instead of documenting and tightly constraining the exception. |
| Choose CUDA automatically from compute capability | It cannot account for wheel tags and driver compatibility and would make builds less reproducible. |

## Consequences

The supported reproducible Torch baseline becomes 2.9.1. Existing environments
must rebuild after setting their local `TORCH_VERSION=2.9.1`. The ARM64
exception is limited to one dependency and one exact validator output; an
unexpected dependency failure remains blocking. Numerical equivalence across
hardware architectures is not implied, and experimental artifacts must retain
their installed software provenance.

## Validation And Traceability

This decision covers `REQ-ENV-TORCH-001` through `REQ-ENV-TORCH-004` and the
acceptance matrix in `PLAN-ENV-TORCH-001`. Mandatory validation includes the
new deterministic ELF-validator unit tests, Docker build, import/dependency
smoke check, real CUDA tensor operation, applicable project tests, and the
same GPU verification command on the x86_64/RTX 50xx host.

## Approval Record

- Approved by: `user`
- Approval evidence: user message `"quindi puoi procedere a risolvere il
  problema? Se sì procedi pure"` on 2026-07-19.
