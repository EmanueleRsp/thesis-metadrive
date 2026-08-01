# ADR-045: Configurable Signal-Camera Range/FOV/Height

- Status: `APPROVED`
- Date: 2026-08-01
- Decision owners: thesis repository maintainer
- Approval evidence: explicit user approval in the current task on
  2026-08-01, after the deviation from OBS-V1.2 SS6.2 and its consequences
  were presented
- Related spec: `docs/specifications/observation_v1.2_specification.md`
  SS6.2

## Context

While auditing junction perception, the user asked whether `signal_fov_degrees`
in `conf/obs/semantic_v3.yaml` had any effect. Investigation found that
`SymbolicSignalVisibilityAdapter` (`src/thesis_rl/envs/observations/perception.py`)
enforced the OBS-V1.2 SS6.2 baseline (range 80 m, horizontal FOV 65 degrees,
camera height 1.2 m) as hard-coded constants, and separately validated any
constructor argument against that exact triple, raising `ValueError`
otherwise. `conf/obs/semantic_v3.yaml`'s `signal_range_m` /
`signal_fov_degrees` / `signal_camera_height_m` keys were not read by any
call site, so editing them silently did nothing.

The immediate fix (same task) threaded the YAML values through
`factory.py` -> `thesis_scenario_env.py` -> the batch builder ->
`mapped_signal_visibility` -> `SymbolicSignalVisibilityAdapter`, but kept the
adapter's `ValueError` guard, so a config value other than 80/65/1.2 would
now be *reachable* but still rejected. The user judged that guard pointless
now that the wiring exists: an experimenter should be free to study a
different sensor placement or field of view without editing source code.

OBS-V1.2 SS6.2 states the 80 m / 65-degree / 1.2 m baseline as a normative
`MUST`. Making it configurable is therefore a deviation from an approved
specification's acceptance behavior, which `AGENTS.md`'s Decision And Change
Control requires to be approved and recorded here rather than silently
implemented.

## Decision

`SymbolicSignalVisibilityAdapter` no longer requires range/FOV/camera-height
to equal the OBS-V1.2 baseline. It keeps the defaults at 80 m / 65 degrees /
1.2 m (so any caller that does not override them, including all currently
approved experiments, reproduces the SS6.2 baseline unchanged) and only
validates that the configured values are physically sane (`range_m > 0`,
`0 < horizontal_fov_deg <= 360`).

`conf/obs/semantic_v3.yaml`'s `signal_range_m` / `signal_fov_degrees` /
`signal_camera_height_m` are the effective source of truth for this geometry
from now on; the OBS-V1.2 SS6.2 numbers are the documented default, not a
hard limit.

## Consequences

Positive:

- `conf/obs/semantic_v3.yaml` now does what it visually appears to do; a run
  manifest that records this file also records the true signal-perception
  geometry used;
- experiments can study signal-detection robustness under a narrower/wider
  FOV or a different camera height without a code change.

Negative:

- OBS-V1.2 SS6.2's "MUST" range/FOV/height clause is no longer a hard
  invariant of the implementation; a run using a non-default value is no
  longer OBS-V1.2-conformant in the strict sense of that clause, only
  conformant with "OBS-V1.2 as amended by ADR-045". Any comparison across
  runs MUST check the recorded config value, not assume the baseline.
- The default remains 80/65/1.2, so no existing experiment's behavior
  changes unless its config is edited.

## Alternatives rejected

- Keep the `ValueError` guard (status quo before this ADR): rejected by the
  user as pointless once the wiring exists — it would let an experimenter
  edit the YAML, see no error, and still get the baseline geometry, which is
  worse than either a hard error or real effect.
- Silently drop the guard without recording a decision: rejected because it
  contradicts a normative `MUST` clause in an approved specification without
  approval or traceability, per `AGENTS.md`.
