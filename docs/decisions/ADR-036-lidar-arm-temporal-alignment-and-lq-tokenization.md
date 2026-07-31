# ADR-036: LiDAR Arm Temporal Alignment and LQ Tokenization

- Status: Approved
- Date: 2026-07-30
- Decision owner: thesis repository maintainer
- Approval date: 2026-07-30
- Affected specifications:
  - `docs/specifications/observation_lidar_v2.0_specification.md`, ID
    `OBS-LIDAR-V2.0` (new)
  - `docs/specifications/encoder_v1.4_specification.md`, ID `ENC-V1.4` (new)
  - `docs/specifications/observation_v1.1_specification.md`, ID `OBS-V1.1`
    (unchanged; retained for reproducibility)
- Affected ExecPlan:
  `docs/implementation/lidar_arm_temporal_alignment_and_lq_tokenization_v2.0_exec_plan.md`
  (`DEC-001` through `DEC-010`)

## Context

The LiDAR observation arm stacked five causal 308-wide frames (0.4 s,
`stacked_lidar_state`). That depth predates Rulebook v2's timers and was
chosen when no temporal rule existed, so it carries no derivation. The
semantic arm's `compliance_history_length = 21` (2.0 s at 10 Hz), by contrast,
is exactly `DASHED_TCAP_S`-derived. Comparing the two arms on the five-frame
baseline therefore compared a designed window against an accidental one, not a
fair baseline-vs-encoder or semantic-vs-perception comparison. Separately, the
LiDAR arm had no latent-query encoder path, so any comparison against
`semantic_v3 + lq_v3` confounded the observation with the encoder choice.

## Decision

1. **Window depth**: 21 frames (2.0 s at 10 Hz), derived from
   `DASHED_TCAP_S = 2.0` s
   (`src/thesis_rl/rulebook/v2/components/road.py`), matching the semantic
   arm's existing window exactly.
2. **New contract, not a redefinition**: a new observation mode
   `stacked_lidar_v2` (`OBS-LIDAR-V2.0`, `D = 6489`) is introduced alongside
   the unchanged `stacked_lidar_state` (`D = 1540`), mirroring how `OBS-V1.3`
   preserved `semantic_v2`.
3. **Warm-up mask, not replication**: at 21 frames, the legacy convention of
   replicating the first frame during warm-up becomes ambiguous (20 replicated
   steps = 2.0 s, indistinguishable from a genuinely static 2.0 s history,
   exactly the discrimination the Rulebook timers require). `stacked_lidar_v2`
   adds a 21-wide validity mask and zero-fills absent frames instead.
4. **Channel-stacking, not per-frame tokens**: the temporal axis enters the
   tokenizer by channel-stacking each group across the 21 frames into one
   wider token, keeping the token count at 38 (not 21x). This keeps the
   frozen latent-query core (16 latents, depth 4) untouched.
5. **Partial STECA adoption**: only sector tokenization (30 sectors of 8 rays)
   and a circular sine/cosine positional encoding over the LiDAR ring are
   adopted from STECA. STECA's two-stage attention (Stage-I sector
   self-attention, Stage-II ego-centric cross-attention) is NOT adopted.
6. **No neighbor-slot identity embedding**: the nearby-vehicle block (4
   *k*-nearest slots) is channel-stacked without a per-slot identity
   embedding, following the `ADR-026` precedent.
7. **`yellow_must_stop` remains unexposed**: this Rulebook latch has an
   unbounded lookback and is forbidden as a policy input by `ADR-033`. This
   ADR does not reopen that decision; it records the resulting gap as a
   declared, scoped (to the `signal` Rulebook component) limitation of the
   LiDAR arm, to be quantified on the catalog and reported as a stratified
   robustness check rather than argued as a modeling shortcoming.
8. **Replay-buffer unification**: `buffer_size` is unified at `300000` across
   `sac_sb3` and `td3_sb3` (previously `1000000` vs `300000`). The Automatic
   Curriculum Learning (ACL) makes the data distribution non-stationary by
   design; a buffer holding 67% of a 1.5M-step run retains superseded
   curriculum stages and confounded any SAC/TD3 comparison on the prior
   asymmetric configuration. This deviates from V-Max's `1e6`, which trains
   without a curriculum.
9. **`fast` diagnostic profile buffer override**: `run_profile=fast`
   (120000-step budget) now overrides `buffer_size` to `24000`, preserving the
   20% buffer/budget ratio the `thesis` profile establishes (`300000/1500000`),
   so the diagnostic profile exercises a replay regime representative of
   production rather than a buffer that never evicts.

## Rationale

Every quantitative choice above is derived from a verified repository constant
(`DASHED_TCAP_S`, the control period, the machine's verified 417 GiB available
RAM) or from an already-approved precedent (`ADR-026`, `ADR-033`), rather than
from a new judgment call with no anchor. This keeps the LiDAR arm a
literature-grounded, fair reference for the semantic arm: both now cover
identical Rulebook-derived temporal windows, and the observation-vs-encoder
confound is resolved by giving the LiDAR arm its own latent-query path.

The machine's verified 417 GiB of available RAM (against ~15.6 GB for a
21-frame, 300k-capacity replay buffer) removes any memory justification for a
strided or mixed-depth compromise, which would otherwise have required
declaring a timer-reconstruction error of up to 0.2 s. Uniform 21-frame
stacking is therefore both derivable and affordable.

## Consequences

- `stacked_lidar_state` (1540) and its `encoder_v1.0` MLP path are unchanged
  and remain available for reproducibility.
- `semantic_v3` / `ENC-V1.3` are unchanged.
- No existing checkpoint migrates to the new observation or encoder; this is
  by design, consistent with `ENC-V1.3` §5 and `OBS-V1.3` §7.
- The LiDAR arm inherits the same three declared representational limitations
  recorded in `OBS-LIDAR-V2.0` §5: unreconstructible `yellow_must_stop`,
  non-identity-stable neighbor slots, and ego-relative (not world-fixed)
  sector bearings under yaw rate.
- While validating this ADR's implementation, three pre-existing defects were
  found and fixed with regression tests in the (declared frozen, reused
  unchanged) `CausalLidarFrameBuilder` and `RayNoiseWrapper`. Two used
  `isinstance(config, dict)` to validate MetaDrive vehicle/sensor
  configuration, which a real MetaDrive `Config` object (not a `dict`
  subclass) always failed; the fix duck-types on `.get` instead of the
  nominal type. The third was in `_lidar_blocks`: `metadrive.component.
  sensors.lidar.Lidar.perceive` returns a plain `(cloud_points,
  detected_objects)` tuple, not the `detect_result` namedtuple
  `DistanceDetector.perceive` returns for the side/lane-line detectors, so
  `getattr(result, "cloud_points", result)` silently fell back to the whole
  tuple (crashing on the array conversion) and `detected_objects` was always
  read as `None` (silently emptying the nearby-vehicle block); the fix
  unpacks the tuple positionally. None of the three had ever been exercised
  outside unit tests with hand-built mocks that happened to paper over the
  real MetaDrive return shapes, so they silently blocked any real training
  run with either the old or the new stacked LiDAR observation. All three are
  bug fixes with no semantic effect on any *correctly* produced observation
  value; end-to-end smoke training now passes for both
  `stacked_lidar_state` and `stacked_lidar_v2`.

## Alternatives considered

**Keep the five-frame window as a designed hypothesis.** Rejected: it has no
derivation and predates the Rulebook timers it would need to justify against.

**Per-frame tokens (21x38 = 798 tokens).** Rejected: inflates attention cost
and forces the encoder to learn temporal alignment the channel-stacking design
gives it for free as a layer-0 linear operation.

**Adopt STECA's full two-stage attention.** Rejected: supported only by an
unreplicated single-seed comparison with an internal numeric inconsistency,
and V-Max's own results show the transformer family (LQ, LQH, MTR, Wayformer)
on a plateau, so the additional architecture is not evidenced to help.

**Engineer per-slot neighbor identity (e.g. via tracking).** Rejected per the
`ADR-026` precedent: the ranking that fills the slots has no stable identity to
engineer around; declaring the limitation is proportionate given the block is
a small fraction of the observation.

**Keep `optimize_memory_usage=True` to reduce buffer memory.** Rejected: it is
forbidden by construction for PER + n-step sampling
(`src/thesis_rl/sb3_extensions/replay/config.py`,
`replay/prioritized.py`), and the verified machine memory removes the need
for it regardless.
