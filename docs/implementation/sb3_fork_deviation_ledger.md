# Stable-Baselines3 Fork Deviation Ledger

## Metadata

- Baseline specification: `RL-BASELINES v1.0`
- Fork path: `third_party/stable-baselines3`
- Fork commit: `4e6c3db1367a4cda96308bc6d0b80e63cd698828`
- Upstream comparison point: `6a196a60c7df3550ac5832caad54ef8dce9a6f31`
- Ledger status: `VERIFIED`
- Date: `2026-07-20`

The comparison point is the repository's parent commit immediately before the
project fork commit. The fork is based on Stable-Baselines3 `2.9.0`.

## Scientific deviations

| Area | Upstream behavior | Fork behavior | Scientific impact | Validation |
|---|---|---|---|---|
| TD3 critic loss | Unweighted per-sample critic losses are averaged. | When replay data provides finite importance-sampling weights, each sample loss is weighted before reduction; otherwise upstream reduction is retained. | PER importance correction changes the critic gradient while leaving actor loss and target construction unchanged. | `tests/test_td3_sb3_porting.py`; fork TD3 implementation |
| SAC critic loss | Unweighted per-sample critic losses are averaged. | When replay data provides finite importance-sampling weights, each sample loss is weighted before reduction; otherwise upstream reduction is retained. | PER importance correction changes the critic gradient while leaving actor and entropy losses unchanged. | `tests/test_sac_sb3_porting.py`; fork SAC implementation |
| Replay sample typing | Replay samples do not expose a typed optional weight field for these algorithms. | `ReplayBufferSamples` includes optional `weights`, defaulting to `None` for unprioritized replay. | The algorithm can consume PER IS weights without changing the unprioritized API contract. | `stable_baselines3/common/type_aliases.py`; replay tests |

## Effective baseline defaults outside the fork

The following are thesis configuration decisions, not fork deviations:

- TD3/SAC transition replay uses 3-step returns and proportional PER.
- Replay persistence is disabled by default because it adds disk and I/O cost;
  model-only restart is therefore distinct from stateful continuation.
- TD3 uses `batch_size=256`; its duration-profile warm-up and batch overrides
  are recorded in the authoritative RL-baselines specification.
- SAC uses `learning_starts=100` and `batch_size=256` as algorithm defaults,
  with duration-profile warm-up overrides.
- PPO uses the upstream rollout/update semantics with the approved project
  encoder and policy-head configuration.

These settings must be read from the resolved Hydra manifest for each run; this
ledger does not override configuration.

## Non-deviations checked

The fork does not change the TD3 target-policy smoothing formula, delayed actor
update cadence, SAC target update interval, automatic entropy tuning, PPO
clipped objective, PPO value clipping default (`None`), termination/truncation
boundary, or deterministic evaluation semantics. Those are governed by the
approved baseline specification and the resolved run manifest.
