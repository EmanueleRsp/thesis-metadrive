"""Microbenchmark for the semantic latent-query encoder variants.

Answers one question: how much of a training step is spent inside the encoder?

It reports, for `lq_v3` (production), `lq_v3_lite` and `lq_v3_micro`
(diagnostic-only):

* exact trainable parameter counts, broken down by functional group;
* forward latency at the action-selection batch;
* forward+backward latency at the learner batch;
* peak allocated memory;
* the derived per-environment-step encoder cost, composed from the measured
  latencies and the SAC pass structure resolved from the repository
  configuration.

Methodology notes that matter for reading the numbers:

* the three encoders are timed **round-robin** rather than one after another,
  so that a contended GPU perturbs all three equally and the comparison between
  them survives even when the absolute numbers do not;
* every timed region is preceded by warm-up iterations and bracketed by
  `torch.cuda.synchronize()` on CUDA;
* the same input batch and the same validity masks are used for all variants;
* the median and the interquartile range are reported, not the mean, because a
  contended device produces a heavy right tail.

This measures the encoder in isolation. It does **not** measure MetaDrive, the
semantic observation construction, worker communication or replay sampling, so
the per-step share it derives is an estimate whose denominator comes from a
previously recorded run rather than from a live profile. See `--help`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from thesis_rl.agent.planners.encoders.factory import build_encoder
from thesis_rl.contracts.observation_schema import SemanticObservationSchemaV12

REPO_ROOT = Path(__file__).resolve().parents[1]
ENCODER_CONF_DIR = REPO_ROOT / "conf" / "agent" / "planner" / "encoder"

# Functional grouping for the parameter breakdown. Order matters: the first
# matching prefix wins, so the more specific names come first.
PARAMETER_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("token_projectors", ("_projection", "projection.")),
    ("embeddings", ("_embedding",)),
    ("latent_queries", ("latent_queries",)),
    ("token_to_latent", ("token_to_latent",)),
    ("attention", ("attention",)),
    ("layer_norms", ("norm",)),
    ("latent_ffn", ("feed_forward", "ff.")),
    ("output_projection", ("output_projection",)),
)


@dataclass
class Timing:
    """Latency samples for one timed region, in milliseconds."""

    samples: list[float] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples)

    @property
    def iqr_ms(self) -> float:
        if len(self.samples) < 4:
            return 0.0
        ordered = sorted(self.samples)
        half = len(ordered) // 2
        lower = statistics.median(ordered[:half])
        upper = statistics.median(ordered[-half:])
        return upper - lower


def load_encoder_config(name: str) -> dict[str, Any]:
    path = ENCODER_CONF_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No encoder configuration at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def classify_parameter(qualified_name: str) -> str:
    for group, prefixes in PARAMETER_GROUPS:
        if any(prefix in qualified_name for prefix in prefixes):
            return group
    return "other"


def parameter_breakdown(module: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        counts[classify_parameter(name)] = counts.get(classify_parameter(name), 0) + parameter.numel()
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_forward(encoder: torch.nn.Module, batch: torch.Tensor, device: torch.device) -> float:
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        encoder(batch)
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0


def time_forward_backward(
    encoder: torch.nn.Module, batch: torch.Tensor, device: torch.device
) -> float:
    encoder.zero_grad(set_to_none=True)
    synchronize(device)
    start = time.perf_counter()
    output = encoder(batch)
    output.sum().backward()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0


def build_variants(names: list[str], device: torch.device) -> dict[str, torch.nn.Module]:
    encoders: dict[str, torch.nn.Module] = {}
    for name in names:
        config = load_encoder_config(name)
        encoder = build_encoder(
            config,
            input_dim=SemanticObservationSchemaV12.flat_dim,
            observation_schema=SemanticObservationSchemaV12(),
        )
        encoders[name] = encoder.to(device).train()
    return encoders


def run_benchmark(
    encoders: dict[str, torch.nn.Module],
    *,
    device: torch.device,
    act_batch: int,
    learn_batch: int,
    repeats: int,
    warmup: int,
    seed: int,
) -> dict[str, dict[str, Timing]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    act_input = torch.randn(
        act_batch, SemanticObservationSchemaV12.flat_dim, generator=generator
    ).to(device)
    learn_input = torch.randn(
        learn_batch, SemanticObservationSchemaV12.flat_dim, generator=generator
    ).to(device)

    results: dict[str, dict[str, Timing]] = {
        name: {"act_forward": Timing(), "learn_forward_backward": Timing()} for name in encoders
    }

    for _ in range(warmup):
        for encoder in encoders.values():
            time_forward(encoder, act_input, device)
            time_forward_backward(encoder, learn_input, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Round-robin: one sample per variant per sweep, so contention is shared.
    for _ in range(repeats):
        for name, encoder in encoders.items():
            results[name]["act_forward"].samples.append(time_forward(encoder, act_input, device))
        for name, encoder in encoders.items():
            results[name]["learn_forward_backward"].samples.append(
                time_forward_backward(encoder, learn_input, device)
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--variants", nargs="+", default=["lq_v3", "lq_v3_lite", "lq_v3_micro"]
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument(
        "--act-batch", type=int, default=20, help="Action-selection batch (one row per env)."
    )
    parser.add_argument("--learn-batch", type=int, default=256, help="Learner batch.")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--encoder-passes-per-step",
        type=float,
        default=5.0,
        help=(
            "Encoder forward passes at the learner batch per environment step. "
            "SAC with share_features_extractor=false performs, per gradient step: "
            "actor(obs), actor(next_obs), critic_target(next_obs), critic(obs) and "
            "critic(obs, actions_pi). gradient_steps resolves to train_freq*n_envs, "
            "so there is one gradient step per environment step."
        ),
    )
    parser.add_argument(
        "--recorded-step-ms",
        type=float,
        default=149.7,
        help=(
            "Measured wall clock per environment step from a previously recorded run, "
            "used as the denominator of the derived encoder share. The default is the "
            "2026-07-27 EXP_sac-lite-cmp_RP_medium run (6.68 fps)."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    encoders = build_variants(args.variants, device)

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"input: D={SemanticObservationSchemaV12.flat_dim}, "
          f"tokens={SemanticObservationSchemaV12.raw_token_count}")
    print()

    print("=== trainable parameters ===")
    totals: dict[str, int] = {}
    for name, encoder in encoders.items():
        breakdown = parameter_breakdown(encoder)
        total = sum(breakdown.values())
        totals[name] = total
        print(f"\n{name}: {total:,}")
        for group, count in breakdown.items():
            print(f"  {group:<20} {count:>10,}  ({100.0 * count / total:5.1f} %)")

    results = run_benchmark(
        encoders,
        device=device,
        act_batch=args.act_batch,
        learn_batch=args.learn_batch,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
    )

    print(f"\n=== latency (median +/- IQR over {args.repeats} round-robin sweeps) ===")
    header = f"{'variant':<14}{'act fwd b=' + str(args.act_batch):>22}{'learn fwd+bwd b=' + str(args.learn_batch):>26}"
    print(header)
    for name in encoders:
        act = results[name]["act_forward"]
        learn = results[name]["learn_forward_backward"]
        print(
            f"{name:<14}"
            f"{act.median_ms:>13.3f} +/-{act.iqr_ms:<7.3f}"
            f"{learn.median_ms:>15.3f} +/-{learn.iqr_ms:<7.3f}"
        )

    if device.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"\npeak allocated across all variants: {peak_mib:.1f} MiB")

    print("\n=== derived per-environment-step encoder cost ===")
    print(
        f"composition: {args.encoder_passes_per_step:g} learner passes + 1 action-selection "
        f"forward per environment step"
    )
    print(f"denominator: {args.recorded_step_ms:.1f} ms/step (recorded run)")
    print(f"\n{'variant':<14}{'encoder ms/step':>18}{'share of step':>16}")
    derived: dict[str, dict[str, float]] = {}
    for name in encoders:
        per_step_ms = (
            args.encoder_passes_per_step * results[name]["learn_forward_backward"].median_ms
            + results[name]["act_forward"].median_ms
        )
        share = 100.0 * per_step_ms / args.recorded_step_ms
        derived[name] = {"encoder_ms_per_step": per_step_ms, "share_percent": share}
        print(f"{name:<14}{per_step_ms:>18.2f}{share:>15.1f} %")

    print(
        "\nLimitation: the denominator is a recorded run on a different day and a "
        "different device load, and the numerator assumes every learner pass costs a "
        "full forward+backward, which overstates the forward-only passes. Treat the "
        "share as an upper bound. A definitive attribution requires a profiled "
        "training run, which this script deliberately does not perform."
    )

    if args.json_out is not None:
        payload = {
            "device": str(device),
            "input_dim": SemanticObservationSchemaV12.flat_dim,
            "raw_token_count": SemanticObservationSchemaV12.raw_token_count,
            "act_batch": args.act_batch,
            "learn_batch": args.learn_batch,
            "repeats": args.repeats,
            "parameters": {name: totals[name] for name in encoders},
            "parameter_breakdown": {
                name: parameter_breakdown(encoder) for name, encoder in encoders.items()
            },
            "latency_ms": {
                name: {
                    region: {"median": timing.median_ms, "iqr": timing.iqr_ms}
                    for region, timing in regions.items()
                }
                for name, regions in results.items()
            },
            "derived_per_step": derived,
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
