"""Thesis-specific integration layer for the local SB3 fork.

This package is intentionally lightweight at first. It will host the bridge
objects that connect thesis-specific modules (encoders, decoders, custom heads,
buffers, policy builders) to the vendored/forked SB3 implementation while
keeping orchestration code in :mod:`thesis_rl` separate from algorithmic core
changes inside the fork.
"""

from thesis_rl.sb3_extensions.builders import (
    build_algorithm_spec,
    build_algorithm_spec_from_planner_cfg,
    build_policy_spec,
    build_policy_spec_from_planner_cfg,
    build_sb3_specs_from_configs,
    normalize_policy_kwargs,
    uses_explicit_custom_sb3_policy,
    validate_sb3_bridge_configs,
)
from thesis_rl.sb3_extensions.specs import Sb3AlgorithmSpec, Sb3PolicySpec
from thesis_rl.sb3_extensions.checkpointing import (
    CheckpointGeneration,
    load_checkpoint_generation,
    publish_checkpoint_generation,
    resolve_explicit_generation,
    resolve_latest_generation,
    sha256_file,
)

__all__ = [
    "build_algorithm_spec",
    "build_algorithm_spec_from_planner_cfg",
    "build_policy_spec",
    "build_policy_spec_from_planner_cfg",
    "build_sb3_specs_from_configs",
    "normalize_policy_kwargs",
    "Sb3AlgorithmSpec",
    "Sb3PolicySpec",
    "uses_explicit_custom_sb3_policy",
    "validate_sb3_bridge_configs",
    "CheckpointGeneration",
    "load_checkpoint_generation",
    "publish_checkpoint_generation",
    "resolve_explicit_generation",
    "resolve_latest_generation",
    "sha256_file",
]
