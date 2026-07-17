"""Resolve the versioned ScenarioNet pipeline configuration.

The YAML file is the single source of truth for scientific/data-policy
parameters.  Host paths, download options and credentials are deliberately
handled by the shell orchestration layer and are not merged into this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


_CONFIG_PATHS: dict[str, tuple[str, ...]] = {
    "SCENARIONET_WAYMO_AUTO_EXPAND": ("waymo", "auto_expand"),
    "SCENARIONET_WAYMO_BATCH_SHARDS": ("waymo", "batch_shards"),
    "SCENARIONET_WAYMO_MAX_NEW_SHARDS": ("waymo", "max_new_shards"),
    "SCENARIONET_WAYMO_WORKERS": ("waymo", "workers"),
    "SCENARIONET_WAYMO_KEEP_RAW_BATCHES": ("waymo", "keep_raw_batches"),
    "SCENARIONET_WAYMO_REQUIRED_A4_VRU": ("waymo", "required_arms", "A4_vru"),
    "SCENARIONET_PG_COUNT": ("pg", "count_per_profile"),
    "SCENARIONET_PG_SEED_START": ("pg", "seed_start"),
    "SCENARIONET_PG_WORKERS": ("pg", "workers"),
    "SCENARIONET_PG_MAX_COMPOSITION_REPLENISHMENT_BLOCKS": (
        "pg",
        "max_composition_replenishment_blocks",
    ),
    "SCENARIONET_PG_REPLENISHMENT_CANDIDATE_BUDGET": (
        "pg",
        "replenishment_candidate_budget",
    ),
    "SCENARIONET_RULEBOOK_V2_ENABLED": ("rulebook_v2", "enabled"),
    "SCENARIONET_RULEBOOK_V2_WORKERS": ("rulebook_v2", "workers"),
    "SCENARIONET_SPLIT_SEED": ("split", "seed"),
    "SCENARIONET_AUTO_SPLIT": ("split", "auto"),
    "SCENARIONET_WAYMO_TRAIN_TARGET": ("split", "targets", "waymo", "train"),
    "SCENARIONET_WAYMO_VALIDATION_TARGET": ("split", "targets", "waymo", "validation"),
    "SCENARIONET_WAYMO_TEST_TARGET": ("split", "targets", "waymo", "test"),
    "SCENARIONET_PG_TRAIN_TARGET": ("split", "targets", "pg", "train"),
    "SCENARIONET_PG_VALIDATION_TARGET": ("split", "targets", "pg", "validation"),
    "SCENARIONET_PG_TEST_TARGET": ("split", "targets", "pg", "test"),
    "SCENARIONET_RUN_SIMULATION_CHECK": ("checks", "simulation"),
    "SCENARIONET_CHECK_WORKERS": ("checks", "workers"),
}

_BOOLEAN_PATHS = (
    ("waymo", "auto_expand"),
    ("waymo", "keep_raw_batches"),
    ("rulebook_v2", "enabled"),
    ("split", "auto"),
    ("checks", "simulation"),
)

_POSITIVE_INTEGER_PATHS = (
    ("waymo", "batch_shards"),
    ("waymo", "workers"),
    ("pg", "count_per_profile"),
    ("pg", "workers"),
    ("pg", "replenishment_candidate_budget"),
    ("rulebook_v2", "workers"),
    ("checks", "workers"),
)

_NON_NEGATIVE_INTEGER_PATHS = (
    ("waymo", "max_new_shards"),
    ("waymo", "required_arms", "A4_vru"),
    ("pg", "seed_start"),
    ("pg", "max_composition_replenishment_blocks"),
    ("split", "seed"),
    *(
        ("split", "targets", source, split)
        for source in ("waymo", "pg")
        for split in ("train", "validation", "test")
    ),
)


def _lookup(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for part in path:
        if not isinstance(value, dict) or part not in value:
            raise KeyError(".".join(path))
        value = value[part]
    return value


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _validate_payload(payload: dict[str, Any]) -> None:
    for path in _BOOLEAN_PATHS:
        value = _lookup(payload, path)
        if not isinstance(value, bool):
            raise ValueError(f"{'.'.join(path)} must be a boolean, got {value!r}")
    for path in _POSITIVE_INTEGER_PATHS:
        value = _lookup(payload, path)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{'.'.join(path)} must be a positive integer, got {value!r}")
    for path in _NON_NEGATIVE_INTEGER_PATHS:
        value = _lookup(payload, path)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{'.'.join(path)} must be a non-negative integer, got {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"pipeline config must be a mapping: {args.config}")
        _validate_payload(payload)
    except (KeyError, OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    for env_name, path in _CONFIG_PATHS.items():
        value = _lookup(payload, path)
        print(f"{env_name}\t{_format_value(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
