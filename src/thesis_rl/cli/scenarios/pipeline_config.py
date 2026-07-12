"""Resolve versioned ScenarioNet pipeline defaults with environment overrides."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


_ENV_PATHS: dict[str, tuple[str, ...]] = {
    "SCENARIONET_PG_COUNT": ("pg", "count_per_profile"),
    "SCENARIONET_PG_SEED_START": ("pg", "seed_start"),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"pipeline config must be a mapping: {args.config}")
    for env_name, path in _ENV_PATHS.items():
        configured = os.environ.get(env_name, "")
        value = configured if configured != "" else _lookup(payload, path)
        print(f"{env_name}\t{_format_value(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
