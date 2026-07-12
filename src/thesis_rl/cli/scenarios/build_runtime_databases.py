"""Build ScenarioNet runtime views for train, validation and test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from thesis_rl.scenarios.catalog import read_scenario_catalog, write_scenario_catalog
from thesis_rl.scenarios.pipeline import SPLITS, assign_catalog_runtime_indices
from thesis_rl.scenarios.runtime_database import build_runtime_database, verify_runtime_mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ScenarioNet runtime database views.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--data-root", default=os.environ.get("SCENARIONET_DATA_ROOT"))
    parser.add_argument("--runtime-root")
    parser.add_argument("--output-catalog")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.data_root:
        raise SystemExit("--data-root or SCENARIONET_DATA_ROOT is required")

    data_root = Path(args.data_root).expanduser().resolve()
    runtime_root = Path(args.runtime_root or data_root / "runtime").expanduser().resolve()
    catalog = read_scenario_catalog(args.catalog)
    assigned = assign_catalog_runtime_indices(catalog.entries)
    summary: dict[str, int] = {}
    for split in SPLITS:
        split_entries = tuple(entry for entry in assigned if entry.record.split == split)
        valid_entries = tuple(
            entry
            for entry in split_entries
            if entry.record.validation_status in {"valid", "warning"}
        )
        if not valid_entries:
            raise ValueError(f"cannot build empty runtime view for split={split!r}")
        runtime_directory = runtime_root / split
        build_runtime_database(
            [entry.record for entry in valid_entries],
            data_root=data_root,
            runtime_directory=runtime_directory,
            overwrite=args.overwrite,
        )
        verify_runtime_mapping(runtime_directory)
        summary[split] = len(valid_entries)

    output_catalog = Path(args.output_catalog or args.catalog).expanduser().resolve()
    write_scenario_catalog(assigned, output_catalog, overwrite=True)
    payload = {
        "catalog": str(output_catalog),
        "runtime_root": str(runtime_root),
        "runtime_counts": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
