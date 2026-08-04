"""Freeze the final ScenarioNet selection into a replayable index."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from thesis_rl.scenarios.frozen import build_frozen_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("SCENARIONET_DATA_ROOT"))
    parser.add_argument("--catalog")
    parser.add_argument("--split-manifest")
    parser.add_argument("--shard-ledger")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("--data-root or SCENARIONET_DATA_ROOT is required")

    root = Path(args.data_root).expanduser().resolve()
    output = build_frozen_index(
        catalog_path=args.catalog or root / "catalog" / "scenario_catalog.parquet",
        split_manifest_path=args.split_manifest or root / "splits" / "split_manifest.yaml",
        data_root=root,
        shard_ledger_path=args.shard_ledger
        or root / "waymo" / "acquisition" / "converted_shards.txt",
        output_path=args.output or root / "frozen" / "scenario_selection_index.json",
        overwrite=args.overwrite,
        require_driving_mission=True,
    )
    print(f"Frozen ScenarioNet selection index: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
