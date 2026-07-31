"""Rebuild canonical ScenarioNet artifacts from a frozen selection index."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from thesis_rl.scenarios.catalog import write_scenario_catalog
from thesis_rl.scenarios.frozen import (
    frozen_catalog,
    load_frozen_index,
    restore_frozen_panels,
    verify_frozen_sources,
)
from thesis_rl.scenarios.runtime_database import build_runtime_database, verify_runtime_mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--data-root", default=os.environ.get("SCENARIONET_DATA_ROOT"))
    parser.add_argument("--catalog-output")
    parser.add_argument("--runtime-root")
    parser.add_argument("--split-manifest-output")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("--data-root or SCENARIONET_DATA_ROOT is required")

    root = Path(args.data_root).expanduser().resolve()
    payload = load_frozen_index(args.index)
    verify_frozen_sources(payload, root)
    catalog = frozen_catalog(payload)
    if args.verify_only:
        print(
            json.dumps(
                {"index": str(Path(args.index).resolve()), "records": len(catalog.entries)},
                indent=2,
            )
        )
        return 0

    catalog_output = Path(args.catalog_output or root / "catalog" / "scenario_catalog.parquet")
    runtime_root = Path(args.runtime_root or root / "runtime")
    manifest_output = Path(args.split_manifest_output or root / "splits" / "split_manifest.yaml")
    write_scenario_catalog(catalog.entries, catalog_output, overwrite=args.overwrite)
    for split in ("train", "validation", "test"):
        records = tuple(entry.record for entry in catalog.entries if entry.record.split == split)
        build_runtime_database(
            records,
            data_root=root,
            runtime_directory=runtime_root / split,
            overwrite=args.overwrite,
        )
        verify_runtime_mapping(runtime_root / split)
    if manifest_output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite canonical split manifest: {manifest_output}")
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        yaml.safe_dump(payload["split_manifest"], sort_keys=True), encoding="utf-8"
    )
    restored_panels = restore_frozen_panels(payload, root, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "catalog": str(catalog_output.resolve()),
                "runtime_root": str(runtime_root.resolve()),
                "split_manifest": str(manifest_output.resolve()),
                "panel_manifests": [str(path) for path in restored_panels],
                "records": len(catalog.entries),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
