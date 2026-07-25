"""Build and persist a frozen EVAL-PROTOCOL v1.0 REQ-004/DEC-005 panel manifest.

Draws a deterministic seed-driven balanced panel across the six scenario arms
(A0-A5) from the real ScenarioNet catalog for a given split, then writes the
versioned, hashed manifest artifact so every run in a comparison block can
load the identical frozen panel via ``provider.panel_manifest_path``.

Usage:
    python scripts/build_panel_manifest.py --split validation --size 100 --seed 20260725
    python scripts/build_panel_manifest.py --split test --size 300 --seed 20260725

This is a one-time (per comparison block) operator action, not part of the
training/evaluation hot path: the manifest is meant to be built once, checked
into the appropriate data artifact location (outside `src/`), and referenced
identically by every algorithm/extension/training-seed run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_rl.scenarios.catalog import read_scenario_catalog
from thesis_rl.scenarios.panel_manifest import (
    build_balanced_panel,
    default_panel_manifest_path,
    save_panel_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--size", type=int, required=True, help="Panel size (thesis: 100 validation, 300 test).")
    parser.add_argument("--seed", type=int, required=True, help="Deterministic draw seed (not a training seed).")
    parser.add_argument(
        "--catalog-path",
        default=None,
        help="Path to scenario_catalog.parquet. Defaults to $SCENARIONET_DATA_ROOT/catalog/scenario_catalog.parquet.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root for the default output path (data/scenarionet/panels/...). Defaults to $DATA_ROOT or ./data.",
    )
    parser.add_argument("--output", default=None, help="Explicit output path (overrides --data-root default).")
    parser.add_argument(
        "--tracked-subset-count-per-arm",
        type=int,
        default=None,
        help=(
            "REQ-014/DEC-014 (amended 2026-07-25): tracked-subset scenarios "
            "per arm, feature-diversity selected from each arm's drawn panel "
            "candidates. Defaults to 4 for --split validation and 10 for "
            "--split test (the approved per-split sizing); pass 0 to disable."
        ),
    )
    args = parser.parse_args()

    import os

    catalog_path = args.catalog_path
    if catalog_path is None:
        scenarionet_root = os.environ.get("SCENARIONET_DATA_ROOT")
        if not scenarionet_root:
            raise SystemExit("--catalog-path or $SCENARIONET_DATA_ROOT is required.")
        catalog_path = str(Path(scenarionet_root) / "catalog" / "scenario_catalog.parquet")

    catalog = read_scenario_catalog(catalog_path)
    records = tuple(
        record
        for record in catalog.records
        if record.split == args.split and record.runtime_index is not None
    )
    if not records:
        raise SystemExit(f"No eligible records for split={args.split!r} in {catalog_path}")

    # REQ-014/DEC-014 (amended 2026-07-25): approved default tracked-subset
    # sizing is 4 scenarios/arm for the periodic validation panel and 10
    # scenarios/arm for the final-test panel (two different counts for two
    # different purposes -- validation is rendered often during training,
    # test only once at the end but with broader coverage).
    if args.tracked_subset_count_per_arm is not None:
        tracked_subset_count_per_arm = int(args.tracked_subset_count_per_arm)
    else:
        tracked_subset_count_per_arm = 4 if args.split == "validation" else 10

    feature_lookup: dict[str, dict] = {}
    for entry in catalog.entries:
        if entry.record.split == args.split and entry.record.runtime_index is not None:
            feature_lookup[entry.record.scenario_uid] = entry.features.to_dict()

    manifest = build_balanced_panel(
        records,
        split=args.split,
        size=args.size,
        seed=args.seed,
        tracked_subset_count_per_arm=tracked_subset_count_per_arm,
        feature_lookup=feature_lookup,
    )

    if args.output is not None:
        output_path = Path(args.output)
    else:
        data_root = args.data_root or os.environ.get("DATA_ROOT") or "./data"
        output_path = default_panel_manifest_path(args.split, data_root=data_root)

    save_panel_manifest(manifest, output_path)
    print(f"Wrote panel manifest -> {output_path}")
    print(f"  split={manifest.split} size={manifest.size} seed={manifest.seed}")
    print(f"  per_arm_counts={dict(zip(manifest.arms, manifest.per_arm_counts))}")
    print(f"  sha256={manifest.sha256}")
    print(
        f"  tracked_subset_uids={len(manifest.tracked_subset_uids)} "
        f"(count_per_arm={tracked_subset_count_per_arm})"
    )


if __name__ == "__main__":
    main()
