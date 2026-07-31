"""Build and persist a frozen named panel manifest (EVAL-PROTOCOL v1.1 REQ-004).

SCENARIONET-INTEGRATION v1.2 / EVAL-PROTOCOL v1.1 declare five named panels,
each with its own split, source, size, and draw policy:

    validation_waymo_empirical  validation  waymo     empirical      primary curve
    validation_pg               validation  pg        empirical      diagnostic
    test_waymo_empirical        test        waymo     empirical      primary endpoint
    test_pg                     test        pg        empirical      secondary
    test_arm_stratified         test        combined  arm_balanced   competence endpoint

This is a companion to the existing `scripts/build_panel_manifest.py` (the
v1.0 single validation/test balanced panel), not a replacement: that script
still works for a v1.1-only dataset. This one requires a catalog produced by
`build_splits_v1_2` (records carrying `holdout_pool`).

Usage:
    python scripts/build_named_panel_manifest.py --panel-name test_waymo_empirical --seed 20260731
    python scripts/build_named_panel_manifest.py --panel-name test_arm_stratified --size 300 --seed 20260731
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis_rl.scenarios.catalog import read_scenario_catalog
from thesis_rl.scenarios.panel_manifest import (
    NAMED_PANELS,
    build_balanced_panel,
    build_empirical_panel,
    named_panel_manifest_path,
    save_panel_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-name", required=True, choices=sorted(NAMED_PANELS))
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Must equal the canonical complete-pool size; omitted uses that frozen contract.",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Deterministic draw seed (not a training seed)."
    )
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
    parser.add_argument(
        "--output", default=None, help="Explicit output path (overrides --data-root default)."
    )
    parser.add_argument(
        "--tracked-subset-count",
        type=int,
        default=None,
        help=(
            "Tracked-subset scenarios, feature-diversity selected. For "
            "test_arm_stratified this is per arm (matching build_balanced_panel); "
            "for the four empirical panels it is a flat count over the whole "
            "panel. Defaults to 4 for a validation_* panel, 10 for a test_* panel; "
            "pass 0 to disable."
        ),
    )
    args = parser.parse_args()

    panel_spec = NAMED_PANELS[args.panel_name]
    expected_size = int(panel_spec["full_size"])
    if args.size is not None and int(args.size) != expected_size:
        raise SystemExit(
            f"{args.panel_name} is a complete frozen pool of {expected_size}; "
            "diagnostic sizes are generated only as profile child manifests during freeze."
        )
    panel_size = expected_size
    catalog_path = args.catalog_path
    if catalog_path is None:
        scenarionet_root = os.environ.get("SCENARIONET_DATA_ROOT")
        if not scenarionet_root:
            raise SystemExit("--catalog-path or $SCENARIONET_DATA_ROOT is required.")
        catalog_path = str(Path(scenarionet_root) / "catalog" / "scenario_catalog.parquet")

    catalog = read_scenario_catalog(catalog_path)
    matching_entries = tuple(
        entry
        for entry in catalog.entries
        if entry.record.split == panel_spec["split"]
        and entry.record.runtime_index is not None
        and entry.record.holdout_pool == panel_spec["holdout_pool"]
        and (panel_spec["source"] == "combined" or entry.record.source == panel_spec["source"])
    )
    if not matching_entries:
        raise SystemExit(
            f"No eligible records for panel={args.panel_name!r} "
            f"(split={panel_spec['split']!r}, source={panel_spec['source']!r}, "
            f"holdout_pool={panel_spec['holdout_pool']!r}) in {catalog_path}. "
            "This panel requires a catalog built by build_splits_v1_2 "
            "(records must carry a non-null holdout_pool)."
        )
    records = tuple(entry.record for entry in matching_entries)
    feature_lookup = {
        entry.record.scenario_uid: entry.features.to_dict() for entry in matching_entries
    }

    if args.tracked_subset_count is not None:
        tracked_subset_count = int(args.tracked_subset_count)
    else:
        tracked_subset_count = 4 if panel_spec["split"] == "validation" else 10

    if panel_spec["draw_policy"] == "empirical":
        manifest = build_empirical_panel(
            records,
            split=panel_spec["split"],
            source=panel_spec["source"],
            size=panel_size,
            seed=args.seed,
            tracked_subset_count=tracked_subset_count,
            feature_lookup=feature_lookup,
        )
    else:
        manifest = build_balanced_panel(
            records,
            split=panel_spec["split"],
            size=panel_size,
            seed=args.seed,
            tracked_subset_count_per_arm=tracked_subset_count,
            feature_lookup=feature_lookup,
        )

    if args.output is not None:
        output_path = Path(args.output)
    else:
        data_root = args.data_root or os.environ.get("DATA_ROOT") or "./data"
        output_path = named_panel_manifest_path(args.panel_name, data_root=data_root)

    save_panel_manifest(manifest, output_path)
    print(f"Wrote panel manifest -> {output_path}")
    print(
        f"  panel={args.panel_name} draw_policy={manifest.draw_policy} "
        f"source={manifest.source} split={manifest.split} size={manifest.size} seed={manifest.seed}"
    )
    print(f"  observed_arm_counts={dict(zip(manifest.arms, manifest.per_arm_counts))}")
    print(f"  sha256={manifest.sha256}")
    print(f"  tracked_subset_uids={len(manifest.tracked_subset_uids)}")


if __name__ == "__main__":
    main()
