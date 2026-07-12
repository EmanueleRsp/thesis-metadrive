"""Compute train-only traffic thresholds and classify the catalog arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_rl.scenarios.arms import classify_catalog_entry
from thesis_rl.scenarios.catalog import read_scenario_catalog, write_scenario_catalog
from thesis_rl.scenarios.reports import compute_arm_distribution, compute_feature_statistics, write_json_report
from thesis_rl.scenarios.thresholds import compute_arm_thresholds, write_arm_thresholds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute balanced train-only Q40/Q75 thresholds and classify arms."
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-catalog", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--feature-version", default="v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    catalog = read_scenario_catalog(args.catalog)
    train_entries = tuple(
        entry
        for entry in catalog.entries
        if entry.record.split == "train"
        and entry.record.validation_status in {"valid", "warning"}
    )
    thresholds = compute_arm_thresholds(
        train_entries,
        feature_version=str(args.feature_version),
    )
    classified = tuple(
        classify_catalog_entry(entry, thresholds) for entry in catalog.entries
    )
    write_arm_thresholds(thresholds, args.thresholds, overwrite=args.overwrite)
    write_scenario_catalog(
        classified,
        args.output_catalog,
        overwrite=args.overwrite,
    )
    report = {
        "catalog": str(Path(args.output_catalog).expanduser().resolve()),
        "thresholds": str(Path(args.thresholds).expanduser().resolve()),
        "train_entries": len(train_entries),
        "threshold_values": thresholds.to_dict(),
        "features": compute_feature_statistics(classified),
        "arms": compute_arm_distribution(classified),
    }
    write_json_report(
        report,
        Path(args.output_catalog).expanduser().resolve().parent / "arm_report.json",
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
