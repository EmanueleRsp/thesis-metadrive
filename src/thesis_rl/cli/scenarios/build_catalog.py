"""Build the unified ScenarioNet catalog from converted Waymo and PG files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from thesis_rl.scenarios.catalog import write_scenario_catalog
from thesis_rl.scenarios.paths import ScenarioDataPaths
from thesis_rl.scenarios.pipeline import group_ids_for_entries
from thesis_rl.scenarios.pg.loader import load_exported_pg_entries
from thesis_rl.scenarios.reports import compute_arm_distribution, compute_feature_statistics, write_json_report
from thesis_rl.scenarios.waymo import load_converted_waymo_entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a unified ScenarioNet catalog from Waymo and PG databases."
    )
    parser.add_argument("--data-root", default=os.environ.get("SCENARIONET_DATA_ROOT"))
    parser.add_argument("--waymo-database")
    parser.add_argument("--pg-database")
    parser.add_argument("--pg-seed-start", type=int)
    parser.add_argument("--pg-count", type=int)
    parser.add_argument("--output")
    parser.add_argument("--groups-output")
    parser.add_argument("--report-output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.data_root:
        raise SystemExit("--data-root or SCENARIONET_DATA_ROOT is required")

    paths = ScenarioDataPaths(Path(args.data_root))
    waymo_database = Path(args.waymo_database or paths.root / "waymo" / "database")
    pg_database = Path(args.pg_database or paths.root / "pg" / "database")
    output = Path(args.output or paths.root / "catalog" / "scenario_catalog.parquet")
    groups_output = Path(
        args.groups_output or paths.root / "splits" / "scenario_groups.json"
    )
    report_output = Path(
        args.report_output or paths.root / "catalog" / "catalog_report.json"
    )

    waymo_entries, _waymo_groups = load_converted_waymo_entries(
        waymo_database, data_root=paths.root
    )
    pg_entries = load_exported_pg_entries(
        pg_database,
        data_root=paths.root,
        split="train",
        seed_start=args.pg_seed_start,
        count_per_profile=args.pg_count,
    )
    entries = tuple(waymo_entries) + tuple(pg_entries)
    write_scenario_catalog(entries, output, overwrite=args.overwrite)

    groups_output.parent.mkdir(parents=True, exist_ok=True)
    if groups_output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite group mapping: {groups_output}")
    groups_output.write_text(
        json.dumps(group_ids_for_entries(entries), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "catalog": str(output.expanduser().resolve()),
        "groups": str(groups_output.expanduser().resolve()),
        "total": len(entries),
        "by_source": {
            source: sum(entry.record.source == source for entry in entries)
            for source in ("waymo", "pg")
        },
        "features": compute_feature_statistics(entries),
        "arms": compute_arm_distribution(entries),
    }
    write_json_report(report, report_output, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
