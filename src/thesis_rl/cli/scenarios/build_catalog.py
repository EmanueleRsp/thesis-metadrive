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
from thesis_rl.scenarios.reports import (
    compute_arm_distribution,
    compute_feature_statistics,
    write_json_report,
)
from thesis_rl.scenarios.waymo import load_converted_waymo_entries
from thesis_rl.cli.scenarios.ui import console, make_progress, print_key_value_table, print_panel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a unified ScenarioNet catalog from Waymo and PG databases."
    )
    parser.add_argument("--data-root", default=os.environ.get("SCENARIONET_DATA_ROOT"))
    parser.add_argument("--waymo-database")
    parser.add_argument("--pg-database")
    parser.add_argument("--pg-seed-start", type=int)
    parser.add_argument("--pg-count", type=int)
    parser.add_argument(
        "--pg-include-all",
        action="store_true",
        help="Include every exported PG seed instead of only the configured seed window.",
    )
    parser.add_argument(
        "--waymo-workers",
        type=int,
        default=1,
        help="Number of spawned workers used to load Waymo scenarios.",
    )
    parser.add_argument(
        "--pg-workers",
        type=int,
        default=1,
        help="Number of spawned workers used to load PG scenarios.",
    )
    parser.add_argument("--output")
    parser.add_argument("--groups-output")
    parser.add_argument("--report-output")
    parser.add_argument(
        "--allow-empty-waymo",
        action="store_true",
        help="Treat a missing converted Waymo directory as an empty candidate pool.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.data_root:
        raise SystemExit("--data-root or SCENARIONET_DATA_ROOT is required")
    if args.waymo_workers < 1 or args.pg_workers < 1:
        parser.error("--waymo-workers and --pg-workers must be positive")

    paths = ScenarioDataPaths(Path(args.data_root))
    waymo_database = Path(args.waymo_database or paths.root / "waymo" / "database")
    pg_database = Path(args.pg_database or paths.root / "pg" / "database")
    output = Path(args.output or paths.root / "catalog" / "scenario_catalog.parquet")
    groups_output = Path(args.groups_output or paths.root / "splits" / "scenario_groups.json")
    report_output = Path(args.report_output or paths.root / "catalog" / "catalog_report.json")

    with make_progress() as progress:
        waymo_task = progress.add_task("Loading Waymo catalog entries", total=None)
        pg_task = progress.add_task("Loading PG catalog entries", total=None)

        def update_waymo(completed: int, total: int) -> None:
            progress.update(waymo_task, completed=completed, total=total)

        def update_pg(completed: int, total: int) -> None:
            progress.update(pg_task, completed=completed, total=total)

        console.log(
            f"Loading catalog entries with Waymo workers={args.waymo_workers}, "
            f"PG workers={args.pg_workers}"
        )
        has_waymo_scenarios = waymo_database.is_dir() and any(waymo_database.rglob("sd_*.pkl"))
        if args.allow_empty_waymo and not has_waymo_scenarios:
            waymo_entries = ()
            progress.update(waymo_task, completed=0, total=0)
            console.log(
                "Waymo catalog directory is missing or contains no converted "
                f"scenario files; continuing with an empty candidate pool: {waymo_database}"
            )
        else:
            waymo_entries, _waymo_groups = load_converted_waymo_entries(
                waymo_database,
                data_root=paths.root,
                workers=args.waymo_workers,
                progress_callback=update_waymo,
            )
        pg_entries = load_exported_pg_entries(
            pg_database,
            data_root=paths.root,
            split="train",
            seed_start=None if args.pg_include_all else args.pg_seed_start,
            count_per_profile=None if args.pg_include_all else args.pg_count,
            workers=args.pg_workers,
            progress_callback=update_pg,
        )
    entries = tuple(waymo_entries) + tuple(pg_entries)
    with console.status("Writing unified ScenarioNet catalog and group mapping", spinner="dots"):
        write_scenario_catalog(entries, output, overwrite=args.overwrite)

        groups_output.parent.mkdir(parents=True, exist_ok=True)
        if groups_output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite group mapping: {groups_output}")
        groups_output.write_text(
            json.dumps(group_ids_for_entries(entries), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    source_counts = {
        source: sum(entry.record.source == source for entry in entries)
        for source in ("waymo", "pg")
    }
    report = {
        "catalog": str(output.expanduser().resolve()),
        "groups": str(groups_output.expanduser().resolve()),
        "total": len(entries),
        "by_source": source_counts,
        "features": compute_feature_statistics(entries),
        "arms": compute_arm_distribution(entries),
    }
    write_json_report(report, report_output, overwrite=args.overwrite)
    print_panel(
        "Catalog built",
        f"Total scenarios: {report['total']}\nCatalog: {output}\nGroups: {groups_output}",
    )
    print_key_value_table(
        "Catalog source counts",
        list(source_counts.items()),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
