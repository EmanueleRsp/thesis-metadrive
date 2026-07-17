from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

from thesis_rl.scenarios.bootstrap import collect_local_api_inventory
from thesis_rl.scenarios.pg.profiles import PG_PROFILES
from thesis_rl.scenarios.pg.report import run_pg_pilot, write_pg_pilot_report
from thesis_rl.scenarios.reports import write_json_report
from thesis_rl.cli.scenarios.ui import make_progress, print_key_value_table, print_panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate offline native MetaDrive PG scenarios.")
    parser.add_argument("--data-root", default=os.environ.get("SCENARIONET_DATA_ROOT"))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--count", type=int, default=20, help="Scenarios per profile.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Optional separate JSON report path; the default is the canonical PG pilot report.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of isolated MetaDrive generation processes.",
    )
    parser.add_argument(
        "--profile-counts-json",
        help="Optional JSON object overriding counts per PG profile.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.data_root:
        raise SystemExit("--data-root or SCENARIONET_DATA_ROOT is required")
    profile_counts = None
    if args.profile_counts_json:
        try:
            parsed_profile_counts = json.loads(args.profile_counts_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid --profile-counts-json: {exc}") from exc
        if not isinstance(parsed_profile_counts, dict):
            raise SystemExit("--profile-counts-json must be a JSON object")
        profile_counts = parsed_profile_counts

    inventory = collect_local_api_inventory(Path(args.repo_root))
    git = cast(dict[str, Any], inventory["git"])
    total = (
        sum(profile_counts.values())
        if profile_counts is not None
        else len(PG_PROFILES) * int(args.count)
    )
    failed = 0
    progress = make_progress()
    task_id = progress.add_task("Generating PG scenarios", total=total)

    def on_progress(
        completed: int,
        _total: int,
        profile: str,
        seed: int,
        succeeded: bool,
    ) -> None:
        nonlocal failed
        if not succeeded:
            failed += 1
        progress.update(
            task_id,
            completed=completed,
            description=f"Generating PG • {profile} • seed={seed} • failed={failed}",
        )

    with progress:
        report, _results = run_pg_pilot(
            data_root=args.data_root,
            count_per_profile=args.count,
            seed_start=args.seed_start,
            workers=args.workers,
            overwrite=args.overwrite,
            generator_commit=git["metadrive"]["commit"],
            exporter_commit=git["metadrive"]["commit"],
            progress_callback=on_progress,
            profile_counts=profile_counts,
        )
    report_path = (
        write_json_report(
            report.to_dict(), args.report_output.expanduser().resolve(), overwrite=args.overwrite
        )
        if args.report_output is not None
        else write_pg_pilot_report(report, args.data_root, overwrite=args.overwrite)
    )
    style = "green" if report.failed == 0 else "red"
    print_panel(
        "PG generation completed"
        if report.failed == 0
        else "PG generation completed with failures",
        f"Generated {report.generated}/{report.requested} scenarios\n"
        f"Failed: {report.failed}\n"
        f"Report: {report_path}",
        style=style,
    )
    print_key_value_table(
        "PG profile summary",
        [
            (profile, ", ".join(f"{arm}={count}" for arm, count in arms.items()) or "none")
            for profile, arms in report.by_profile_arm.items()
        ],
    )
    print(
        json.dumps({**report.to_dict(), "report_path": str(report_path)}, indent=2, sort_keys=True)
    )
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
