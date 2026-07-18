"""Generate exactly the PG profile/seed tasks recorded by a frozen index."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

from thesis_rl.scenarios.bootstrap import collect_local_api_inventory
from thesis_rl.scenarios.frozen import load_frozen_index
from thesis_rl.scenarios.pg.report import run_pg_tasks
from thesis_rl.scenarios.reports import write_json_report
from thesis_rl.cli.scenarios.ui import make_progress, print_panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--data-root", default=os.environ.get("SCENARIONET_DATA_ROOT"))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.data_root:
        parser.error("--data-root or SCENARIONET_DATA_ROOT is required")

    payload = load_frozen_index(args.index)
    generations = payload["source_inventory"]["pg"]["generations"]
    data_root = Path(args.data_root).expanduser().resolve()
    tasks = [
        (str(item["profile"]), int(item["seed"]))
        for item in generations
        if args.overwrite or not (data_root / str(item["relative_path"])).is_file()
    ]
    skipped = len(generations) - len(tasks)
    if not tasks:
        print(json.dumps({"requested": len(generations), "generated": 0, "skipped": skipped}))
        return 0
    inventory = cast(dict[str, Any], collect_local_api_inventory(Path(args.repo_root)))
    git = cast(dict[str, Any], inventory["git"])
    progress = make_progress()
    task_id = progress.add_task("Generating frozen PG scenarios", total=len(tasks))
    failed = 0

    def on_progress(completed: int, _total: int, profile: str, seed: int, succeeded: bool) -> None:
        nonlocal failed
        if not succeeded:
            failed += 1
        progress.update(
            task_id,
            completed=completed,
            description=f"Generating frozen PG • {profile} • seed={seed} • failed={failed}",
        )

    with progress:
        report, _results = run_pg_tasks(
            tasks,
            data_root=data_root,
            workers=args.workers,
            overwrite=args.overwrite,
            generator_commit=git["metadrive"]["commit"],
            exporter_commit=git["metadrive"]["commit"],
            progress_callback=on_progress,
        )
    report_path = args.report_output or (data_root / "pg" / "frozen_generation_report.json")
    # The generation report is a derived progress artifact. Source scenario
    # files remain protected unless --overwrite is explicitly requested.
    report_path = write_json_report(report.to_dict(), report_path, overwrite=True)
    if report.failed:
        raise SystemExit(
            f"frozen PG generation failed for {report.failed}/{report.requested} tasks; "
            f"report: {report_path}"
        )
    print_panel(
        "Frozen PG generation completed",
        f"Generated {report.generated}/{report.requested} scenarios\nReport: {report_path}",
    )
    print(
        json.dumps({**report.to_dict(), "report_path": str(report_path)}, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
