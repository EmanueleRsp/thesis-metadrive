from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

from thesis_rl.scenarios.bootstrap import collect_local_api_inventory
from thesis_rl.scenarios.pg.report import run_pg_pilot, write_pg_pilot_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate offline native MetaDrive PG scenarios.")
    parser.add_argument("--data-root", default=os.environ.get("SCENARIONET_DATA_ROOT"))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--count", type=int, default=20, help="Scenarios per profile.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.data_root:
        raise SystemExit("--data-root or SCENARIONET_DATA_ROOT is required")

    inventory = collect_local_api_inventory(Path(args.repo_root))
    git = cast(dict[str, Any], inventory["git"])
    report, _results = run_pg_pilot(
        data_root=args.data_root,
        count_per_profile=args.count,
        seed_start=args.seed_start,
        overwrite=args.overwrite,
        generator_commit=git["metadrive"]["commit"],
        exporter_commit=git["metadrive"]["commit"],
    )
    report_path = write_pg_pilot_report(report, args.data_root, overwrite=args.overwrite)
    print(json.dumps({**report.to_dict(), "report_path": str(report_path)}, indent=2, sort_keys=True))
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
