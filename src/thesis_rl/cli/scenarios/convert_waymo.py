from __future__ import annotations

import argparse
import os
from pathlib import Path

from thesis_rl.scenarios.waymo import WaymoConversionError, convert_waymo_training_20s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Waymo training_20s using the checked-out ScenarioNet converter."
    )
    parser.add_argument("--raw-data-path", required=True)
    parser.add_argument("--database-path", default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    database_path = args.database_path or str(
        Path(os.environ.get("SCENARIONET_DATA_ROOT", "/workspace/data/scenarionet"))
        / "waymo"
        / "database"
    )
    try:
        command = convert_waymo_training_20s(
            raw_data_path=args.raw_data_path,
            database_path=database_path,
            num_workers=args.num_workers,
            num_files=args.num_files,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, WaymoConversionError) as exc:
        parser.error(str(exc))
    print("Executed:", " ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
