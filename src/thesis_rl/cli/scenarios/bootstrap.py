from __future__ import annotations

import argparse
import os
from pathlib import Path

from thesis_rl.scenarios.bootstrap import initialize_bootstrap_artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect local ScenarioNet APIs and create the initial dataset manifest."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("SCENARIONET_DATA_ROOT"),
        help="ScenarioNet data root (default: SCENARIONET_DATA_ROOT).",
    )
    parser.add_argument("--repo-root", default=".", help="Project repository root.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing bootstrap artifacts explicitly.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.data_root:
        raise SystemExit("--data-root or SCENARIONET_DATA_ROOT is required")
    command = "python -m thesis_rl.cli.scenarios.bootstrap"
    manifest_path, inventory_path = initialize_bootstrap_artifacts(
        data_root=args.data_root,
        repo_root=Path(args.repo_root),
        created_by_command=command,
        overwrite=bool(args.overwrite),
    )
    print(f"manifest={manifest_path}")
    print(f"api_inventory={inventory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
