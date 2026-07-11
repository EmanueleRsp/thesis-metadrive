from __future__ import annotations

import argparse
import json
import os
from thesis_rl.scenarios.runtime_database import verify_runtime_mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a ScenarioNet runtime mapping.")
    parser.add_argument("runtime_directory")
    parser.add_argument("--data-root", default=os.environ.get("SCENARIONET_DATA_ROOT"))
    parser.add_argument("--skip-feature-extraction", action="store_true")
    args = parser.parse_args()
    if not args.data_root:
        raise SystemExit("--data-root or SCENARIONET_DATA_ROOT is required")
    filenames = verify_runtime_mapping(args.runtime_directory)
    # Full catalog-driven validation is exposed by the Python API; this CLI
    # performs the always-safe mapping check until a catalog reader is wired in.
    print(json.dumps({"runtime_files": len(filenames), "feature_extraction_skipped": bool(args.skip_feature_extraction)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
