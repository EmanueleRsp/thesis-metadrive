from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from thesis_rl.scenarios.catalog import read_scenario_catalog
from thesis_rl.scenarios.runtime_database import verify_runtime_mapping
from thesis_rl.scenarios.runtime_database import sha256_file
from thesis_rl.scenarios.validation import validate_records, write_validation_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a ScenarioNet runtime mapping.")
    parser.add_argument("runtime_directory")
    parser.add_argument("--data-root", default=os.environ.get("SCENARIONET_DATA_ROOT"))
    parser.add_argument(
        "--catalog",
        help="Optional scenario_catalog.parquet to run catalog-driven validation.",
    )
    parser.add_argument(
        "--validation-summary",
        help="Output JSON path (default: <runtime_directory>/validation_summary.json).",
    )
    parser.add_argument("--skip-feature-extraction", action="store_true")
    args = parser.parse_args()
    if not args.data_root:
        raise SystemExit("--data-root or SCENARIONET_DATA_ROOT is required")
    filenames = verify_runtime_mapping(args.runtime_directory)
    payload: dict[str, object] = {
        "runtime_files": len(filenames),
        "feature_extraction_skipped": bool(args.skip_feature_extraction),
    }
    if args.catalog:
        catalog_path = Path(args.catalog).expanduser().resolve()
        catalog = read_scenario_catalog(catalog_path)
        results = validate_records(
            catalog.records,
            data_root=args.data_root,
            run_feature_extraction=not args.skip_feature_extraction,
        )
        summary_path = Path(args.validation_summary or (Path(args.runtime_directory) / "validation_summary.json"))
        write_validation_summary(
            results,
            summary_path,
            catalog_hash=sha256_file(catalog_path),
            overwrite=True,
        )
        payload["catalog_records"] = len(catalog.records)
        payload["validation_summary"] = str(summary_path.expanduser().resolve())
        payload["validation_counts"] = {
            status: sum(result.status == status for result in results)
            for status in ("valid", "warning", "invalid")
        }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
