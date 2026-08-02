"""Promote an approved M7a candidate to new canonical frozen artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_rl.mission.materialize import promote_candidate_index, write_candidate_index
from thesis_rl.scenarios.catalog import ScenarioCatalogEntry, write_scenario_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(args.frozen_parent.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    promoted = promote_candidate_index(candidate, parent)
    if args.output_catalog.expanduser().resolve().exists():
        raise FileExistsError(f"refusing to overwrite mission catalog: {args.output_catalog}")
    write_candidate_index(promoted, args.output_index)
    write_scenario_catalog(
        (ScenarioCatalogEntry.from_flat_dict(record) for record in promoted["records"]),
        args.output_catalog,
    )
    print(
        json.dumps(
            {
                "index": str(args.output_index.resolve()),
                "catalog": str(args.output_catalog.resolve()),
                "records": len(promoted["records"]),
                "mission_selection_hash": promoted["mission_selection_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
