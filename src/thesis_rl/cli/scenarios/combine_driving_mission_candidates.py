"""Combine contiguous read-only M7a mission candidate batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_rl.mission.materialize import combine_candidate_indexes, write_candidate_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [json.loads(path.read_text(encoding="utf-8")) for path in args.candidate]
    combined = combine_candidate_indexes(candidates)
    write_candidate_index(combined, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "records": len(combined["records"]),
                "mission_selection_hash": combined["mission_selection_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
