"""Create a read-only M7a mission-aware frozen-index candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_rl.mission.materialize import build_candidate_index, write_candidate_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--record-end", type=int)
    args = parser.parse_args()
    index = json.loads(args.frozen_index.read_text(encoding="utf-8"))
    candidate = build_candidate_index(
        index,
        args.data_root,
        record_start=args.record_start,
        record_end=args.record_end,
    )
    write_candidate_index(candidate, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "records": len(candidate["records"]),
                "record_start": candidate["record_start"],
                "record_end": candidate["record_end"],
                "mission_selection_hash": candidate["mission_selection_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
