"""Validate one complete read-only M7a mission candidate against its frozen parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_rl.mission.materialize import validate_candidate_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-index", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(args.frozen_index.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    print(json.dumps(validate_candidate_index(candidate, parent), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
