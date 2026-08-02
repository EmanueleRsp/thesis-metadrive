"""Run the authorized read-only unified-driving-mission frozen-catalog audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_rl.scenarios.driving_mission_audit import audit_frozen_index, write_audit_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.frozen_index.read_text(encoding="utf-8"))
    result = audit_frozen_index(payload, args.data_root)
    write_audit_report(result, args.output_dir)
    print(json.dumps(result.summary(), sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
