from __future__ import annotations

import argparse
import json

from thesis_rl.scenarios.pg.replenishment import load_report, plan_profile_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan targeted PG replenishment profile counts.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--budget", type=int, default=1750)
    args = parser.parse_args()
    print(
        json.dumps(
            plan_profile_counts(load_report(args.report), budget=args.budget), sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
