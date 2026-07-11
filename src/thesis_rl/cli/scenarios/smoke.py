from __future__ import annotations

import argparse
import json

from thesis_rl.scenarios.smoke import smoke_test_scenario_env


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a short headless ScenarioEnv rollout.")
    parser.add_argument("data_directory")
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--no-reactive-traffic",
        action="store_true",
        help="Disable reactive traffic for API diagnostics.",
    )
    args = parser.parse_args()
    result = smoke_test_scenario_env(
        args.data_directory,
        scenario_index=args.scenario_index,
        steps=args.steps,
        reactive_traffic=not args.no_reactive_traffic,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
