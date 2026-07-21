"""Generate the checked-in synthetic Rulebook ScenarioDescription fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

from rulebook_scenario_fixtures import write_persistent_fixtures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/fixtures/rulebook_scenarios"),
    )
    args = parser.parse_args()
    print(write_persistent_fixtures(args.output_dir))


if __name__ == "__main__":
    main()
