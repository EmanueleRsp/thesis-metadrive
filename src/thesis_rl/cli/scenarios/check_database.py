"""Run the checked-out ScenarioNet official database verifiers."""

from __future__ import annotations

import argparse

from thesis_rl.scenarios.official_checks import run_official_check
from thesis_rl.cli.scenarios.ui import console, print_panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an official ScenarioNet database check.")
    parser.add_argument("check", choices=("existence", "simulation", "overlap"))
    parser.add_argument("database_path")
    parser.add_argument("--other-database-path")
    parser.add_argument("--error-file-path")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show-id", action="store_true")
    args = parser.parse_args()
    with console.status(
        f"Running ScenarioNet official {args.check} check on {args.database_path}",
        spinner="dots",
    ):
        result = run_official_check(
            args.check,
            database_path=args.database_path,
            other_database_path=args.other_database_path,
            error_file_path=args.error_file_path,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
            show_id=args.show_id,
        )
    if result.returncode != 0:
        # The official verifier output contains one progress bar per worker.
        # Keep successful checks quiet; expose only a compact tail on failure.
        diagnostics = "\n".join(
            line
            for stream in (result.stdout, result.stderr)
            for line in stream.splitlines()[-20:]
        )
        if diagnostics:
            console.print("Verifier diagnostics (last 20 lines):", style="bold red")
            console.print(diagnostics, style="red")
    print_panel(
        f"Official {args.check} check completed",
        f"Database: {args.database_path}\nExit code: {result.returncode}",
        style="green" if result.returncode == 0 else "red",
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
