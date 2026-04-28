from __future__ import annotations

import argparse
from pathlib import Path

from analysis.aggregate_runs import aggregate_runs
from analysis.make_curriculum_tables import build_curriculum_tables
from analysis.make_final_tables import build_final_tables
from analysis.make_plots import make_plots
from analysis.make_rulebook_tables import build_rulebook_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full analysis pipeline (aggregate + tables + plots).")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--analysis-root", default="analysis")
    parser.add_argument("--total-timesteps", default=None)
    parser.add_argument("--eval-episodes", default=None)
    parser.add_argument("--final-eval-episodes", default=None)
    parser.add_argument("--seed-list", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--only", choices=("all", "aggregate", "tables", "plots"), default="all")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    analysis_root = Path(args.analysis_root)
    seed_list = [int(item.strip()) for item in str(args.seed_list).split(",") if item.strip()]

    if args.only in ("all", "aggregate"):
        aggregate_runs(
            outputs_root=outputs_root,
            analysis_root=analysis_root,
            total_timesteps=args.total_timesteps,
            eval_episodes=args.eval_episodes,
            final_eval_episodes=args.final_eval_episodes,
            expected_seeds=seed_list,
        )

    if args.only in ("all", "tables"):
        build_final_tables(
            aggregated_dir=analysis_root / "aggregated",
            tables_dir=analysis_root / "tables",
        )
        build_curriculum_tables(
            aggregated_dir=analysis_root / "aggregated",
            tables_dir=analysis_root / "tables",
        )
        build_rulebook_tables(
            aggregated_dir=analysis_root / "aggregated",
            tables_dir=analysis_root / "tables",
        )

    if args.only in ("all", "plots"):
        make_plots(
            aggregated_dir=analysis_root / "aggregated",
            plots_dir=analysis_root / "plots",
        )


if __name__ == "__main__":
    main()
