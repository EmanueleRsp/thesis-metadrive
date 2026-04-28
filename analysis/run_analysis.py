from __future__ import annotations

import argparse
from pathlib import Path

from analysis.aggregate_runs import aggregate_runs
from analysis.make_curriculum_tables import build_curriculum_tables
from analysis.make_final_tables import build_final_tables
from analysis.make_plots import make_plots
from analysis.make_rulebook_tables import build_rulebook_tables
from analysis.render_selected_videos import render_selected_videos
from analysis.select_video_episodes import select_video_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full analysis pipeline (aggregate + tables + plots).")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--analysis-root", default="analysis")
    parser.add_argument("--total-timesteps", default=None)
    parser.add_argument("--eval-episodes", default=None)
    parser.add_argument("--final-eval-episodes", default=None)
    parser.add_argument("--seed-list", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--only", choices=("all", "aggregate", "tables", "plots"), default="all")
    parser.add_argument("--no-videos", action="store_true", help="Disable video selection/rendering stage.")
    parser.add_argument("--video-max", type=int, default=5, help="Max selected videos per run.")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    analysis_root = Path(args.analysis_root)
    seed_list = [int(item.strip()) for item in str(args.seed_list).split(",") if item.strip()]

    selected_runs = []
    if args.only in ("all", "aggregate"):
        selected_runs = aggregate_runs(
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

    if args.only == "all" and not bool(args.no_videos):
        if not selected_runs:
            selected_runs = aggregate_runs(
                outputs_root=outputs_root,
                analysis_root=analysis_root,
                total_timesteps=args.total_timesteps,
                eval_episodes=args.eval_episodes,
                final_eval_episodes=args.final_eval_episodes,
                expected_seeds=seed_list,
            )
        for run in selected_runs:
            try:
                select_video_episodes(
                    run_dir=run.run_dir,
                    source="final",
                    max_videos=int(args.video_max),
                )
                render_selected_videos(run_dir=run.run_dir)
            except Exception as exc:
                print(f"[video] Skipped run {run.run_dir}: {exc}")


if __name__ == "__main__":
    main()
