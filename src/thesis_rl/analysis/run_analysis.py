from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from thesis_rl.analysis.aggregate.aggregate_runs import aggregate_runs
from thesis_rl.analysis.comparisons.make_comparison_views import make_comparison_views
from thesis_rl.analysis.tables.make_curriculum_tables import build_curriculum_tables
from thesis_rl.analysis.tables.make_factor_effect_tables import build_factor_effect_tables
from thesis_rl.analysis.tables.make_final_tables import build_final_tables
from thesis_rl.analysis.tables.make_generalization_tables import build_generalization_tables
from thesis_rl.analysis.videos.make_qualitative_manifest import build_qualitative_manifest
from thesis_rl.analysis.tables.make_rulebook_tables import build_rulebook_tables
from thesis_rl.analysis.tables.make_sample_efficiency_tables import build_sample_efficiency_tables


def _build_tables_for_root(
    *,
    aggregated_dir: Path,
    tables_dir: Path,
    include_effects_tables: bool,
    success_threshold: float,
    collision_threshold: float,
    route_completion_threshold: float,
) -> None:
    build_final_tables(
        aggregated_dir=aggregated_dir,
        tables_dir=tables_dir,
    )
    build_curriculum_tables(
        aggregated_dir=aggregated_dir,
        tables_dir=tables_dir,
    )
    build_rulebook_tables(
        aggregated_dir=aggregated_dir,
        tables_dir=tables_dir,
    )
    build_sample_efficiency_tables(
        aggregated_dir=aggregated_dir,
        tables_dir=tables_dir,
        success_threshold=success_threshold,
        collision_threshold=collision_threshold,
        route_completion_threshold=route_completion_threshold,
    )
    build_generalization_tables(
        aggregated_dir=aggregated_dir,
        tables_dir=tables_dir,
    )
    if include_effects_tables:
        build_factor_effect_tables(
            aggregated_dir=aggregated_dir,
            tables_dir=tables_dir,
        )


def _build_plots_for_root(
    *,
    aggregated_dir: Path,
    plots_dir: Path,
    include_diagnostic_plots: bool,
) -> None:
    from thesis_rl.analysis.plots.make_plots import make_plots

    make_plots(
        aggregated_dir=aggregated_dir,
        plots_dir=plots_dir,
        include_diagnostics=include_diagnostic_plots,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full analysis pipeline (aggregate + tables + plots).")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--analysis-root", default="outputs/analysis")
    parser.add_argument(
        "--run-profile",
        required=True,
        help="Run profile scope for this analysis (e.g. smoke|fast|medium|long). Outputs are written to outputs/analysis/<run_profile>/.",
    )
    parser.add_argument("--total-timesteps", default=None)
    parser.add_argument("--eval-episodes", default=None)
    parser.add_argument("--final-eval-episodes", default=None)
    parser.add_argument("--seed-list", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--only", choices=("all", "aggregate", "tables", "plots"), default="all")
    parser.add_argument(
        "--comparison-dimension",
        choices=("none", "curriculum", "reward", "algorithm"),
        default="none",
        help="Build reports for one varying factor only (A/B/C style comparisons).",
    )
    parser.add_argument(
        "--comparison-id",
        default=None,
        help="Optional specific comparison id (from comparisons/*/comparison_manifest.csv).",
    )
    parser.add_argument("--algorithm", default=None, help="Optional fixed algorithm filter.")
    parser.add_argument("--reward-type", default=None, help="Optional fixed reward_type filter.")
    parser.add_argument("--reward-behavior", default=None, help="Optional fixed reward_behavior filter.")
    parser.add_argument(
        "--reward-granularity",
        choices=("semantic", "raw"),
        default="semantic",
        help="How reward variants are grouped in comparison views.",
    )
    parser.add_argument("--curriculum-name", default=None, help="Optional fixed curriculum_name filter.")
    parser.add_argument("--rulebook-config", default=None, help="Optional fixed rulebook_config filter.")
    parser.add_argument("--success-threshold", type=float, default=0.70)
    parser.add_argument("--collision-threshold", type=float, default=0.20)
    parser.add_argument("--route-completion-threshold", type=float, default=0.80)
    parser.add_argument(
        "--include-effects-tables",
        action="store_true",
        help="Include ablation/effect tables (non-core).",
    )
    parser.add_argument(
        "--include-diagnostic-plots",
        action="store_true",
        help="Include non-core diagnostic plots.",
    )
    parser.add_argument(
        "--include-qualitative-pack",
        action="store_true",
        help="Build curated qualitative manifest and GIFs for comparison views.",
    )
    parser.add_argument(
        "--qualitative-max-per-category",
        type=int,
        default=1,
        help="Max episodes per category in curated qualitative pack (currently supports 1).",
    )
    parser.add_argument("--no-videos", action="store_true", help="Disable video selection/rendering stage.")
    parser.add_argument("--video-max", type=int, default=5, help="Max selected videos per run.")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    analysis_root = Path(args.analysis_root) / str(args.run_profile).strip()
    comparison_dimension = str(args.comparison_dimension).strip().lower()
    seed_list = [int(item.strip()) for item in str(args.seed_list).split(",") if item.strip()]

    selected_runs = []
    if args.only in ("all", "aggregate"):
        selected_runs = aggregate_runs(
            outputs_root=outputs_root,
            analysis_root=analysis_root,
            run_profile=str(args.run_profile).strip(),
            total_timesteps=args.total_timesteps,
            eval_episodes=args.eval_episodes,
            final_eval_episodes=args.final_eval_episodes,
            expected_seeds=seed_list,
        )

    comparison_roots: list[Path] = []
    if comparison_dimension != "none":
        comparison_roots = make_comparison_views(
            aggregated_dir=analysis_root / "aggregated",
            comparisons_root=analysis_root / "comparisons",
            dimension=comparison_dimension,
            comparison_id=args.comparison_id,
            algorithm=args.algorithm,
            reward_type=args.reward_type,
            reward_behavior=args.reward_behavior,
            reward_granularity=str(args.reward_granularity).strip().lower(),
            curriculum_name=args.curriculum_name,
            rulebook_config=args.rulebook_config,
        )

    if args.only in ("all", "tables"):
        if comparison_roots:
            for root in comparison_roots:
                _build_tables_for_root(
                    aggregated_dir=root / "aggregated",
                    tables_dir=root / "tables",
                    include_effects_tables=bool(args.include_effects_tables),
                    success_threshold=float(args.success_threshold),
                    collision_threshold=float(args.collision_threshold),
                    route_completion_threshold=float(args.route_completion_threshold),
                )
        else:
            _build_tables_for_root(
                aggregated_dir=analysis_root / "aggregated",
                tables_dir=analysis_root / "tables",
                include_effects_tables=bool(args.include_effects_tables),
                success_threshold=float(args.success_threshold),
                collision_threshold=float(args.collision_threshold),
                route_completion_threshold=float(args.route_completion_threshold),
            )

    if args.only in ("all", "plots"):
        if comparison_roots:
            for root in comparison_roots:
                _build_plots_for_root(
                    aggregated_dir=root / "aggregated",
                    plots_dir=root / "plots",
                    include_diagnostic_plots=bool(args.include_diagnostic_plots),
                )
        else:
            _build_plots_for_root(
                aggregated_dir=analysis_root / "aggregated",
                plots_dir=analysis_root / "plots",
                include_diagnostic_plots=bool(args.include_diagnostic_plots),
            )

    if bool(args.include_qualitative_pack):
        if not comparison_roots:
            print(
                "[qualitative] Skipped: requires comparison views. "
                "Set --comparison-dimension to curriculum/reward/algorithm."
            )
        else:
            for root in comparison_roots:
                build_qualitative_manifest(
                    comparison_root=root,
                    max_per_category=int(args.qualitative_max_per_category),
                )
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "thesis_rl.analysis.videos.render_qualitative_videos",
                            "--comparison-root",
                            str(root),
                        ],
                        check=True,
                    )
                except Exception as exc:
                    print(f"[qualitative] Skipped render for {root}: {exc}")

    if args.only == "all" and not bool(args.no_videos):
        from thesis_rl.analysis.videos.select_video_episodes import select_video_episodes

        if not selected_runs:
            selected_runs = aggregate_runs(
                outputs_root=outputs_root,
                analysis_root=analysis_root,
                run_profile=str(args.run_profile).strip(),
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
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "thesis_rl.analysis.videos.render_selected_videos",
                        "--run-dir",
                        str(run.run_dir),
                    ],
                    check=True,
                )
            except Exception as exc:
                print(f"[video] Skipped run {run.run_dir}: {exc}")


if __name__ == "__main__":
    main()
