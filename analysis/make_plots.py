from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required to generate plots. Install it in your environment first."
    ) from exc


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    mean_v = sum(values) / len(values)
    if len(values) <= 1:
        return mean_v, 0.0
    var = sum((x - mean_v) ** 2 for x in values) / (len(values) - 1)
    ci = 1.96 * math.sqrt(var / len(values))
    return mean_v, ci


def _plot_learning_curve(
    rows: list[dict[str, str]],
    promotions_rows: list[dict[str, str]],
    metric: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    # algo -> step -> [values across seeds]
    bucket: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        algo = str(row.get("algorithm", "")).strip()
        step = _to_float(row.get("global_step"))
        value = _to_float(row.get(metric))
        if not algo or step is None or value is None:
            continue
        bucket[algo][int(step)].append(float(value))

    if not bucket:
        return

    plt.figure(figsize=(8, 5))
    color_by_algo: dict[str, Any] = {}
    for algo in sorted(bucket.keys()):
        xs = sorted(bucket[algo].keys())
        ys: list[float] = []
        ci: list[float] = []
        for x in xs:
            m, c = _mean_ci95(bucket[algo][x])
            ys.append(m)
            ci.append(c)
        lower = [y - c for y, c in zip(ys, ci)]
        upper = [y + c for y, c in zip(ys, ci)]
        line = plt.plot(xs, ys, label=algo)[0]
        color_by_algo[algo] = line.get_color()
        plt.fill_between(xs, lower, upper, alpha=0.2)

    # Promotion markers: per algorithm, draw light dashed vertical lines at mean promotion steps.
    # Group by (algorithm, seed, run_id) to avoid duplicated rows.
    per_run_promotions: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in promotions_rows:
        algo = str(row.get("algorithm", "")).strip()
        seed = str(row.get("seed", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        event_type = str(row.get("event_type", "")).strip().lower()
        step = _to_float(row.get("global_step"))
        if not algo or not seed or not run_id or step is None:
            continue
        if event_type != "promoted":
            continue
        per_run_promotions[(algo, seed, run_id)].append(float(step))

    # Convert per-run ordered promotions into per-algo transitions.
    per_algo_transition_steps: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (algo, _seed, _run_id), steps in per_run_promotions.items():
        ordered = sorted(steps)
        for idx, step in enumerate(ordered):
            per_algo_transition_steps[algo][idx].append(step)

    for algo, transitions in per_algo_transition_steps.items():
        color = color_by_algo.get(algo)
        if color is None:
            continue
        for _t_idx, vals in transitions.items():
            if not vals:
                continue
            mean_step, _ci = _mean_ci95(vals)
            plt.axvline(
                x=mean_step,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.25,
            )

    plt.title(title)
    plt.xlabel("global_step")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Wrote plot -> {output_path}")


def _plot_tradeoff(rows: list[dict[str, str]], output_path: Path) -> None:
    by_algo_x: dict[str, list[float]] = defaultdict(list)
    by_algo_y: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        algo = str(row.get("algorithm", "")).strip()
        x = _to_float(row.get("collision_rate"))
        y = _to_float(row.get("success_rate"))
        if algo and x is not None and y is not None:
            by_algo_x[algo].append(float(x))
            by_algo_y[algo].append(float(y))

    if not by_algo_x:
        return

    plt.figure(figsize=(7, 5))
    for algo in sorted(by_algo_x.keys()):
        x_m, _ = _mean_ci95(by_algo_x[algo])
        y_m, _ = _mean_ci95(by_algo_y[algo])
        plt.scatter([x_m], [y_m], label=algo)
        plt.annotate(algo, (x_m, y_m))

    plt.title("Safety-Performance Tradeoff")
    plt.xlabel("collision_rate (lower is better)")
    plt.ylabel("success_rate (higher is better)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Wrote plot -> {output_path}")


def _plot_rule_violation_by_rule(rows: list[dict[str, str]], output_path: Path) -> None:
    # rule -> algo -> values
    bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rule = str(row.get("rule_name", "")).strip()
        algo = str(row.get("algorithm", "")).strip()
        val = _to_float(row.get("violation_rate"))
        if rule and algo and val is not None:
            bucket[rule][algo].append(float(val))
    if not bucket:
        return

    rules = sorted(bucket.keys())
    algos = sorted({algo for data in bucket.values() for algo in data.keys()})
    width = 0.8 / max(len(algos), 1)
    x_positions = list(range(len(rules)))

    plt.figure(figsize=(max(8, len(rules) * 0.8), 5))
    for idx, algo in enumerate(algos):
        ys = []
        xs = []
        for r_idx, rule in enumerate(rules):
            values = bucket[rule].get(algo, [])
            if values:
                m, _ = _mean_ci95(values)
                ys.append(m)
            else:
                ys.append(0.0)
            xs.append(r_idx + idx * width - 0.4 + width / 2.0)
        plt.bar(xs, ys, width=width, label=algo)

    plt.xticks(x_positions, rules, rotation=45, ha="right")
    plt.title("Violation Rate by Rule")
    plt.ylabel("violation_rate")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Wrote plot -> {output_path}")


def _plot_error_boxplot(rows: list[dict[str, str]], output_path: Path) -> None:
    bucket: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        algo = str(row.get("algorithm", "")).strip()
        val = _to_float(row.get("error_value"))
        if algo and val is not None:
            bucket[algo].append(float(val))
    if not bucket:
        return

    algos = sorted(bucket.keys())
    data = [bucket[algo] for algo in algos]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=algos, showfliers=False)
    plt.title("Episode Error Value Distribution")
    plt.ylabel("error_value")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Wrote plot -> {output_path}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_plots(aggregated_dir: Path, plots_dir: Path) -> None:
    eval_rows = _read_rows(aggregated_dir / "evals_all_runs.csv")
    final_rows = _read_rows(aggregated_dir / "final_eval_all_runs.csv")
    rule_rows = _read_rows(aggregated_dir / "rule_metrics_all_runs.csv")
    episode_rows = _read_rows(aggregated_dir / "eval_episodes_all_runs.csv")
    promotions_rows = _read_rows(aggregated_dir / "promotions_all_runs.csv")

    _plot_learning_curve(eval_rows, promotions_rows, "success_rate", plots_dir / "learning_success_vs_global_step.png", "Success vs Global Step", "success_rate")
    _plot_learning_curve(eval_rows, promotions_rows, "collision_rate", plots_dir / "learning_collision_vs_global_step.png", "Collision vs Global Step", "collision_rate")
    _plot_learning_curve(eval_rows, promotions_rows, "out_of_road_rate", plots_dir / "learning_out_of_road_vs_global_step.png", "Out-of-road vs Global Step", "out_of_road_rate")
    _plot_learning_curve(eval_rows, promotions_rows, "route_completion", plots_dir / "learning_route_completion_vs_global_step.png", "Route Completion vs Global Step", "route_completion")
    _plot_learning_curve(eval_rows, promotions_rows, "top_rule_violation_rate", plots_dir / "learning_rule_top_violation_vs_global_step.png", "Top Rule Violation vs Global Step", "top_rule_violation_rate")
    _plot_learning_curve(eval_rows, promotions_rows, "avg_error_value", plots_dir / "learning_avg_error_value_vs_global_step.png", "Avg Error Value vs Global Step", "avg_error_value")
    _plot_learning_curve(eval_rows, promotions_rows, "stage_index", plots_dir / "curriculum_stage_index_vs_global_step.png", "Curriculum Stage vs Global Step", "stage_index")

    _plot_tradeoff(final_rows, plots_dir / "safety_performance_tradeoff.png")
    _plot_rule_violation_by_rule(rule_rows, plots_dir / "rule_metrics_violation_rate_by_rule.png")
    _plot_error_boxplot(episode_rows, plots_dir / "episode_error_distribution_boxplot.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison plots from aggregated CSVs.")
    parser.add_argument("--analysis-root", default="analysis")
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    make_plots(
        aggregated_dir=analysis_root / "aggregated",
        plots_dir=analysis_root / "plots",
    )


if __name__ == "__main__":
    main()
