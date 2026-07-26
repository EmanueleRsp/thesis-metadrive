from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from thesis_rl.common.paths import default_analysis_root_str

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required to generate plots. Install it in your environment first."
    ) from exc

REQUIRED_CONDITION_COLUMNS = (
    "condition_id",
    "algorithm",
    "reward_type",
    "reward_behavior",
    "curriculum_name",
    "curriculum_enabled",
    "rulebook_config",
)

DESCRIPTOR_COLUMNS = (
    "algorithm",
    "task_contract",
    "reward_type",
    "reward_behavior",
    "curriculum",
    "rulebook_config",
)

DISPLAY_NAME = {
    "algorithm": "Algorithm",
    "task_contract": "Task Contract",
    "reward_type": "Reward Type",
    "reward_behavior": "Reward Behavior",
    "curriculum": "Curriculum",
    "rulebook_config": "Rulebook Config",
}


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


def _mean_sd(values: list[float]) -> tuple[float, float]:
    """Cross-seed mean and sample standard deviation.

    EVAL-PROTOCOL v1.0 REQ-009/DEC-003: no confidence interval, bootstrap
    estimate, or significance test is computed in the core report. This
    replaces the removed ``1.96 * s / sqrt(n)`` 95% CI band with a raw
    mean +/- sample-SD band.
    """
    mean_v = sum(values) / len(values)
    if len(values) <= 1:
        return mean_v, 0.0
    var = sum((x - mean_v) ** 2 for x in values) / (len(values) - 1)
    return mean_v, math.sqrt(var)


def _condition_label(row: dict[str, str]) -> str:
    missing = [c for c in REQUIRED_CONDITION_COLUMNS if str(row.get(c, "")).strip() == ""]
    if missing:
        raise ValueError(f"Missing required condition fields in plot input row: {missing}")
    return str(row["condition_id"]).strip()


def _collect_condition_descriptors(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    descriptors: dict[str, dict[str, str]] = {}
    for row in rows:
        condition_id = _condition_label(row)
        payload = {
            "algorithm": str(row.get("algorithm", "")).strip(),
            "task_contract": str(row.get("task_contract", "")).strip(),
            "reward_type": str(row.get("reward_type", "")).strip(),
            "reward_behavior": str(row.get("reward_behavior", "")).strip(),
            "curriculum": str(row.get("curriculum_name", "")).strip(),
            "rulebook_config": str(row.get("rulebook_config", "")).strip(),
        }
        previous = descriptors.get(condition_id)
        if previous is None:
            descriptors[condition_id] = payload
            continue
        if previous != payload:
            raise ValueError(f"Inconsistent condition descriptors for condition_id '{condition_id}'")
    return descriptors


def _legend_and_common_info(
    rows: list[dict[str, str]],
    condition_ids: list[str],
) -> tuple[dict[str, str], str | None, str]:
    descriptors = _collect_condition_descriptors(rows)
    selected = [descriptors[cid] for cid in condition_ids if cid in descriptors]
    if not selected:
        return ({cid: cid for cid in condition_ids}, None, "Condition")

    varying_keys: list[str] = []
    common_keys: list[str] = []
    for key in DESCRIPTOR_COLUMNS:
        values = {desc[key] for desc in selected}
        if len(values) <= 1:
            common_keys.append(key)
        else:
            varying_keys.append(key)

    legend_title = "Condition"
    if len(varying_keys) == 1:
        legend_title = DISPLAY_NAME.get(varying_keys[0], varying_keys[0])
    elif len(varying_keys) > 1:
        legend_title = "Varies"

    labels: dict[str, str] = {}
    for condition_id in condition_ids:
        desc = descriptors.get(condition_id)
        if desc is None:
            labels[condition_id] = condition_id
            continue
        if len(varying_keys) == 1:
            labels[condition_id] = desc[varying_keys[0]]
        elif len(varying_keys) > 1:
            labels[condition_id] = " | ".join(
                f"{DISPLAY_NAME.get(k, k)}={desc[k]}" for k in varying_keys
            )
        else:
            labels[condition_id] = condition_id

    # Ensure labels remain unique if values collide.
    counts: dict[str, int] = defaultdict(int)
    for value in labels.values():
        counts[value] += 1
    for condition_id, value in list(labels.items()):
        if counts[value] > 1:
            labels[condition_id] = f"{value} ({condition_id})"

    common_lines = [
        f"{DISPLAY_NAME.get(key, key)}: {selected[0][key]}"
        for key in common_keys
        if str(selected[0][key]).strip() != ""
    ]
    common_text = "\n".join(common_lines) if common_lines else None
    return labels, common_text, legend_title


def _add_common_box(common_text: str | None, *, x: float = 0.02, y: float = 0.98) -> None:
    if not common_text:
        return
    ax = plt.gca()
    ax.text(
        x,
        y,
        common_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7", "boxstyle": "round,pad=0.3"},
    )


def _place_common_box_below_legend(common_text: str | None, legend: Any) -> None:
    if not common_text:
        return
    if legend is None:
        _add_common_box(common_text)
        return

    fig = plt.gcf()
    ax = plt.gca()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox_display = legend.get_window_extent(renderer=renderer)
    legend_bbox_axes = legend_bbox_display.transformed(ax.transAxes.inverted())
    x = float(legend_bbox_axes.x0)
    y = float(legend_bbox_axes.y0) - 0.02
    _add_common_box(common_text, x=max(0.01, x), y=min(0.98, y))


def _plot_learning_curve(
    rows: list[dict[str, str]],
    promotions_rows: list[dict[str, str]],
    metric: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    # condition_id -> step -> [values across seeds]
    bucket: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        condition_id = _condition_label(row)
        step = _to_float(row.get("global_step"))
        value = _to_float(row.get(metric))
        if step is None or value is None:
            continue
        bucket[condition_id][int(step)].append(float(value))

    if not bucket:
        return

    condition_ids = sorted(bucket.keys())
    labels_by_condition, common_text, legend_title = _legend_and_common_info(rows, condition_ids)

    plt.figure(figsize=(10, 6))
    color_by_condition: dict[str, Any] = {}
    for condition_id in condition_ids:
        xs = sorted(bucket[condition_id].keys())
        ys: list[float] = []
        ci: list[float] = []
        for x in xs:
            m, c = _mean_sd(bucket[condition_id][x])
            ys.append(m)
            ci.append(c)
        lower = [y - c for y, c in zip(ys, ci)]
        upper = [y + c for y, c in zip(ys, ci)]
        line = plt.plot(xs, ys, label=labels_by_condition.get(condition_id, condition_id))[0]
        color_by_condition[condition_id] = line.get_color()
        plt.fill_between(xs, lower, upper, alpha=0.2)

    per_run_promotions: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in promotions_rows:
        condition_id = _condition_label(row)
        seed = str(row.get("seed", "")).strip()
        run_id = str(row.get("run_id", "")).strip()
        event_type = str(row.get("event_type", "")).strip().lower()
        step = _to_float(row.get("global_step"))
        if seed == "" or run_id == "" or step is None:
            raise ValueError("Missing required promotions fields: seed/run_id/global_step")
        if event_type != "promoted":
            continue
        per_run_promotions[(condition_id, seed, run_id)].append(float(step))

    per_condition_transition_steps: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (condition_id, _seed, _run_id), steps in per_run_promotions.items():
        ordered = sorted(steps)
        for idx, step in enumerate(ordered):
            per_condition_transition_steps[condition_id][idx].append(step)

    for condition_id, transitions in per_condition_transition_steps.items():
        color = color_by_condition.get(condition_id)
        if color is None:
            continue
        for _t_idx, vals in transitions.items():
            if not vals:
                continue
            mean_step, _ci = _mean_sd(vals)
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
    legend = plt.legend(fontsize=8, title=legend_title, loc="upper left")
    _place_common_box_below_legend(common_text, legend)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Wrote plot -> {output_path}")


def _plot_tradeoff(rows: list[dict[str, str]], output_path: Path) -> None:
    by_condition_x: dict[str, list[float]] = defaultdict(list)
    by_condition_y: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        condition_id = _condition_label(row)
        x = _to_float(row.get("collision_rate"))
        y = _to_float(row.get("success_rate"))
        if x is not None and y is not None:
            by_condition_x[condition_id].append(float(x))
            by_condition_y[condition_id].append(float(y))

    if not by_condition_x:
        return

    condition_ids = sorted(by_condition_x.keys())
    labels_by_condition, common_text, legend_title = _legend_and_common_info(rows, condition_ids)

    plt.figure(figsize=(9, 6))
    for condition_id in condition_ids:
        x_m, _ = _mean_sd(by_condition_x[condition_id])
        y_m, _ = _mean_sd(by_condition_y[condition_id])
        label = labels_by_condition.get(condition_id, condition_id)
        plt.scatter([x_m], [y_m], label=label)
        plt.annotate(label, (x_m, y_m), fontsize=8)

    plt.title("Safety-Performance Tradeoff")
    plt.xlabel("collision_rate (lower is better)")
    plt.ylabel("success_rate (higher is better)")
    legend = plt.legend(fontsize=8, title=legend_title, loc="upper left")
    _place_common_box_below_legend(common_text, legend)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Wrote plot -> {output_path}")


def _plot_rule_violation_by_rule(rows: list[dict[str, str]], output_path: Path) -> None:
    # rule -> condition -> values
    bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rule = str(row.get("rule_name", "")).strip()
        condition_id = _condition_label(row)
        val = _to_float(row.get("violation_rate"))
        if rule and val is not None:
            bucket[rule][condition_id].append(float(val))
    if not bucket:
        return

    rules = sorted(bucket.keys())
    conditions = sorted({cond for data in bucket.values() for cond in data.keys()})
    labels_by_condition, common_text, legend_title = _legend_and_common_info(rows, conditions)
    width = 0.8 / max(len(conditions), 1)
    x_positions = list(range(len(rules)))

    plt.figure(figsize=(max(10, len(rules) * 0.9), 6))
    for idx, condition_id in enumerate(conditions):
        ys = []
        xs = []
        for r_idx, rule in enumerate(rules):
            values = bucket[rule].get(condition_id, [])
            if values:
                m, _ = _mean_sd(values)
                ys.append(m)
            else:
                ys.append(0.0)
            xs.append(r_idx + idx * width - 0.4 + width / 2.0)
        plt.bar(xs, ys, width=width, label=labels_by_condition.get(condition_id, condition_id))

    plt.xticks(x_positions, rules, rotation=45, ha="right")
    plt.title("Violation Rate by Rule")
    plt.ylabel("violation_rate")
    legend = plt.legend(fontsize=8, title=legend_title, loc="upper left")
    _place_common_box_below_legend(common_text, legend)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Wrote plot -> {output_path}")


def _plot_error_boxplot(rows: list[dict[str, str]], output_path: Path) -> None:
    bucket: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        condition_id = _condition_label(row)
        val = _to_float(row.get("error_value"))
        if val is not None:
            bucket[condition_id].append(float(val))
    if not bucket:
        return

    conditions = sorted(bucket.keys())
    labels_by_condition, common_text, _ = _legend_and_common_info(rows, conditions)
    data = [bucket[condition_id] for condition_id in conditions]
    labels = [labels_by_condition.get(condition_id, condition_id) for condition_id in conditions]
    plt.figure(figsize=(max(10, len(conditions) * 0.7), 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=25, ha="right")
    plt.title("Episode Error Value Distribution")
    plt.ylabel("error_value")
    _add_common_box(common_text)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Wrote plot -> {output_path}")


def _plot_subrule_dominance_stacked_bar(rows: list[dict[str, str]], output_path: Path) -> None:
    """EP-SUBRULE-DIAG (`REQ-SUB-02`/`REQ-SUB-04`): diagnostic only, never a
    primary comparison metric (`DEC-SUB-001`). One segment filling a bar
    means that macro rule is, in practice, a single sub-rule."""

    # (condition, source, macro_rule) -> subrule -> dominance_share values
    bucket: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        condition_id = _condition_label(row)
        source = str(row.get("scenario_source", "")).strip() or "unknown"
        macro_rule = str(row.get("macro_rule", "")).strip()
        subrule = str(row.get("subrule_name", "")).strip()
        share = _to_float(row.get("dominance_share"))
        if macro_rule and subrule and share is not None:
            bucket[(condition_id, source, macro_rule)][subrule].append(float(share))
    if not bucket:
        return

    categories = sorted(bucket.keys())
    labels_by_condition, common_text, _legend_title = _legend_and_common_info(
        rows, sorted({key[0] for key in categories})
    )
    x_labels = [
        f"{labels_by_condition.get(cond, cond)} | {source} | {macro}"
        for cond, source, macro in categories
    ]
    all_subrules = sorted({name for data in bucket.values() for name in data.keys()})

    plt.figure(figsize=(max(10, len(categories) * 1.1), 6))
    bottoms = [0.0] * len(categories)
    for subrule in all_subrules:
        heights = []
        for category in categories:
            values = bucket[category].get(subrule, [])
            heights.append(_mean_sd(values)[0] if values else 0.0)
        plt.bar(x_labels, heights, bottom=bottoms, label=subrule)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    plt.xticks(rotation=45, ha="right")
    plt.title("Sub-Rule Dominance Share Within Macro Rule (diagnostic, EP-SUBRULE-DIAG)")
    plt.ylabel("dominance_share (of macro-violated steps)")
    legend = plt.legend(fontsize=8, title="Sub-rule", loc="upper left")
    _place_common_box_below_legend(common_text, legend)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Wrote plot -> {output_path}")


def _plot_subrule_cost_distribution_boxplot(rows: list[dict[str, str]], output_path: Path) -> None:
    """EP-SUBRULE-DIAG (`REQ-SUB-03`): calibration view -- why a dominance
    pattern occurs. Each box is the cross-seed distribution of a sub-rule's
    seed-level mean cost (over applicable steps), grouped by macro rule;
    `subrule_metrics.csv` carries seed-level statistics, not raw per-step
    costs, so this is not a per-step distribution."""

    bucket: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        macro_rule = str(row.get("macro_rule", "")).strip()
        subrule = str(row.get("subrule_name", "")).strip()
        cost = _to_float(row.get("mean_cost"))
        if macro_rule and subrule and cost is not None:
            bucket[f"{macro_rule}:{subrule}"].append(float(cost))
    if not bucket:
        return

    keys = sorted(bucket.keys())
    data = [bucket[key] for key in keys]
    plt.figure(figsize=(max(10, len(keys) * 0.9), 6))
    plt.boxplot(data, labels=keys, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.title("Sub-Rule Cost Distribution (diagnostic, EP-SUBRULE-DIAG)")
    plt.ylabel("seed-level mean cost (applicable steps)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Wrote plot -> {output_path}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregated file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _read_optional_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _read_rows(path)


def make_plots(aggregated_dir: Path, plots_dir: Path, *, include_diagnostics: bool = False) -> None:
    eval_rows = _read_rows(aggregated_dir / "evals_all_runs.csv")
    rule_rows = _read_rows(aggregated_dir / "rule_metrics_all_runs.csv")
    promotions_rows = _read_optional_rows(aggregated_dir / "promotions_all_runs.csv")

    # Core thesis figures (behavior, safety, compliance, curriculum progression)
    _plot_learning_curve(eval_rows, promotions_rows, "success_rate", plots_dir / "learning_success_vs_global_step.png", "Success vs Global Step", "success_rate")
    _plot_learning_curve(eval_rows, promotions_rows, "collision_rate", plots_dir / "learning_collision_vs_global_step.png", "Collision vs Global Step", "collision_rate")
    _plot_learning_curve(eval_rows, promotions_rows, "out_of_road_rate", plots_dir / "learning_out_of_road_vs_global_step.png", "Out-of-road vs Global Step", "out_of_road_rate")
    _plot_learning_curve(eval_rows, promotions_rows, "route_completion", plots_dir / "learning_route_completion_vs_global_step.png", "Route Completion vs Global Step", "route_completion")
    _plot_learning_curve(eval_rows, promotions_rows, "top_rule_violation_rate", plots_dir / "learning_rule_top_violation_vs_global_step.png", "Top Rule Violation vs Global Step", "top_rule_violation_rate")
    _plot_learning_curve(eval_rows, promotions_rows, "stage_index", plots_dir / "curriculum_stage_index_vs_global_step.png", "Curriculum Stage vs Global Step", "stage_index")
    _plot_rule_violation_by_rule(rule_rows, plots_dir / "rule_metrics_violation_rate_by_rule.png")

    if include_diagnostics:
        final_rows = _read_rows(aggregated_dir / "final_eval_all_runs.csv")
        episode_rows = _read_rows(aggregated_dir / "eval_episodes_all_runs.csv")
        _plot_learning_curve(eval_rows, promotions_rows, "avg_error_value", plots_dir / "learning_avg_error_value_vs_global_step.png", "Avg Error Value vs Global Step", "avg_error_value")
        _plot_tradeoff(final_rows, plots_dir / "safety_performance_tradeoff.png")
        _plot_error_boxplot(episode_rows, plots_dir / "episode_error_distribution_boxplot.png")
        # EP-SUBRULE-DIAG: additive R2/R3 diagnostics (`DEC-SUB-003`: absent
        # for runs recorded before this feature, handled by `_read_optional_rows`).
        subrule_rows = _read_optional_rows(aggregated_dir / "subrule_metrics_all_runs.csv")
        if subrule_rows:
            _plot_subrule_dominance_stacked_bar(
                subrule_rows, plots_dir / "subrule_dominance_stacked_bar.png"
            )
            _plot_subrule_cost_distribution_boxplot(
                subrule_rows, plots_dir / "subrule_cost_distribution_boxplot.png"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison plots from aggregated CSVs (by condition).")
    parser.add_argument("--analysis-root", default=default_analysis_root_str())
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Include non-core diagnostic plots (error-value curve, tradeoff scatter, error boxplot).",
    )
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    make_plots(
        aggregated_dir=analysis_root / "aggregated",
        plots_dir=analysis_root / "plots",
        include_diagnostics=bool(args.include_diagnostics),
    )


if __name__ == "__main__":
    main()
