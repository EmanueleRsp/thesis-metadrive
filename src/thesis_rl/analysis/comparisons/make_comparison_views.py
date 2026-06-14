from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from thesis_rl.common.paths import default_analysis_root_str


ALL_RUNS_FILES = (
    "train_chunks_all_runs.csv",
    "evals_all_runs.csv",
    "eval_episodes_all_runs.csv",
    "promotions_all_runs.csv",
    "rule_metrics_all_runs.csv",
    "final_eval_all_runs.csv",
    "selected_runs.csv",
)


@dataclass(frozen=True)
class ComparisonView:
    comparison_id: str
    dimension: str
    condition_ids: tuple[str, ...]
    summary: str


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregated file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _collect_descriptors(final_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    descriptors: dict[str, dict[str, str]] = {}
    required = (
        "condition_id",
        "algorithm",
        "reward_type",
        "reward_behavior",
        "curriculum_name",
        "rulebook_config",
    )
    for row in final_rows:
        for key in required:
            if str(row.get(key, "")).strip() == "":
                raise ValueError(f"Missing required descriptor field '{key}' in final_eval_all_runs.csv")
        condition_id = str(row["condition_id"]).strip()
        payload = {
            "condition_id": condition_id,
            "algorithm": str(row["algorithm"]).strip(),
            "reward_type": str(row["reward_type"]).strip(),
            "reward_behavior": str(row["reward_behavior"]).strip(),
            "curriculum_name": str(row["curriculum_name"]).strip(),
            "rulebook_config": str(row["rulebook_config"]).strip(),
        }
        prev = descriptors.get(condition_id)
        if prev is None:
            descriptors[condition_id] = payload
        elif prev != payload:
            raise ValueError(f"Inconsistent descriptors for condition_id '{condition_id}' in final_eval_all_runs.csv")
    return descriptors


def _apply_descriptor_filters(
    descriptors: dict[str, dict[str, str]],
    *,
    algorithm: str | None = None,
    reward_type: str | None = None,
    reward_behavior: str | None = None,
    curriculum_name: str | None = None,
    rulebook_config: str | None = None,
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for condition_id, desc in descriptors.items():
        if algorithm and desc["algorithm"] != algorithm:
            continue
        if reward_type and desc["reward_type"] != reward_type:
            continue
        if reward_behavior and desc["reward_behavior"] != reward_behavior:
            continue
        if curriculum_name and desc["curriculum_name"] != curriculum_name:
            continue
        if rulebook_config and desc["rulebook_config"] != rulebook_config:
            continue
        out[condition_id] = desc
    return out


def _build_curriculum_views(descriptors: dict[str, dict[str, str]]) -> list[ComparisonView]:
    groups: dict[tuple[str, str, str, str], dict[str, list[str]]] = {}
    for condition_id, desc in descriptors.items():
        key = (
            desc["algorithm"],
            desc["reward_type"],
            desc["reward_behavior"],
            desc["rulebook_config"],
        )
        bucket = groups.setdefault(key, {})
        bucket.setdefault(desc["curriculum_name"], []).append(condition_id)

    views: list[ComparisonView] = []
    for key, bucket in sorted(groups.items()):
        curricula = sorted(bucket.keys())
        if len(curricula) < 2:
            continue
        algorithm, reward_type, reward_behavior, rulebook_config = key
        comparison_id = f"{algorithm}__{reward_type}__{reward_behavior}__{rulebook_config}"
        summary = (
            f"curriculum effect | algorithm={algorithm} | reward_type={reward_type} | "
            f"reward_behavior={reward_behavior} | rulebook_config={rulebook_config}"
        )
        condition_ids: list[str] = []
        for curriculum in curricula:
            condition_ids.extend(bucket[curriculum])
        views.append(
            ComparisonView(
                comparison_id=comparison_id,
                dimension="curriculum",
                condition_ids=tuple(sorted(condition_ids)),
                summary=summary,
            )
        )
    return views


def _reward_value(desc: dict[str, str], reward_granularity: str) -> str:
    if reward_granularity == "raw":
        return desc["reward_behavior"]
    return desc["reward_type"]


def _build_reward_views(
    descriptors: dict[str, dict[str, str]],
    *,
    reward_granularity: str,
) -> list[ComparisonView]:
    groups: dict[tuple[str, str], dict[str, list[str]]] = {}
    for condition_id, desc in descriptors.items():
        key = (
            desc["algorithm"],
            desc["curriculum_name"],
        )
        reward_value = _reward_value(desc, reward_granularity)
        groups.setdefault(key, {}).setdefault(reward_value, []).append(condition_id)

    views: list[ComparisonView] = []
    for key, by_reward in sorted(groups.items()):
        reward_values = sorted(by_reward.keys())
        if len(reward_values) < 2:
            continue
        algorithm, curriculum_name = key
        comparison_id = f"{algorithm}__{curriculum_name}"
        if reward_granularity == "raw":
            comparison_id = f"{comparison_id}__raw_reward"
        summary = (
            f"reward effect | algorithm={algorithm} | curriculum={curriculum_name} "
            f"| reward_granularity={reward_granularity}"
        )
        condition_ids: list[str] = []
        for reward_value in reward_values:
            condition_ids.extend(by_reward[reward_value])
        views.append(
            ComparisonView(
                comparison_id=comparison_id,
                dimension="reward",
                condition_ids=tuple(sorted(condition_ids)),
                summary=summary,
            )
        )
    return views


def _build_algorithm_views(descriptors: dict[str, dict[str, str]]) -> list[ComparisonView]:
    groups: dict[tuple[str, str, str, str], dict[str, list[str]]] = {}
    for condition_id, desc in descriptors.items():
        key = (
            desc["reward_type"],
            desc["reward_behavior"],
            desc["curriculum_name"],
            desc["rulebook_config"],
        )
        groups.setdefault(key, {}).setdefault(desc["algorithm"], []).append(condition_id)

    views: list[ComparisonView] = []
    for key, by_algorithm in sorted(groups.items()):
        algorithms = sorted(by_algorithm.keys())
        if len(algorithms) < 2:
            continue
        reward_type, reward_behavior, curriculum_name, rulebook_config = key
        comparison_id = (
            f"{reward_type}__{reward_behavior}__{curriculum_name}__{rulebook_config}"
        )
        summary = (
            f"algorithm effect | reward_type={reward_type} | reward_behavior={reward_behavior} "
            f"| curriculum={curriculum_name} | rulebook_config={rulebook_config}"
        )
        condition_ids: list[str] = []
        for algorithm in algorithms:
            condition_ids.extend(by_algorithm[algorithm])
        views.append(
            ComparisonView(
                comparison_id=comparison_id,
                dimension="algorithm",
                condition_ids=tuple(sorted(condition_ids)),
                summary=summary,
            )
        )
    return views


def _write_view_aggregated(
    source_rows: list[dict[str, str]],
    source_fieldnames: list[str],
    condition_ids: set[str],
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        row for row in source_rows if str(row.get("condition_id", "")).strip() in condition_ids
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return len(selected)


def make_comparison_views(
    *,
    aggregated_dir: Path,
    comparisons_root: Path,
    dimension: str,
    comparison_id: str | None = None,
    algorithm: str | None = None,
    reward_type: str | None = None,
    reward_behavior: str | None = None,
    reward_granularity: str = "semantic",
    curriculum_name: str | None = None,
    rulebook_config: str | None = None,
) -> list[Path]:
    if dimension not in {"curriculum", "reward", "algorithm"}:
        raise ValueError(f"Unsupported comparison dimension: {dimension}")
    if reward_granularity not in {"semantic", "raw"}:
        raise ValueError(f"Unsupported reward_granularity: {reward_granularity}")

    final_path = aggregated_dir / "final_eval_all_runs.csv"
    final_rows = _read_rows(final_path)
    descriptors = _collect_descriptors(final_rows)
    descriptors = _apply_descriptor_filters(
        descriptors,
        algorithm=algorithm,
        reward_type=reward_type,
        reward_behavior=reward_behavior,
        curriculum_name=curriculum_name,
        rulebook_config=rulebook_config,
    )

    if dimension == "curriculum":
        views = _build_curriculum_views(descriptors)
    elif dimension == "reward":
        views = _build_reward_views(descriptors, reward_granularity=reward_granularity)
    else:
        views = _build_algorithm_views(descriptors)

    if comparison_id:
        views = [view for view in views if view.comparison_id == comparison_id]

    if not views:
        raise ValueError(
            f"No comparison views found for dimension='{dimension}'"
            + (f" and comparison_id='{comparison_id}'." if comparison_id else ".")
        )

    rows_by_file: dict[str, list[dict[str, str]]] = {}
    fieldnames_by_file: dict[str, list[str]] = {}
    for filename in ALL_RUNS_FILES:
        source_path = aggregated_dir / filename
        if not source_path.exists():
            continue
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {source_path}")
            rows_by_file[filename] = list(reader)
            fieldnames_by_file[filename] = list(reader.fieldnames)

    manifest_path = comparisons_root / dimension / "comparison_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    built_roots: list[Path] = []

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dimension", "comparison_id", "summary", "n_conditions", "condition_ids"],
        )
        writer.writeheader()
        for view in views:
            view_root = comparisons_root / dimension / view.comparison_id
            view_aggregated = view_root / "aggregated"
            condition_ids = set(view.condition_ids)
            for filename, rows in rows_by_file.items():
                _ = _write_view_aggregated(
                    source_rows=rows,
                    source_fieldnames=fieldnames_by_file[filename],
                    condition_ids=condition_ids,
                    output_path=view_aggregated / filename,
                )
            writer.writerow(
                {
                    "dimension": view.dimension,
                    "comparison_id": view.comparison_id,
                    "summary": view.summary,
                    "n_conditions": len(view.condition_ids),
                    "condition_ids": ";".join(view.condition_ids),
                }
            )
            built_roots.append(view_root)
            print(f"Wrote comparison view -> {view_root}")

    print(f"Wrote manifest -> {manifest_path}")
    return built_roots


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comparison-specific aggregated views (one varying factor).")
    parser.add_argument("--analysis-root", default=default_analysis_root_str())
    parser.add_argument("--dimension", choices=("curriculum", "reward", "algorithm"), required=True)
    parser.add_argument("--comparison-id", default=None)
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--reward-type", default=None)
    parser.add_argument("--reward-behavior", default=None)
    parser.add_argument("--reward-granularity", choices=("semantic", "raw"), default="semantic")
    parser.add_argument("--curriculum-name", default=None)
    parser.add_argument("--rulebook-config", default=None)
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root)
    _ = make_comparison_views(
        aggregated_dir=analysis_root / "aggregated",
        comparisons_root=analysis_root / "comparisons",
        dimension=args.dimension,
        comparison_id=args.comparison_id,
        algorithm=args.algorithm,
        reward_type=args.reward_type,
        reward_behavior=args.reward_behavior,
        reward_granularity=str(args.reward_granularity).strip().lower(),
        curriculum_name=args.curriculum_name,
        rulebook_config=args.rulebook_config,
    )


if __name__ == "__main__":
    main()
