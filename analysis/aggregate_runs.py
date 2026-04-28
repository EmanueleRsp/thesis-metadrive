from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CSV_FILENAMES = (
    "train_chunks.csv",
    "evals.csv",
    "eval_episodes.csv",
    "promotions.csv",
    "rule_metrics.csv",
    "final_eval.csv",
)

EXTRA_FIELDS = ("run_dir", "run_name", "seed_dir", "timestamp_dir", "metadata_status")


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    run_name: str
    algorithm: str
    reward_mode: str
    curriculum_name: str
    experiment_group: str
    seed: int
    timestamp_dir: str
    metadata_status: str
    include_in_comparison: bool
    total_timesteps: str
    eval_episodes: str
    final_eval_episodes: str


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or ":" not in clean:
            continue
        key, value = clean.split(":", 1)
        out[key.strip()] = value.strip().strip("'\"")
    return out


def _iter_run_dirs(outputs_root: Path) -> Iterable[Path]:
    for csv_dir in outputs_root.glob("**/csv"):
        if csv_dir.is_dir():
            yield csv_dir.parent


def _read_single_value(csv_path: Path, key: str) -> str:
    if not csv_path.exists():
        return ""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return str(row.get(key, ""))
    return ""


def _discover_runs(outputs_root: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for run_dir in _iter_run_dirs(outputs_root):
        metadata = _parse_simple_yaml(run_dir / "artifacts" / "run_metadata.yaml")
        status = metadata.get("status", "")
        if status != "completed":
            continue
        include_in_comparison = str(metadata.get("include_in_comparison", "true")).lower() in {"1", "true", "yes"}
        if not include_in_comparison:
            continue

        final_eval_path = run_dir / "csv" / "final_eval.csv"
        if not final_eval_path.exists():
            continue

        algorithm = _read_single_value(final_eval_path, "algorithm")
        reward_mode = _read_single_value(final_eval_path, "reward_mode")
        curriculum_name = _read_single_value(final_eval_path, "curriculum_name")
        seed_raw = _read_single_value(final_eval_path, "seed")
        total_timesteps = _read_single_value(final_eval_path, "total_timesteps")
        final_eval_episodes = _read_single_value(final_eval_path, "final_eval_episodes")
        eval_episodes = metadata.get("eval_episodes", "")
        experiment_group = str(metadata.get("experiment_group", "")).strip()
        if not algorithm or not seed_raw:
            continue
        try:
            seed = int(seed_raw)
        except ValueError:
            continue

        run_name = run_dir.parents[1].name if len(run_dir.parents) >= 2 else ""
        runs.append(
            RunInfo(
                run_dir=run_dir,
                run_name=run_name or algorithm,
                algorithm=algorithm,
                reward_mode=reward_mode,
                curriculum_name=curriculum_name,
                experiment_group=experiment_group,
                seed=seed,
                timestamp_dir=run_dir.name,
                metadata_status=status,
                include_in_comparison=include_in_comparison,
                total_timesteps=total_timesteps,
                eval_episodes=eval_episodes,
                final_eval_episodes=final_eval_episodes,
            )
        )
    return runs


def _dedupe_latest_by_algorithm_seed(runs: list[RunInfo]) -> list[RunInfo]:
    selected: dict[tuple[str, str, str, str, int], RunInfo] = {}
    for run in runs:
        key = (
            run.experiment_group or "ungrouped",
            run.algorithm,
            run.reward_mode,
            run.curriculum_name,
            run.seed,
        )
        previous = selected.get(key)
        if previous is None or run.timestamp_dir > previous.timestamp_dir:
            selected[key] = run
    return list(selected.values())


def _filter_protocol(
    runs: list[RunInfo],
    total_timesteps: str | None,
    eval_episodes: str | None,
    final_eval_episodes: str | None,
) -> list[RunInfo]:
    out: list[RunInfo] = []
    for run in runs:
        if total_timesteps and run.total_timesteps and run.total_timesteps != total_timesteps:
            continue
        if eval_episodes and run.eval_episodes and run.eval_episodes != eval_episodes:
            continue
        if final_eval_episodes and run.final_eval_episodes and run.final_eval_episodes != final_eval_episodes:
            continue
        out.append(run)
    return out


def _warn_seed_coverage(runs: list[RunInfo], expected_seeds: list[int]) -> None:
    by_algo: dict[str, set[int]] = {}
    for run in runs:
        algo_key = run.experiment_group or run.run_name
        by_algo.setdefault(algo_key, set()).add(int(run.seed))
    expected = set(expected_seeds)
    for algo, seeds in sorted(by_algo.items()):
        missing = sorted(expected - seeds)
        if missing:
            warnings.warn(
                f"Algorithm '{algo}' missing seeds: {missing}. Included with warning as requested.",
                stacklevel=2,
            )


def _row_with_context(row: dict[str, str], run: RunInfo) -> dict[str, str]:
    out = dict(row)
    out["run_dir"] = str(run.run_dir)
    out["run_name"] = run.run_name
    out["seed_dir"] = f"seed_{run.seed}"
    out["timestamp_dir"] = run.timestamp_dir
    out["metadata_status"] = run.metadata_status
    return out


def aggregate_runs(
    outputs_root: Path,
    analysis_root: Path,
    *,
    total_timesteps: str | None = None,
    eval_episodes: str | None = None,
    final_eval_episodes: str | None = None,
    expected_seeds: list[int] | None = None,
) -> None:
    expected_seeds = expected_seeds or list(range(10))
    aggregated_dir = analysis_root / "aggregated"
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(outputs_root)
    runs = _filter_protocol(runs, total_timesteps, eval_episodes, final_eval_episodes)
    runs = _dedupe_latest_by_algorithm_seed(runs)
    _warn_seed_coverage(runs, expected_seeds)

    selected_dirs = {run.run_dir: run for run in runs}
    collected_rows: dict[str, list[dict[str, str]]] = {name: [] for name in CSV_FILENAMES}
    fieldnames_by_file: dict[str, list[str]] = {}

    for run_dir, run in selected_dirs.items():
        csv_dir = run_dir / "csv"
        for filename in CSV_FILENAMES:
            path = csv_dir / filename
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    continue
                if filename not in fieldnames_by_file:
                    fieldnames_by_file[filename] = list(reader.fieldnames) + list(EXTRA_FIELDS)
                for row in reader:
                    collected_rows[filename].append(_row_with_context(row, run))

    for filename, rows in collected_rows.items():
        if not rows:
            continue
        output_path = aggregated_dir / filename.replace(".csv", "_all_runs.csv")
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames_by_file[filename])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate latest completed runs per experiment/algorithm/reward/curriculum/seed.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--analysis-root", default="analysis")
    parser.add_argument("--total-timesteps", default=None)
    parser.add_argument("--eval-episodes", default=None)
    parser.add_argument("--final-eval-episodes", default=None)
    parser.add_argument("--seed-list", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()

    seed_list = [int(item.strip()) for item in str(args.seed_list).split(",") if item.strip()]
    aggregate_runs(
        outputs_root=Path(args.outputs_root),
        analysis_root=Path(args.analysis_root),
        total_timesteps=args.total_timesteps,
        eval_episodes=args.eval_episodes,
        final_eval_episodes=args.final_eval_episodes,
        expected_seeds=seed_list,
    )


if __name__ == "__main__":
    main()
