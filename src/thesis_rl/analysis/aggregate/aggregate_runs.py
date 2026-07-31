from __future__ import annotations

import argparse
import csv
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from thesis_rl.common.paths import default_analysis_root_str, default_outputs_root_str

CSV_FILENAMES = (
    "train_chunks.csv",
    "evals.csv",
    "eval_episodes.csv",
    "promotions.csv",
    "rule_metrics.csv",
    # EP-SUBRULE-DIAG: additive diagnostic, `DEC-SUB-003` -- runs recorded
    # before this feature simply have no rows here, not an error.
    "subrule_metrics.csv",
    "final_eval.csv",
)

FINAL_EVAL_REQUIRED_COLUMNS = (
    "run_id",
    "eval_type",
    "scenario_set",
    "algorithm",
    "reward_type",
    "reward_behavior",
    "curriculum_name",
    "curriculum_enabled",
    "rulebook_config",
    "seed",
    "total_timesteps",
    "final_eval_episodes",
)

SCENARIONET_FINAL_PANEL_SETS = {
    "full": {"test_waymo_empirical", "test_pg", "test_arm_stratified"},
    "smoke": {"test_waymo_empirical", "test_pg", "test_arm_stratified"},
    "fast": {"test_waymo_empirical", "test_pg", "test_arm_stratified"},
}

CONTEXT_FIELDS = (
    "run_dir",
    "run_name",
    "seed_dir",
    "timestamp_dir",
    "metadata_status",
    "condition_id",
    "algorithm",
    "task_contract",
    "reward_type",
    "reward_behavior",
    "curriculum_name",
    "curriculum_enabled",
    "rulebook_config",
    "experiment_group",
    "run_profile",
)

INFERRED_FIELDS_BY_FILE = {
    "final_eval.csv": ("eval_type", "scenario_set", "steps_to_final_stage"),
    "evals.csv": ("eval_type", "scenario_set"),
    "eval_episodes.csv": ("eval_type", "scenario_set"),
    "rule_metrics.csv": ("eval_type", "scenario_set"),
    "subrule_metrics.csv": ("eval_type", "scenario_set"),
}


def _extend_unique(target: list[str], items: Iterable[str]) -> None:
    seen = set(target)
    for item in items:
        key = str(item).strip()
        if key == "" or key in seen:
            continue
        target.append(key)
        seen.add(key)


def _to_int(value: object) -> int | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    run_name: str
    algorithm: str
    task_contract: str
    reward_type: str
    reward_behavior: str
    curriculum_name: str
    curriculum_enabled: str
    rulebook_config: str
    condition_id: str
    experiment_group: str
    run_profile: str
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


def _read_final_eval_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required file: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        missing_columns = [col for col in FINAL_EVAL_REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {csv_path}: {missing_columns}. "
                "Regenerate runs with updated CSV schema."
            )

        rows = [{k: str(v) for k, v in row.items()} for row in reader]
        if rows:
            _validate_final_panel_rows(rows, csv_path)
            return rows

    raise ValueError(f"final_eval.csv has no data rows: {csv_path}")


def _validate_final_panel_rows(rows: list[dict[str, str]], csv_path: Path) -> None:
    """Fail closed on incompatible multi-panel ScenarioNet final results."""
    scopes = {str(row.get("evaluation_scope", "")).strip() for row in rows}
    scopes.discard("")
    if not scopes:
        # Legacy single-final-row files remain analyzable under their original
        # protocol and cannot be combined with v1.2 panel-labelled runs.
        return
    if len(scopes) != 1:
        raise ValueError(f"mixed evaluation scopes in one final_eval.csv: {csv_path}")
    scope = next(iter(scopes))
    expected = SCENARIONET_FINAL_PANEL_SETS.get(scope)
    if expected is None:
        raise ValueError(f"unsupported ScenarioNet evaluation scope {scope!r} in {csv_path}")
    names = [str(row.get("panel_name") or row.get("scenario_set", "")).strip() for row in rows]
    if set(names) != expected or len(names) != len(expected):
        raise ValueError(
            f"final_eval.csv must contain exactly {sorted(expected)} for scope {scope!r}; got {names}"
        )
    for identity_field in ("frozen_selection_hash", "panel_sha256"):
        if any(str(row.get(identity_field, "")).strip() == "" for row in rows):
            raise ValueError(f"panel-labelled final result lacks {identity_field}: {csv_path}")


def _build_condition_id(
    *,
    algorithm: str,
    task_contract: str,
    reward_type: str,
    reward_behavior: str,
    curriculum_name: str,
    curriculum_enabled: str,
    rulebook_config: str,
) -> str:
    parts = [
        algorithm.strip(),
        task_contract.strip(),
        reward_type.strip(),
        reward_behavior.strip(),
        curriculum_name.strip(),
        curriculum_enabled.strip(),
        rulebook_config.strip(),
    ]
    if any(part == "" for part in parts):
        raise ValueError(f"Cannot build condition_id from empty parts: {parts}")
    return "__".join(parts)


def _infer_run_profile(*, metadata: dict[str, str], experiment_group: str) -> str:
    profile = str(metadata.get("run_profile", "")).strip()
    if profile != "":
        return profile

    # Expected format in conf/config.yaml:
    # EXP_<...>_RP_<run_profile>_CUR_<...>_REW_<...>
    match = re.search(r"_RP_(.*?)_CUR_", experiment_group)
    if match:
        return str(match.group(1)).strip()
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
        final_rows = _read_final_eval_rows(final_eval_path)
        first_row = next(
            (
                row
                for row in final_rows
                if str(row.get("panel_name") or row.get("scenario_set", "")).strip()
                == "test_waymo_empirical"
            ),
            final_rows[0],
        )

        algorithm = first_row["algorithm"].strip()
        task_contract = str(metadata.get("task_contract", "")).strip()
        if task_contract == "":
            task_contract = "unknown"
        reward_type = first_row["reward_type"].strip()
        reward_behavior = first_row["reward_behavior"].strip()
        curriculum_name = first_row["curriculum_name"].strip()
        curriculum_enabled = first_row["curriculum_enabled"].strip()
        rulebook_config = first_row["rulebook_config"].strip()
        seed_raw = first_row["seed"].strip()
        total_timesteps = first_row["total_timesteps"].strip()
        final_eval_episodes = first_row["final_eval_episodes"].strip()

        required_values = {
            "algorithm": algorithm,
            "reward_type": reward_type,
            "reward_behavior": reward_behavior,
            "curriculum_name": curriculum_name,
            "curriculum_enabled": curriculum_enabled,
            "rulebook_config": rulebook_config,
            "seed": seed_raw,
        }
        missing_values = [k for k, v in required_values.items() if v == ""]
        if missing_values:
            raise ValueError(
                f"Missing required values in {final_eval_path}: {missing_values}. "
                "No fallback is enabled in strict mode."
            )

        try:
            seed = int(seed_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid integer seed '{seed_raw}' in {final_eval_path}") from exc

        condition_id = _build_condition_id(
            algorithm=algorithm,
            task_contract=task_contract,
            reward_type=reward_type,
            reward_behavior=reward_behavior,
            curriculum_name=curriculum_name,
            curriculum_enabled=curriculum_enabled,
            rulebook_config=rulebook_config,
        )

        eval_episodes = metadata.get("eval_episodes", "")
        experiment_group = str(metadata.get("experiment_group", "")).strip()
        if experiment_group == "":
            raise ValueError(
                f"Missing required metadata field 'experiment_group' in {run_dir / 'artifacts' / 'run_metadata.yaml'}"
            )
        run_profile = _infer_run_profile(metadata=metadata, experiment_group=experiment_group)
        if run_profile == "":
            raise ValueError(
                "Cannot infer run_profile for run "
                f"{run_dir}. Add 'run_profile' in run_metadata.yaml or use standard experiment_group naming."
            )

        run_name = run_dir.parents[1].name if len(run_dir.parents) >= 2 else ""
        runs.append(
            RunInfo(
                run_dir=run_dir,
                run_name=run_name or condition_id,
                algorithm=algorithm,
                task_contract=task_contract,
                reward_type=reward_type,
                reward_behavior=reward_behavior,
                curriculum_name=curriculum_name,
                curriculum_enabled=curriculum_enabled,
                rulebook_config=rulebook_config,
                condition_id=condition_id,
                experiment_group=experiment_group,
                run_profile=run_profile,
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


def _dedupe_latest_by_condition_seed(runs: list[RunInfo]) -> list[RunInfo]:
    selected: dict[tuple[str, str, int], RunInfo] = {}
    for run in runs:
        key = (run.experiment_group, run.condition_id, run.seed)
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
        if total_timesteps:
            if run.total_timesteps == "":
                raise ValueError(f"Run missing total_timesteps but protocol filter requested: {run.run_dir}")
            if run.total_timesteps != total_timesteps:
                continue

        if eval_episodes:
            if run.eval_episodes == "":
                raise ValueError(f"Run missing eval_episodes but protocol filter requested: {run.run_dir}")
            if run.eval_episodes != eval_episodes:
                continue

        if final_eval_episodes:
            if run.final_eval_episodes == "":
                raise ValueError(f"Run missing final_eval_episodes but protocol filter requested: {run.run_dir}")
            if run.final_eval_episodes != final_eval_episodes:
                continue
        out.append(run)
    return out


def _filter_run_profile(runs: list[RunInfo], run_profile: str) -> list[RunInfo]:
    expected = str(run_profile).strip().lower()
    if expected == "":
        raise ValueError("run_profile filter cannot be empty.")
    out = [run for run in runs if str(run.run_profile).strip().lower() == expected]
    if not out:
        raise ValueError(
            f"No runs found for run_profile='{run_profile}'. "
            "Check outputs metadata and ensure completed runs exist for this profile."
        )
    return out


def _warn_seed_coverage(runs: list[RunInfo], expected_seeds: list[int]) -> None:
    by_condition: dict[str, set[int]] = {}
    for run in runs:
        by_condition.setdefault(run.condition_id, set()).add(int(run.seed))
    expected = set(expected_seeds)
    for condition_id, seeds in sorted(by_condition.items()):
        missing = sorted(expected - seeds)
        if missing:
            warnings.warn(
                f"Condition '{condition_id}' missing seeds: {missing}.",
                stacklevel=2,
            )


def _row_with_context(row: dict[str, str], run: RunInfo, *, filename: str) -> dict[str, str]:
    out = dict(row)
    if filename == "final_eval.csv":
        if str(out.get("eval_type", "")).strip() == "":
            out["eval_type"] = "final"
        if str(out.get("scenario_set", "")).strip() == "":
            out["scenario_set"] = "test"
        if str(out.get("steps_to_final_stage", "")).strip() == "":
            reached = str(out.get("final_stage_reached", "")).strip().lower() in {"true", "1", "yes"}
            curriculum_enabled = str(run.curriculum_enabled).strip().lower() in {"true", "1", "yes"}
            if curriculum_enabled:
                out["steps_to_final_stage"] = "0" if reached else "-1"
            else:
                out["steps_to_final_stage"] = "0"

    if filename in {"evals.csv", "eval_episodes.csv", "rule_metrics.csv"}:
        step = _to_int(out.get("global_step"))
        final_step = _to_int(run.total_timesteps)
        is_final = step is not None and final_step is not None and step == final_step
        eval_type_default = "final" if is_final else "intermediate"
        scenario_set_default = "test" if is_final else "curriculum_eval"
        if str(out.get("eval_type", "")).strip() == "":
            out["eval_type"] = eval_type_default
        if str(out.get("scenario_set", "")).strip() == "":
            out["scenario_set"] = scenario_set_default

    out["run_dir"] = str(run.run_dir)
    out["run_name"] = run.run_name
    out["seed_dir"] = f"seed_{run.seed}"
    out["timestamp_dir"] = run.timestamp_dir
    out["metadata_status"] = run.metadata_status
    out["condition_id"] = run.condition_id
    out["algorithm"] = run.algorithm
    out["task_contract"] = run.task_contract
    out["reward_type"] = run.reward_type
    out["reward_behavior"] = run.reward_behavior
    out["curriculum_name"] = run.curriculum_name
    out["curriculum_enabled"] = run.curriculum_enabled
    out["rulebook_config"] = run.rulebook_config
    out["experiment_group"] = run.experiment_group
    out["run_profile"] = run.run_profile
    return out


def _reject_mixed_frozen_panel_identities(rows: list[dict[str, str]]) -> None:
    """Prevent cross-condition comparison of non-identical frozen endpoints."""
    grouped: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    for row in rows:
        scope = str(row.get("evaluation_scope", "")).strip()
        panel = str(row.get("panel_name") or row.get("scenario_set", "")).strip()
        selection = str(row.get("frozen_selection_hash", "")).strip()
        panel_hash = str(row.get("panel_sha256", "")).strip()
        if not scope:
            continue
        key = (str(row.get("experiment_group", "")), scope, panel)
        grouped.setdefault(key, set()).add((selection, panel_hash))
    conflicts = {key: identities for key, identities in grouped.items() if len(identities) > 1}
    if conflicts:
        raise ValueError(
            "Cannot aggregate mixed frozen ScenarioNet panel identities: "
            + "; ".join(f"{key}={sorted(identities)}" for key, identities in sorted(conflicts.items()))
        )


def aggregate_runs(
    outputs_root: Path,
    analysis_root: Path,
    *,
    run_profile: str,
    total_timesteps: str | None = None,
    eval_episodes: str | None = None,
    final_eval_episodes: str | None = None,
    expected_seeds: list[int] | None = None,
) -> list[RunInfo]:
    expected_seeds = expected_seeds or list(range(10))
    aggregated_dir = analysis_root / "aggregated"
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(outputs_root)
    runs = _filter_run_profile(runs, run_profile)
    runs = _filter_protocol(runs, total_timesteps, eval_episodes, final_eval_episodes)
    runs = _dedupe_latest_by_condition_seed(runs)
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
                    raise ValueError(f"CSV has no header: {path}")
                fieldnames = fieldnames_by_file.setdefault(filename, [])
                _extend_unique(fieldnames, list(reader.fieldnames))
                _extend_unique(fieldnames, INFERRED_FIELDS_BY_FILE.get(filename, ()))
                _extend_unique(fieldnames, list(CONTEXT_FIELDS))
                for row in reader:
                    collected_rows[filename].append(_row_with_context(row, run, filename=filename))

    _reject_mixed_frozen_panel_identities(collected_rows["final_eval.csv"])

    for filename, rows in collected_rows.items():
        if not rows:
            continue
        output_path = aggregated_dir / filename.replace(".csv", "_all_runs.csv")
        fieldnames = fieldnames_by_file[filename]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        print(f"Wrote {len(rows)} rows -> {output_path}")

    selected_runs_path = aggregated_dir / "selected_runs.csv"
    with selected_runs_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "run_dir",
            "run_name",
            "condition_id",
            "algorithm",
            "task_contract",
            "reward_type",
            "reward_behavior",
            "curriculum_name",
            "curriculum_enabled",
            "rulebook_config",
            "experiment_group",
            "run_profile",
            "seed",
            "timestamp_dir",
            "metadata_status",
            "include_in_comparison",
            "total_timesteps",
            "eval_episodes",
            "final_eval_episodes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "run_dir": str(run.run_dir),
                    "run_name": run.run_name,
                    "condition_id": run.condition_id,
                    "algorithm": run.algorithm,
                    "task_contract": run.task_contract,
                    "reward_type": run.reward_type,
                    "reward_behavior": run.reward_behavior,
                    "curriculum_name": run.curriculum_name,
                    "curriculum_enabled": run.curriculum_enabled,
                    "rulebook_config": run.rulebook_config,
                    "experiment_group": run.experiment_group,
                    "run_profile": run.run_profile,
                    "seed": run.seed,
                    "timestamp_dir": run.timestamp_dir,
                    "metadata_status": run.metadata_status,
                    "include_in_comparison": run.include_in_comparison,
                    "total_timesteps": run.total_timesteps,
                    "eval_episodes": run.eval_episodes,
                    "final_eval_episodes": run.final_eval_episodes,
                }
            )
    print(f"Wrote {len(runs)} rows -> {selected_runs_path}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate latest completed runs per condition/seed.")
    parser.add_argument("--outputs-root", default=default_outputs_root_str())
    parser.add_argument("--analysis-root", default=default_analysis_root_str())
    parser.add_argument("--run-profile", required=True)
    parser.add_argument("--total-timesteps", default=None)
    parser.add_argument("--eval-episodes", default=None)
    parser.add_argument("--final-eval-episodes", default=None)
    parser.add_argument("--seed-list", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()

    seed_list = [int(item.strip()) for item in str(args.seed_list).split(",") if item.strip()]
    _ = aggregate_runs(
        outputs_root=Path(args.outputs_root),
        analysis_root=Path(args.analysis_root),
        run_profile=str(args.run_profile).strip(),
        total_timesteps=args.total_timesteps,
        eval_episodes=args.eval_episodes,
        final_eval_episodes=args.final_eval_episodes,
        expected_seeds=seed_list,
    )


if __name__ == "__main__":
    main()
