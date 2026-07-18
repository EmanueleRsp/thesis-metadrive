from __future__ import annotations

import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from thesis_rl.scenarios.arms import assign_primary_arm
from thesis_rl.scenarios.pg.generator import PGGenerationResult, generate_pg_scenario
from thesis_rl.scenarios.pg.profiles import PG_PROFILES
from thesis_rl.scenarios.reports import write_json_report


@dataclass(frozen=True, slots=True)
class PGPilotReport:
    requested: int
    generated: int
    failed: int
    by_profile_arm: dict[str, dict[str, int]]
    failures: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "generated": self.generated,
            "failed": self.failed,
            "invalid_rate": self.failed / self.requested if self.requested else 0.0,
            "by_profile_arm": self.by_profile_arm,
            "failures": list(self.failures),
        }


PGProgressCallback = Callable[[int, int, str, int, bool], None]


@dataclass(frozen=True, slots=True)
class _PGTask:
    profile: str
    seed: int
    data_root: str
    overwrite: bool
    generator_commit: str | None
    exporter_commit: str | None


def _resolve_profile_counts(
    count_per_profile: int, profile_counts: Mapping[str, int] | None
) -> dict[str, int]:
    if count_per_profile < 1:
        raise ValueError("count_per_profile must be positive")
    configured_counts = (
        {profile.name: count_per_profile for profile in PG_PROFILES}
        if profile_counts is None
        else {profile.name: 0 for profile in PG_PROFILES}
    )
    if profile_counts is not None:
        unknown = sorted(set(profile_counts).difference(configured_counts))
        if unknown:
            raise ValueError(f"unknown PG profiles: {unknown}")
        for profile, count in profile_counts.items():
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"profile count must be a non-negative integer: {profile}")
            configured_counts[profile] = count
    if sum(configured_counts.values()) < 1:
        raise ValueError("at least one PG profile count must be positive")
    return configured_counts


def _run_pg_task(
    task: _PGTask,
) -> tuple[_PGTask, PGGenerationResult | None, dict[str, Any] | None]:
    """Generate one scenario in an isolated process without leaking logs."""

    try:
        # MetaDrive's exporter prints a summary for every scenario. Workers
        # must keep that output private; the parent owns the single Rich bar.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = generate_pg_scenario(
                task.profile,
                seed=task.seed,
                data_root=task.data_root,
                overwrite=task.overwrite,
                generator_commit=task.generator_commit,
                exporter_commit=task.exporter_commit,
            )
    except Exception as exc:
        return (
            task,
            None,
            {
                "profile": task.profile,
                "seed": task.seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
    return task, result, None


def run_pg_pilot(
    *,
    data_root: str | Path,
    count_per_profile: int,
    seed_start: int = 0,
    overwrite: bool = False,
    generator_commit: str | None = None,
    exporter_commit: str | None = None,
    workers: int = 1,
    progress_callback: PGProgressCallback | None = None,
    profile_counts: Mapping[str, int] | None = None,
) -> tuple[PGPilotReport, tuple[PGGenerationResult, ...]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    configured_counts = _resolve_profile_counts(count_per_profile, profile_counts)
    requested = sum(configured_counts.values())

    results: list[PGGenerationResult] = []
    failures: list[dict[str, Any]] = []
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    profile_stride = 1_000_000
    profile_indices = {profile.name: index for index, profile in enumerate(PG_PROFILES)}
    tasks = [
        _PGTask(
            profile=profile.name,
            seed=int(seed_start) + profile_index * profile_stride + offset,
            data_root=str(data_root),
            overwrite=overwrite,
            generator_commit=generator_commit,
            exporter_commit=exporter_commit,
        )
        for profile in PG_PROFILES
        for offset in range(configured_counts[profile.name])
        for profile_index in (profile_indices[profile.name],)
    ]

    def consume(
        completed: int,
        task: _PGTask,
        result: PGGenerationResult | None,
        failure: dict[str, Any] | None,
    ) -> None:
        if result is not None:
            results.append(result)
            matrix[task.profile][assign_primary_arm(result.entry.features)] += 1
            succeeded = True
        else:
            assert failure is not None
            failures.append(failure)
            succeeded = False
        if progress_callback is not None:
            progress_callback(completed, requested, task.profile, task.seed, succeeded)

    processed = 0
    if workers == 1:
        for task in tasks:
            task, result, failure = _run_pg_task(task)
            processed += 1
            consume(processed, task, result, failure)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, requested), mp_context=context
        ) as executor:
            futures = {executor.submit(_run_pg_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    completed_task, result, failure = future.result()
                except Exception as exc:
                    completed_task, result, failure = (
                        task,
                        None,
                        {
                            "profile": task.profile,
                            "seed": task.seed,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                processed += 1
                consume(processed, completed_task, result, failure)
    results.sort(key=lambda item: (str(item.spec.profile), item.spec.seed))
    failures.sort(key=lambda item: (str(item["profile"]), int(item["seed"])))
    report = PGPilotReport(
        requested=requested,
        generated=len(results),
        failed=len(failures),
        by_profile_arm={
            profile: dict(sorted(counts.items())) for profile, counts in sorted(matrix.items())
        },
        failures=tuple(failures),
    )
    return report, tuple(results)


def run_pg_tasks(
    tasks: Sequence[tuple[str, int]],
    *,
    data_root: str | Path,
    overwrite: bool = False,
    generator_commit: str | None = None,
    exporter_commit: str | None = None,
    workers: int = 1,
    progress_callback: PGProgressCallback | None = None,
) -> tuple[PGPilotReport, tuple[PGGenerationResult, ...]]:
    """Generate an explicit, deterministic set of profile/seed PG tasks."""

    if workers < 1:
        raise ValueError("workers must be positive")
    known_profiles = {profile.name for profile in PG_PROFILES}
    normalized_tasks = [(str(profile), int(seed)) for profile, seed in tasks]
    unknown_profiles = sorted({profile for profile, _seed in normalized_tasks} - known_profiles)
    if unknown_profiles:
        raise ValueError(f"unknown PG profiles in frozen task list: {unknown_profiles}")
    if len(set(normalized_tasks)) != len(normalized_tasks):
        raise ValueError("frozen PG task list contains duplicate profile/seed pairs")
    if not normalized_tasks:
        raise ValueError("frozen PG task list must not be empty")

    worker_tasks = [
        _PGTask(
            profile=profile,
            seed=seed,
            data_root=str(data_root),
            overwrite=overwrite,
            generator_commit=generator_commit,
            exporter_commit=exporter_commit,
        )
        for profile, seed in normalized_tasks
    ]
    results: list[PGGenerationResult] = []
    failures: list[dict[str, Any]] = []
    matrix: dict[str, Counter[str]] = defaultdict(Counter)

    def consume(
        completed: int,
        task: _PGTask,
        result: PGGenerationResult | None,
        failure: dict[str, Any] | None,
    ) -> None:
        if result is not None:
            results.append(result)
            matrix[task.profile][assign_primary_arm(result.entry.features)] += 1
        else:
            assert failure is not None
            failures.append(failure)
        if progress_callback is not None:
            progress_callback(
                completed,
                len(worker_tasks),
                task.profile,
                task.seed,
                result is not None,
            )

    processed = 0
    if workers == 1:
        for task in worker_tasks:
            completed_task, result, failure = _run_pg_task(task)
            processed += 1
            consume(processed, completed_task, result, failure)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(worker_tasks)), mp_context=context
        ) as executor:
            futures = {executor.submit(_run_pg_task, task): task for task in worker_tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    completed_task, result, failure = future.result()
                except Exception as exc:
                    completed_task, result, failure = (
                        task,
                        None,
                        {
                            "profile": task.profile,
                            "seed": task.seed,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                processed += 1
                consume(processed, completed_task, result, failure)
    results.sort(key=lambda item: (str(item.spec.profile), item.spec.seed))
    failures.sort(key=lambda item: (str(item["profile"]), int(item["seed"])))
    return (
        PGPilotReport(
            requested=len(worker_tasks),
            generated=len(results),
            failed=len(failures),
            by_profile_arm={
                profile: dict(sorted(counts.items())) for profile, counts in sorted(matrix.items())
            },
            failures=tuple(failures),
        ),
        tuple(results),
    )


def write_pg_pilot_report(
    report: PGPilotReport, data_root: str | Path, *, overwrite: bool = False
) -> Path:
    path = Path(data_root).expanduser().resolve() / "pg" / "pilot" / "pg_pilot_report.json"
    return write_json_report(report.to_dict(), path, overwrite=overwrite)
