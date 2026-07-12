from __future__ import annotations

import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
) -> tuple[PGPilotReport, tuple[PGGenerationResult, ...]]:
    if count_per_profile < 1:
        raise ValueError("count_per_profile must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    results: list[PGGenerationResult] = []
    failures: list[dict[str, Any]] = []
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    profile_stride = 1_000_000
    requested = len(PG_PROFILES) * count_per_profile
    tasks = [
        _PGTask(
            profile=profile.name,
            seed=int(seed_start) + profile_index * profile_stride + offset,
            data_root=str(data_root),
            overwrite=overwrite,
            generator_commit=generator_commit,
            exporter_commit=exporter_commit,
        )
        for profile_index, profile in enumerate(PG_PROFILES)
        for offset in range(count_per_profile)
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


def write_pg_pilot_report(
    report: PGPilotReport, data_root: str | Path, *, overwrite: bool = False
) -> Path:
    path = Path(data_root).expanduser().resolve() / "pg" / "pilot" / "pg_pilot_report.json"
    return write_json_report(report.to_dict(), path, overwrite=overwrite)
