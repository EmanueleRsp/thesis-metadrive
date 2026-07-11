from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def run_pg_pilot(
    *,
    data_root: str | Path,
    count_per_profile: int,
    seed_start: int = 0,
    overwrite: bool = False,
    generator_commit: str | None = None,
    exporter_commit: str | None = None,
) -> tuple[PGPilotReport, tuple[PGGenerationResult, ...]]:
    if count_per_profile < 1:
        raise ValueError("count_per_profile must be positive")
    results: list[PGGenerationResult] = []
    failures: list[dict[str, Any]] = []
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    profile_stride = 1_000_000
    for profile_index, profile in enumerate(PG_PROFILES):
        for offset in range(count_per_profile):
            seed = int(seed_start) + profile_index * profile_stride + offset
            try:
                result = generate_pg_scenario(
                    profile.name,
                    seed=seed,
                    data_root=data_root,
                    overwrite=overwrite,
                    generator_commit=generator_commit,
                    exporter_commit=exporter_commit,
                )
            except Exception as exc:
                failures.append(
                    {
                        "profile": profile.name,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            results.append(result)
            matrix[profile.name][assign_primary_arm(result.entry.features)] += 1
    requested = len(PG_PROFILES) * count_per_profile
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
