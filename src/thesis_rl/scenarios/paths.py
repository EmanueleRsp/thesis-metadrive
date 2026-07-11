from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


DATA_DIRECTORIES = (
    "waymo/database",
    "waymo/metadata",
    "waymo/validation",
    "pg/database",
    "pg/generation_manifests",
    "pg/pilot",
    "pg/validation",
    "runtime/train",
    "runtime/validation",
    "runtime/test",
    "catalog",
    "splits",
)


@dataclass(frozen=True, slots=True)
class ScenarioDataPaths:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    def resolve_relative(self, relative_path: str | PurePosixPath) -> Path:
        relative = PurePosixPath(relative_path)
        original = str(relative_path)
        if (
            original != relative.as_posix()
            or relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
        ):
            raise ValueError(f"unsafe ScenarioNet relative path: {str(relative)!r}")
        if any(part in {"", "."} for part in relative.parts):
            raise ValueError(f"ScenarioNet path must be normalized: {str(relative)!r}")
        candidate = self.root.joinpath(*relative.parts).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"ScenarioNet path escapes data root: {str(relative)!r}")
        return candidate

    def make_relative(self, path: str | Path) -> str:
        resolved = Path(path).expanduser().resolve()
        try:
            relative = resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"path is outside SCENARIONET_DATA_ROOT: {resolved}") from exc
        if not relative.parts:
            raise ValueError("data root itself is not a scenario-relative path")
        return relative.as_posix()

    def ensure_layout(self) -> tuple[Path, ...]:
        created_or_existing = []
        self.root.mkdir(parents=True, exist_ok=True)
        for relative in DATA_DIRECTORIES:
            directory = self.resolve_relative(relative)
            directory.mkdir(parents=True, exist_ok=True)
            created_or_existing.append(directory)
        return tuple(created_or_existing)

    def runtime_split(self, split: str) -> Path:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported runtime split: {split!r}")
        return self.resolve_relative(f"runtime/{split}")
