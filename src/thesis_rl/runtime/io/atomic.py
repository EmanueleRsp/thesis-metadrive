"""Crash-safe file publication: write a temporary sibling, then ``os.replace``.

Every artifact a resume reads must be either the previous complete version or
the new complete version, never a partially written one (`RESUME-ABRUPT-001`
`REQ-RES-001`). The helpers here implement that contract on the existing file
names, so no consumer of the checkpoint layout has to change.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _fsync_path(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def temporary_sibling(path: Path) -> Path:
    """Return the temporary path used while publishing ``path``.

    The name keeps the target's suffix last so writers that append a default
    extension when none is present (SB3 ``model.save`` adds ``.zip`` only to
    suffix-less paths) leave the temporary name unchanged.
    """

    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_publish(path: str | Path, writer: Callable[[Path], None]) -> Path:
    """Run ``writer`` on a temporary sibling of ``path`` and atomically commit it.

    On any failure the temporary file is removed and the previous version of
    ``path``, if any, is left untouched. The committed file and its directory
    are fsynced so the rename is durable across a power loss.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(target)
    temporary.unlink(missing_ok=True)
    try:
        writer(temporary)
        if not temporary.is_file():
            raise RuntimeError(f"Writer did not create the temporary artifact {temporary}.")
        _fsync_path(temporary)
        os.replace(temporary, target)
        _fsync_path(target.parent)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_publish(path, lambda tmp: tmp.write_text(text, encoding=encoding))


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    return atomic_publish(path, lambda tmp: tmp.write_bytes(payload))


def atomic_pickle_dump(path: str | Path, payload: Any) -> Path:
    def _write(tmp: Path) -> None:
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle)

    return atomic_publish(path, _write)


def atomic_omegaconf_save(path: str | Path, payload: Any) -> Path:
    return atomic_publish(
        path, lambda tmp: OmegaConf.save(config=OmegaConf.create(payload), f=str(tmp))
    )


__all__ = [
    "atomic_omegaconf_save",
    "atomic_pickle_dump",
    "atomic_publish",
    "atomic_write_bytes",
    "atomic_write_text",
    "temporary_sibling",
]
