"""Atomic checkpoint generations and fail-fast compatibility validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from thesis_rl.contracts.checkpoint_manifest import (
    CheckpointCompatibilityError,
    CheckpointManifest,
    assert_checkpoint_compatible,
)

MODEL_FILENAME = "model.zip"
MANIFEST_FILENAME = "manifest.json"
LATEST_FILENAME = "latest.json"
POINTER_VERSION = "1"


@dataclass(frozen=True)
class CheckpointGeneration:
    """A validated immutable checkpoint generation."""

    generation_dir: Path
    generation_id: str
    manifest: CheckpointManifest

    @property
    def model_path(self) -> Path:
        return self.generation_dir / MODEL_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self.generation_dir / MANIFEST_FILENAME


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
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


def _require_complete_generation(generation_dir: Path) -> tuple[Path, Path]:
    model_path = generation_dir / MODEL_FILENAME
    manifest_path = generation_dir / MANIFEST_FILENAME
    if not generation_dir.is_dir() or not model_path.is_file() or not manifest_path.is_file():
        raise CheckpointCompatibilityError(
            f"Checkpoint generation is incomplete: {generation_dir}."
        )
    return model_path, manifest_path


def _read_manifest(path: Path) -> CheckpointManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCompatibilityError(f"Cannot read checkpoint manifest: {path}") from exc
    return CheckpointManifest.from_mapping(payload)


def _validate_generation(
    generation_dir: Path,
    expected_manifest: CheckpointManifest | Mapping[str, Any],
    *,
    expected_model_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> CheckpointGeneration:
    model_path, manifest_path = _require_complete_generation(generation_dir)
    model_digest = sha256_file(model_path)
    manifest_digest = sha256_file(manifest_path)
    if expected_model_sha256 is not None and model_digest != expected_model_sha256:
        raise CheckpointCompatibilityError(
            f"Checkpoint model digest mismatch: checkpoint={model_digest!r}, "
            f"pointer={expected_model_sha256!r}."
        )
    if expected_manifest_sha256 is not None and manifest_digest != expected_manifest_sha256:
        raise CheckpointCompatibilityError(
            f"Checkpoint manifest digest mismatch: checkpoint={manifest_digest!r}, "
            f"pointer={expected_manifest_sha256!r}."
        )
    manifest = _read_manifest(manifest_path)
    assert_checkpoint_compatible(manifest, expected_manifest)
    generation_id = generation_dir.name.rsplit("-", maxsplit=1)[-1]
    return CheckpointGeneration(generation_dir, generation_id, manifest)


def publish_checkpoint_generation(
    checkpoint_root: str | Path,
    checkpoint_name: str,
    save_model: Callable[[Path], None],
    manifest: CheckpointManifest | Mapping[str, Any],
) -> CheckpointGeneration:
    """Publish a complete immutable generation and atomically update latest."""

    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    resolved_manifest = (
        manifest
        if isinstance(manifest, CheckpointManifest)
        else CheckpointManifest.from_mapping(manifest)
    )
    safe_name = str(checkpoint_name).strip()
    if not safe_name or Path(safe_name).name != safe_name:
        raise ValueError("checkpoint_name must be a non-empty filename component")
    generation_id = uuid.uuid4().hex
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{safe_name}-{generation_id}-", dir=root))
    final_dir = root / f"{safe_name}-{generation_id}"
    try:
        save_model(temporary_dir / MODEL_FILENAME)
        model_path = temporary_dir / MODEL_FILENAME
        manifest_path = temporary_dir / MANIFEST_FILENAME
        if not model_path.is_file():
            raise CheckpointCompatibilityError(
                f"Checkpoint saver did not create the required model: {model_path}"
            )
        _write_json(manifest_path, resolved_manifest.to_dict())
        _validate_generation(temporary_dir, resolved_manifest)
        _fsync_directory(temporary_dir)
        os.replace(temporary_dir, final_dir)
        _fsync_directory(root)

        pointer = {
            "pointer_version": POINTER_VERSION,
            "checkpoint_name": safe_name,
            "generation_id": generation_id,
            "generation_dir": final_dir.name,
            "model_sha256": sha256_file(final_dir / MODEL_FILENAME),
            "manifest_sha256": sha256_file(final_dir / MANIFEST_FILENAME),
        }
        pointer_path = root / LATEST_FILENAME
        pointer_tmp = root / f".{LATEST_FILENAME}.{uuid.uuid4().hex}.tmp"
        try:
            _write_json(pointer_tmp, pointer)
            os.replace(pointer_tmp, pointer_path)
            _fsync_directory(root)
        finally:
            pointer_tmp.unlink(missing_ok=True)
        return CheckpointGeneration(final_dir, generation_id, resolved_manifest)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def resolve_latest_generation(
    checkpoint_root: str | Path,
    expected_manifest: CheckpointManifest | Mapping[str, Any],
) -> CheckpointGeneration:
    """Resolve and validate the generation selected by latest.json."""

    root = Path(checkpoint_root)
    pointer_path = root / LATEST_FILENAME
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCompatibilityError(
            f"Cannot read atomic checkpoint pointer: {pointer_path}"
        ) from exc
    if not isinstance(pointer, dict) or pointer.get("pointer_version") != POINTER_VERSION:
        raise CheckpointCompatibilityError(f"Invalid checkpoint pointer: {pointer_path}")
    generation_dir_name = pointer.get("generation_dir")
    generation_id = pointer.get("generation_id")
    if not isinstance(generation_dir_name, str) or not isinstance(generation_id, str):
        raise CheckpointCompatibilityError(f"Incomplete checkpoint pointer: {pointer_path}")
    if Path(generation_dir_name).name != generation_dir_name:
        raise CheckpointCompatibilityError("Checkpoint pointer escapes its root directory.")
    if not generation_dir_name.endswith(f"-{generation_id}"):
        raise CheckpointCompatibilityError(
            "Checkpoint pointer generation id does not match its directory."
        )
    return _validate_generation(
        root / generation_dir_name,
        expected_manifest,
        expected_model_sha256=pointer.get("model_sha256"),
        expected_manifest_sha256=pointer.get("manifest_sha256"),
    )


def resolve_explicit_generation(
    generation_dir: str | Path,
    expected_manifest: CheckpointManifest | Mapping[str, Any],
) -> CheckpointGeneration:
    """Validate a caller-selected complete generation before SB3 loading."""

    return _validate_generation(Path(generation_dir), expected_manifest)


def load_checkpoint_generation(
    checkpoint_root: str | Path,
    expected_manifest: CheckpointManifest | Mapping[str, Any],
    model_loader: Callable[[Path], Any],
    *,
    generation_dir: str | Path | None = None,
) -> tuple[Any, CheckpointGeneration]:
    """Validate a generation, then invoke the supplied SB3 loader callback."""

    generation = (
        resolve_explicit_generation(generation_dir, expected_manifest)
        if generation_dir is not None
        else resolve_latest_generation(checkpoint_root, expected_manifest)
    )
    return model_loader(generation.model_path), generation
