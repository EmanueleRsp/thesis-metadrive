"""Resume-snapshot layout, classification and consistency checks.

`RESUME-ABRUPT-001`. A *resume snapshot* is the set of files a resumed run
reads for one checkpoint name:

- ``latest`` / ``final``: model-only at every chunk boundary (``latest``) or at
  the end of training (``final``); their training state, RNG and quarantine
  files keep the historical ``latest_*`` names.
- ``periodic/step_XXXXXXXX``: the periodic model checkpoint, which under
  `TRANSITION-REPLAY` v1.1 also carries the replay buffer and its pair
  manifest, plus its own training state, RNG, quarantine and (for the scenario
  ACL) curriculum state, all suffixed with the same stem.

``checkpoint.resume.checkpoint_name=periodic`` resolves to the newest periodic
snapshot whose pair is complete.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from thesis_rl.runtime.io.atomic import atomic_write_text

PERIODIC_ALIAS = "periodic"
PERIODIC_DIRNAME = "periodic"
REPLAY_SUFFIX = "_replay_buffer.pkl"
PAIR_SUFFIX = "_checkpoint_pair.json"
TRAINING_STATE_SUFFIX = "_training_state.yaml"
RNG_STATE_SUFFIX = "_rng_state.pkl"
QUARANTINE_STATE_SUFFIX = "_quarantine_state.json"
ACL_STATE_SUFFIX = "_acl"
_PERIODIC_STEM_RE = re.compile(r"^step_\d{8}$")

ReplayResumeMode = Literal["continuation", "reset", "not_applicable"]


@dataclass(frozen=True)
class ResumeArtifactPaths:
    """Every file that belongs to one checkpoint name."""

    checkpoint_name: str
    checkpoint_stem: Path
    model_path: Path
    replay_path: Path
    pair_path: Path
    training_state_path: Path
    rng_state_path: Path
    quarantine_state_path: Path
    acl_state_dir: Path | None

    @property
    def is_periodic(self) -> bool:
        return self.acl_state_dir is not None


def is_periodic_checkpoint_name(checkpoint_name: str) -> bool:
    parts = Path(checkpoint_name).parts
    return (
        len(parts) == 2 and parts[0] == PERIODIC_DIRNAME and bool(_PERIODIC_STEM_RE.match(parts[1]))
    )


def resume_artifact_paths(checkpoints_dir: Path, checkpoint_name: str) -> ResumeArtifactPaths:
    """Map a checkpoint name to its snapshot files without touching the disk."""

    name = str(checkpoint_name).strip()
    stem = checkpoints_dir / name
    periodic = is_periodic_checkpoint_name(name)
    if periodic:
        state_stem = stem
        acl_dir: Path | None = stem.with_name(f"{stem.name}{ACL_STATE_SUFFIX}")
    else:
        state_stem = checkpoints_dir / "latest"
        acl_dir = None
    return ResumeArtifactPaths(
        checkpoint_name=name,
        checkpoint_stem=stem,
        model_path=stem.with_name(f"{stem.name}.zip"),
        replay_path=stem.with_name(f"{stem.name}{REPLAY_SUFFIX}"),
        pair_path=stem.with_name(f"{stem.name}{PAIR_SUFFIX}"),
        training_state_path=state_stem.with_name(f"{state_stem.name}{TRAINING_STATE_SUFFIX}"),
        rng_state_path=state_stem.with_name(f"{state_stem.name}{RNG_STATE_SUFFIX}"),
        quarantine_state_path=state_stem.with_name(f"{state_stem.name}{QUARANTINE_STATE_SUFFIX}"),
        acl_state_dir=acl_dir,
    )


def periodic_snapshot_complete(paths: ResumeArtifactPaths) -> bool:
    """A periodic snapshot is complete when model, replay and pair all exist."""

    return paths.model_path.is_file() and paths.replay_path.is_file() and paths.pair_path.is_file()


def list_periodic_checkpoint_names(checkpoints_dir: Path) -> list[str]:
    periodic_dir = checkpoints_dir / PERIODIC_DIRNAME
    if not periodic_dir.is_dir():
        return []
    names = sorted(
        f"{PERIODIC_DIRNAME}/{path.stem}"
        for path in periodic_dir.glob("step_*.zip")
        if path.is_file() and _PERIODIC_STEM_RE.match(path.stem)
    )
    return names


def resolve_resume_checkpoint_name(checkpoints_dir: Path, requested: str) -> str:
    """Resolve the ``periodic`` alias to the newest complete periodic snapshot."""

    name = str(requested).strip()
    if name != PERIODIC_ALIAS:
        return name
    candidates = list_periodic_checkpoint_names(checkpoints_dir)
    for candidate in reversed(candidates):
        if periodic_snapshot_complete(resume_artifact_paths(checkpoints_dir, candidate)):
            return candidate
    raise FileNotFoundError(
        "checkpoint.resume.checkpoint_name=periodic requires a periodic checkpoint carrying a "
        "complete replay pair (model, replay buffer and pair manifest) under "
        f"{checkpoints_dir / PERIODIC_DIRNAME}; found periodic checkpoints: "
        f"{candidates or 'none'}. Enable transition_replay.persistence with "
        "trigger=periodic_and_final, or resume model-only from `latest` with "
        "checkpoint.resume.allow_replay_reset=true."
    )


def periodic_snapshot_due(previous_step: int, current_step: int, interval_steps: int) -> bool:
    """True when the run crossed a multiple of ``interval_steps`` in this chunk.

    Crossing semantics rather than exact equality: with ``eval_interval=50000``
    and ``periodic_interval_steps=125000`` the boundary 125 000 is never a chunk
    end, so an equality test would silently skip every other snapshot.
    """

    interval = int(interval_steps)
    if interval <= 0 or int(current_step) <= int(previous_step):
        return False
    return int(current_step) // interval > int(previous_step) // interval


def planner_trained_timesteps(planner: Any) -> int | None:
    """Return the number of environment steps the planner's model was trained on."""

    model = getattr(planner, "model", None)
    value = getattr(model, "num_timesteps", None)
    if value is None:
        value = getattr(planner, "num_timesteps", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def assert_snapshot_model_consistent(
    *,
    recorded_model_timesteps: Any,
    loaded_model_timesteps: int | None,
    checkpoint_name: str,
    state_path: Path,
    logger: logging.Logger | None = None,
) -> None:
    """Refuse a torn snapshot: model and state must come from the same save.

    Snapshots written before `RESUME-ABRUPT-001` carry no
    ``model_num_timesteps``; they are accepted with a warning so old runs stay
    resumable.
    """

    if recorded_model_timesteps is None:
        if logger is not None:
            logger.warning(
                "Resume snapshot %s has no model_num_timesteps in %s; torn-snapshot check skipped.",
                checkpoint_name,
                state_path,
            )
        return
    if loaded_model_timesteps is None:
        return
    if int(recorded_model_timesteps) != int(loaded_model_timesteps):
        raise ValueError(
            "Resume snapshot is torn: the loaded model was trained for "
            f"{int(loaded_model_timesteps)} steps but {state_path} recorded "
            f"model_num_timesteps={int(recorded_model_timesteps)} for checkpoint "
            f"{checkpoint_name!r}. The model and the training state come from different "
            "saves; resume from another checkpoint (e.g. checkpoint.resume.checkpoint_name="
            "periodic) or restart the run."
        )


def assert_snapshot_seed_consistent(*, recorded_seed: Any, configured_seed: int) -> None:
    if recorded_seed is None:
        return
    if int(recorded_seed) != int(configured_seed):
        raise ValueError(
            f"Resume seed mismatch: the snapshot was written by seed={int(recorded_seed)} but "
            f"the run is configured with seed={int(configured_seed)}. Resume with the original "
            "seed or start a fresh run."
        )


def classify_replay_resume(
    *,
    learner_has_replay: bool,
    persistence_enabled: bool,
    pair_path: Path,
    replay_path: Path,
    allow_replay_reset: bool,
    checkpoint_name: str,
) -> ReplayResumeMode:
    """`TRANSITION-REPLAY` REQ-025 resume classification.

    ``continuation`` when a complete pair exists; ``reset`` when the model is
    loaded without its replay state and the user explicitly allowed it;
    otherwise fail closed with the exact override that would proceed.
    """

    if not learner_has_replay:
        return "not_applicable"
    if persistence_enabled and pair_path.is_file() and replay_path.is_file():
        return "continuation"
    if allow_replay_reset:
        return "reset"
    if persistence_enabled and pair_path.is_file() and not replay_path.is_file():
        raise ValueError(
            f"Checkpoint pair {pair_path} references a replay artifact that is missing "
            f"({replay_path}); the pair is partially committed and cannot be resumed."
        )
    raise ValueError(
        f"Checkpoint {checkpoint_name!r} carries no replay buffer (pair manifest "
        f"{pair_path.name} "
        f"{'present' if pair_path.is_file() else 'absent'}, persistence "
        f"{'enabled' if persistence_enabled else 'disabled'}). Resuming would start a new "
        "empty replay segment, which is not a replay-equivalent continuation (REQ-025). "
        "Either resume from the newest paired periodic snapshot with "
        "checkpoint.resume.checkpoint_name=periodic, or accept the empty segment explicitly "
        "with checkpoint.resume.allow_replay_reset=true."
    )


def write_checkpoint_pair(
    path: Path,
    *,
    checkpoint_name: str,
    replay_name: str | None,
    training_timestep: int,
    beta_progress_env_steps: int = 0,
    model_num_timesteps: int | None = None,
    replay_segment_id: int = 0,
) -> dict[str, Any]:
    """Atomically publish the model/replay pair identity (REQ-024, REQ-033)."""

    payload: dict[str, Any] = {
        "checkpoint_id": uuid.uuid4().hex,
        "training_timestep": int(training_timestep),
        "replay_segment_id": int(replay_segment_id),
        "beta_progress_env_steps": int(beta_progress_env_steps),
        "model_num_timesteps": None if model_num_timesteps is None else int(model_num_timesteps),
        "model_path": f"{Path(checkpoint_name).name}.zip",
        "replay_path": replay_name,
    }
    atomic_write_text(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return payload


def prune_replay_snapshots(checkpoints_dir: Path, *, keep_checkpoint_name: str) -> list[Path]:
    """Keep exactly one periodic replay pair (``keep_last=1``): the one just committed."""

    periodic_dir = checkpoints_dir / PERIODIC_DIRNAME
    keep = resume_artifact_paths(checkpoints_dir, keep_checkpoint_name)
    removed: list[Path] = []
    if not periodic_dir.is_dir():
        return removed
    for pattern in (f"step_*{PAIR_SUFFIX}", f"step_*{REPLAY_SUFFIX}"):
        for path in sorted(periodic_dir.glob(pattern)):
            if path in (keep.pair_path, keep.replay_path):
                continue
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def prune_periodic_companions(checkpoints_dir: Path) -> list[Path]:
    """Remove snapshot companions whose periodic model zip was pruned."""

    periodic_dir = checkpoints_dir / PERIODIC_DIRNAME
    removed: list[Path] = []
    if not periodic_dir.is_dir():
        return removed
    live_stems = {path.stem for path in periodic_dir.glob("step_*.zip")}
    for path in sorted(periodic_dir.iterdir()):
        match = re.match(r"^(step_\d{8})(_.+)$", path.name)
        if match is None or match.group(1) in live_stems:
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                else:
                    child.rmdir()
            path.rmdir()
        else:
            path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def prune_old_periodic_checkpoints(checkpoints_dir: Path, *, keep_last: int) -> list[Path]:
    """Keep the newest ``keep_last`` periodic model zips and drop their companions with them."""

    periodic_dir = checkpoints_dir / PERIODIC_DIRNAME
    removed: list[Path] = []
    if not periodic_dir.is_dir():
        return removed
    zips = sorted(
        (
            p
            for p in periodic_dir.glob("step_*.zip")
            if p.is_file() and _PERIODIC_STEM_RE.match(p.stem)
        ),
        key=lambda p: p.name,
    )
    keep_n = max(int(keep_last), 0)
    to_remove = zips if keep_n <= 0 else zips[:-keep_n]
    for zip_path in to_remove:
        for sidecar in sorted(periodic_dir.glob(f"{zip_path.stem}.*")):
            sidecar.unlink(missing_ok=True)
            removed.append(sidecar)
    removed.extend(prune_periodic_companions(checkpoints_dir))
    return removed


def remove_replay_snapshots_after_final(checkpoints_dir: Path) -> list[Path]:
    """`DEC-RES-006`: once ``final`` carries the pair, intermediate replay copies go."""

    removed: list[Path] = []
    for legacy in (
        checkpoints_dir / f"latest{REPLAY_SUFFIX}",
        checkpoints_dir / f"latest{PAIR_SUFFIX}",
    ):
        if legacy.exists():
            legacy.unlink()
            removed.append(legacy)
    periodic_dir = checkpoints_dir / PERIODIC_DIRNAME
    if periodic_dir.is_dir():
        for pattern in (f"step_*{REPLAY_SUFFIX}", f"step_*{PAIR_SUFFIX}"):
            for path in sorted(periodic_dir.glob(pattern)):
                path.unlink(missing_ok=True)
                removed.append(path)
    return removed


__all__ = [
    "ACL_STATE_SUFFIX",
    "PERIODIC_ALIAS",
    "PERIODIC_DIRNAME",
    "ReplayResumeMode",
    "ResumeArtifactPaths",
    "assert_snapshot_model_consistent",
    "assert_snapshot_seed_consistent",
    "classify_replay_resume",
    "is_periodic_checkpoint_name",
    "list_periodic_checkpoint_names",
    "periodic_snapshot_complete",
    "periodic_snapshot_due",
    "planner_trained_timesteps",
    "prune_old_periodic_checkpoints",
    "prune_periodic_companions",
    "prune_replay_snapshots",
    "remove_replay_snapshots_after_final",
    "resolve_resume_checkpoint_name",
    "resume_artifact_paths",
    "write_checkpoint_pair",
]
