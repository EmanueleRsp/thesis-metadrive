"""Parent-side deterministic state and attribution for vectorized Scenario ACL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from thesis_rl.curriculum.scenario_acl.usefulness import (
    compute_ppo_learning_potential,
    compute_sac_learning_potential,
    compute_td3_learning_potential,
)


VECTOR_STATE_VERSION = 2


@dataclass(frozen=True)
class AclSlotSelection:
    """Immutable parent decision installed in exactly one worker slot."""

    slot_id: int
    episode_id: int
    generation: int
    mode: str
    arm_index: int
    arm_name: str
    reset_seed: int
    scenario_uid: str | None = None
    runtime_index: int | None = None
    source: str | None = None
    selection_probability: float | None = None

    def __post_init__(self) -> None:
        if self.slot_id < 0 or self.episode_id < 0 or self.generation < 0:
            raise ValueError("ACL slot, episode, and generation identifiers must be non-negative.")
        if self.mode not in {"generate", "replay"}:
            raise ValueError("ACL selection mode must be 'generate' or 'replay'.")
        if not 0 <= self.arm_index < 6:
            raise ValueError("ScenarioNet ACL vector selections require one of six arms A0-A5.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AclSlotSelection":
        return cls(**dict(payload))


@dataclass
class AclEpisodeAccumulator:
    """Collected transition metrics retained across vector collection chunks."""

    slot_id: int
    episode_id: int
    rewards: list[float] = field(default_factory=list)
    td_residuals: list[float] = field(default_factory=list)
    sac_residuals: list[float] = field(default_factory=list)
    ppo_rewards: list[float] = field(default_factory=list)
    ppo_values: list[float] = field(default_factory=list)
    ppo_next_values: list[float] = field(default_factory=list)
    ppo_dones: list[bool] = field(default_factory=list)

    def add_transition(self, *, reward: float, done: bool, **values: float) -> None:
        if not np.isfinite(float(reward)):
            raise ValueError("ACL episode reward must be finite.")
        self.rewards.append(float(reward))
        if "td3_residual" in values:
            self.td_residuals.append(float(values["td3_residual"]))
        if "sac_residual" in values:
            self.sac_residuals.append(float(values["sac_residual"]))
        if "value" in values and "next_value" in values:
            self.ppo_rewards.append(float(reward))
            self.ppo_values.append(float(values["value"]))
            self.ppo_next_values.append(float(values["next_value"]))
            self.ppo_dones.append(bool(done))

    def learning_potential(self, algorithm: str) -> float:
        name = algorithm.lower()
        if name.startswith("ppo"):
            return compute_ppo_learning_potential(
                rewards=self.ppo_rewards,
                values=self.ppo_values,
                next_values=self.ppo_next_values,
                dones=self.ppo_dones,
            )
        if name.startswith("sac"):
            return compute_sac_learning_potential(self.sac_residuals)
        if name.startswith("td3") or name.startswith("ddpg"):
            return compute_td3_learning_potential(self.td_residuals)
        raise ValueError(f"Unsupported ACL learning-potential backend: {algorithm!r}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AclEpisodeAccumulator":
        return cls(**dict(payload))


@dataclass(frozen=True)
class AclCompletion:
    """Terminal outcome keyed to the selection that owned the episode."""

    collection_tick: int
    worker_id: int
    episode_id: int
    selection: AclSlotSelection
    metrics: Mapping[str, Any] = field(default_factory=dict)


class AclVectorTransaction:
    """Deterministic parent-side completion transaction.

    The callback is deliberately supplied by the driver: it can update the
    existing ScenarioBuffer/MAB and publish an artifact in one parent process,
    while this class guarantees that process arrival order is never observable.
    """

    def __init__(self, state: "AclVectorState") -> None:
        self.state = state

    def commit_tick(
        self,
        completions: Iterable[AclCompletion],
        *,
        learning_potentials: Mapping[tuple[int, int], float],
        commit: Any,
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            completions,
            key=lambda item: (
                int(item.collection_tick),
                int(item.worker_id),
                int(item.episode_id),
            ),
        )
        events: list[dict[str, Any]] = []
        for completion in ordered:
            key = (int(completion.worker_id), int(completion.episode_id))
            if key not in learning_potentials:
                raise ValueError(f"Missing per-episode LP for ACL completion {key}.")
            event = {
                "collection_tick": int(completion.collection_tick),
                "worker_id": int(completion.worker_id),
                "episode_id": int(completion.episode_id),
                "selection_generation": int(completion.selection.generation),
                "scenario_uid": completion.selection.scenario_uid,
                "learning_potential": float(learning_potentials[key]),
                "metrics": dict(completion.metrics),
            }
            commit(event)
            events.append(event)
        if ordered:
            self.state.collection_tick = max(
                self.state.collection_tick,
                max(int(item.collection_tick) for item in ordered) + 1,
            )
        return events


class AclVectorSelectionCoordinator:
    """Coordinate parent-owned selection and selective worker reset operations."""

    def __init__(self, state: "AclVectorState") -> None:
        self.state = state

    def select_batch(
        self,
        slots: Sequence[int],
        *,
        selector: Callable[[int, frozenset[str]], AclSlotSelection],
    ) -> dict[int, AclSlotSelection]:
        """Select one decision per slot using one immutable fresh exclusion set."""

        selected: dict[int, AclSlotSelection] = {}
        excluded: set[str] = {
            selection.scenario_uid
            for selection in self.state.active_selections.values()
            if selection.scenario_uid is not None
        }
        for slot in sorted({int(value) for value in slots}):
            selection = selector(slot, frozenset(excluded))
            if selection.slot_id != slot:
                raise ValueError("ACL selector returned a selection for the wrong worker slot.")
            if selection.generation < self.state.next_generation:
                raise ValueError("ACL selector returned a stale selection generation.")
            if selection.mode == "generate" and selection.scenario_uid is not None:
                if selection.scenario_uid in excluded:
                    raise ValueError(
                        f"Duplicate fresh ScenarioNet selection in ACL batch: "
                        f"{selection.scenario_uid}."
                    )
                excluded.add(selection.scenario_uid)
            selected[slot] = selection
        validate_fresh_batch_unique(selected.values())
        self.state.active_selections.update(selected)
        self.state.next_generation = max(
            self.state.next_generation,
            max((selection.generation for selection in selected.values()), default=-1) + 1,
        )
        return selected

    def configure_and_reset(
        self,
        env: Any,
        selections: Mapping[int, AclSlotSelection],
    ) -> dict[int, tuple[Any, dict[str, Any]]]:
        """Install selections first, then reset exactly the selected slots."""

        if not bool(getattr(env, "acl_mode", False)):
            raise RuntimeError("ACL vector coordinator requires an ACL-mode vector environment.")
        for slot in sorted(selections):
            env.configure_acl_selection(int(slot), selections[slot].to_dict())
        return env.reset_slots(
            sorted(selections),
            seeds={slot: int(selection.reset_seed) for slot, selection in selections.items()},
        )

    def restart_active_slots_for_resume(self, env: Any) -> dict[int, tuple[Any, dict[str, Any]]]:
        """Restart persisted logical episodes on a newly spawned worker pool.

        Learner checkpoints do not contain the MetaDrive/Rulebook simulator
        state. A resumed worker must therefore be reset before its first step.
        The parent keeps the persisted selection token, episode id,
        provenance, and seed; only the simulator trajectory prefix is
        restarted.
        """

        if not self.state.active_selections:
            return {}
        return self.configure_and_reset(env, self.state.active_selections)

    def clear_completed(self, slots: Iterable[int]) -> None:
        for slot in slots:
            self.state.active_selections.pop(int(slot), None)
            self.state.accumulators.pop(int(slot), None)


@dataclass
class AclVectorState:
    """Serializable parent control-plane state for deterministic resume."""

    n_envs: int
    collection_tick: int = 0
    next_episode_id: int = 0
    next_generation: int = 0
    active_selections: dict[int, AclSlotSelection] = field(default_factory=dict)
    accumulators: dict[int, AclEpisodeAccumulator] = field(default_factory=dict)
    pending_completions: list[AclCompletion] = field(default_factory=list)
    last_observations: Any | None = None
    rng_state: dict[str, Any] | None = None
    quarantined_scenario_uids: list[str] = field(default_factory=list)
    version: int = VECTOR_STATE_VERSION

    def __post_init__(self) -> None:
        if self.version != VECTOR_STATE_VERSION:
            raise ValueError(f"Unsupported ACL vector state version: {self.version}.")
        if self.n_envs <= 1:
            raise ValueError("ACL vector state requires n_envs > 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n_envs": self.n_envs,
            "collection_tick": self.collection_tick,
            "next_episode_id": self.next_episode_id,
            "next_generation": self.next_generation,
            "active_selections": {str(k): v.to_dict() for k, v in self.active_selections.items()},
            "accumulators": {str(k): v.to_dict() for k, v in self.accumulators.items()},
            "pending_completions": [
                {
                    "collection_tick": int(item.collection_tick),
                    "worker_id": int(item.worker_id),
                    "episode_id": int(item.episode_id),
                    "selection": item.selection.to_dict(),
                    "metrics": dict(item.metrics),
                }
                for item in self.pending_completions
            ],
            "last_observations": _json_observation(self.last_observations),
            "rng_state": self.rng_state,
            "quarantined_scenario_uids": sorted(str(uid) for uid in self.quarantined_scenario_uids),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, expected_n_envs: int | None = None
    ) -> "AclVectorState":
        if int(payload.get("version", -1)) != VECTOR_STATE_VERSION:
            raise ValueError("ACL vector checkpoint version is incompatible.")
        n_envs = int(payload["n_envs"])
        if expected_n_envs is not None and n_envs != int(expected_n_envs):
            raise ValueError(
                f"ACL vector checkpoint n_envs={n_envs} does not match {expected_n_envs}."
            )
        return cls(
            n_envs=n_envs,
            collection_tick=int(payload.get("collection_tick", 0)),
            next_episode_id=int(payload.get("next_episode_id", 0)),
            next_generation=int(payload.get("next_generation", 0)),
            active_selections={
                int(k): AclSlotSelection.from_dict(v)
                for k, v in dict(payload.get("active_selections", {})).items()
            },
            accumulators={
                int(k): AclEpisodeAccumulator.from_dict(v)
                for k, v in dict(payload.get("accumulators", {})).items()
            },
            pending_completions=[
                AclCompletion(
                    collection_tick=int(item["collection_tick"]),
                    worker_id=int(item["worker_id"]),
                    episode_id=int(item["episode_id"]),
                    selection=AclSlotSelection.from_dict(item["selection"]),
                    metrics=dict(item.get("metrics", {})),
                )
                for item in payload.get("pending_completions", [])
            ],
            last_observations=_restore_json_observation(payload.get("last_observations")),
            rng_state=payload.get("rng_state"),
            quarantined_scenario_uids=[
                str(uid) for uid in payload.get("quarantined_scenario_uids", ())
            ],
        )


def _json_observation(value: Any) -> Any:
    """Encode numpy observations without changing the versioned JSON format."""

    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _json_observation(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_json_observation(item) for item in value]}
    if isinstance(value, list):
        return [_json_observation(item) for item in value]
    return value


def _restore_json_observation(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "__ndarray__" in value:
            return np.asarray(value["__ndarray__"], dtype=np.dtype(str(value["dtype"])))
        if "__tuple__" in value:
            return tuple(_restore_json_observation(item) for item in value["__tuple__"])
        return {key: _restore_json_observation(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_json_observation(item) for item in value]
    return value


def save_acl_vector_state(path: str | Path, state: AclVectorState) -> None:
    """Publish vector state atomically so interrupted commits cannot corrupt it."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_acl_vector_state(
    path: str | Path, *, expected_n_envs: int | None = None
) -> AclVectorState:
    with Path(path).open(encoding="utf-8") as handle:
        return AclVectorState.from_dict(json.load(handle), expected_n_envs=expected_n_envs)


def order_acl_completions(completions: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return completion outcomes in the schedule-independent commit order."""

    def key(item: Any) -> tuple[int, int, int]:
        if isinstance(item, Mapping):
            return (
                int(item["collection_tick"]),
                int(item["worker_id"]),
                int(item.get("episode_id", 0)),
            )
        return int(item.collection_tick), int(item.worker_id), int(item.episode_id)

    return sorted(completions, key=key)


def retain_unresolved_acl_completions(
    pending: Iterable[AclCompletion],
    completed: Iterable[AclCompletion],
    *,
    resolved_keys: Iterable[tuple[int, int]],
) -> list[AclCompletion]:
    """Merge unresolved completions without duplicating a pending episode."""

    by_key = {(int(item.worker_id), int(item.episode_id)): item for item in pending}
    by_key.update({(int(item.worker_id), int(item.episode_id)): item for item in completed})
    resolved = {(int(slot), int(episode)) for slot, episode in resolved_keys}
    return [item for key, item in by_key.items() if key not in resolved]


def validate_fresh_batch_unique(
    selections: Iterable[AclSlotSelection],
    *,
    already_replayed: Iterable[str] = (),
) -> None:
    """Reject duplicate fresh catalog identities within one ACL parent batch."""

    seen = set(already_replayed)
    for selection in selections:
        if selection.mode != "generate" or selection.scenario_uid is None:
            continue
        if selection.scenario_uid in seen:
            raise ValueError(
                f"Duplicate fresh ScenarioNet selection in ACL batch: {selection.scenario_uid}."
            )
        seen.add(selection.scenario_uid)


def derive_worker_rng(global_seed: int, worker_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(global_seed), int(worker_id)]))
