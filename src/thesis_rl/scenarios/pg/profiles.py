from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


PGProfileName = Literal[
    "P0_simple",
    "P1_vehicle_interaction",
    "P2_merge_or_roundabout",
    "P3_intersection",
    "P5_complex_mixed",
]


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    profile: PGProfileName
    seed: int
    block_sequence: tuple[str, ...]
    traffic_density: float
    random_lane_num: bool
    random_lane_width: bool
    lane_num: int | None
    lane_width: float | None
    accident_prob: float
    max_episode_length: int


@dataclass(frozen=True, slots=True)
class PGProfile:
    name: PGProfileName
    native_tokens: tuple[str, ...]
    traffic_density_min: float
    traffic_density_max: float
    num_blocks_choices: tuple[int, ...]
    complex_tokens: tuple[str, ...] = ()
    min_complex_tokens: int = 0
    accident_prob: float = 0.0
    max_episode_length: int = 500

    def __post_init__(self) -> None:
        if not self.native_tokens:
            raise ValueError(f"{self.name} has no native block tokens")
        if not 0 <= self.traffic_density_min <= self.traffic_density_max <= 1:
            raise ValueError(f"invalid traffic density range for {self.name}")
        if not self.num_blocks_choices or any(value < 2 for value in self.num_blocks_choices):
            raise ValueError(f"{self.name} requires at least two blocks")
        if self.accident_prob != 0.0:
            raise ValueError("ScenarioNet v1 PG profiles require accident_prob=0")

    @property
    def required_native_tokens(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.native_tokens + self.complex_tokens))

    def sample(self, *, seed: int) -> GenerationSpec:
        rng = np.random.default_rng(int(seed))
        num_blocks = int(rng.choice(self.num_blocks_choices))
        token_count = num_blocks - 1  # the initial FirstPGBlock is implicit
        if self.min_complex_tokens:
            if len(self.complex_tokens) < self.min_complex_tokens:
                raise ValueError(f"{self.name} lacks enough complex native tokens")
            complex_count = self.min_complex_tokens
            selected = [
                str(token)
                for token in rng.choice(
                    np.asarray(self.complex_tokens), size=complex_count, replace=False
                )
            ]
            filler = [
                str(rng.choice(np.asarray(self.native_tokens)))
                for _ in range(token_count - complex_count)
            ]
            tokens = selected + filler
            rng.shuffle(tokens)
        else:
            tokens = [
                str(rng.choice(np.asarray(self.native_tokens))) for _ in range(token_count)
            ]
        return GenerationSpec(
            profile=self.name,
            seed=int(seed),
            block_sequence=tuple(str(token) for token in tokens),
            traffic_density=float(
                rng.uniform(self.traffic_density_min, self.traffic_density_max)
            ),
            random_lane_num=True,
            random_lane_width=True,
            lane_num=None,
            lane_width=None,
            accident_prob=self.accident_prob,
            max_episode_length=self.max_episode_length,
        )


PG_PROFILES: tuple[PGProfile, ...] = (
    PGProfile("P0_simple", ("S", "C"), 0.00, 0.05, (2, 3)),
    PGProfile("P1_vehicle_interaction", ("S", "C"), 0.07, 0.18, (2, 3, 4)),
    PGProfile(
        "P2_merge_or_roundabout",
        ("S", "C"),
        0.08,
        0.18,
        (2, 3, 4),
        complex_tokens=("y", "r", "R", "O"),
        min_complex_tokens=1,
    ),
    PGProfile(
        "P3_intersection",
        ("S", "C"),
        0.08,
        0.18,
        (2, 3, 4),
        complex_tokens=("X", "T"),
        min_complex_tokens=1,
    ),
    PGProfile(
        "P5_complex_mixed",
        ("S", "C"),
        0.15,
        0.25,
        (4, 5),
        complex_tokens=("y", "r", "R", "O", "X", "T"),
        min_complex_tokens=2,
    ),
)


def get_pg_profile(name: str) -> PGProfile:
    for profile in PG_PROFILES:
        if profile.name == name:
            return profile
    raise KeyError(f"unknown PG profile: {name!r}")


def validate_native_tokens(profile: PGProfile, available_tokens: Sequence[str]) -> None:
    missing = sorted(set(profile.required_native_tokens).difference(available_tokens))
    if missing:
        raise RuntimeError(
            f"PG profile {profile.name} requires unavailable native block IDs: {missing}"
        )
