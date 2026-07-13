from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from thesis_rl.scenarios.arms import ARMS as SCENARIO_ARMS


# These are the canonical labels assigned to realized ScenarioNet scenarios.
# They intentionally do not replace GENERATOR_ARMS: ACL currently samples
# procedural MetaDrive distributions, while ScenarioNet labels the resulting
# scenarios after feature extraction. In particular, PG does not generate the
# VRU context represented by A4.
SCENARIO_ARM_NAMES = SCENARIO_ARMS


@dataclass(frozen=True)
class GeneratorArm:
    """Procedural MetaDrive generation profile used by the ACL MAB.

    ``GeneratorArm.name`` is a generator-profile identifier, not a realized
    ScenarioNet semantic arm. The latter is exposed as ``SCENARIO_ARM_NAMES``.
    """

    name: str
    map_choices: tuple[int, ...]
    traffic_density_min: float
    traffic_density_max: float
    random_lane_width_prob: float
    random_lane_num_prob: float

    def sample_env_overrides(
        self,
        *,
        rng: np.random.Generator,
        scenario_seed: int,
        base_env_config: dict[str, Any],
    ) -> dict[str, object]:
        traffic_density = float(
            rng.uniform(self.traffic_density_min, self.traffic_density_max)
        )
        map_value = int(rng.choice(np.asarray(self.map_choices, dtype=np.int64)))

        return {
            "map": map_value,
            "traffic_density": traffic_density,
            "random_lane_width": bool(
                rng.random() < float(self.random_lane_width_prob)
            ),
            "random_lane_num": bool(rng.random() < float(self.random_lane_num_prob)),
            # Keep observation dimensionality stable across iterations.
            "random_agent_model": bool(base_env_config.get("random_agent_model", True)),
            "start_seed": int(scenario_seed),
            "num_scenarios": 1,
            "log_level": int(base_env_config.get("log_level", 50)),
        }


@dataclass(frozen=True)
class ScenarioArm:
    """Selector for a realized ScenarioNet semantic arm.

    The provider performs the actual filtering over catalog records. This
    object only adapts the shared arm names to the ACL MAB interface.
    """

    name: str

    def __post_init__(self) -> None:
        if self.name not in SCENARIO_ARM_NAMES:
            raise ValueError(
                f"unsupported ScenarioNet arm {self.name!r}; "
                f"expected one of {SCENARIO_ARM_NAMES}"
            )

    def sample_env_overrides(
        self,
        *,
        rng: np.random.Generator,
        scenario_seed: int,
        base_env_config: dict[str, Any],
    ) -> dict[str, object]:
        del rng, scenario_seed, base_env_config
        provider: dict[str, object] = {"arm": self.name}
        # The frozen ScenarioNet catalog has no selected Waymo A0 records and
        # no PG A4 records. Avoid asking the strict provider for an impossible
        # source × split × arm combination.
        if self.name == "A0_simple_low_traffic":
            provider["source_probability"] = {"waymo": 0.0, "pg": 1.0}
        elif self.name == "A4_vru":
            provider["source_probability"] = {"waymo": 1.0, "pg": 0.0}
        return {"provider": provider}


def build_default_generator_arms() -> tuple[GeneratorArm, ...]:
    return (
        GeneratorArm(
            name="broad_random",
            map_choices=(3, 4, 5, 6, 7, 8),
            traffic_density_min=0.00,
            traffic_density_max=0.50,
            random_lane_width_prob=0.50,
            random_lane_num_prob=0.50,
        ),
        GeneratorArm(
            name="simple_low_risk",
            map_choices=(3, 4),
            traffic_density_min=0.00,
            traffic_density_max=0.10,
            random_lane_width_prob=0.0,
            random_lane_num_prob=0.0,
        ),
        GeneratorArm(
            name="traffic_heavy",
            map_choices=(4, 5, 6),
            traffic_density_min=0.25,
            traffic_density_max=0.50,
            random_lane_width_prob=0.30,
            random_lane_num_prob=0.30,
        ),
        GeneratorArm(
            name="obstacle_heavy",
            map_choices=(4, 5, 6),
            traffic_density_min=0.05,
            traffic_density_max=0.25,
            random_lane_width_prob=0.25,
            random_lane_num_prob=0.25,
        ),
        GeneratorArm(
            name="complex_topology",
            map_choices=(6, 7, 8),
            traffic_density_min=0.10,
            traffic_density_max=0.30,
            random_lane_width_prob=0.30,
            random_lane_num_prob=0.30,
        ),
        GeneratorArm(
            name="lane_randomized",
            map_choices=(4, 5, 6, 7),
            traffic_density_min=0.10,
            traffic_density_max=0.30,
            random_lane_width_prob=1.0,
            random_lane_num_prob=1.0,
        ),
        GeneratorArm(
            name="safety_critical_mix",
            map_choices=(5, 6, 7, 8),
            traffic_density_min=0.20,
            traffic_density_max=0.50,
            random_lane_width_prob=0.60,
            random_lane_num_prob=0.60,
        ),
    )


def build_default_scenario_arms() -> tuple[ScenarioArm, ...]:
    """Return the canonical A0-A5 arms in curriculum order."""

    return tuple(ScenarioArm(name=name) for name in SCENARIO_ARM_NAMES)
