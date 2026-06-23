from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeneratorArm:
    """Safe subset of the specification that preserves observation shape."""

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
