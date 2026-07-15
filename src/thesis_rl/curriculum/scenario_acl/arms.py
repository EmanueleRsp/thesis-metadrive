from __future__ import annotations

from dataclasses import dataclass
from thesis_rl.scenarios.arms import ARMS as SCENARIO_ARMS


# Canonical labels assigned to realized ScenarioNet scenarios.
SCENARIO_ARM_NAMES = SCENARIO_ARMS


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
        rng: object | None = None,
        scenario_seed: int | None = None,
        base_env_config: dict[str, object] | None = None,
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


def build_default_scenario_arms() -> tuple[ScenarioArm, ...]:
    """Return the canonical A0-A5 arms in curriculum order."""

    return tuple(ScenarioArm(name=name) for name in SCENARIO_ARM_NAMES)
