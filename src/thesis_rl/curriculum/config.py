from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _to_plain_mapping(data: DictConfig | dict[str, Any] | None) -> dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, DictConfig):
        return OmegaConf.to_container(data, resolve=True)  # type: ignore[return-value]
    return dict(data)


@dataclass(frozen=True)
class StageConfig:
    name: str
    env: dict[str, Any]
    eval_env: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromotionSafetyGates:
    collision_rate_max: float = 0.05
    top_rule_violation_rate_max: float = 0.02
    out_of_road_rate_max: float = 0.03


@dataclass(frozen=True)
class PromotionTaskGates:
    success_rate_min: float = 0.80
    route_completion_min: float = 0.85


@dataclass(frozen=True)
class PromotionQualityGates:
    mean_reward_min: float = -5.0


@dataclass(frozen=True)
class PromotionGates:
    safety: PromotionSafetyGates = field(default_factory=PromotionSafetyGates)
    task: PromotionTaskGates = field(default_factory=PromotionTaskGates)
    quality: PromotionQualityGates = field(default_factory=PromotionQualityGates)


@dataclass(frozen=True)
class PromotionConfig:
    consecutive_evals: int = 3
    warmup_evals: int = 2
    min_stage_steps: int = 25_000
    no_demotion: bool = True
    gates: PromotionGates = field(default_factory=PromotionGates)


@dataclass(frozen=True)
class CurriculumConfig:
    enabled: bool = False
    mode: str = "fixed"
    fixed_stage: str = "stage1"
    stages: tuple[StageConfig, ...] = field(default_factory=tuple)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)

    @classmethod
    def from_experiment_cfg(cls, experiment_cfg: DictConfig | dict[str, Any]) -> CurriculumConfig:
        payload = _to_plain_mapping(experiment_cfg)
        curriculum_payload = _to_plain_mapping(payload.get("curriculum"))
        return cls.from_mapping(curriculum_payload)

    @classmethod
    def from_mapping(cls, data: DictConfig | dict[str, Any] | None) -> CurriculumConfig:
        payload = _to_plain_mapping(data)

        stages_payload = payload.get("stages", [])
        stages: list[StageConfig] = []
        for stage in stages_payload:
            stage_dict = _to_plain_mapping(stage)
            stages.append(
                StageConfig(
                    name=str(stage_dict.get("name", "stage")),
                    env=_to_plain_mapping(stage_dict.get("env")),
                    eval_env=_to_plain_mapping(stage_dict.get("eval_env")),
                )
            )

        promotion_payload = _to_plain_mapping(payload.get("promotion"))
        gates_payload = _to_plain_mapping(promotion_payload.get("gates"))
        safety_payload = _to_plain_mapping(gates_payload.get("safety"))
        task_payload = _to_plain_mapping(gates_payload.get("task"))
        quality_payload = _to_plain_mapping(gates_payload.get("quality"))

        promotion = PromotionConfig(
            consecutive_evals=int(promotion_payload.get("consecutive_evals", 3)),
            warmup_evals=int(promotion_payload.get("warmup_evals", 2)),
            min_stage_steps=int(promotion_payload.get("min_stage_steps", 25_000)),
            no_demotion=bool(promotion_payload.get("no_demotion", True)),
            gates=PromotionGates(
                safety=PromotionSafetyGates(
                    collision_rate_max=float(safety_payload.get("collision_rate_max", 0.05)),
                    top_rule_violation_rate_max=float(
                        safety_payload.get("top_rule_violation_rate_max", 0.02)
                    ),
                    out_of_road_rate_max=float(safety_payload.get("out_of_road_rate_max", 0.03)),
                ),
                task=PromotionTaskGates(
                    success_rate_min=float(task_payload.get("success_rate_min", 0.80)),
                    route_completion_min=float(task_payload.get("route_completion_min", 0.85)),
                ),
                quality=PromotionQualityGates(
                    mean_reward_min=float(quality_payload.get("mean_reward_min", -5.0)),
                ),
            ),
        )

        return cls(
            enabled=bool(payload.get("enabled", False)),
            mode=str(payload.get("mode", "fixed")),
            fixed_stage=str(payload.get("fixed_stage", "stage1")),
            stages=tuple(stages),
            promotion=promotion,
        )
