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
class PromotionStabilityGates:
    success_rate_std_max: float = 0.10
    collision_rate_std_max: float = 0.03


@dataclass(frozen=True)
class PromotionGates:
    safety: PromotionSafetyGates = field(default_factory=PromotionSafetyGates)
    task: PromotionTaskGates = field(default_factory=PromotionTaskGates)
    stability: PromotionStabilityGates = field(default_factory=PromotionStabilityGates)


@dataclass(frozen=True)
class PromotionConfig:
    consecutive_evals: int = 3
    warmup_evals: int = 2
    default_min_stage_steps: int = 25_000
    per_stage_min_steps: dict[str, int] = field(default_factory=dict)
    no_demotion: bool = True
    gates: PromotionGates = field(default_factory=PromotionGates)


@dataclass(frozen=True)
class StagedCurriculumConfig:
    mode: str = "fixed"
    fixed_stage: str = "stage1"
    stages: tuple[StageConfig, ...] = field(default_factory=tuple)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)


@dataclass(frozen=True)
class CurriculumConfig:
    enabled: bool = False
    kind: str = "disabled"
    staged: StagedCurriculumConfig = field(default_factory=StagedCurriculumConfig)

    @property
    def is_active(self) -> bool:
        return bool(self.enabled) and str(self.kind).lower() != "disabled"

    @property
    def is_staged(self) -> bool:
        return str(self.kind).lower() == "staged"

    @classmethod
    def from_curriculum_cfg(
        cls, curriculum_cfg: DictConfig | dict[str, Any] | None
    ) -> CurriculumConfig:
        """Create CurriculumConfig from the dedicated `cfg.curriculum` group."""
        return cls.from_mapping(curriculum_cfg)

    @classmethod
    def from_mapping(cls, data: DictConfig | dict[str, Any] | None) -> CurriculumConfig:
        '''Creates a CurriculumConfig instance from the given mapping.'''
        payload = _to_plain_mapping(data)

        kind_raw = payload.get("kind")
        if kind_raw is None:
            raise ValueError(
                "Curriculum config requires `kind` (e.g. staged, disabled)."
            )
        kind = str(kind_raw).strip().lower()
        if not kind:
            raise ValueError("Curriculum kind must be a non-empty string.")

        enabled = bool(payload.get("enabled", False))

        if kind in {"disabled", "none"}:
            if enabled:
                raise ValueError("Curriculum kind 'disabled' requires enabled=false.")
            return cls(enabled=False, kind="disabled")

        if not enabled:
            raise ValueError(f"Curriculum kind '{kind}' requires enabled=true.")

        if kind == "staged":
            staged_payload = payload.get("staged")
            if staged_payload is None:
                raise ValueError("Curriculum kind 'staged' requires a 'staged' block.")
            staged = _parse_staged_config(staged_payload)
            return cls(enabled=True, kind="staged", staged=staged)

        raise ValueError(f"Unsupported curriculum kind '{kind}'.")


def _parse_staged_config(data: DictConfig | dict[str, Any]) -> StagedCurriculumConfig:
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
    stability_payload = _to_plain_mapping(gates_payload.get("stability"))
    per_stage_payload = _to_plain_mapping(promotion_payload.get("per_stage"))

    if "min_stage_steps" in promotion_payload:
        raise ValueError(
            "Unsupported legacy key `promotion.min_stage_steps`. "
            "Use `promotion.default_min_stage_steps` and optional "
            "`promotion.per_stage.<stage>.min_stage_steps`."
        )
    default_min_stage_steps = int(promotion_payload.get("default_min_stage_steps", 25_000))
    per_stage_min_steps: dict[str, int] = {}
    for stage_name, stage_cfg in per_stage_payload.items():
        stage_cfg_map = _to_plain_mapping(stage_cfg)
        if "min_stage_steps" in stage_cfg_map:
            per_stage_min_steps[str(stage_name)] = int(stage_cfg_map["min_stage_steps"])

    promotion = PromotionConfig(
        consecutive_evals=int(promotion_payload.get("consecutive_evals", 3)),
        warmup_evals=int(promotion_payload.get("warmup_evals", 2)),
        default_min_stage_steps=default_min_stage_steps,
        per_stage_min_steps=per_stage_min_steps,
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
            stability=PromotionStabilityGates(
                success_rate_std_max=float(stability_payload.get("success_rate_std_max", 0.10)),
                collision_rate_std_max=float(stability_payload.get("collision_rate_std_max", 0.03)),
            ),
        ),
    )

    return StagedCurriculumConfig(
        mode=str(payload.get("mode", "fixed")),
        fixed_stage=str(payload.get("fixed_stage", "stage1")),
        stages=tuple(stages),
        promotion=promotion,
    )
