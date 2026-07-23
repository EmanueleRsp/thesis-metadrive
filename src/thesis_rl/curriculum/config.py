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
class ScenarioAclMabConfig:
    num_arms: int = 6
    eta: float = 0.2
    update_method: str = "ema"
    alpha: float = 0.10
    initial_score: float = 0.50
    temperature: float = 0.50
    use_importance_correction: bool = False
    use_target_mab: bool = False
    target_sync_interval: int = 1
    feedback: str = "rank_normalized_learning_potential"


@dataclass(frozen=True)
class ScenarioAclReplaySamplingConfig:
    omega: float = 0.7
    beta: float = 1.0
    staleness_offset: int = 1


@dataclass(frozen=True)
class ScenarioAclScenarioEnvConfig:
    horizon: int = 1000
    truncate_as_terminate: bool = False
    out_of_route_done: bool = False
    out_of_road_done: bool = False
    on_continuous_line_done: bool = False
    on_broken_line_done: bool = False
    crash_vehicle_done: bool = True
    crash_object_done: bool = True
    crash_human_done: bool = True
    reactive_traffic: bool = True
    agent_policy: str = "EnvInputPolicy"
    log_level: int = 50


@dataclass(frozen=True)
class ScenarioAclConfig:
    buffer_capacity: int = 1000
    warmup_buffer_size: int = 100
    generate_probability: float = 0.4
    exploit_probability: float = 0.6
    use_mab: bool = True
    use_scenario_buffer: bool = True
    use_replay: bool = False
    use_staleness: bool = True
    recent_window_size: int = 100
    mab: ScenarioAclMabConfig = field(default_factory=ScenarioAclMabConfig)
    replay_sampling: ScenarioAclReplaySamplingConfig = field(
        default_factory=ScenarioAclReplaySamplingConfig
    )
    scenario_env: ScenarioAclScenarioEnvConfig = field(default_factory=ScenarioAclScenarioEnvConfig)


@dataclass(frozen=True)
class CurriculumConfig:
    enabled: bool = False
    kind: str = "disabled"
    staged: StagedCurriculumConfig = field(default_factory=StagedCurriculumConfig)
    scenario_acl: ScenarioAclConfig = field(default_factory=ScenarioAclConfig)

    @property
    def is_active(self) -> bool:
        return bool(self.enabled) and str(self.kind).lower() != "disabled"

    @property
    def is_staged(self) -> bool:
        return str(self.kind).lower() == "staged"

    @property
    def is_scenario_acl(self) -> bool:
        return str(self.kind).lower() == "scenario_acl"

    @classmethod
    def from_curriculum_cfg(
        cls, curriculum_cfg: DictConfig | dict[str, Any] | None
    ) -> CurriculumConfig:
        """Create CurriculumConfig from the dedicated `cfg.curriculum` group."""
        return cls.from_mapping(curriculum_cfg)

    @classmethod
    def from_mapping(cls, data: DictConfig | dict[str, Any] | None) -> CurriculumConfig:
        """Creates a CurriculumConfig instance from the given mapping."""
        payload = _to_plain_mapping(data)

        kind_raw = payload.get("kind")
        if kind_raw is None:
            raise ValueError("Curriculum config requires `kind` (e.g. staged, disabled).")
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

        if kind == "scenario_acl":
            scenario_acl_payload = payload.get("scenario_acl")
            if scenario_acl_payload is None:
                raise ValueError("Curriculum kind 'scenario_acl' requires a 'scenario_acl' block.")
            scenario_acl = _parse_scenario_acl_config(scenario_acl_payload)
            return cls(enabled=True, kind="scenario_acl", scenario_acl=scenario_acl)

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


def _parse_scenario_acl_config(data: DictConfig | dict[str, Any]) -> ScenarioAclConfig:
    payload = _to_plain_mapping(data)

    buffer_capacity = int(payload.get("buffer_capacity", 1000))
    warmup_buffer_size = int(payload.get("warmup_buffer_size", 100))
    recent_window_size = int(payload.get("recent_window_size", 100))
    exploit_probability = float(payload.get("exploit_probability", 0.6))
    generate_probability = float(payload.get("generate_probability", 1.0 - exploit_probability))

    if buffer_capacity <= 0:
        raise ValueError("scenario_acl.buffer_capacity must be > 0.")
    if warmup_buffer_size < 0:
        raise ValueError("scenario_acl.warmup_buffer_size must be >= 0.")
    if warmup_buffer_size > buffer_capacity:
        raise ValueError("scenario_acl.warmup_buffer_size must be <= scenario_acl.buffer_capacity.")
    if bool(payload.get("use_mutation", False)) or "mutation" in payload:
        raise ValueError("Scenario ACL mutation is out of scope. Use generation and replay only.")
    if bool(payload.get("use_rule_criticality", False)):
        raise ValueError(
            "Scenario ACL usefulness is learning-potential-only; "
            "Rulebook criticality is diagnostic-only."
        )
    if recent_window_size <= 0:
        raise ValueError("scenario_acl.recent_window_size must be > 0.")
    if not 0.0 <= exploit_probability <= 1.0 or not 0.0 <= generate_probability <= 1.0:
        raise ValueError("scenario_acl Generate/Replay probabilities must be in [0, 1].")
    if not abs(generate_probability + exploit_probability - 1.0) <= 1.0e-8:
        raise ValueError("scenario_acl Generate/Replay probabilities must sum to one.")
    if bool(payload.get("use_replay", False)) and not bool(
        payload.get("use_scenario_buffer", True)
    ):
        raise ValueError("scenario_acl.use_replay=true requires use_scenario_buffer=true.")

    mab_payload = _to_plain_mapping(payload.get("mab"))
    replay_payload = _to_plain_mapping(payload.get("replay_sampling"))
    scenario_env_payload = _to_plain_mapping(payload.get("scenario_env"))

    legacy_mab_keys = {
        "initial_weight",
        "initial_weight_decay",
        "weight_clip_min",
        "weight_clip_max",
    }
    present_legacy_keys = sorted(legacy_mab_keys.intersection(mab_payload))
    if present_legacy_keys:
        raise ValueError(
            "Legacy cumulative Scenario ACL MAB fields are incompatible with ACL v1.1: "
            f"{present_legacy_keys}"
        )

    feedback = str(mab_payload.get("feedback", "rank_normalized_learning_potential")).strip()
    if feedback != "rank_normalized_learning_potential":
        raise ValueError(
            "Unsupported scenario_acl.mab.feedback "
            f"'{feedback}'. Expected 'rank_normalized_learning_potential'."
        )

    num_arms = int(mab_payload.get("num_arms", 6))
    update_method = str(mab_payload.get("update_method", "ema"))
    alpha = float(mab_payload.get("alpha", 0.10))
    initial_score = float(mab_payload.get("initial_score", 0.50))
    temperature = float(mab_payload.get("temperature", 0.50))
    if num_arms != 6:
        raise ValueError("scenario_acl.mab.num_arms must be 6 for semantic arms A0-A5.")
    if update_method != "ema":
        raise ValueError("scenario_acl.mab.update_method must be 'ema'.")
    if not 0.0 < alpha <= 1.0 or not 0.0 <= initial_score <= 1.0 or temperature <= 0.0:
        raise ValueError("Invalid EMA MAB alpha, initial_score, or temperature.")
    if bool(mab_payload.get("use_importance_correction", False)):
        raise ValueError("ACL v1.1 does not support importance correction.")
    use_target_mab = bool(mab_payload.get("use_target_mab", False))
    target_sync_interval = int(mab_payload.get("target_sync_interval", 1))
    if target_sync_interval <= 0:
        raise ValueError("scenario_acl.mab.target_sync_interval must be > 0.")

    return ScenarioAclConfig(
        buffer_capacity=buffer_capacity,
        warmup_buffer_size=warmup_buffer_size,
        generate_probability=generate_probability,
        exploit_probability=exploit_probability,
        use_mab=bool(payload.get("use_mab", True)),
        use_scenario_buffer=bool(payload.get("use_scenario_buffer", True)),
        use_replay=bool(payload.get("use_replay", False)),
        use_staleness=bool(payload.get("use_staleness", True)),
        recent_window_size=recent_window_size,
        mab=ScenarioAclMabConfig(
            num_arms=num_arms,
            eta=float(mab_payload.get("eta", 0.2)),
            update_method=update_method,
            alpha=alpha,
            initial_score=initial_score,
            temperature=temperature,
            use_importance_correction=bool(mab_payload.get("use_importance_correction", False)),
            use_target_mab=use_target_mab,
            target_sync_interval=target_sync_interval,
            feedback=feedback,
        ),
        replay_sampling=ScenarioAclReplaySamplingConfig(
            omega=float(replay_payload.get("omega", 0.7)),
            beta=float(replay_payload.get("beta", 1.0)),
            staleness_offset=int(replay_payload.get("staleness_offset", 1)),
        ),
        scenario_env=ScenarioAclScenarioEnvConfig(
            horizon=int(scenario_env_payload.get("horizon", 1000)),
            truncate_as_terminate=bool(scenario_env_payload.get("truncate_as_terminate", False)),
            out_of_route_done=bool(scenario_env_payload.get("out_of_route_done", False)),
            out_of_road_done=bool(scenario_env_payload.get("out_of_road_done", False)),
            on_continuous_line_done=bool(
                scenario_env_payload.get("on_continuous_line_done", False)
            ),
            on_broken_line_done=bool(scenario_env_payload.get("on_broken_line_done", False)),
            crash_vehicle_done=bool(scenario_env_payload.get("crash_vehicle_done", True)),
            crash_object_done=bool(scenario_env_payload.get("crash_object_done", True)),
            crash_human_done=bool(scenario_env_payload.get("crash_human_done", True)),
            reactive_traffic=bool(scenario_env_payload.get("reactive_traffic", True)),
            agent_policy=str(scenario_env_payload.get("agent_policy", "EnvInputPolicy")),
            log_level=int(scenario_env_payload.get("log_level", 50)),
        ),
    )
