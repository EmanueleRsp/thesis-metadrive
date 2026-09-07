"""Pure scalarization of ordered Rulebook margin vectors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


SCALARIZATION_SPECIFICATION_ID = "SCAL-V1.4"
SCALARIZATION_VERSION = "1.4"
BOUNDED_VECTOR_SCHEMA_ID = "rulebook_v2_macro_v4"
# RULEBOOK-V5.1 §3.4. A distinct schema id, because a six-level vector is not a
# four-level one with two entries appended: L5 and L6 sit *below* progress, so a
# consumer that read the old schema and ignored the tail would be reading a
# different preference order rather than a truncated one.
SIX_LEVEL_VECTOR_SCHEMA_ID = "rulebook_v5_1_six_level_v1"
SCALARIZATION_MODES = (
    "legacy_scaled_sigmoid",
    "bounded_centered_sigmoid",
    "bounded_satisfaction_rank",
    "bounded_priority_weighted_rank",
    "six_level_priority_weighted_rank",
)
REWARD_COMPRESSION_MODES = ("none", "symlog")
# ADR-079: `a = 2.5`, raised from RULEBOOK-V5.1 §5.5's 2.2. §5.4 caps `sigma` at
# 0.1227 when `a = 2.2`, and no admissible `sigma` at that base makes the
# reward's local gradient point the right way in a conflict a real vehicle could
# still brake out of. §5.5's 2.2 was the first round value above §5.4's lower
# bound of 2.12, not an optimum, and that document sets no upper bound.
#
# Public, and read by the tests instead of being restated there: a required
# value written down twice is a required value that can drift.
SIX_LEVEL_PRIORITY_BASE = 2.5
# SCAL-V1.1 REQ-SCAL11-003: each mode requires its own frozen priority_base;
# legacy/centered-sigmoid/satisfaction-rank keep SCAL-V1.0's 2.01, while the
# priority-weighted-rank mode requires the re-derived bound a'=3 (SCAL-V1.1 §7.6).
_REQUIRED_PRIORITY_BASE_BY_MODE = {
    "legacy_scaled_sigmoid": 2.01,
    "bounded_centered_sigmoid": 2.01,
    "bounded_satisfaction_rank": 2.01,
    "bounded_priority_weighted_rank": 3.0,
    "six_level_priority_weighted_rank": SIX_LEVEL_PRIORITY_BASE,
}

# The four legacy modes consume v4.7's four-margin vector; only the new mode
# consumes six. Kept as data rather than as an `if` chain so that adding a mode
# cannot forget to declare its arity (`DEC-RB51-003` keeps the legacy modes so
# earlier runs stay reproducible). This table is the *only* place an arity is
# written: `scalarize_rulebook_margins` reads it instead of restating 4 and 6 at
# its two call sites, because an arity written twice is an arity that can drift.
_REQUIRED_MARGIN_COUNT_BY_MODE = {
    "bounded_centered_sigmoid": 4,
    "bounded_satisfaction_rank": 4,
    "bounded_priority_weighted_rank": 4,
    "six_level_priority_weighted_rank": 6,
}


class ScalarizationConfigurationError(ValueError):
    """Raised when scalarization configuration violates the approved contract."""


class ScalarizationEvaluationError(ValueError):
    """Raised when a margin vector cannot be scalarized safely."""


@dataclass(frozen=True, slots=True)
class ScalarizationConfig:
    """Immutable configuration shared by all algorithm backends."""

    mode: str = "bounded_priority_weighted_rank"
    priority_base: float = 3.0
    sigmoid_sharpness: float = 30.0
    numerical_tolerance: float = 1.0e-8
    vector_schema_id: str = BOUNDED_VECTOR_SCHEMA_ID
    legacy_vector_schema_id: str | None = None
    legacy_rule_scales: tuple[float, ...] | None = None
    native_environment_reward_weight: float = 0.0
    reward_compression_mode: str = "none"
    # RULEBOOK-V5.1 §5.5 as amended by ADR-079, read only by
    # `six_level_priority_weighted_rank`. The defaults are the selected weights:
    # `sigma = 0.30`, `phi = 0.25`, `lambda4 = 2.0`, `eta = 1.0`,
    # `lambda6 = 0.2`, at `a = 2.5`.
    severity: float = 0.30
    flat_tie_breaker: float = 0.25
    progress_weight: float = 2.0
    relaxable_weight: float = 1.0
    progress_rate_weight: float = 0.2
    step_dt_s: float = 0.1
    reference_time_s: float = 1.0
    specification_id: str = SCALARIZATION_SPECIFICATION_ID
    version: str = SCALARIZATION_VERSION

    def __post_init__(self) -> None:
        mode = str(self.mode)
        if mode not in SCALARIZATION_MODES:
            raise ScalarizationConfigurationError(
                f"Unknown scalarization mode {mode!r}; expected one of {SCALARIZATION_MODES}."
            )
        if not math.isfinite(float(self.priority_base)) or float(self.priority_base) <= 1.0:
            raise ScalarizationConfigurationError(
                "priority_base must be finite and greater than 1."
            )
        required_base = _REQUIRED_PRIORITY_BASE_BY_MODE[mode]
        if float(self.priority_base) != required_base:
            raise ScalarizationConfigurationError(
                f"Mode {mode!r} requires priority_base={required_base}, "
                f"got {float(self.priority_base)!r}."
            )
        compression_mode = str(self.reward_compression_mode)
        if compression_mode not in REWARD_COMPRESSION_MODES:
            raise ScalarizationConfigurationError(
                f"Unknown reward_compression mode {compression_mode!r}; "
                f"expected one of {REWARD_COMPRESSION_MODES}."
            )
        if not math.isfinite(float(self.sigmoid_sharpness)) or float(self.sigmoid_sharpness) <= 0.0:
            raise ScalarizationConfigurationError(
                "sigmoid_sharpness must be finite and greater than 0."
            )
        if (
            not math.isfinite(float(self.numerical_tolerance))
            or float(self.numerical_tolerance) < 0.0
        ):
            raise ScalarizationConfigurationError(
                "numerical_tolerance must be finite and non-negative."
            )
        if float(self.native_environment_reward_weight) != 0.0:
            raise ScalarizationConfigurationError(
                "native_environment_reward_weight must be exactly 0 for conformant scalarization."
            )
        if not str(self.vector_schema_id):
            raise ScalarizationConfigurationError("vector_schema_id must be a non-empty string.")
        if mode == "legacy_scaled_sigmoid":
            if not self.legacy_vector_schema_id:
                raise ScalarizationConfigurationError(
                    "legacy_vector_schema_id is required for legacy_scaled_sigmoid."
                )
            _validate_scales(self.legacy_rule_scales)
        elif self.legacy_vector_schema_id is not None or self.legacy_rule_scales is not None:
            raise ScalarizationConfigurationError(
                "Bounded modes reject legacy_vector_schema_id and legacy_rule_scales."
            )
        if mode == "six_level_priority_weighted_rank":
            self._validate_six_level_weights()

    def _validate_six_level_weights(self) -> None:
        """RULEBOOK-V5.1 §5.4, checked at construction rather than at use.

        The rank-preservation condition is what makes the priority weights an
        *ordering* rather than a set of numbers: below it, one step of progress
        or of relaxation can overturn a higher-level violation. Inadmissible
        weights are refused here so that no such reward is ever emitted — the
        same discipline the offline weight grid uses when it declines to price a
        non-rank-preserving member instead of pricing it and reporting it.
        """

        for name in (
            "severity",
            "flat_tie_breaker",
            "progress_weight",
            "relaxable_weight",
            "progress_rate_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ScalarizationConfigurationError(
                    f"{name} must be finite and non-negative, got {value!r}."
                )
        for name in ("step_dt_s", "reference_time_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ScalarizationConfigurationError(
                    f"{name} must be finite and strictly positive, got {value!r}."
                )

        base = float(self.priority_base)
        severity = float(self.severity)
        flat = float(self.flat_tie_breaker)
        dt_ratio = float(self.step_dt_s) / float(self.reference_time_s)
        # The utility tail: one maximal step of progress (`DELTA_Q_MAX = 1`,
        # §4.1) plus one maximal step of each level below it.
        tail = (
            float(self.progress_weight)
            + float(self.relaxable_weight) * dt_ratio
            + float(self.progress_rate_weight) * dt_ratio
        )
        weights = (base**3, base**2, base)
        for index, weight in enumerate(weights):
            lower = weights[index + 1 :]
            bound = (1.0 + severity) * sum(lower) + flat * len(lower) + tail
            if weight <= bound:
                raise ScalarizationConfigurationError(
                    "Weights violate the rank-preservation condition of "
                    f"RULEBOOK-V5.1 §5.4 at level {index + 1}: {weight} <= {bound}."
                )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "ScalarizationConfig":
        """Construct configuration from a plain or OmegaConf mapping."""

        if values is None:
            return cls()
        raw = dict(values)
        legacy = raw.get("legacy")
        if isinstance(legacy, Mapping):
            raw.setdefault("legacy_vector_schema_id", legacy.get("vector_schema_id"))
            raw.setdefault("legacy_rule_scales", legacy.get("rule_scales"))
        sigmoid = raw.get("sigmoid")
        if isinstance(sigmoid, Mapping):
            raw.setdefault("sigmoid_sharpness", sigmoid.get("sharpness"))
        reward_compression = raw.get("reward_compression")
        if isinstance(reward_compression, Mapping):
            raw.setdefault("reward_compression_mode", reward_compression.get("mode"))
        raw.pop("legacy", None)
        raw.pop("sigmoid", None)
        raw.pop("reward_compression", None)
        if raw.get("legacy_rule_scales") is not None:
            raw["legacy_rule_scales"] = tuple(float(v) for v in raw["legacy_rule_scales"])
        allowed = {
            "mode",
            "priority_base",
            "sigmoid_sharpness",
            "numerical_tolerance",
            "vector_schema_id",
            "legacy_vector_schema_id",
            "legacy_rule_scales",
            "native_environment_reward_weight",
            "reward_compression_mode",
            "severity",
            "flat_tie_breaker",
            "progress_weight",
            "relaxable_weight",
            "progress_rate_weight",
            "step_dt_s",
            "reference_time_s",
            "specification_id",
            "version",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ScalarizationConfigurationError(
                f"Unknown scalarization configuration field(s): {', '.join(unknown)}."
            )
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class ScalarizationResult:
    """Decomposed, reproducible result of one scalarization operation."""

    reward: float
    mode: str
    canonical_margins: tuple[float, ...]
    priority_contributions: tuple[float, ...]
    continuous_tie_breaker: float
    satisfaction_pattern: tuple[bool, ...] | None
    priority_base: float
    sigmoid_sharpness: float
    vector_schema_id: str
    legacy_rule_scales: tuple[float, ...] | None
    raw_reward: float
    reward_compression_mode: str = "none"
    specification_id: str = SCALARIZATION_SPECIFICATION_ID
    version: str = SCALARIZATION_VERSION

    @property
    def scalar_reward(self) -> float:
        """Compatibility alias used by transition and logging interfaces."""

        return self.reward

    @property
    def raw_scalar_reward(self) -> float:
        """Pre-compression scalar reward, equal to ``scalar_reward`` when uncompressed."""

        return self.raw_reward

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe diagnostics for info and run artifacts."""

        return {
            "reward": self.reward,
            "scalar_reward": self.reward,
            "raw_reward": self.raw_reward,
            "raw_scalar_reward": self.raw_reward,
            "reward_compression_mode": self.reward_compression_mode,
            "mode": self.mode,
            "canonical_margins": list(self.canonical_margins),
            "priority_contributions": list(self.priority_contributions),
            "continuous_tie_breaker": self.continuous_tie_breaker,
            "satisfaction_pattern": (
                None if self.satisfaction_pattern is None else list(self.satisfaction_pattern)
            ),
            "priority_base": self.priority_base,
            "sigmoid_sharpness": self.sigmoid_sharpness,
            "vector_schema_id": self.vector_schema_id,
            "legacy_rule_scales": (
                None if self.legacy_rule_scales is None else list(self.legacy_rule_scales)
            ),
            "specification_id": self.specification_id,
            "version": self.version,
        }


def _validate_scales(scales: Iterable[float] | None) -> None:
    if scales is None:
        raise ScalarizationConfigurationError(
            "legacy_rule_scales must contain one positive finite scale per margin."
        )
    values = tuple(float(value) for value in scales)
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ScalarizationConfigurationError(
            "legacy_rule_scales must contain positive finite values."
        )


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _symlog(value: float) -> float:
    """SCAL-V1.1 §7.7: sign(r) * log(1 + |r|), odd, strictly increasing, h(0)=0."""

    return math.copysign(math.log1p(abs(value)), value) if value != 0.0 else 0.0


def _canonicalize_bounded(
    margins: tuple[float, ...], tolerance: float, *, expected: int
) -> tuple[float, ...]:
    """Clamp near-zero margins to exactly zero and range-check every entry.

    The progress level is the one entry that may be positive; every other level
    is a negated cost and therefore lies in ``[-1, 0]``. Its index is 3 in both
    the four-level and the six-level vector, because ADR-072 added L5 and L6
    *below* progress rather than around it.
    """

    if len(margins) != expected:
        raise ScalarizationEvaluationError(
            f"Scalarization requires {expected} macro margins, got {len(margins)}."
        )
    result: list[float] = []
    for index, value in enumerate(margins):
        lower, upper = (-1.0, 1.0) if index == 3 else (-1.0, 0.0)
        if value < lower - tolerance or value > upper + tolerance:
            raise ScalarizationEvaluationError(
                f"Bounded margin at index {index} is outside [{lower}, {upper}]: {value!r}."
            )
        if abs(value) <= tolerance:
            value = 0.0
        elif abs(value - lower) <= tolerance:
            value = lower
        elif abs(value - upper) <= tolerance:
            value = upper
        result.append(float(value))
    return tuple(result)


def scalarize_rulebook_margins(
    margins: Iterable[float],
    cfg: ScalarizationConfig,
) -> ScalarizationResult:
    """Apply the configured pure scalarization formula to one margin vector."""

    try:
        values = tuple(float(value) for value in margins)
    except (TypeError, ValueError) as exc:
        raise ScalarizationEvaluationError("Margins must be a finite numeric vector.") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise ScalarizationEvaluationError("Margins must be a non-empty finite numeric vector.")

    if cfg.mode == "legacy_scaled_sigmoid":
        scales = tuple(cfg.legacy_rule_scales or ())
        if len(scales) != len(values):
            raise ScalarizationEvaluationError(
                "Legacy scales must contain exactly one value per margin: "
                f"scales={len(scales)}, margins={len(values)}."
            )
        rho = tuple(math.tanh(value / scale) for value, scale in zip(values, scales))
        exponents = range(len(values), 0, -1)
        priority_terms = tuple(
            float(cfg.priority_base**exponent)
            * _stable_sigmoid(float(cfg.sigmoid_sharpness) * normalized)
            for exponent, normalized in zip(exponents, rho)
        )
        continuous = float(sum(rho) / len(rho))
        pattern = None
        canonical = values
    elif cfg.mode == "six_level_priority_weighted_rank":
        # RULEBOOK-V5.1 §5.1:
        #   r = sum_k a^(4-k) [ (step(m_k) - 1) + sigma * m_k ]
        #       + phi * sum_k m_k  +  lambda4 * dq
        #       - eta * c_L5 * (dt / T_REF)  -  lambda6 * c_L6 * (dt / T_REF)
        # L1-L3 are SCAL-V1.2 verbatim, so their per-step dominance is inherited
        # rather than re-argued; L4, L5 and L6 form a finite exchange, because no
        # finite weight can make a continuous increment dominate a bounded cost
        # as the increment tends to zero (§5.2).
        canonical = _canonicalize_bounded(
            values,
            float(cfg.numerical_tolerance),
            expected=_REQUIRED_MARGIN_COUNT_BY_MODE[cfg.mode],
        )
        pattern = tuple(value == 0.0 for value in canonical[:3])
        base = float(cfg.priority_base)
        severity = float(cfg.severity)
        flat = float(cfg.flat_tie_breaker)
        dt_ratio = float(cfg.step_dt_s) / float(cfg.reference_time_s)
        priority_terms = tuple(
            float(weight) * ((float(is_satisfied) - 1.0) + severity * margin) + flat * margin
            for weight, is_satisfied, margin in zip(
                (base**3, base**2, base), pattern, canonical[:3]
            )
        )
        # `canonical[4]` and `canonical[5]` are negated costs, so adding their
        # weighted value subtracts the cost. Writing it as an addition keeps the
        # sign convention of the vector in one place.
        continuous = (
            float(cfg.progress_weight) * canonical[3]
            + float(cfg.relaxable_weight) * canonical[4] * dt_ratio
            + float(cfg.progress_rate_weight) * canonical[5] * dt_ratio
        )
    else:
        canonical = _canonicalize_bounded(
            values,
            float(cfg.numerical_tolerance),
            expected=_REQUIRED_MARGIN_COUNT_BY_MODE[cfg.mode],
        )
        pattern = tuple(value == 0.0 for value in canonical[:3])
        continuous = float(sum(canonical) / 4.0)
        if cfg.mode == "bounded_centered_sigmoid":
            priority_terms = tuple(
                float(base) * (2.0 * _stable_sigmoid(float(cfg.sigmoid_sharpness) * value) - 1.0)
                for base, value in zip(
                    (cfg.priority_base**3, cfg.priority_base**2, cfg.priority_base),
                    canonical[:3],
                )
            )
        elif cfg.mode == "bounded_satisfaction_rank":
            priority_terms = tuple(
                float(base) * (float(is_satisfied) - 1.0)
                for base, is_satisfied in zip(
                    (cfg.priority_base**3, cfg.priority_base**2, cfg.priority_base),
                    pattern,
                )
            )
        elif cfg.mode == "bounded_priority_weighted_rank":
            # SCAL-V1.1 §7.6: embed each margin's severity inside its own
            # priority-weighted term instead of a shared, diluted tie-breaker;
            # m_4 keeps unit weight instead of the four-way average above.
            priority_terms = tuple(
                float(base) * ((float(is_satisfied) - 1.0) + margin)
                for base, is_satisfied, margin in zip(
                    (cfg.priority_base**3, cfg.priority_base**2, cfg.priority_base),
                    pattern,
                    canonical[:3],
                )
            )
            continuous = canonical[3]
        else:  # pragma: no cover - ScalarizationConfig validates this branch.
            raise ScalarizationConfigurationError(cfg.mode)

    raw_reward = float(sum(priority_terms) + continuous)
    if not math.isfinite(raw_reward) or not math.isfinite(continuous):
        raise ScalarizationEvaluationError("Scalarization produced a non-finite result.")
    if any(not math.isfinite(value) for value in priority_terms):
        raise ScalarizationEvaluationError("Scalarization produced a non-finite contribution.")
    compression_mode = str(cfg.reward_compression_mode)
    reward = _symlog(raw_reward) if compression_mode == "symlog" else raw_reward
    if not math.isfinite(reward):
        raise ScalarizationEvaluationError("Reward compression produced a non-finite result.")
    return ScalarizationResult(
        reward=reward,
        mode=cfg.mode,
        canonical_margins=canonical,
        priority_contributions=priority_terms,
        continuous_tie_breaker=continuous,
        satisfaction_pattern=pattern,
        priority_base=float(cfg.priority_base),
        sigmoid_sharpness=float(cfg.sigmoid_sharpness),
        vector_schema_id=cfg.vector_schema_id,
        legacy_rule_scales=(
            None
            if cfg.legacy_rule_scales is None
            else tuple(float(value) for value in cfg.legacy_rule_scales)
        ),
        raw_reward=raw_reward,
        reward_compression_mode=compression_mode,
        specification_id=cfg.specification_id,
        version=cfg.version,
    )


class RulebookScalarizer:
    """Callable adapter that owns immutable scalarization configuration."""

    def __init__(self, config: ScalarizationConfig) -> None:
        self.config = config

    def __call__(self, margins: Iterable[float]) -> ScalarizationResult:
        return scalarize_rulebook_margins(margins, self.config)
