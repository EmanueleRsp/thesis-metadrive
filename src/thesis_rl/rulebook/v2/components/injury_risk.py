"""Injury-risk curves backing the R1 collision cost (Rulebook v4.9 §4).

The cost of a new contact is the probability that the impact produces an
at-least-serious injury (``MAIS3+F``), read from the weighted binary logistic
regression of:

    N. Lubbe, Y. Wu, H. Jeppsson, "Safe speeds: fatality and injury risks of
    pedestrians, cyclists, motorcyclists, and car drivers impacting the front
    of another passenger car as a function of closing speed and age",
    Traffic Safety Research, vol. 2, 000006, 2022. DOI: 10.55329/vfma7555.

That source is used because its independent variable is the *closing speed*
between the two crash partners -- exactly the quantity the monitor already
derives in v4.7 §5.3 -- and because a single dataset and methodology covers
every actor class the rulebook needs, so the curves are comparable with each
other by construction.  Its codomain is ``(0, 1)``, so the bounded-cost
requirement of the rulebook contract holds without any tunable normalization
constant.  This replaces the v4.7 §5.4 normalization by the configured speed
cap, which made the cost depend on scenario configuration at equal physical
impact severity (v4.9 §1.1, ADR-027).

Coefficients are stored exactly as published (per km/h), not pre-multiplied,
so that every value can be checked against Tables 2, 3 and 5 of the source by
direct comparison.  ``test_published_ten_percent_anchors_are_reproduced``
verifies the transcription against the anchors printed in the source's §4.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log

from thesis_rl.rulebook.v2.types import ActorClass


MPS_TO_KMH = 3.6

# v4.9 §4.3 (DEC-R1-03): a single reference age for every class isolates
# vulnerability from the sample's age composition.  65 is the value the source
# itself uses for its cross-user vulnerability comparison (its §4.3).
REFERENCE_AGE_YEARS = 65.0

INJURY_SEVERITY = "MAIS3+F"
INJURY_RISK_SOURCE = "Lubbe et al. (2022), Traffic Safety Research 2:000006"


@dataclass(frozen=True, slots=True)
class InjuryRiskModel:
    """Logistic injury-risk coefficients, in the source's published units."""

    curve_id: str
    intercept: float
    per_kmh: float
    per_year: float

    def __post_init__(self) -> None:
        values = (self.intercept, self.per_kmh, self.per_year)
        if not all(isfinite(value) for value in values):
            raise ValueError("Injury-risk coefficients must be finite")
        if self.per_kmh <= 0.0:
            raise ValueError("Injury-risk closing-speed coefficient must be positive")

    def logit(self, *, closing_speed_mps: float, age_years: float) -> float:
        """Return ``z`` of Equation (1) of the source, for a m/s input."""

        return (
            self.intercept
            + self.per_kmh * MPS_TO_KMH * closing_speed_mps
            + self.per_year * age_years
        )


# Source Tables 2 (pedestrian), 3 (cyclist) and 5 (car driver), MAIS3+F rows.
_PEDESTRIAN_MAIS3F = InjuryRiskModel("pedestrian_mais3f", -6.190, 0.078, 0.038)
_CYCLIST_MAIS3F = InjuryRiskModel("cyclist_mais3f", -7.467, 0.079, 0.047)
_CAR_DRIVER_MAIS3F = InjuryRiskModel("car_driver_mais3f", -7.654, 0.041, 0.021)

# v4.9 §4.4.  STATIC_COLLIDABLE has no curve in the source: in an ego-versus-
# fixed-object impact the exposed party is the ego occupant, so the car-driver
# (car-occupant) curve applies by analogy (DEC-R1-04).  Declared limitation and
# its direction, v4.9 §9.1: the source curve describes impacts against another
# car's deformable front, while a rigid obstacle concentrates load, so this
# approximation *under-estimates* the real risk.
MAIS3F_MODEL_BY_ACTOR_CLASS: dict[ActorClass, InjuryRiskModel] = {
    ActorClass.PEDESTRIAN: _PEDESTRIAN_MAIS3F,
    ActorClass.CYCLIST: _CYCLIST_MAIS3F,
    ActorClass.VEHICLE: _CAR_DRIVER_MAIS3F,
    ActorClass.STATIC_COLLIDABLE: _CAR_DRIVER_MAIS3F,
}

# Median ages of the source's own sample, used only to reproduce its published
# anchors when verifying the coefficient transcription.  They are deliberately
# NOT the runtime age: see REFERENCE_AGE_YEARS and DEC-R1-03.
SOURCE_MEDIAN_AGE_YEARS: dict[ActorClass, float] = {
    ActorClass.PEDESTRIAN: 46.0,
    ActorClass.CYCLIST: 39.0,
    ActorClass.VEHICLE: 39.0,
    ActorClass.STATIC_COLLIDABLE: 39.0,
}


def _stable_logistic(value: float) -> float:
    """Overflow-safe logistic, mirroring ``reward.scalarization._stable_sigmoid``."""

    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    exponential = exp(value)
    return exponential / (1.0 + exponential)


def model_for(actor_class: ActorClass) -> InjuryRiskModel:
    """Return the mapped curve; an unmapped class is fatal, never defaulted."""

    model = MAIS3F_MODEL_BY_ACTOR_CLASS.get(actor_class)
    if model is None:
        raise ValueError(f"Actor class {actor_class.value!r} has no MAIS3+F injury-risk curve")
    return model


def injury_risk_cost(
    *,
    actor_class: ActorClass,
    normal_closing_speed_mps: float,
    age_years: float = REFERENCE_AGE_YEARS,
) -> float:
    """Return the MAIS3+F probability for one contact (v4.9 §4.1).

    ``normal_closing_speed_mps`` is the pre-state normal component ``u_i`` of
    v4.7 §5.3, not the magnitude of the relative velocity.  Keeping the normal
    projection preserves v4.7's deliberate graze-versus-frontal discrimination;
    since ``u_i <= ||v_e - v_i||`` the substitution is conservative in the
    under-estimating direction (v4.9 §4.2, DEC-R1-05).
    """

    if not isfinite(normal_closing_speed_mps) or normal_closing_speed_mps < 0.0:
        raise ValueError("Normal closing speed must be finite and non-negative")
    if not isfinite(age_years):
        raise ValueError("Injury-risk reference age must be finite")
    model = model_for(actor_class)
    risk = _stable_logistic(
        model.logit(closing_speed_mps=normal_closing_speed_mps, age_years=age_years)
    )
    if not isfinite(risk) or not 0.0 <= risk <= 1.0:
        raise ValueError("Injury-risk probability must be finite and in [0, 1]")
    return float(risk)


def closing_speed_at_risk_kmh(
    *,
    actor_class: ActorClass,
    risk: float,
    age_years: float,
) -> float:
    """Invert the curve; used only to verify the published anchors, not at runtime."""

    if not 0.0 < risk < 1.0:
        raise ValueError("Target injury risk must be in (0, 1)")
    model = model_for(actor_class)
    return (log(risk / (1.0 - risk)) - model.intercept - model.per_year * age_years) / (
        model.per_kmh
    )
