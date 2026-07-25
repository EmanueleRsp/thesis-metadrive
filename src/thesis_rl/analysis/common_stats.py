from __future__ import annotations

import math


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float], mean_value: float) -> float:
    if len(values) <= 1:
        return 0.0
    return math.sqrt(sum((x - mean_value) ** 2 for x in values) / (len(values) - 1))


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Cross-seed mean and sample standard deviation.

    EVAL-PROTOCOL v1.0 REQ-009/DEC-003: no confidence interval, bootstrap
    estimate, or significance test is computed or presented. This replaces
    the former ``mean_ci95`` (``1.96 * s / sqrt(n)``) helper.
    """
    m = mean(values)
    s = sample_std(values, m)
    return m, s


def ci95(sd: float, n: int) -> float:
    """Optional, off-by-default 95% normal-approximation confidence-interval
    half-width: ``1.96 * sd_a(x) / sqrt(n)``.

    EVAL-PROTOCOL v1.0 REQ-009/DEC-003 (amended 2026-07-25): a confidence
    interval MAY additionally be computed and reported when the analyst
    explicitly opts in, using this formula for consistency with the
    historical convention; it is never computed or shown by default and
    never substitutes for the mandatory raw-values/mean/SD reporting.
    """
    if n <= 0:
        return 0.0
    return 1.96 * sd / math.sqrt(n)


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None

