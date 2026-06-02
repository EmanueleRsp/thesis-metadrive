from __future__ import annotations

import math


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float], mean_value: float) -> float:
    if len(values) <= 1:
        return 0.0
    return math.sqrt(sum((x - mean_value) ** 2 for x in values) / (len(values) - 1))


def mean_ci95(values: list[float]) -> tuple[float, float]:
    m = mean(values)
    s = sample_std(values, m)
    if len(values) <= 1:
        return m, 0.0
    ci = 1.96 * (s / math.sqrt(len(values)))
    return m, ci


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

