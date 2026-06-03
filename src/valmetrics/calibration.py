from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from valmetrics.utils import prepare_binary_inputs


@dataclass(frozen=True)
class HosmerLemeshowResult:
    statistic: float
    p_value: float
    degrees_of_freedom: int
    n_groups: int


def hosmer_lemeshow(
    y_true,
    y_prob,
    *,
    n_groups: int = 10,
    dropna: bool = False,
) -> HosmerLemeshowResult:
    y, p = prepare_binary_inputs(y_true, y_prob, values_name="y_prob", dropna=dropna)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("y_prob must be in [0, 1]")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2")
    if y.size < n_groups:
        raise ValueError("n_groups must not exceed number of observations")

    order = np.argsort(p, kind="mergesort")
    y_sorted = y[order]
    p_sorted = p[order]
    groups = np.array_split(np.arange(y.size), n_groups)
    statistic = 0.0
    used_groups = 0
    for group in groups:
        y_g = y_sorted[group]
        p_g = p_sorted[group]
        n_g = y_g.size
        observed_bad = np.sum(y_g)
        expected_bad = np.sum(p_g)
        observed_good = n_g - observed_bad
        expected_good = n_g - expected_bad
        if expected_bad <= 0.0 or expected_good <= 0.0:
            continue
        statistic += (observed_bad - expected_bad) ** 2 / expected_bad
        statistic += (observed_good - expected_good) ** 2 / expected_good
        used_groups += 1
    degrees_of_freedom = used_groups - 2
    if degrees_of_freedom <= 0:
        raise ValueError("Not enough valid groups to compute Hosmer-Lemeshow statistic")
    p_value = 1.0 - chi2.cdf(statistic, degrees_of_freedom)
    return HosmerLemeshowResult(
        statistic=float(statistic),
        p_value=float(p_value),
        degrees_of_freedom=int(degrees_of_freedom),
        n_groups=int(used_groups),
    )
