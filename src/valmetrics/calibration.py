from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import binom, binomtest, chi2

from valmetrics.utils import ArrayLike, prepare_binary_inputs, validate_probabilities

Alternative = Literal["two-sided", "greater", "less"]


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


@dataclass(frozen=True)
class BinomialBinResult:
    bin_id: int
    n_observations: int
    observed_defaults: int
    expected_defaults: float
    observed_dr: float
    average_pd: float
    lower_default_bound: float
    upper_default_bound: float
    lower_default_rate: float
    upper_default_rate: float
    p_value: float


@dataclass(frozen=True)
class BinomialTestResult:
    bins: tuple[BinomialBinResult, ...]
    confidence_level: float
    alternative: Alternative
    method: Literal["binomial_exact"]


def _compute_exact_binomial_bin_result(
    y_g: np.ndarray,
    p_g: np.ndarray,
    bin_id: int,
    confidence_level: float = 0.95,
    alternative: Alternative = "two-sided",
) -> BinomialBinResult:
    n_observations = int(y_g.size)
    observed_defaults = int(y_g.sum())
    expected_defaults = float(p_g.sum())
    average_pd = float(expected_defaults / n_observations)
    observed_dr = float(observed_defaults / n_observations)

    if not 0.0 < average_pd < 1.0:
        raise ValueError("average PD must be strictly between 0 and 1 in each bin")

    test = binomtest(observed_defaults, n_observations, average_pd, alternative=alternative)
    p_value = float(test.pvalue)
    alpha = 1.0 - confidence_level

    if alternative == "two-sided":
        lower_default_bound = int(binom.ppf(alpha / 2.0, n_observations, average_pd))
        upper_default_bound = int(binom.ppf(1 - alpha / 2.0, n_observations, average_pd))
    elif alternative == "greater":
        lower_default_bound = 0
        upper_default_bound = int(binom.ppf(1 - alpha, n_observations, average_pd))
    else:  # "less"
        lower_default_bound = int(binom.ppf(alpha, n_observations, average_pd))
        upper_default_bound = n_observations

    return BinomialBinResult(
        bin_id=bin_id,
        n_observations=n_observations,
        observed_defaults=observed_defaults,
        expected_defaults=expected_defaults,
        observed_dr=observed_dr,
        average_pd=average_pd,
        lower_default_bound=lower_default_bound,
        upper_default_bound=upper_default_bound,
        lower_default_rate=lower_default_bound / n_observations,
        upper_default_rate=upper_default_bound / n_observations,
        p_value=p_value,
    )


def binomial_test(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_groups: int = 10,
    confidence_level: float = 0.95,
    alternative: Alternative = "two-sided",
    dropna: bool = False,
) -> BinomialTestResult:
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2")

    y, p = prepare_binary_inputs(y_true, y_prob, values_name="y_prob", dropna=dropna)
    validate_probabilities(p, name="y_prob")

    if y.size < n_groups:
        raise ValueError("n_groups must not exceed number of observations")

    order = np.argsort(p, kind="mergesort")
    y_sorted = y[order]
    p_sorted = p[order]
    groups = np.array_split(np.arange(y.size), n_groups)

    bin_results: list[BinomialBinResult] = []
    for bin_id, group in enumerate(groups):
        y_g = y_sorted[group]
        p_g = p_sorted[group]
        bin_results.append(
            _compute_exact_binomial_bin_result(
                y_g=y_g,
                p_g=p_g,
                bin_id=bin_id,
                confidence_level=confidence_level,
                alternative=alternative,
            )
        )
    return BinomialTestResult(
        bins=tuple(bin_results),
        confidence_level=confidence_level,
        alternative=alternative,
        method="binomial_exact",
    )
