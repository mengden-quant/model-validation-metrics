from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import binom, binomtest, chi2

from valmetrics.utils import (
    ArrayLike,
    GroupLike,
    prepare_binary_inputs,
    prepare_grouped_binary_inputs,
    validate_probabilities,
)

Alternative = Literal["two-sided", "greater", "less"]
GroupLabel = int | float | str | bool


def _validate_n_groups(
    n_groups: int,
    *,
    n_observations: int,
) -> None:
    """Validate the requested number of automatic groups."""
    if isinstance(n_groups, bool) or not isinstance(n_groups, (int, np.integer)):
        raise ValueError("n_groups must be an integer")

    if n_groups < 1:
        raise ValueError("n_groups must be positive")

    if n_groups > n_observations:
        raise ValueError("n_groups must not exceed number of observations")


def _make_equal_frequency_groups(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort observations by PD and create approximately equal-sized groups.

    Observations with identical predicted probabilities are always assigned
    to the same group. Therefore, the actual number of groups can be smaller
    than n_groups.
    """
    _validate_n_groups(
        n_groups,
        n_observations=y.size,
    )

    order = np.argsort(p, kind="mergesort")
    y_sorted = y[order]
    p_sorted = p[order]

    _, inverse, counts = np.unique(
        p_sorted,
        return_inverse=True,
        return_counts=True,
    )

    group_starts = np.cumsum(counts) - counts
    group_midpoints = group_starts + counts / 2.0

    raw_group_ids = np.floor(group_midpoints * n_groups / y.size).astype(int)
    raw_group_ids = np.clip(
        raw_group_ids,
        0,
        n_groups - 1,
    )

    _, dense_group_ids = np.unique(
        raw_group_ids,
        return_inverse=True,
    )

    group_labels = dense_group_ids[inverse]

    return y_sorted, p_sorted, group_labels


@dataclass(frozen=True)
class HosmerLemeshowResult:
    statistic: float
    p_value: float
    degrees_of_freedom: int
    n_groups: int


def _compute_hosmer_lemeshow(
    y: np.ndarray,
    p: np.ndarray,
    group_labels: np.ndarray,
) -> HosmerLemeshowResult:
    """Compute the Hosmer-Lemeshow statistic for predefined groups."""
    if not (y.size == p.size == group_labels.size):
        raise ValueError("y, p, and group_labels must have the same length")

    unique_groups = tuple(dict.fromkeys(group_labels.tolist()))

    if len(unique_groups) < 3:
        raise ValueError("Hosmer-Lemeshow test requires at least 3 groups")

    statistic = 0.0

    for group in unique_groups:
        mask = group_labels == group

        y_group = y[mask]
        p_group = p[mask]

        n_observations = y_group.size
        observed_defaults = int(np.sum(y_group))
        expected_defaults = float(np.sum(p_group))

        observed_non_defaults = n_observations - observed_defaults
        expected_non_defaults = n_observations - expected_defaults

        if expected_defaults <= 0.0 or expected_non_defaults <= 0.0:
            raise ValueError(
                "Each group must have expected defaults and non-defaults "
                "strictly greater than zero"
            )

        statistic += (observed_defaults - expected_defaults) ** 2 / expected_defaults
        statistic += (observed_non_defaults - expected_non_defaults) ** 2 / expected_non_defaults

    n_groups = len(unique_groups)
    degrees_of_freedom = n_groups - 2
    p_value = chi2.sf(statistic, degrees_of_freedom)

    return HosmerLemeshowResult(
        statistic=float(statistic),
        p_value=float(p_value),
        degrees_of_freedom=int(degrees_of_freedom),
        n_groups=int(n_groups),
    )


def hosmer_lemeshow(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_groups: int = 10,
    dropna: bool = False,
) -> HosmerLemeshowResult:
    """Compute the Hosmer-Lemeshow test using approximately equal-sized groups.

    Equal predicted probabilities are not split between groups. Therefore,
    the result can contain fewer groups than requested.
    """
    y, p = prepare_binary_inputs(y_true, y_prob, values_name="y_prob", dropna=dropna)
    validate_probabilities(p, name="y_prob")

    y_grouped, p_grouped, group_labels = _make_equal_frequency_groups(
        y,
        p,
        n_groups=n_groups,
    )

    return _compute_hosmer_lemeshow(y_grouped, p_grouped, group_labels)


def grouped_hosmer_lemeshow(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    groups: GroupLike,
    *,
    dropna: bool = False,
) -> HosmerLemeshowResult:
    """Compute the Hosmer-Lemeshow test using predefined groups."""
    y, p, group_labels = prepare_grouped_binary_inputs(
        y_true,
        y_prob,
        groups,
        values_name="y_prob",
        groups_name="groups",
        dropna=dropna,
    )
    validate_probabilities(p, name="y_prob")

    return _compute_hosmer_lemeshow(
        y,
        p,
        group_labels,
    )


@dataclass(frozen=True)
class BinomialBinResult:
    group: GroupLabel
    n_observations: int
    observed_defaults: int
    expected_defaults: float
    observed_dr: float
    average_pd: float
    lower_default_bound: int
    upper_default_bound: int
    lower_default_rate: float
    upper_default_rate: float
    p_value: float


@dataclass(frozen=True)
class BinomialTestResult:
    bins: tuple[BinomialBinResult, ...]
    confidence_level: float
    alternative: Alternative
    method: Literal["binomial_exact"]


def _validate_binomial_parameters(
    *,
    confidence_level: float,
    alternative: Alternative,
) -> None:
    """Validate common binomial-test parameters."""
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")


def _compute_exact_binomial_bin_result(
    y_group: np.ndarray,
    p_group: np.ndarray,
    group: GroupLabel,
    confidence_level: float,
    alternative: Alternative,
) -> BinomialBinResult:
    """Compute an exact binomial test using the group's average PD.

    Individual predicted probabilities are replaced by their group average,
    so this is a homogeneous binomial approximation rather than an exact
    Poisson-binomial test.
    """
    n_observations = int(y_group.size)
    observed_defaults = int(y_group.sum())
    expected_defaults = float(p_group.sum())

    average_pd = float(expected_defaults / n_observations)
    observed_dr = float(observed_defaults / n_observations)

    test = binomtest(observed_defaults, n_observations, average_pd, alternative=alternative)
    p_value = float(test.pvalue)
    alpha = 1.0 - confidence_level

    if alternative == "two-sided":
        lower_default_bound = int(binom.ppf(alpha / 2.0, n_observations, average_pd))
        upper_default_bound = int(binom.isf(alpha / 2.0, n_observations, average_pd))
    elif alternative == "greater":
        lower_default_bound = 0
        upper_default_bound = int(binom.isf(alpha, n_observations, average_pd))
    else:  # "less"
        lower_default_bound = int(binom.ppf(alpha, n_observations, average_pd))
        upper_default_bound = n_observations

    return BinomialBinResult(
        group=group,
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


def _compute_binomial_test(
    y: np.ndarray,
    p: np.ndarray,
    group_labels: np.ndarray,
    *,
    confidence_level: float,
    alternative: Alternative,
) -> BinomialTestResult:
    """Compute binomial calibration results for predefined groups."""
    if not (y.size == p.size == group_labels.size):
        raise ValueError("y, p, and group_labels must have the same length")

    unique_groups = tuple(dict.fromkeys(group_labels.tolist()))

    bin_results = tuple(
        _compute_exact_binomial_bin_result(
            y[group_labels == group],
            p[group_labels == group],
            group=group,
            confidence_level=confidence_level,
            alternative=alternative,
        )
        for group in unique_groups
    )

    return BinomialTestResult(
        bins=bin_results,
        confidence_level=confidence_level,
        alternative=alternative,
        method="binomial_exact",
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
    """Compute group-level binomial tests using automatic PD groups."""
    _validate_binomial_parameters(
        confidence_level=confidence_level,
        alternative=alternative,
    )

    y, p = prepare_binary_inputs(y_true, y_prob, values_name="y_prob", dropna=dropna)
    validate_probabilities(p, name="y_prob")

    y_grouped, p_grouped, group_labels = _make_equal_frequency_groups(y, p, n_groups=n_groups)

    return _compute_binomial_test(
        y_grouped,
        p_grouped,
        group_labels,
        confidence_level=confidence_level,
        alternative=alternative,
    )


def grouped_binomial_test(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    groups: GroupLike,
    *,
    confidence_level: float = 0.95,
    alternative: Alternative = "two-sided",
    dropna: bool = False,
) -> BinomialTestResult:
    """Compute group-level binomial tests using predefined groups."""
    _validate_binomial_parameters(
        confidence_level=confidence_level,
        alternative=alternative,
    )

    y, p, group_labels = prepare_grouped_binary_inputs(
        y_true, y_prob, groups, values_name="y_prob", groups_name="groups", dropna=dropna
    )
    validate_probabilities(p, name="y_prob")

    return _compute_binomial_test(
        y,
        p,
        group_labels,
        confidence_level=confidence_level,
        alternative=alternative,
    )
