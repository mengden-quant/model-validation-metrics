from dataclasses import dataclass
from typing import Literal

import numpy as np

from valmetrics.utils import ArrayLike, GroupLike, as_1d_float_array, as_1d_group_array

MissingPolicy = Literal["raise", "drop", "separate"]


def _validate_psi_parameters(
    *,
    missing: MissingPolicy = "raise",
    epsilon: float = 1e-6,
) -> None:
    """Validatie common PSI parameters."""
    if missing not in {"raise", "drop", "separate"}:
        raise ValueError("missing must be one of: raise, drop, separate")

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be between 0 and 1")


def _smooth_proportions(proportions: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Apply an epsilon floor and renormalize proportions."""
    smoothed = np.clip(proportions, epsilon, None)
    return smoothed / smoothed.sum()


@dataclass(frozen=True)
class _PSICalculation:
    value: float
    expected_proportions: np.ndarray
    actual_proportions: np.ndarray
    contributions: np.ndarray


def _compute_psi_from_counts(
    expected_counts: np.ndarray, actual_counts: np.ndarray, *, epsilon: float = 1e-6
) -> _PSICalculation:
    """Compute PSI components from aligned bin counts."""
    expected_proportions = expected_counts / expected_counts.sum()
    actual_proportions = actual_counts / actual_counts.sum()

    expected_smoothed = _smooth_proportions(
        expected_proportions,
        epsilon=epsilon,
    )
    actual_smoothed = _smooth_proportions(
        actual_proportions,
        epsilon=epsilon,
    )

    contributions = (actual_smoothed - expected_smoothed) * np.log(
        actual_smoothed / expected_smoothed
    )

    return _PSICalculation(
        value=float(np.sum(contributions)),
        expected_proportions=expected_proportions,
        actual_proportions=actual_proportions,
        contributions=contributions,
    )


def _missing_category_mask(values: np.ndarray) -> np.ndarray:
    """Return a mask identifying missing categorical labels."""
    return np.fromiter(
        (
            value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
            for value in values
        ),
        dtype=bool,
        count=values.size,
    )


@dataclass(frozen=True)
class ContinuousPSIBinResult:
    lower_bound: float | None
    upper_bound: float | None
    expected_count: int
    actual_count: int
    expected_proportion: float
    actual_proportion: float
    contribution: float


@dataclass(frozen=True)
class ContinuousPSIResult:
    value: float
    bins: tuple[ContinuousPSIBinResult, ...]


def psi_continuous(
    expected: ArrayLike,
    actual: ArrayLike,
    *,
    bins: int = 10,
    missing: MissingPolicy = "raise",
    epsilon: float = 1e-6,
) -> ContinuousPSIResult:
    """Compute PSI for continuous values using reference-sample quantile bins."""
    _validate_psi_parameters(missing=missing, epsilon=epsilon)

    if bins < 2:
        raise ValueError("bins must be at least 2")

    expected_array = as_1d_float_array(expected, name="expected")
    actual_array = as_1d_float_array(actual, name="actual")

    if expected_array.size == 0:
        raise ValueError("expected must contain at least one observation")
    if actual_array.size == 0:
        raise ValueError("actual must contain at least one observation")

    expected_missing = np.isnan(expected_array)
    actual_missing = np.isnan(actual_array)

    if missing == "raise" and (np.any(expected_missing) or np.any(actual_missing)):
        raise ValueError("expected and actual contain missing values")

    expected_non_missing = expected_array[~expected_missing]
    actual_non_missing = actual_array[~actual_missing]

    if expected_non_missing.size == 0:
        raise ValueError("expected must contain at least one non-missing observation")

    if missing == "drop" and actual_non_missing.size == 0:
        raise ValueError("actual must contain at least one non-missing observation")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected_non_missing, quantiles))

    if edges.size < 2:
        raise ValueError("expected must contain at least two distinct values for continuous PSI")

    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_counts, _ = np.histogram(expected_non_missing, bins=edges)
    actual_counts, _ = np.histogram(actual_non_missing, bins=edges)

    bins_definitions: list[tuple[float | None.float | None]] = [
        (
            float(lower),
            float(upper),
        )
        for lower, upper in zip(edges[:-1], edges[1:], strict=True)
    ]

    if missing == "separate":
        expected_counts = np.append(expected_counts, int(expected_missing.sum()))
        actual_counts = np.append(actual_counts, int(actual_missing.sum()))
        bins_definitions.append((None, None))

    calculation = _compute_psi_from_counts(expected_counts, actual_counts, epsilon=epsilon)

    bin_results: list[ContinuousPSIBinResult] = []
    for index, (lower_bound, upper_bound) in enumerate(bins_definitions):
        bin_results.append(
            ContinuousPSIBinResult(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                expected_count=int(expected_counts[index]),
                actual_count=int(actual_counts[index]),
                expected_proportion=float(calculation.expected_proportions[index]),
                actual_proportion=float(calculation.actual_proportions[index]),
                contribution=float(calculation.contributions[index]),
            )
        )
    return ContinuousPSIResult(
        value=calculation.value,
        bins=tuple(bin_results),
    )


@dataclass(frozen=True)
class CategoricalPSIBinResult:
    category: int | float | str | bool | None
    expected_count: int
    actual_count: int
    expected_proportion: float
    actual_proportion: float
    contribution: float


@dataclass(frozen=True)
class CategoricalPSIResult:
    value: float
    bins: tuple[CategoricalPSIBinResult, ...]


def psi_categorical(
    expected: GroupLike,
    actual: GroupLike,
    *,
    missing: MissingPolicy = "raise",
    epsilon: float = 1e-6,
) -> CategoricalPSIResult:
    """Compute PSI for categorical values."""
    _validate_psi_parameters(missing=missing, epsilon=epsilon)

    expected_array = as_1d_group_array(expected, name="expected")
    actual_array = as_1d_group_array(actual, name="actual")

    if expected_array.size == 0:
        raise ValueError("expected must contain at least one observation")
    if actual_array.size == 0:
        raise ValueError("actual must contain at least one observation")

    expected_missing = _missing_category_mask(expected_array)
    actual_missing = _missing_category_mask(actual_array)

    if missing == "raise" and (np.any(expected_missing) or np.any(actual_missing)):
        raise ValueError("expected and actual contain missing values")

    expected_non_missing = expected_array[~expected_missing]
    actual_non_missing = actual_array[~actual_missing]

    if missing == "drop":
        if expected_non_missing.size == 0:
            raise ValueError("expected must contain at least one non-missing observation")
        if actual_non_missing.size == 0:
            raise ValueError("actual must contain at least one non-missing observation")

    categories = list(dict.fromkeys(expected_non_missing.tolist()))
    categories.extend(
        category
        for category in dict.fromkeys(actual_non_missing.tolist())
        if category not in categories
    )
    expected_counts = np.array(
        [np.sum(expected_non_missing == category) for category in categories], dtype=float
    )
    actual_counts = np.array(
        [np.sum(actual_non_missing == category) for category in categories],
        dtype=float,
    )

    result_categories: list[int | float | str | bool | None] = list(categories)

    if missing == "separate":
        expected_counts = np.append(expected_counts, int(expected_missing.sum()))
        actual_counts = np.append(actual_counts, int(actual_missing.sum()))
        result_categories.append(None)

    calculation = _compute_psi_from_counts(expected_counts, actual_counts, epsilon=epsilon)

    bin_results: list[CategoricalPSIBinResult] = []
    for index, category in enumerate(result_categories):
        bin_results.append(
            CategoricalPSIBinResult(
                category=category,
                expected_count=int(expected_counts[index]),
                actual_count=int(actual_counts[index]),
                expected_proportion=float(calculation.expected_proportions[index]),
                actual_proportion=float(calculation.actual_proportions[index]),
                contribution=float(calculation.contributions[index]),
            )
        )
    return CategoricalPSIResult(
        value=calculation.value,
        bins=tuple(bin_results),
    )
