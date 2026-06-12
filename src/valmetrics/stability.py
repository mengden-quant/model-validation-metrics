from typing import Literal

import numpy as np

from valmetrics.utils import ArrayLike, GroupLike, as_1d_array, as_1d_float_array

MissingPolicy = Literal["raise", "drop", "separate"]


def _validate_psi_parameters(
    *,
    missing: MissingPolicy = "raise",
    epsilon: float = 1e-6,
) -> None:
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be between 0 and 1")

    if missing not in {"raise", "drop", "separate"}:
        raise ValueError("missing must be one of: raise, drop, separate")


def _smooth_proportions(proportions: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Apply an epsilon floor and renormalize."""
    smoothed = np.clip(proportions, epsilon, None)
    return smoothed / smoothed.sum()


def _compute_psi_from_counts(
    expected_counts: np.ndarray, actual_counts: np.ndarray, *, epsilon: float = 1e-6
):
    if expected_counts.shape != actual_counts.shape:
        raise ValueError("expected_counts and actual_counts must have the same shape")
    if expected_counts.sum() <= 0 or actual_counts.sum() <= 0:
        raise ValueError("expected and actual counts must have positive totals")

    expected_proportions = expected_counts / expected_counts.sum()
    actual_proportions = actual_counts / actual_counts.sum()
    expected_proportions = _smooth_proportions(
        expected_proportions,
        epsilon=epsilon,
    )
    actual_proportions = _smooth_proportions(
        actual_proportions,
        epsilon=epsilon,
    )

    psi_values = (actual_proportions - expected_proportions) * np.log(
        actual_proportions / expected_proportions
    )

    return float(np.sum(psi_values))


def _missing_category_mask(values: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (
            value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
            for value in values
        ),
        dtype=bool,
        count=values.size,
    )


def psi_continuous(
    expected: ArrayLike,
    actual: ArrayLike,
    *,
    bins: int = 10,
    missing: MissingPolicy = "raise",
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI for continuous values using expected-sample quantile bins."""
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

    if missing == "separate":
        expected_counts = np.append(expected_counts, expected_missing.sum())
        actual_counts = np.append(actual_counts, actual_missing.sum())

    return _compute_psi_from_counts(expected_counts, actual_counts, epsilon=epsilon)


def psi_categorical(
    expected: GroupLike,
    actual: GroupLike,
    *,
    missing: MissingPolicy = "raise",
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI for categorical values."""
    _validate_psi_parameters(missing=missing, epsilon=epsilon)

    expected_array = as_1d_array(expected, name="expected")
    actual_array = as_1d_array(actual, name="actual")

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
        value for value in dict.fromkeys(actual_non_missing.tolist()) if value not in categories
    )
    expected_counts = np.array(
        [np.sum(expected_non_missing == category) for category in categories], dtype=float
    )
    actual_counts = np.array(
        [np.sum(actual_non_missing == category) for category in categories],
        dtype=float,
    )

    if missing == "separate":
        expected_counts = np.append(expected_counts, expected_missing.sum())
        actual_counts = np.append(actual_counts, actual_missing.sum())

    return _compute_psi_from_counts(expected_counts, actual_counts, epsilon=epsilon)
