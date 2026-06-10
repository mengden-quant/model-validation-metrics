from typing import Literal

import numpy as np

from valmetrics.utils import ArrayLike, as_1d_float_array

MissingPolicy = Literal["raise", "drop", "separate"]


def _smooth_proportions(proportions: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Replace zero proportions and renormalize."""
    smoothed = np.clip(proportions, epsilon, None)
    return smoothed / smoothed.sum()


def psi_continuous(
    expected: ArrayLike,
    actual: ArrayLike,
    *,
    bins: int = 10,
    missing: MissingPolicy = "raise",
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI for continuous values using expected-sample quantile bins."""
    if bins < 2:
        raise ValueError("bins must be at least 2")

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be between 0 and 1")

    if missing not in {"raise", "drop", "separate"}:
        raise ValueError("missing must be one of: raise, drop, separate")

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
