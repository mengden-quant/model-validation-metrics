from collections.abc import Iterable

import numpy as np

ArrayLike = np.ndarray | Iterable[int] | Iterable[float] | Iterable[bool]
GroupLike = np.ndarray | Iterable[int] | Iterable[float] | Iterable[bool] | Iterable[str]


def as_1d_numeric_array(values: ArrayLike, *, name: str) -> np.ndarray:
    """Convert values to a 1D finite numeric NumPy array."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {array.shape}")
    try:
        array = array.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def as_1d_float_array_allow_nan(values: ArrayLike, *, name: str) -> np.ndarray:
    """Convert values to a 1D float array while allowing NaN values."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {array.shape}")
    try:
        array = array.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} contains infinite values")
    return array


def binary_class_counts(y_true: np.ndarray) -> tuple[int, int]:
    """Return counts of positive and negative classes."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    return n_pos, n_neg


def check_contains_both_binary_classes(y_true: np.ndarray, *, name: str = "y_true") -> None:
    """Validate binary target contains at least one positive and one negative class."""
    n_pos, n_neg = binary_class_counts(y_true)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"{name} must contain at least one 0 and one 1")


def prepare_binary_target(y_true: ArrayLike, *, name: str = "y_true") -> np.ndarray:
    """Validate and convert binary target to an integer NumPy array."""
    y = as_1d_numeric_array(y_true, name=name)
    unique_values = np.unique(y)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(f"{name} must be binary in {{0,1}}, got unique={unique_values.tolist()}")
    y_int = y.astype(int, copy=False)
    check_contains_both_binary_classes(y_int, name=name)
    return y_int


def check_same_length(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Validate two arrays have the same length."""
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"{left_name} and {right_name} must have the same length")


def validate_probabilities(values: np.ndarray, *, name: str = "y_prob") -> None:
    """Validate values are finite probabilities in [0, 1]."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError(f"{name} must be in [0, 1]")


def prepare_binary_inputs(
    y_true: ArrayLike,
    values: ArrayLike,
    *,
    values_name: str,
    dropna: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary target and numeric values.
    If dropna=True, NaN values are removed pairwise from values and y_true.
    Infinite values are always rejected.
    """
    y = prepare_binary_target(y_true)
    if dropna:
        x = as_1d_float_array_allow_nan(values, name=values_name)
    else:
        x = as_1d_numeric_array(values, name=values_name)
    check_same_length(y, x, left_name="y_true", right_name=values_name)
    if dropna:
        mask = ~np.isnan(x)
        y = y[mask]
        x = x[mask]
        check_contains_both_binary_classes(y, name="y_true after dropna")
    return y, x
