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


def prepare_binary_target(y_true: ArrayLike, *, name: str = "y_true") -> np.ndarray:
    """Validate and convert binary target to an integer NumPy array."""
    y = as_1d_numeric_array(y_true, name=name)
    if not np.all(np.equal(y, np.floor(y))):
        raise ValueError(f"{name} must be integer/bool (0/1), got non-integers")
    y_int = y.astype(int, copy=False)
    unique_values = np.unique(y_int)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(f"{name} must be binary in {{0,1}}, got unique={unique_values.tolist()}")

    if np.sum(y_int == 1) == 0 or np.sum(y_int == 0) == 0:
        raise ValueError(f"{name} must contain at least one 0 and one 1")
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
    """Validate values are probabilities in [0, 1]."""
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError(f"{name} must be in [0, 1]")


def binary_class_counts(y_true: np.ndarray) -> tuple[int, int]:
    """Return counts of positive and negative classes."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    return n_pos, n_neg


def prepare_binary_inputs(
    y_true: ArrayLike,
    values: ArrayLike,
    *,
    values_name: str,
    dropna: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary target and numeric values with optional pairwise missing-value removal."""
    y_raw = np.asarray(y_true)
    x_raw = np.asarray(values)
    if y_raw.ndim != 1 or x_raw.ndim != 1:
        raise ValueError(
            f"y_true and {values_name} must be 1D, got shapes {y_raw.shape} and {x_raw.shape}"
        )
    if y_raw.shape[0] != x_raw.shape[0]:
        raise ValueError(f"y_true and {values_name} must have the same length")
    try:
        y_float = y_raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must be numeric or bool") from exc
    try:
        x_float = x_raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{values_name} must be numeric") from exc
    if dropna:
        mask = np.isfinite(y_float) & np.isfinite(x_float)
        y_float = y_float[mask]
        x_float = x_float[mask]
    y = prepare_binary_target(y_float)
    x = as_1d_numeric_array(x_float, name=values_name)
    check_same_length(y, x, left_name="y_true", right_name=values_name)
    return y, x
