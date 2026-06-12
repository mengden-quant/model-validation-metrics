from collections.abc import Sequence

import numpy as np

ArrayLike = np.ndarray | Sequence[int] | Sequence[float] | Sequence[bool]
GroupLike = np.ndarray | Sequence[int] | Sequence[float] | Sequence[bool] | Sequence[str]


def as_1d_group_array(values: GroupLike, *, name: str) -> np.ndarray:
    """Convert group labels to a one-dimensional object array."""
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {array.shape}")
    return array


def as_1d_float_array(values: ArrayLike, *, name: str) -> np.ndarray:
    """Convert values to a one-dimensional float array.

    NaN values are allowed. Positive and negative infinities are rejected.
    """
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


def as_1d_binary_array(y_true: ArrayLike, *, name: str = "y_true") -> np.ndarray:
    """Convert values to a one-dimensional binary float array.

    NaN values are allowed.
    """
    y = as_1d_float_array(y_true, name=name)
    valid_unique_values = np.unique(y[~np.isnan(y)])
    if np.any(~np.isin(valid_unique_values, [0.0, 1.0])):
        raise ValueError(f"{name} must be binary, got unique={valid_unique_values.tolist()}")
    return y


def binary_class_counts(y_true: np.ndarray) -> tuple[int, int]:
    """Return counts of positive and negative classes."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    return n_pos, n_neg


def check_contains_both_binary_classes(y_true: np.ndarray, *, name: str = "y_true") -> None:
    """Validate that a binary target contains both classes."""
    n_pos, n_neg = binary_class_counts(y_true)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"{name} must contain at least one 0 and one 1")


def check_same_length(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Validate two one-dimensional arrays have equal length."""
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"{left_name} and {right_name} must have the same length")


def validate_probabilities(values: np.ndarray, *, name: str = "y_prob") -> None:
    """Validate values are finite probabilities in [0, 1]."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError(f"{name} must be in [0, 1]")


def _missing_group_mask(groups: np.ndarray) -> np.ndarray:
    """Return a mask for missing group labels."""
    return np.fromiter(
        (
            value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
            for value in groups
        ),
        dtype=bool,
        count=groups.size,
    )


def _prepare_binary_arrays(
    y_true: ArrayLike,
    values: ArrayLike,
    *,
    values_name: str,
    dropna: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare binary target and numeric values and return the retained mask."""
    y = as_1d_binary_array(y_true, name="y_true")
    x = as_1d_float_array(values, name=values_name)

    check_same_length(
        y,
        x,
        left_name="y_true",
        right_name=values_name,
    )

    if dropna:
        mask = ~np.isnan(y) & ~np.isnan(x)
    else:
        if np.any(np.isnan(y)):
            raise ValueError("y_true contains NaN values")
        if np.any(np.isnan(x)):
            raise ValueError(f"{values_name} contains NaN values")

        mask = np.ones(y.size, dtype=bool)

    y = y[mask]
    x = x[mask]

    if y.size == 0:
        raise ValueError("No valid observations remain after dropping NaN values")

    return y.astype(int, copy=False), x, mask


def prepare_binary_inputs(
    y_true: ArrayLike,
    values: ArrayLike,
    *,
    values_name: str,
    dropna: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare aligned binary target and numeric values."""
    y, x, _ = _prepare_binary_arrays(
        y_true,
        values,
        values_name=values_name,
        dropna=dropna,
    )
    return y, x


def prepare_grouped_binary_inputs(
    y_true: ArrayLike,
    values: ArrayLike,
    groups: GroupLike,
    *,
    values_name: str,
    groups_name: str = "groups",
    dropna: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare aligned binary target, numeric values, and group labels."""
    group_array = as_1d_group_array(groups, name=groups_name)

    y, x, mask = _prepare_binary_arrays(
        y_true,
        values,
        values_name=values_name,
        dropna=dropna,
    )
    check_same_length(
        mask,
        group_array,
        left_name="y_true",
        right_name=groups_name,
    )

    if np.any(_missing_group_mask(group_array)):
        raise ValueError(f"{groups_name} contains missing values")

    return y, x, group_array[mask]
