from dataclasses import dataclass

import numpy as np

from valmetrics.utils import GroupLike, as_1d_group_array, missing_group_mask


def _prepare_groups(
    groups: GroupLike,
    *,
    dropna: bool,
) -> np.ndarray:
    """Prepare one-dimensional non-empty group labels."""
    group_array = as_1d_group_array(groups, name="groups")

    missing_mask = missing_group_mask(group_array)

    if np.any(missing_mask):
        if dropna:
            group_array = group_array[~missing_mask]
        else:
            raise ValueError("groups contains missing values")

    if group_array.size == 0:
        raise ValueError("groups must contain at least one observation")

    for value in group_array:
        if isinstance(value, (float, np.floating)) and np.isinf(value):
            raise ValueError("groups contains infinite values")

    return group_array


def _resolve_n_groups(
    observed_n_groups: int,
    n_groups: int | None,
) -> int:
    """Resolve and validate the number of groups used for HHI normalization."""
    if n_groups is None:
        return observed_n_groups

    if isinstance(n_groups, bool) or not isinstance(n_groups, (int, np.integer)):
        raise ValueError("n_groups must be an integer")

    if n_groups < 1:
        raise ValueError("n_groups must be positive")

    if n_groups < observed_n_groups:
        raise ValueError("n_groups must be at least the number of observed groups")

    return int(n_groups)


def herfindahl_hirschman(
    groups: GroupLike,
    *,
    n_groups: int | None = None,
    normalized: bool = False,
    dropna: bool = False,
) -> float:
    """Compute the Herfindahl-Hirschman Index over group frequencies."""
    group_array = _prepare_groups(groups, dropna=dropna)

    _, counts = np.unique(group_array, return_counts=True)
    observed_n_groups = counts.size
    effective_n_groups = _resolve_n_groups(observed_n_groups, n_groups)

    shares = counts / group_array.size
    hhi = np.sum(shares**2)

    if not normalized:
        return float(hhi)

    if effective_n_groups == 1:
        return 1.0

    normalized_hhi = (hhi - 1.0 / effective_n_groups) / (1.0 - 1.0 / effective_n_groups)
    return float(normalized_hhi)


@dataclass(frozen=True)
class HCIResult:
    groups: tuple[int | float | str, ...]
    value: float


def hci(
    groups: GroupLike,
    *,
    dropna: bool = False,
) -> HCIResult:
    """Compute the Highest Concentration Index and return all groups with maximum share."""
    group_array = _prepare_groups(groups, dropna=dropna)

    labels, counts = np.unique(group_array, return_counts=True)

    max_count = np.max(counts)
    max_groups = tuple(labels[counts == max_count])

    return HCIResult(groups=max_groups, value=float(max_count / group_array.size))
