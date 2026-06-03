from dataclasses import dataclass

import numpy as np

from valmetrics.utils import GroupLike


def _is_missing_group(groups: np.ndarray) -> np.ndarray:
    """Return mask of missing group labels."""
    if np.issubdtype(groups.dtype, np.number):
        return ~np.isfinite(groups.astype(float, copy=False))
    return np.array([x is None for x in groups], dtype=bool)


def herfindahl_hirschman(
    groups: GroupLike,
    *,
    n_groups: int | None = None,
    normalized: bool = False,
    dropna: bool = False,
) -> float:
    """Compute the Herfindahl-Hirschman Index over group frequencies."""
    group_array = np.asarray(groups)
    if group_array.ndim != 1:
        raise ValueError(f"groups must be 1D, got shape {group_array.shape}")
    if dropna:
        group_array = group_array[~_is_missing_group(group_array)]
    if group_array.size == 0:
        raise ValueError("groups must contain at least one observation")
    _, counts = np.unique(group_array, return_counts=True)
    observed_n_groups = counts.size
    if n_groups is None:
        effective_n_groups = observed_n_groups
    else:
        if n_groups < 1:
            raise ValueError("n_groups must be positive")
        if n_groups < observed_n_groups:
            raise ValueError("n_groups must be at least the number of observed groups")
        effective_n_groups = n_groups

    shares = counts.astype(float) / group_array.size
    hhi = np.sum(shares**2)

    if not normalized:
        return float(hhi)

    if effective_n_groups == 1:
        return 1.0

    normalized_hhi = (hhi - 1.0 / effective_n_groups) / (1.0 - 1.0 / effective_n_groups)
    return float(normalized_hhi)


@dataclass(frozen=True)
class HCIResult:
    groups: tuple[int | float | str]
    value: float


def hci(
    groups: GroupLike,
    *,
    dropna: bool = False,
) -> HCIResult:
    """Compute the Highest Concentration Index and return all groups with maximum share."""
    group_array = np.asarray(groups)

    if group_array.ndim != 1:
        raise ValueError(f"groups must be 1D, got shape {group_array.shape}")

    if dropna:
        group_array = group_array[~_is_missing_group(group_array)]

    if group_array.size == 0:
        raise ValueError("groups must contain at least one observation")

    labels, counts = np.unique(group_array, return_counts=True)

    max_count = np.max(counts)
    max_labels = labels[counts == max_count]
    max_share = max_count / group_array.size

    return HCIResult(groups=tuple(max_labels), value=float(max_share))
