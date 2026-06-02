from collections.abc import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score

ArrayLike = np.ndarray | Iterable[float]


def _prepare_inputs(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    dropna: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Validate and sanitize inputs; returns (y:int, s:float, n_pos, n_neg)."""
    y = np.asarray(y_true)
    s = np.asarray(y_score)

    if y.ndim != 1 or s.ndim != 1:
        raise ValueError(f"y_true and y_score must be 1D, got shapes {y.shape} and {s.shape}")
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    # Convert inputs to float for finite checks and numeric validation
    try:
        y_float = y.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must be numeric or bool") from exc
    try:
        s = s.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_score must be numeric") from exc

    if dropna:
        m = np.isfinite(s) & np.isfinite(y_float)
        s = s[m]
        y_float = y_float[m]

    # y must be {0,1} (allow bool/int; reject non-integer floats)
    if not np.all(np.isfinite(y_float)):
        raise ValueError("y_true contains non-finite values")
    if not np.all(np.isfinite(s)):
        raise ValueError("y_score contains non-finite values")
    if not np.all(np.equal(y_float, np.floor(y_float))):
        raise ValueError("y_true must be integer/bool (0/1), got non-integers")

    y = y_float.astype(int, copy=False)

    unique_values = np.unique(y)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(f"y_true must be binary in {{0,1}}, got unique={unique_values.tolist()}")

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need at least one positive (1) and one negative (0) sample")

    return y, s, n_pos, n_neg


def _gini_from_auc(auc: float) -> float:
    """Convert AUC to Gini: G = 2*AUC - 1."""
    return float(2.0 * auc - 1.0)


def roc_auc(y_true: ArrayLike, y_score: ArrayLike, *, dropna: bool = False) -> float:
    """Standard ROC AUC (ties treated neutrally as 0.5 in sklearn)."""
    y, s, _, _ = _prepare_inputs(y_true, y_score, dropna=dropna)
    return float(roc_auc_score(y, s))


def conservative_tie_correction(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    dropna: bool = False,
) -> float:
    """
    Conservative tie correction based on tie groups:

        T = sum_j pos_j * neg_j  over groups with equal score
        corr = T / (N_pos * N_neg)

    Depends only on tie structure (permutation-invariant).
    """
    y, s, n_pos, n_neg = _prepare_inputs(y_true, y_score, dropna=dropna)

    # group by identical scores
    _, inv = np.unique(s, return_inverse=True)
    pos_j = np.bincount(inv, weights=y.astype(float))
    cnt_j = np.bincount(inv)
    neg_j = cnt_j - pos_j

    ties_pairs = np.sum(pos_j * neg_j)
    return float(ties_pairs / (n_pos * n_neg))


def gini_standard(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    dropna: bool = False,
) -> float:
    """
    Compute standard Gini coefficient as 2 * ROC AUC - 1.
    """
    auc = roc_auc(y_true, y_score, dropna=dropna)
    return _gini_from_auc(auc)


def gini_conservative(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    dropna: bool = False,
) -> float:
    """
    Compute conservative Gini coefficient by penalizing mixed tie groups.
    """
    return float(
        gini_standard(y_true, y_score, dropna=dropna)
        - conservative_tie_correction(y_true, y_score, dropna=dropna)
    )


def ks_statistic(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    dropna: bool = False,
) -> float:
    """Compute the Kolmogorov-Smirnov statistic between score distributions by class."""
    y, s, n_pos, n_neg = _prepare_inputs(y_true, y_score, dropna=dropna)
    order = np.argsort(s, kind="mergesort")

    y_sorted = y[order]
    s_sorted = s[order]

    is_pos = y_sorted == 1
    is_neg = y_sorted == 0

    cum_pos = np.cumsum(is_pos) / n_pos
    cum_neg = np.cumsum(is_neg) / n_neg

    _, group_counts = np.unique(s_sorted, return_counts=True)
    group_last_indices = np.cumsum(group_counts) - 1
    ks_values = np.abs(cum_pos[group_last_indices] - cum_neg[group_last_indices])

    return float(np.max(ks_values))
