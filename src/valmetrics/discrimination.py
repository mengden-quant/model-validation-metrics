import numpy as np
from sklearn.metrics import roc_auc_score

from valmetrics.utils import (
    ArrayLike,
    binary_class_counts,
    check_contains_both_binary_classes,
    prepare_binary_inputs,
)


def _gini_from_auc(auc: float) -> float:
    """Convert AUC to Gini: G = 2*AUC - 1."""
    return float(2.0 * auc - 1.0)


def __roc_auc_prepared(y: np.ndarray, s: np.ndarray) -> float:
    """Compute ROC AUC for already prepared arrays."""
    check_contains_both_binary_classes(y)
    return float(roc_auc_score(y, s))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike, *, dropna: bool = False) -> float:
    """Standard ROC AUC (ties are treated neutrally as 0.5 in sklearn)."""
    y, s = prepare_binary_inputs(y_true, y_score, values_name="y_score", dropna=dropna)
    return __roc_auc_prepared(y, s)


def _conservative_tie_correction_prepared(y: np.ndarray, s: np.ndarray) -> float:
    """Compute conservaive tie correction for already prepared arrays."""
    check_contains_both_binary_classes(y)

    n_pos, n_neg = binary_class_counts(y)

    _, inv = np.unique(s, return_inverse=True)
    pos_j = np.bincount(inv, weights=y.astype(float))
    cnt_j = np.bincount(inv)
    neg_j = cnt_j - pos_j

    tie_pairs = np.sum(pos_j * neg_j)

    return float(tie_pairs / (n_pos * n_neg))


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
    y, s = prepare_binary_inputs(y_true, y_score, values_name="y_score", dropna=dropna)

    return _conservative_tie_correction_prepared(y, s)


def gini_standard(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    dropna: bool = False,
) -> float:
    """
    Compute standard Gini coefficient as 2 * ROC AUC - 1.
    """
    y, s = prepare_binary_inputs(y_true, y_score, values_name="y_score", dropna=dropna)
    auc = __roc_auc_prepared(y, s)
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
    y, s = prepare_binary_inputs(y_true, y_score, values_name="y_score", dropna=dropna)
    auc = __roc_auc_prepared(y, s)
    tie_correction = _conservative_tie_correction_prepared(y, s)
    return float(_gini_from_auc(auc) - tie_correction)


def ks_statistic(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    dropna: bool = False,
) -> float:
    """Compute the Kolmogorov-Smirnov statistic between score distributions by class."""
    y, s = prepare_binary_inputs(y_true, y_score, values_name="y_score", dropna=dropna)
    check_contains_both_binary_classes(y)
    n_pos, n_neg = binary_class_counts(y)
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
