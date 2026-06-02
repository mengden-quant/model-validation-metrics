import numpy as np
import pandas as pd
import pytest

from valmetrics.discrimination import (
    conservative_tie_correction,
    gini_conservative,
    gini_standard,
    ks_statistic,
)


def test_perfect_ranking():
    df = pd.DataFrame(
        {"PD": [0.23, 0.17, 0.13, 0.11, 0.07, 0.04, 0.01], "target": [1, 1, 1, 1, 0, 0, 0]}
    )
    assert gini_standard(df["target"], df["PD"]) == pytest.approx(1.0)


def test_reverse_ranking():
    df = pd.DataFrame(
        {"PD": [0.23, 0.17, 0.13, 0.11, 0.07, 0.04, 0.01], "target": [0, 0, 0, 0, 1, 1, 1]}
    )
    assert gini_standard(df["target"], df["PD"]) == pytest.approx(-1.0)


def test_no_ties_correction():
    df = pd.DataFrame(
        {"PD": [0.23, 0.17, 0.13, 0.11, 0.07, 0.04, 0.01], "target": [0, 0, 0, 0, 1, 1, 1]}
    )
    assert conservative_tie_correction(df["target"], df["PD"]) == 0.0


def test_mixed_ties_correction():
    df = pd.DataFrame(
        {"PD": [0.23, 0.23, 0.13, 0.11, 0.07, 0.04, 0.01], "target": [0, 1, 0, 0, 1, 1, 1]}
    )
    assert conservative_tie_correction(df["target"], df["PD"]) > 0.0


def test_conservative_gini_is_less_than_standard_with_mixed_ties():
    df = pd.DataFrame(
        {"PD": [0.23, 0.23, 0.13, 0.11, 0.07, 0.04, 0.01], "target": [0, 1, 0, 0, 1, 1, 1]}
    )
    standard = gini_standard(df["target"], df["PD"])
    conservative = gini_conservative(df["target"], df["PD"])
    assert conservative < standard


def test_nan_score_raises():
    df = pd.DataFrame(
        {"PD": [np.nan, 0.23, 0.13, 0.11, 0.07, 0.04, 0.01], "target": [0, 1, 0, 0, 1, 1, 1]}
    )
    with pytest.raises(ValueError, match="y_score contains non-finite values"):
        gini_standard(df["target"], df["PD"])


def test_non_binary_target_raises():
    df = pd.DataFrame(
        {"PD": [0.23, 0.23, 0.13, 0.11, 0.07, 0.04, 0.01], "target": [2, 1, 0, 0, 1, 1, 1]}
    )
    with pytest.raises(ValueError, match="y_true must be binary"):
        gini_standard(df["target"], df["PD"])


def test_ks_perfect_ranking():
    df = pd.DataFrame(
        {"PD": [0.23, 0.17, 0.13, 0.11, 0.07, 0.04, 0.01], "target": [1, 1, 1, 1, 0, 0, 0]}
    )
    assert ks_statistic(df["target"], df["PD"]) == pytest.approx(1.0)


def test_ks_reversed_ranking():
    df = pd.DataFrame(
        {"PD": [0.23, 0.17, 0.13, 0.11, 0.07, 0.04, 0.01], "target": [0, 0, 0, 0, 1, 1, 1]}
    )
    assert ks_statistic(df["target"], df["PD"]) == pytest.approx(1.0)


def test_ks_is_invariant_to_tie_order():
    df_1 = pd.DataFrame({"PD": [0.17, 0.17, 0.04, 0.04, 0.04], "target": [1, 0, 1, 0, 1]})
    df_2 = pd.DataFrame({"PD": [0.17, 0.17, 0.04, 0.04, 0.04], "target": [0, 1, 0, 1, 0]})
    assert ks_statistic(df_1["target"], df_1["PD"]) == pytest.approx(
        ks_statistic(df_2["target"], df_2["PD"])
    )


def test_ks_constant_score():
    df = pd.DataFrame(
        {"PD": [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07], "target": [0, 0, 0, 0, 1, 1, 1]}
    )
    assert ks_statistic(df["target"], df["PD"]) == pytest.approx(0.0)
