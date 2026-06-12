import numpy as np
import pytest

from valmetrics.stability import psi_categorical, psi_continuous


def test_psi_identical_distributions_is_zero():
    expected = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    actual = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    assert psi_continuous(expected, actual, bins=3).value == pytest.approx(0.0)


def test_psi_shifted_distribution_is_positive():
    expected = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    actual = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    assert psi_continuous(expected, actual, bins=3).value > 0.0


def test_psi_handles_zero_count_bins_with_epsilon():
    expected = [0.01, 0.02, 0.03, 0.80, 0.90, 1.00]
    actual = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    result = psi_continuous(expected, actual, bins=3, epsilon=1e-6).value
    assert np.isfinite(result)
    assert result > 0.0


def test_psi_rejects_missing_by_default():
    expected = [0.01, 0.02, np.nan, 0.04]
    actual = [0.01, 0.02, 0.03, 0.04]
    with pytest.raises(ValueError, match="missing values"):
        psi_continuous(expected, actual, bins=2).value


def test_psi_can_drop_missing_values():
    expected = [0.01, 0.02, np.nan, 0.04]
    actual = [0.01, 0.02, 0.03, 0.04]
    result = psi_continuous(expected, actual, bins=2, missing="drop").value
    assert np.isfinite(result)


def test_psi_can_treat_missing_as_separate_bin():
    expected = [0.01, 0.02, np.nan, 0.04]
    actual = [0.01, np.nan, 0.03, 0.04]
    result = psi_continuous(expected, actual, bins=2, missing="separate").value
    assert np.isfinite(result)


def test_psi_rejects_infinite_values():
    expected = [0.01, 0.02, np.inf, 0.04]
    actual = [0.01, 0.02, 0.03, 0.04]
    with pytest.raises(ValueError, match="expected contains infinite values"):
        psi_continuous(expected, actual, bins=2).value


def test_psi_rejects_empty_expected():
    with pytest.raises(ValueError, match="expected must contain at least one observation"):
        psi_continuous([], [0.01, 0.02], bins=2).value


def test_psi_rejects_empty_actual():
    with pytest.raises(ValueError, match="actual must contain at least one observation"):
        psi_continuous([0.01, 0.02], [], bins=2).value


def test_psi_rejects_too_few_bins():
    with pytest.raises(ValueError, match="bins must be at least 2"):
        psi_continuous([0.01, 0.02], [0.01, 0.02], bins=1).value


def test_psi_rejects_invalid_epsilon():
    with pytest.raises(ValueError, match="epsilon must be between 0 and 1"):
        psi_continuous([0.01, 0.02], [0.01, 0.02], epsilon=0.0).value


def test_psi_rejects_invalid_missing_policy():
    with pytest.raises(ValueError, match="missing must be one of"):
        psi_continuous([0.01, 0.02], [0.01, 0.02], missing="bad_policy").value


def test_psi_rejects_constant_expected_values():
    expected = [0.05, 0.05, 0.05, 0.05]
    actual = [0.05, 0.05, 0.06, 0.07]
    with pytest.raises(ValueError, match="at least two distinct values"):
        psi_continuous(expected, actual, bins=3).value


def test_psi_categorical_identical_distributions_is_zero():
    expected = ["A", "A", "B", "C"]
    actual = ["A", "A", "B", "C"]
    assert psi_categorical(expected, actual).value == pytest.approx(0.0)


def test_psi_categorical_changed_distribution_is_positive():
    expected = ["A", "A", "A", "B"]
    actual = ["A", "B", "B", "B"]
    assert psi_categorical(expected, actual).value > 0.0


def test_psi_categorical_handles_category_only_in_actual():
    result = psi_categorical(
        ["A", "A", "B", "B"],
        ["A", "A", "B", "C"],
    ).value
    assert np.isfinite(result)
    assert result > 0.0


def test_psi_categorical_handles_category_only_in_expected():
    result = psi_categorical(
        ["A", "A", "B", "C"],
        ["A", "A", "B", "B"],
    ).value
    assert np.isfinite(result)
    assert result > 0.0


def test_psi_categorical_accepts_numeric_categories():
    result = psi_categorical(
        [1, 1, 2, 2],
        [1, 2, 2, 3],
    ).value
    assert np.isfinite(result)


def test_psi_categorical_rejects_missing_by_default():
    with pytest.raises(ValueError, match="contain missing values"):
        psi_categorical(
            ["A", None, "B"],
            ["A", "B", "B"],
        ).value


def test_psi_categorical_can_drop_missing():
    result = psi_categorical(
        ["A", None, "B"],
        ["A", "B", None],
        missing="drop",
    ).value
    assert result == pytest.approx(0.0)


def test_psi_categorical_treats_missing_as_separate_category():
    result = psi_categorical(
        ["A", None, None, "B"],
        ["A", None, "B", "B"],
        missing="separate",
    ).value
    assert np.isfinite(result)
    assert result > 0.0


def test_psi_categorical_rejects_empty_expected():
    with pytest.raises(ValueError, match="expected must contain at least one observation"):
        psi_categorical([], ["A"]).value


def test_psi_categorical_rejects_empty_actual():
    with pytest.raises(ValueError, match="actual must contain at least one observation"):
        psi_categorical(["A"], []).value


def test_psi_categorical_rejects_all_missing_after_drop():
    with pytest.raises(
        ValueError,
        match="expected must contain at least one non-missing observation",
    ):
        psi_categorical(
            [None, np.nan],
            ["A", "B"],
            missing="drop",
        ).value
