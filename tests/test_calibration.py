import re

import numpy as np
import pytest

from valmetrics.calibration import (
    BinomialTestResult,
    binomial_test,
    hosmer_lemeshow,
)


def test_hosmer_lemeshow_returns_result():
    y = [0, 0, 1, 1, 0, 1, 0, 1]
    p = [0.1, 0.2, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6]
    result = hosmer_lemeshow(y, p, n_groups=4)
    assert result.statistic >= 0.0
    assert 0.0 <= result.p_value <= 1.0
    assert result.degrees_of_freedom == 2
    assert result.n_groups == 4


def test_hosmer_lemeshow_bad_calibration_has_larger_statistic():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    good_p = [0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95]
    bad_p = [0.95, 0.90, 0.85, 0.80, 0.20, 0.15, 0.10, 0.05]
    good = hosmer_lemeshow(y, good_p, n_groups=4)
    bad = hosmer_lemeshow(y, bad_p, n_groups=4)
    assert bad.statistic > good.statistic


def test_hosmer_lemeshow_rejects_probabilities_outside_unit_interval():
    y = [0, 1, 0, 1]
    p = [0.1, 0.2, 1.2, 0.4]
    with pytest.raises(ValueError, match=re.escape("y_prob must be in [0, 1]")):
        hosmer_lemeshow(y, p, n_groups=2)


def test_hosmer_lemeshow_rejects_non_finite_probabilities():
    y = [0, 1, 0, 1]
    p = [0.1, np.nan, 0.3, 0.4]
    with pytest.raises(ValueError, match="y_prob contains NaN values"):
        hosmer_lemeshow(y, p, n_groups=2)


def test_hosmer_lemeshow_rejects_too_few_groups():
    with pytest.raises(ValueError, match="n_groups must be at least 2"):
        hosmer_lemeshow([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], n_groups=1)


def test_hosmer_lemeshow_rejects_too_many_groups():
    with pytest.raises(ValueError, match="n_groups must not exceed number of observations"):
        hosmer_lemeshow([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], n_groups=5)


def test_binomial_test_returns_result_with_one_entry_per_group():
    y_true = [0, 0, 1, 0, 1, 1]
    y_prob = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    result = binomial_test(y_true, y_prob, n_groups=3)
    assert isinstance(result, BinomialTestResult)
    assert len(result.bins) == 3
    assert result.confidence_level == pytest.approx(0.95)
    assert result.alternative == "two-sided"
    assert result.method == "binomial_exact"


def test_binomial_test_computes_expected_fields_per_bin():
    y_true = [0, 1, 0, 1]
    y_prob = [0.10, 0.20, 0.30, 0.40]
    result = binomial_test(y_true, y_prob, n_groups=2)
    first_bin = result.bins[0]
    second_bin = result.bins[1]
    assert first_bin.group == 0
    assert first_bin.n_observations == 2
    assert first_bin.observed_defaults == 1
    assert first_bin.expected_defaults == pytest.approx(0.30)
    assert first_bin.average_pd == pytest.approx(0.15)
    assert first_bin.observed_dr == pytest.approx(0.50)
    assert second_bin.group == 1
    assert second_bin.n_observations == 2
    assert second_bin.observed_defaults == 1
    assert second_bin.expected_defaults == pytest.approx(0.70)
    assert second_bin.average_pd == pytest.approx(0.35)
    assert second_bin.observed_dr == pytest.approx(0.50)


def test_binomial_test_sorts_by_probability_before_splitting():
    y_true = [1, 0, 1, 0]
    y_prob = [0.40, 0.10, 0.30, 0.20]
    result = binomial_test(y_true, y_prob, n_groups=2)
    first_bin = result.bins[0]
    second_bin = result.bins[1]
    assert first_bin.expected_defaults == pytest.approx(0.10 + 0.20)
    assert first_bin.observed_defaults == 0
    assert second_bin.expected_defaults == pytest.approx(0.30 + 0.40)
    assert second_bin.observed_defaults == 2


def test_binomial_test_returns_valid_p_values_and_bounds():
    y_true = [0, 0, 1, 0, 1, 1, 0, 1]
    y_prob = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
    result = binomial_test(y_true, y_prob, n_groups=4)
    for bin_result in result.bins:
        assert 0.0 <= bin_result.p_value <= 1.0
        assert 0 <= bin_result.lower_default_bound <= bin_result.upper_default_bound
        assert bin_result.upper_default_bound <= bin_result.n_observations
        assert bin_result.lower_default_rate == pytest.approx(
            bin_result.lower_default_bound / bin_result.n_observations
        )
        assert bin_result.upper_default_rate == pytest.approx(
            bin_result.upper_default_bound / bin_result.n_observations
        )


def test_binomial_test_different_alternatives_can_produce_different_p_values():
    y_true = [1, 0, 1, 0]
    y_prob = [0.40, 0.10, 0.30, 0.20]
    greater_result = binomial_test(
        y_true,
        y_prob,
        n_groups=2,
        alternative="greater",
    )
    less_result = binomial_test(
        y_true,
        y_prob,
        n_groups=2,
        alternative="less",
    )
    assert greater_result.bins[0].p_value != pytest.approx(less_result.bins[0].p_value)


def test_binomial_test_rejects_invalid_alternative():
    with pytest.raises(ValueError, match="alternative must be"):
        binomial_test(
            [0, 1, 0, 1],
            [0.10, 0.20, 0.30, 0.40],
            n_groups=2,
            alternative="bad",
        )


def test_binomial_test_rejects_invalid_confidence_level():
    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        binomial_test(
            [0, 1, 0, 1],
            [0.10, 0.20, 0.30, 0.40],
            n_groups=2,
            confidence_level=1.0,
        )


def test_binomial_test_rejects_too_few_groups():
    with pytest.raises(ValueError, match="n_groups must be at least 2"):
        binomial_test(
            [0, 1, 0, 1],
            [0.10, 0.20, 0.30, 0.40],
            n_groups=1,
        )


def test_binomial_test_rejects_more_groups_than_observations():
    with pytest.raises(ValueError, match="n_groups must not exceed number of observations"):
        binomial_test(
            [0, 1],
            [0.10, 0.20],
            n_groups=3,
        )


def test_binomial_test_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="y_prob"):
        binomial_test(
            [0, 1, 0, 1],
            [0.10, 0.20, -0.30, 0.40],
            n_groups=2,
        )


def test_binomial_test_rejects_average_pd_equal_to_zero():
    with pytest.raises(ValueError, match="average PD must be strictly between 0 and 1"):
        binomial_test(
            [0, 0, 1, 1],
            [0.00, 0.00, 0.20, 0.30],
            n_groups=2,
        )


def test_binomial_test_rejects_average_pd_equal_to_one():
    with pytest.raises(ValueError, match="average PD must be strictly between 0 and 1"):
        binomial_test(
            [0, 0, 1, 1],
            [0.20, 0.30, 1.00, 1.00],
            n_groups=2,
        )


def test_binomial_test_handles_missing_values_when_dropna_is_true():
    y_true = [0, 1, 1, 1]
    y_prob = [0.10, np.nan, 0.30, 0.40]
    result = binomial_test(y_true, y_prob, n_groups=2, dropna=True)
    assert len(result.bins) == 2
    assert sum(bin_result.n_observations for bin_result in result.bins) == 3


def test_binomial_test_rejects_missing_values_by_default():
    y_true = [0, 1, 1, 1]
    y_prob = [0.10, np.nan, 0.30, 0.40]
    with pytest.raises(ValueError):
        binomial_test(y_true, y_prob, n_groups=2)
