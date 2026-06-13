import re

import numpy as np
import pytest

from valmetrics.calibration import (
    BinomialTestResult,
    binomial_test,
    grouped_binomial_test,
    grouped_hosmer_lemeshow,
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
    y = [0, 1, 0, 1, 0, 1]
    p = [0.1, 0.2, 1.2, 0.4, 0.5, 0.6]
    with pytest.raises(ValueError, match=re.escape("y_prob must be in [0, 1]")):
        hosmer_lemeshow(y, p, n_groups=3)


def test_hosmer_lemeshow_rejects_non_finite_probabilities():
    y = [0, 1, 0, 1, 0, 1]
    p = [0.1, np.nan, 0.3, 0.4, 0.5, 0.6]
    with pytest.raises(ValueError, match="y_prob contains NaN values"):
        hosmer_lemeshow(y, p, n_groups=3)


@pytest.mark.parametrize("n_groups", [1, 2])
def test_hosmer_lemeshow_rejects_too_few_groups(n_groups):
    with pytest.raises(ValueError, match="Hosmer-Lemeshow test requires at least 3"):
        hosmer_lemeshow([0, 1, 0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], n_groups=n_groups)


def test_hosmer_lemeshow_rejects_non_positive_number_of_groups():
    with pytest.raises(ValueError, match="n_groups must be positive"):
        hosmer_lemeshow(
            [0, 1, 0, 1],
            [0.1, 0.2, 0.3, 0.4],
            n_groups=0,
        )


def test_hosmer_lemeshow_rejects_too_many_groups():
    with pytest.raises(ValueError, match="n_groups must not exceed number of observations"):
        hosmer_lemeshow([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], n_groups=5)


def test_grouped_hosmer_lemeshow_accepts_string_groups():
    result = grouped_hosmer_lemeshow(
        y_true=[0, 1, 0, 1, 0, 1],
        y_prob=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        groups=["A", "A", "B", "B", "C", "C"],
    )
    assert result.n_groups == 3
    assert result.degrees_of_freedom == 1
    assert result.statistic >= 0.0
    assert 0.0 <= result.p_value <= 1.0


def test_grouped_hosmer_lemeshow_rejects_fewer_than_three_groups():
    with pytest.raises(
        ValueError,
        match="Hosmer-Lemeshow test requires at least 3 groups",
    ):
        grouped_hosmer_lemeshow(
            y_true=[0, 1, 0, 1],
            y_prob=[0.1, 0.2, 0.3, 0.4],
            groups=["A", "A", "B", "B"],
        )


def test_grouped_hosmer_lemeshow_dropna_preserves_alignment():
    result = grouped_hosmer_lemeshow(
        y_true=[0, np.nan, 1, 0, 1, 0, 1],
        y_prob=[0.1, 0.2, 0.3, np.nan, 0.5, 0.6, 0.7],
        groups=["A", "A", "B", "B", "C", "C", "D"],
        dropna=True,
    )
    assert result.n_groups == 4
    assert result.degrees_of_freedom == 2


def test_grouped_hosmer_lemeshow_matches_automatic_groups():
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    groups = ["A", "A", "B", "B", "C", "C", "D", "D"]
    automatic = hosmer_lemeshow(y, p, n_groups=4)
    grouped = grouped_hosmer_lemeshow(y, p, groups)
    assert grouped.statistic == pytest.approx(automatic.statistic)
    assert grouped.p_value == pytest.approx(automatic.p_value)
    assert grouped.degrees_of_freedom == automatic.degrees_of_freedom
    assert grouped.n_groups == automatic.n_groups


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


def test_binomial_test_does_not_split_equal_probabilities_between_groups():
    y_true = [0, 1, 0, 1, 0, 1]
    y_prob = [0.10, 0.10, 0.10, 0.80, 0.80, 0.80]
    result = binomial_test(y_true, y_prob, n_groups=4)
    assert len(result.bins) == 2
    first_group = result.bins[0]
    second_group = result.bins[1]
    assert first_group.n_observations == 3
    assert first_group.expected_defaults == pytest.approx(0.30)
    assert first_group.average_pd == pytest.approx(0.10)
    assert second_group.n_observations == 3
    assert second_group.expected_defaults == pytest.approx(2.40)
    assert second_group.average_pd == pytest.approx(0.80)


def test_binomial_test_is_invariant_to_order_within_probability_ties():
    y_true_1 = [0, 1, 0, 1, 0, 1]
    y_prob_1 = [0.10, 0.10, 0.10, 0.80, 0.80, 0.80]
    y_true_2 = [1, 0, 0, 0, 1, 1]
    y_prob_2 = [0.10, 0.10, 0.10, 0.80, 0.80, 0.80]
    result_1 = binomial_test(y_true_1, y_prob_1, n_groups=4)
    result_2 = binomial_test(y_true_2, y_prob_2, n_groups=4)
    assert len(result_1.bins) == len(result_2.bins)
    for bin_1, bin_2 in zip(result_1.bins, result_2.bins, strict=True):
        assert bin_1.n_observations == bin_2.n_observations
        assert bin_1.observed_defaults == bin_2.observed_defaults
        assert bin_1.expected_defaults == pytest.approx(bin_2.expected_defaults)
        assert bin_1.average_pd == pytest.approx(bin_2.average_pd)
        assert bin_1.p_value == pytest.approx(bin_2.p_value)


def test_grouped_binomial_returns_one_result_per_group():
    y_true = [0, 1, 0, 1, 1, 0]
    y_prob = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    groups = ["A", "A", "B", "B", "C", "C"]
    result = grouped_binomial_test(y_true, y_prob, groups)
    assert len(result.bins) == 3
    assert tuple(bin_result.group for bin_result in result.bins) == ("A", "B", "C")


def test_grouped_binomial_computes_group_level_values():
    y_true = [0, 1, 0, 1]
    y_prob = [0.10, 0.20, 0.30, 0.40]
    groups = ["A", "A", "B", "B"]
    result = grouped_binomial_test(y_true, y_prob, groups)
    first = result.bins[0]
    second = result.bins[1]
    assert first.group == "A"
    assert first.n_observations == 2
    assert first.observed_defaults == 1
    assert first.expected_defaults == pytest.approx(0.30)
    assert first.average_pd == pytest.approx(0.15)
    assert second.group == "B"
    assert second.n_observations == 2
    assert second.observed_defaults == 1
    assert second.expected_defaults == pytest.approx(0.70)
    assert second.average_pd == pytest.approx(0.35)


def test_grouped_binomial_accepts_numeric_groups():
    result = grouped_binomial_test(
        [0, 1, 0, 1],
        [0.10, 0.20, 0.30, 0.40],
        [1, 1, 2, 2],
    )
    assert tuple(bin_result.group for bin_result in result.bins) == (1, 2)


def test_grouped_binomial_preserves_group_order():
    result = grouped_binomial_test(
        [0, 1, 0, 1],
        [0.10, 0.20, 0.30, 0.40],
        ["BBB", "BBB", "AAA", "AAA"],
    )
    assert tuple(bin_result.group for bin_result in result.bins) == ("BBB", "AAA")


def test_grouped_binomial_dropna_filters_groups_consistently():
    result = grouped_binomial_test(
        [0, np.nan, 1, 0],
        [0.10, 0.20, 0.30, np.nan],
        ["A", "B", "C", "D"],
        dropna=True,
    )
    assert tuple(bin_result.group for bin_result in result.bins) == ("A", "C")
    assert sum(bin_result.n_observations for bin_result in result.bins) == 2


def test_grouped_binomial_rejects_group_length_mismatch():
    with pytest.raises(ValueError, match="y_true and groups must have the same length"):
        grouped_binomial_test(
            [0, 1, 0],
            [0.10, 0.20, 0.30],
            ["A", "B"],
        )


def test_grouped_binomial_rejects_missing_group_labels():
    with pytest.raises(ValueError, match="groups contains missing values"):
        grouped_binomial_test(
            [0, 1, 0],
            [0.10, 0.20, 0.30],
            ["A", None, "B"],
        )
