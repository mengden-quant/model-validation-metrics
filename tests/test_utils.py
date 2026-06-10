import numpy as np
import pytest

from valmetrics.utils import (
    as_1d_array,
    as_1d_binary_array,
    as_1d_float_array,
    check_contains_both_binary_classes,
    check_same_length,
    prepare_binary_inputs,
    prepare_binary_inputs_and_groups,
    validate_probabilities,
)


def test_as_1d_array_accepts_string_groups():
    result = as_1d_array(["AAA", "AA", "A"], name="groups")
    assert result.shape == (3,)
    assert result.tolist() == ["AAA", "AA", "A"]


def test_as_1d_array_rejects_two_dimensional_input():
    with pytest.raises(ValueError, match="groups must be 1D"):
        as_1d_array([["AAA", "AA"], ["A", "BBB"]], name="groups")


def test_as_1d_float_array_allows_nan():
    result = as_1d_float_array([1.0, np.nan, 3.0], name="values")
    assert np.isnan(result[1])


def test_as_1d_float_array_rejects_non_numeric_values():
    with pytest.raises(ValueError, match="values must be numeric"):
        as_1d_float_array([1.0, "bad", 3.0], name="values")


@pytest.mark.parametrize("infinite_value", [np.inf, -np.inf])
def test_as_1d_float_array_rejects_infinite_values(infinite_value):
    with pytest.raises(ValueError, match="values contains infinite values"):
        as_1d_float_array([1.0, infinite_value], name="values")


def test_as_1d_binary_array_accepts_binary_values_and_nan():
    result = as_1d_binary_array([0, 1, np.nan, 0])
    assert np.array_equal(result[:2], np.array([0.0, 1.0]))
    assert np.isnan(result[2])


def test_as_1d_binary_array_rejects_non_binary_values():
    with pytest.raises(ValueError, match="y_true must be binary"):
        as_1d_binary_array([0, 1, 2])


@pytest.mark.parametrize("y_true", [np.array([0, 0]), np.array([1, 1])])
def test_check_contains_both_binary_classes_rejects_single_class(y_true):
    with pytest.raises(ValueError, match="must contain at least one 0 and one 1"):
        check_contains_both_binary_classes(y_true)


def test_check_same_length_accepts_equal_lengths():
    check_same_length(
        np.array([1, 2]),
        np.array([3, 4]),
        left_name="left",
        right_name="right",
    )


def test_check_same_length_rejects_different_lengths():
    with pytest.raises(ValueError, match="left and right must have the same length"):
        check_same_length(
            np.array([1, 2]),
            np.array([3]),
            left_name="left",
            right_name="right",
        )


def test_validate_probabilities_accepts_unit_interval_boundaries():
    validate_probabilities(np.array([0.0, 0.2, 1.0]))


@pytest.mark.parametrize(
    "values, expected_message",
    [
        (np.array([-0.1, 0.5]), r"y_prob must be in \[0, 1\]"),
        (np.array([0.5, 1.1]), r"y_prob must be in \[0, 1\]"),
        (np.array([0.5, np.nan]), "y_prob contains non-finite values"),
        (np.array([0.5, np.inf]), "y_prob contains non-finite values"),
    ],
)
def test_validate_probabilities_rejects_invalid_values(values, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        validate_probabilities(values)


def test_prepare_binary_inputs_drops_nan_pairwise():
    y, x = prepare_binary_inputs(
        [0, np.nan, 1, 0],
        [0.1, 0.2, 0.3, np.nan],
        values_name="y_score",
        dropna=True,
    )
    assert np.array_equal(y, np.array([0, 1]))
    assert np.array_equal(x, np.array([0.1, 0.3]))


def test_prepare_binary_inputs_rejects_empty_result_after_dropna():
    with pytest.raises(ValueError, match="No valid observations"):
        prepare_binary_inputs(
            [np.nan, np.nan],
            [np.nan, np.nan],
            values_name="y_score",
            dropna=True,
        )


def test_prepare_binary_inputs_and_groups_accepts_string_groups():
    y, x, groups = prepare_binary_inputs_and_groups(
        [0, 1, 0],
        [0.1, 0.2, 0.3],
        ["AAA", "AA", "A"],
        values_name="y_prob",
    )
    assert np.array_equal(y, np.array([0, 1, 0]))
    assert np.array_equal(x, np.array([0.1, 0.2, 0.3]))
    assert groups.tolist() == ["AAA", "AA", "A"]


def test_prepare_binary_inputs_and_groups_accepts_numeric_groups():
    _, _, groups = prepare_binary_inputs_and_groups(
        [0, 1, 0],
        [0.1, 0.2, 0.3],
        [1, 2, 3],
        values_name="y_prob",
    )
    assert groups.tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "groups",
    [
        ["AAA", None, "A"],
        ["AAA", np.nan, "A"],
    ],
)
def test_prepare_binary_inputs_and_groups_rejects_missing_groups(groups):
    with pytest.raises(ValueError, match="groups contains missing values"):
        prepare_binary_inputs_and_groups(
            [0, 1, 0],
            [0.1, 0.2, 0.3],
            groups,
            values_name="y_prob",
        )


def test_prepare_binary_inputs_and_groups_rejects_empty_result_after_dropna():
    with pytest.raises(ValueError, match="No valid observations"):
        prepare_binary_inputs_and_groups(
            [np.nan, np.nan],
            [np.nan, np.nan],
            ["AAA", "AA"],
            values_name="y_prob",
            dropna=True,
        )
