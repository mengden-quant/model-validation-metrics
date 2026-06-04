import numpy as np
import pytest

from valmetrics.utils import prepare_binary_inputs, prepare_binary_target


def test_prepare_binary_target_accepts_float_binary():
    result = prepare_binary_target([1.0, 0.0, 1.0])
    assert result.dtype == int
    assert np.array_equal(result, np.array([1, 0, 1]))


def test_prepare_binary_target_rejects_non_binary_float():
    with pytest.raises(ValueError, match="binary"):
        prepare_binary_target([1.0, 0.2, 0.0])


def test_prepare_binary_inputs_rejects_nan_when_dropna_false():
    with pytest.raises(ValueError, match="non-finite"):
        prepare_binary_inputs([0, 1, 1], [0.1, np.nan, 0.3], values_name="score")


def test_prepare_binary_inputs_drops_nan_when_dropna_true():
    y, x = prepare_binary_inputs(
        [0, 1, 1],
        [0.1, np.nan, 0.3],
        values_name="score",
        dropna=True,
    )
    assert np.array_equal(y, np.array([0, 1]))
    assert np.array_equal(x, np.array([0.1, 0.3]))


def test_prepare_binary_inputs_rejects_single_class_after_dropna():
    with pytest.raises(ValueError, match="after dropna"):
        prepare_binary_inputs(
            [0, 1, 1],
            [0.1, np.nan, np.nan],
            values_name="score",
            dropna=True,
        )
