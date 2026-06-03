import numpy as np
import pytest

from valmetrics.diagnostics import hci, herfindahl_hirschman


def test_hhi_equal_groups():
    groups = ["A", "A", "B", "B"]
    assert herfindahl_hirschman(groups) == pytest.approx(0.5)


def test_hhi_single_group():
    groups = ["A", "A", "A"]
    assert herfindahl_hirschman(groups) == pytest.approx(1.0)


def test_normalized_hhi_equal_groups():
    groups = ["A", "A", "B", "B"]
    assert herfindahl_hirschman(groups, normalized=True) == pytest.approx(0.0)


def test_normalized_hhi_single_group():
    groups = ["A", "A", "A"]
    assert herfindahl_hirschman(groups, normalized=True) == pytest.approx(1.0)


def test_hhi_rejects_empty_groups():
    with pytest.raises(ValueError, match="groups must contain at least one observation"):
        herfindahl_hirschman([])


def test_hhi_rejects_2d_groups():
    with pytest.raises(ValueError, match="groups must be 1D"):
        herfindahl_hirschman([["A", "B"], ["A", "B"]])


def test_hci_returns_largest_group_share():
    result = hci(["A", "A", "A", "B"])
    assert result.groups == ("A",)
    assert result.value == pytest.approx(0.75)


def test_hci_single_group():
    result = hci(["A", "A", "A"])
    assert result.groups == ("A",)
    assert result.value == pytest.approx(1.0)


def test_hci_rejects_empty_groups():
    with pytest.raises(ValueError, match="groups must contain at least one observation"):
        hci([])


def test_hci_dropna_numeric_groups():
    result = hci([1, 1, 2, np.nan], dropna=True)
    assert result.groups == (1,)
    assert result.value == pytest.approx(2 / 3)


def test_hci_returns_all_groups_with_maximum_share():
    result = hci(["A", "A", "B", "B", "C"])
    assert result.groups == ("A", "B")
    assert result.value == pytest.approx(0.4)
