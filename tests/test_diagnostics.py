import pytest

from valmetrics.diagnostics import herfindahl_hirschman


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
