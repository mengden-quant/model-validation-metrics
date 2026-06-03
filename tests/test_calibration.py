import re

import numpy as np
import pytest

from valmetrics.calibration import (
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
    with pytest.raises(ValueError, match="y_prob contains non-finite values"):
        hosmer_lemeshow(y, p, n_groups=2)


def test_hosmer_lemeshow_rejects_too_few_groups():
    with pytest.raises(ValueError, match="n_groups must be at least 2"):
        hosmer_lemeshow([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], n_groups=1)


def test_hosmer_lemeshow_rejects_too_many_groups():
    with pytest.raises(ValueError, match="n_groups must not exceed number of observations"):
        hosmer_lemeshow([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], n_groups=5)
