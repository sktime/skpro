"""Tests for Empirical distributions."""

import pandas as pd
import pytest

from skpro.distributions.empirical import Empirical
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_empirical_iat_index():
    """Test that the index is correctly set after iat call."""
    spl_idx = pd.MultiIndex.from_product([[0, 1], [0, 1, 2]], names=["sample", "time"])
    spl = pd.DataFrame(
        [[0, 1], [2, 3], [10, 11], [6, 7], [8, 9], [4, 5]],
        index=spl_idx,
        columns=["a", "b"],
    )
    emp = Empirical(spl, columns=["a", "b"])

    emp_iat = emp.iat[0, 0]
    assert emp_iat.shape == ()

    assert not isinstance(emp_iat.spl.index, pd.MultiIndex)
    assert (emp_iat.spl.index == [0, 1]).all()
