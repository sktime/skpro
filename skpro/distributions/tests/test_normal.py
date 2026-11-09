"""Tests for Normal distributions."""

import numpy as np
import pandas as pd
import pytest

from skpro.distributions.normal import Normal
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_energy_nan_propagation_normal():
    """Test that Normal.energy returns NaN when parameters are NaN.

    Regression test: np.sum on a pandas DataFrame (produced by cdf/pdf calls)
    used skipna=True by default, silently converting NaN energy to 0.0.
    """
    # NaN sigma → energy should be NaN
    n = Normal(mu=[[0.0, 1.0]], sigma=[[np.nan, 1.0]])

    x = pd.DataFrame([[0.5, 0.5]], index=n.index, columns=n.columns)
    energy_x = n.energy(x)

    assert np.isnan(
        energy_x.iloc[0, 0]
    ), "energy(x) should be NaN when Normal sigma is NaN"
