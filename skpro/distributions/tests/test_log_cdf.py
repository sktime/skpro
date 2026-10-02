"""Tests for the ``log_cdf`` method on the base class and the scipy adapter."""

__author__ = ["Ashish-Kumar-Dash"]

import numpy as np
import pytest

from skpro.distributions import Gamma, Normal
from skpro.distributions.adapters.scipy import _ScipyAdapter
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_scipy_log_cdf_exact():
    """``_ScipyAdapter.log_cdf`` wraps scipy ``logcdf`` and is exact."""
    from scipy.stats import gamma

    assert issubclass(Gamma, _ScipyAdapter)
    d = Gamma(alpha=[[2.0]], beta=[[1.0]])
    for x in [0.01, 1.0, 5.0]:
        got = d.log_cdf(np.array([[x]])).values[0, 0]
        assert np.isclose(got, gamma.logcdf(x, 2.0, scale=1.0))


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_default_log_cdf_is_log_of_cdf():
    """Base-class default ``log_cdf`` equals ``log(cdf)`` for a non-scipy dist."""
    assert not issubclass(Normal, _ScipyAdapter)
    d = Normal(mu=[[0.0]], sigma=[[1.0]])
    x = np.array([[0.0]])
    got = d.log_cdf(x).values[0, 0]
    assert np.isclose(got, np.log(d.cdf(x).values[0, 0]))
