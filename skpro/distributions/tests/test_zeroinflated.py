"""Tests for ZeroInflated distribution."""

import numpy as np
import pytest

from skpro.distributions import Poisson, Uniform, ZeroInflated
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_zeroinflated_ppf_p1():
    """Test that ppf(1.0) matches the base distribution's supremum / ppf(1.0)."""
    # Poisson(mu=2.0) has support up to +inf
    base_p = Poisson(mu=2.0)
    zi_p = ZeroInflated(p=0.7, distribution=base_p)
    assert np.isinf(zi_p.ppf(1.0))

    # Uniform(0, 5) has upper bound 5.0
    base_u = Uniform(lower=0.0, upper=5.0)
    zi_u = ZeroInflated(p=0.7, distribution=base_u)
    assert np.isclose(float(zi_u.ppf(1.0)), 5.0)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_zeroinflated_ppf_out_of_bounds():
    """Test that probabilities outside [0, 1] return NaN."""
    base = Poisson(mu=2.0)
    zi = ZeroInflated(p=0.7, distribution=base)

    assert np.isnan(float(zi.ppf(-0.1)))
    assert np.isnan(float(zi.ppf(1.1)))

    # Array distribution check
    mu = np.array([[2.0, 2.0], [2.0, 2.0]])
    base_arr = Poisson(mu=mu)
    zi_arr = ZeroInflated(p=0.7, distribution=base_arr)
    p_grid = np.array([[-0.5, 0.0], [1.0, 1.5]])
    res = zi_arr.ppf(p_grid)
    assert np.isnan(res.iloc[0, 0])
    assert res.iloc[0, 1] == 0.0
    assert np.isinf(res.iloc[1, 0])
    assert np.isnan(res.iloc[1, 1])


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_zeroinflated_ppf_zero_p_zi():
    """Test that p_zi == 0 returns 0.0 for valid p without division by zero."""
    base = Poisson(mu=2.0)
    zi = ZeroInflated(p=0.0, distribution=base)

    assert float(zi.ppf(0.5)) == 0.0
    assert float(zi.ppf(0.0)) == 0.0
    assert float(zi.ppf(1.0)) == 0.0
    assert np.isnan(float(zi.ppf(-0.1)))
    assert np.isnan(float(zi.ppf(1.1)))
