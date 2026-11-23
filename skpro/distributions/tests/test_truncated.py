"""Tests for TruncatedDistribution and LeftTruncated."""
import numpy as np

from skpro.distributions import LeftTruncated, Normal, Poisson, TruncatedDistribution


def test_truncated_normal_ppf_bounds():
    """Test that TruncatedDistribution evaluates quantiles correctly at boundaries."""
    # Right-unbounded: lower=0.0, upper=None
    t_unbounded = TruncatedDistribution(Normal(mu=0.0, sigma=1.0), lower=0.0)
    assert np.isposinf(float(t_unbounded.ppf(1.0)))
    assert np.isclose(float(t_unbounded.ppf(0.0)), 0.0)
    assert np.isnan(float(t_unbounded.ppf(-0.5)))
    assert np.isnan(float(t_unbounded.ppf(1.5)))

    # Bounded: lower=0.0, upper=1.0
    t_bounded = TruncatedDistribution(Normal(mu=0.0, sigma=1.0), lower=0.0, upper=1.0)
    assert np.isclose(float(t_bounded.ppf(0.0)), 0.0)
    assert np.isclose(float(t_bounded.ppf(1.0)), 1.0)
    assert np.isnan(float(t_bounded.ppf(-0.1)))
    assert np.isnan(float(t_bounded.ppf(1.1)))


def test_left_truncated_poisson_ppf():
    """Test LeftTruncated with discrete Poisson distribution."""
    lt = LeftTruncated(Poisson(mu=2.0), lower=0)
    assert np.isposinf(float(lt.ppf(1.0)))
    assert np.isnan(float(lt.ppf(-0.5)))
    assert np.isnan(float(lt.ppf(1.5)))
    assert float(lt.ppf(0.5)) > 0.0


def test_truncated_array_quantiles():
    """Test TruncatedDistribution with array-valued parameters."""
    mu = np.array([[0.0, 1.0], [2.0, 3.0]])
    sigma = np.ones((2, 2))
    base = Normal(mu=mu, sigma=sigma)
    t = TruncatedDistribution(base, lower=0.0, upper=5.0)

    q0 = t.ppf(0.0)
    q1 = t.ppf(1.0)
    assert np.allclose(q0.to_numpy(), 0.0)
    assert np.allclose(q1.to_numpy(), 5.0)
