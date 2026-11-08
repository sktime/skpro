"""Tests for the Student's t-distribution."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from scipy.stats import t as student_t

from skpro.distributions.t import TDistribution
from skpro.tests.test_switch import run_test_module_changed


def _energy_by_quadrature(mu, sigma, df, x=None):
    """Energy by numerical integration in the standardised variable.

    Returns E|X-x| if x is passed, else E|X-Y| for i.i.d. X, Y.
    """
    kwargs = {"epsabs": 1e-13, "epsrel": 1e-13, "limit": 500}
    if x is None:
        # E|X-Y| = 2 sigma int F(1-F), F symmetric around 0
        integrand = lambda z: student_t.cdf(z, df) * student_t.sf(z, df)  # noqa: E731
        return 4 * sigma * quad(integrand, 0, np.inf, **kwargs)[0]
    # E|Z-a| = int_{-inf}^a F + int_a^inf (1-F)
    a = (x - mu) / sigma
    left = quad(lambda z: student_t.cdf(z, df), -np.inf, a, **kwargs)[0]
    right = quad(lambda z: student_t.sf(z, df), a, np.inf, **kwargs)[0]
    return sigma * (left + right)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
@pytest.mark.parametrize("df", [0.5, 1])
def test_t_energy_infinite_without_mean(df):
    """Energy is infinite when df <= 1, since E|X| does not exist."""
    d = TDistribution(mu=1, sigma=2, df=df)
    assert d.energy() == np.inf
    assert d.energy(0.0) == np.inf

    d2 = TDistribution(mu=[[0, 1]], sigma=1, df=[[df, 3]])
    assert np.isinf(d2.energy().values).all()
    assert np.isinf(d2.energy(0.0).values).all()


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
@pytest.mark.parametrize("df", [1 + 1e-12, 1 + 2**-52])
def test_t_energy_df_just_above_one(df):
    """Energy for df slightly above 1 is huge, as for the Cauchy limit, not finite."""
    # for df -> 1+: E|X-Y| ~ 4 sigma / (pi (df-1)), E|X-x| ~ 2 sigma / (pi (df-1))
    d = TDistribution(mu=1, sigma=2, df=df)
    np.testing.assert_allclose(d.energy(), 8 / (np.pi * (df - 1)), rtol=1e-6)
    np.testing.assert_allclose(d.energy(3.0), 4 / (np.pi * (df - 1)), rtol=1e-6)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
@pytest.mark.parametrize(
    "mu, sigma, df, x",
    [
        (100, 0.1, 10, 100.05),
        (50, 0.01, 5, 50.02),
        (9.25, 0.059, 29.7, 9.0),
    ],
)
def test_t_energy_vs_quadrature(mu, sigma, df, x):
    """Energy agrees with quadrature, also for small scale relative to location."""
    d = TDistribution(mu=mu, sigma=sigma, df=df)
    expected_self = _energy_by_quadrature(mu, sigma, df)
    expected_x = _energy_by_quadrature(mu, sigma, df, x)
    np.testing.assert_allclose(d.energy(), expected_self, rtol=1e-7)
    np.testing.assert_allclose(d.energy(x), expected_x, rtol=1e-7)

    # 2D case with a second column of different parameters sums the columns
    d2 = TDistribution(mu=[[mu, 0]], sigma=[[sigma, 1]], df=[[df, 3]])
    other_self = _energy_by_quadrature(0, 1, 3)
    other_x = _energy_by_quadrature(0, 1, 3, 0.5)
    np.testing.assert_allclose(
        d2.energy().values, [[expected_self + other_self]], rtol=1e-7
    )
    np.testing.assert_allclose(
        d2.energy([[x, 0.5]]).values, [[expected_x + other_x]], rtol=1e-7
    )


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_t_energy_large_df_approaches_normal():
    """Energy for huge df is finite and close to that of the normal distribution."""
    d = TDistribution(mu=100, sigma=0.1, df=1e12)
    np.testing.assert_allclose(d.energy(), 0.2 / np.sqrt(np.pi), rtol=1e-6)
    expected_x = 0.1 * (0.5 * (2 * norm.cdf(0.5) - 1) + 2 * norm.pdf(0.5))
    np.testing.assert_allclose(d.energy(100.05), expected_x, rtol=1e-6)
