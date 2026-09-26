"""Tests for TruncatedPareto distribution."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import integrate
from scipy.stats import pareto

from skpro.distributions.truncated_pareto import TruncatedPareto
from skpro.tests.test_switch import run_test_module_changed

# (b, scale, lower, upper), spanning wide windows and the b == 1 / b == 2
# moment boundaries (where the naive power-law formula becomes logarithmic).
MOMENT_CASES = [
    (2.0, 1.0, 1.0, 10.0),
    (3.0, 2.0, 2.0, 20.0),
    (2.0, 1.0, 1.0, 100.0),  # wide window: base ppf-average is ~33% low here
    (0.5, 1.0, 1.0, 50.0),
    (1.0, 1.0, 1.0, 10.0),  # b == 1: mean uses the log form
    (2.0, 1.0, 1.0, 50.0),  # b == 2: second moment uses the log form
    (4.0, 1.0, 1.0, 1000.0),
]


def _quad_moments(b, scale, lower, upper):
    """Mean/var by adaptive pdf integration on the bounded support (oracle)."""
    norm = pareto.cdf(upper, b, scale=scale) - pareto.cdf(lower, b, scale=scale)
    pdf = lambda x: pareto.pdf(x, b, scale=scale) / norm  # noqa: E731
    m1, _ = integrate.quad(lambda x: x * pdf(x), lower, upper, limit=200)
    m2, _ = integrate.quad(lambda x: x * x * pdf(x), lower, upper, limit=200)
    return m1, m2 - m1 * m1


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
@pytest.mark.parametrize("b,scale,lower,upper", MOMENT_CASES)
def test_truncated_pareto_moments_exact(b, scale, lower, upper):
    """mean/var match the closed form / adaptive quadrature, no approx warning."""
    dist = TruncatedPareto(b=b, scale=scale, lower=lower, upper=upper)
    exp_mean, exp_var = _quad_moments(b, scale, lower, upper)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # exact tag must not fall back + warn
        mean = float(np.asarray(dist.mean()).ravel()[0])
        var = float(np.asarray(dist.var()).ravel()[0])

    assert_allclose(mean, exp_mean, rtol=1e-9)
    assert_allclose(var, exp_var, rtol=1e-9)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_truncated_pareto_moments_broadcast():
    """Closed-form moments broadcast and preserve shape/index/columns."""
    b = [[2.0, 3.0], [1.0, 2.0]]  # includes b == 1 and b == 2 boundaries
    dist = TruncatedPareto(b=b, scale=1.0, lower=1.0, upper=25.0)

    mean = dist.mean()
    var = dist.var()

    assert mean.shape == dist.shape
    assert var.shape == dist.shape
    assert mean.index.equals(dist.index)
    assert var.columns.equals(dist.columns)

    for i in range(2):
        for j in range(2):
            exp_mean, exp_var = _quad_moments(b[i][j], 1.0, 1.0, 25.0)
            assert_allclose(mean.iloc[i, j], exp_mean, rtol=1e-9)
            assert_allclose(var.iloc[i, j], exp_var, rtol=1e-9)
