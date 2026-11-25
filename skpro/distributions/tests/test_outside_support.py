# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for pdf, log_pdf and cdf at and outside of the support boundary."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from skpro.distributions import Histogram, LogNormal, Weibull

X_GRID = [-3.0, -1.0, -0.5, 0.0, 0.5, 2.0]


def _check_against_scipy(d, ref, x):
    np.testing.assert_allclose(d.pdf(x), ref.pdf(x), rtol=1e-10)
    np.testing.assert_allclose(d.log_pdf(x), ref.logpdf(x), rtol=1e-10)
    np.testing.assert_allclose(d.cdf(x), ref.cdf(x), rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize("x", X_GRID)
@pytest.mark.parametrize("mu, sigma", [(0.0, 1.0), (1.5, 0.3)])
def test_lognormal_matches_scipy(mu, sigma, x):
    """LogNormal pdf, log_pdf and cdf must be 0, -inf and 0 for x <= 0."""
    d = LogNormal(mu=mu, sigma=sigma)
    ref = stats.lognorm(s=sigma, scale=np.exp(mu))
    _check_against_scipy(d, ref, x)


@pytest.mark.parametrize("x", X_GRID)
@pytest.mark.parametrize("k", [0.5, 1.0, 1.5, 2.0])
def test_weibull_matches_scipy(k, x):
    """Weibull pdf, log_pdf and cdf must be 0, -inf and 0 for x < 0."""
    d = Weibull(scale=2.0, k=k)
    ref = stats.weibull_min(c=k, scale=2.0)
    _check_against_scipy(d, ref, x)


def test_outside_support_array():
    """Values outside the support are handled elementwise on 2D distributions."""
    x = np.array([[-1.0, 0.0], [1.0, -2.0]])
    for d in [
        LogNormal(mu=[[0.0, 1.0], [0.5, 2.0]], sigma=1.0),
        Weibull(scale=[[1.0, 2.0], [3.0, 4.0]], k=1.5),
    ]:
        outside = x < 0
        assert np.all(d.pdf(x).values[outside] == 0)
        assert np.all(d.log_pdf(x).values[outside] == -np.inf)
        assert np.all(d.cdf(x).values[outside] == 0)
        assert not np.isnan(d.pdf(x).values).any()


@pytest.mark.parametrize("x", [-1.0, 5.0])
def test_histogram_log_pdf_outside_bins(x):
    """Histogram log_pdf must be -inf where its pdf is 0."""
    d = Histogram(bins=[0, 1, 2, 4], bin_mass=[0.2, 0.5, 0.3])
    assert d.pdf(x) == 0
    assert d.log_pdf(x) == -np.inf


def test_histogram_log_pdf_outside_bins_array():
    """Histogram log_pdf is -inf outside the bins, also for array distributions."""
    d = Histogram(
        bins=[[[0, 1, 2, 4], [0, 2, 3]]],
        bin_mass=[[[0.2, 0.5, 0.3], [0.5, 0.5]]],
        index=pd.Index([0]),
        columns=pd.Index(["a", "b"]),
    )
    x = np.array([[5.0, 2.5]])
    log_pdf = d.log_pdf(x).values
    assert log_pdf[0, 0] == -np.inf
    assert np.isclose(log_pdf[0, 1], np.log(0.5))
