"""Test base class logic for scalar distributions.

Distributions behave polymorphically - if no index is passed,
all methods should accept and return scalars.

This is tested by using the Normal distribution as a test case,
which invokes the boilerplate for scalar distributions.
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from skpro.distributions.normal import Normal


def test_scalar_distribution():
    """Test scalar distribution logic."""
    # test params
    mu = 1
    sigma = 2

    # instantiate distribution
    d = Normal(mu=mu, sigma=sigma)
    assert d.ndim == 0
    assert d.shape == ()
    assert d.index is None
    assert d.columns is None

    # test scalar input
    x = 0.5
    assert np.isscalar(d.mean())
    assert np.isscalar(d.var())
    assert np.isscalar(d.energy())
    assert np.isscalar(d.energy(x))
    assert np.isscalar(d.pdf(x))
    assert np.isscalar(d.log_pdf(x))
    assert np.isscalar(d.cdf(x))
    assert np.isscalar(d.ppf(x))
    assert np.isscalar(d.sample())

    spl_mult = d.sample(5)
    assert isinstance(spl_mult, pd.DataFrame)
    assert spl_mult.shape == (5, 1)
    assert spl_mult.index.equals(pd.RangeIndex(5))


def test_broadcast_ambiguous():
    """Test broadcasting in cases of ambiguous parameter dimensions."""
    mu = [1]
    sigma = 2
    # this should result in 2D array distribution
    # anything that is not scalar is broadcast to 2D
    d = Normal(mu=mu, sigma=sigma)
    assert d.ndim == 2
    assert d.shape == (1, 1)
    assert d.index.equals(pd.RangeIndex(1))
    assert d.columns.equals(pd.RangeIndex(1))

    def is_expected_2d(output, col=None):
        assert isinstance(output, pd.DataFrame)
        assert output.ndim == 2
        assert output.shape == (1, 1)
        assert output.index.equals(pd.RangeIndex(1))
        if col is None:
            col = pd.RangeIndex(1)
        assert output.columns.equals(pd.Index(col))
        return True

    # test scalar input
    x = 0.5

    assert is_expected_2d(d.mean())
    assert is_expected_2d(d.var())
    assert is_expected_2d(d.energy(), ["energy"])
    assert is_expected_2d(d.energy(x), ["energy"])
    assert is_expected_2d(d.pdf(x))
    assert is_expected_2d(d.log_pdf(x))
    assert is_expected_2d(d.cdf(x))
    assert is_expected_2d(d.ppf(x))
    assert is_expected_2d(d.sample())

    spl_mult = d.sample(5)
    assert isinstance(spl_mult, pd.DataFrame)
    assert spl_mult.shape == (5, 1)
    assert isinstance(spl_mult.index, pd.MultiIndex)
    assert spl_mult.index.nlevels == 2
    assert spl_mult.columns.equals(pd.RangeIndex(1))
