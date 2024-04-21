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
    # test data
    mu = 1
    sigma = 2

    # instantiate distribution
    distr = Normal(mu=mu, sigma=sigma)
    assert distr.ndim == 0
    assert distr.shape == ()

    # test scalar input
    x = 0.5
    assert np.isscalar(distr.mean(x))
    assert np.isscalar(distr.var(x))
    assert np.isscalar(distr.energy(x))
    assert np.isscalar(distr.pdf(x))
    assert np.isscalar(distr.log_pdf(x))
    assert np.isscalar(distr.cdf(x))
    assert np.isscalar(distr.ppf(x))


def test_broad