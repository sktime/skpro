# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the MeanScale distribution.

The generic distribution suite mostly uses a component with mean zero, where
the scale factor does not affect the mean, so these tests pin the moments of
``mu + sigma * X`` for a component ``X`` with non-zero mean.
"""

__author__ = ["VividhDesign"]

import numpy as np

from skpro.distributions import MeanScale, Normal


def test_meanscale_moments():
    """Mean is mu + sigma * E[X] and variance is sigma^2 * Var[X]."""
    mu_x = np.array([[0, 1], [2, 3], [4, 5]])
    n = Normal(mu=mu_x, sigma=2)
    d = MeanScale(d=n, mu=2, sigma=3)

    np.testing.assert_allclose(d.mean().values, 2 + 3 * mu_x)
    np.testing.assert_allclose(d.var().values, 3**2 * 2**2)


def test_meanscale_mean_matches_ppf_median():
    """For a symmetric component, the mean must equal the median from ppf."""
    n = Normal(mu=[[1.0, -2.0]], sigma=0.5)
    d = MeanScale(d=n, mu=[[0.5, 4.0]], sigma=[[2.0, 3.0]])

    np.testing.assert_allclose(d.mean().values, d.ppf(0.5).values)
