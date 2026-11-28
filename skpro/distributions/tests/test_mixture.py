# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the Mixture distribution."""

__author__ = ["VividhDesign"]

import numpy as np
import pytest
from scipy.stats import norm

from skpro.distributions import Mixture, Normal
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(Mixture),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_mixture_scalar_respects_weights():
    """Scalar mixture moments, pdf and cdf must use the mixture weights."""
    w = [0.9, 0.1]
    m = Mixture([Normal(mu=0, sigma=1), Normal(mu=10, sigma=1)], weights=w)

    assert m.ndim == 0
    # mean = sum w_i mu_i; var = sum w_i (sigma_i^2 + mu_i^2) - mean^2
    np.testing.assert_allclose(m.mean(), 1.0)
    np.testing.assert_allclose(m.var(), 0.9 * 1 + 0.1 * 101 - 1.0)

    x = 5.0
    np.testing.assert_allclose(m.cdf(x), w[0] * norm.cdf(x) + w[1] * norm.cdf(x - 10))
    np.testing.assert_allclose(m.pdf(x), w[0] * norm.pdf(x) + w[1] * norm.pdf(x - 10))


@pytest.mark.skipif(
    not run_test_for_class(Mixture),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_mixture_iat_matches_array_entry():
    """A scalar entry from iat must agree with the corresponding array entry."""
    n1 = Normal(mu=[[0.0, 1.0], [2.0, 3.0]], sigma=1)
    n2 = Normal(mu=[[10.0, 11.0], [12.0, 13.0]], sigma=2)
    m = Mixture([n1, n2], weights=[0.7, 0.3])

    x = np.full((2, 2), 4.0)
    s = m.iat[1, 0]
    np.testing.assert_allclose(s.mean(), m.mean().iloc[1, 0])
    np.testing.assert_allclose(s.var(), m.var().iloc[1, 0])
    np.testing.assert_allclose(s.cdf(4.0), m.cdf(x).iloc[1, 0])
    np.testing.assert_allclose(s.pdf(4.0), m.pdf(x).iloc[1, 0])
