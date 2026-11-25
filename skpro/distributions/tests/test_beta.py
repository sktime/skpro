# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the Beta distribution energy."""

import numpy as np
import pytest
from scipy.stats import beta

from skpro.distributions import Beta


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 0.3, 0.8, 1.0, 1.5, 6.0])
@pytest.mark.parametrize("a, b", [(1.0, 3.0), (2.0, 0.5), (0.5, 0.5)])
def test_energy_x_matches_scipy(a, b, x):
    """E|X - x| must match scipy, including for x outside of [0, 1]."""
    d = Beta(alpha=a, beta=b)
    expected = beta(a, b).expect(lambda t: abs(t - x))
    assert np.isclose(d.energy(x), expected, rtol=1e-6)


def test_energy_x_above_support_array():
    """For x >= 1, E|X - x| = x - E[X], on a 2D distribution."""
    d = Beta(alpha=[[1.0], [2.0]], beta=[[3.0], [0.5]])
    x = np.array([[1.5], [6.0]])
    expected = x - np.array([[0.25], [0.8]])
    np.testing.assert_allclose(d.energy(x).values, expected)
