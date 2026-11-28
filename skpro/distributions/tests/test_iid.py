# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the IID composite distribution."""

__author__ = ["VividhDesign"]

import numpy as np
import pandas as pd
import pytest

from skpro.distributions import IID, Normal
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(IID),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("scalar_dist", [True, False])
def test_iid_energy_matches_array_distribution(scalar_dist):
    """IID energy must equal the energy of the equivalent array distribution.

    Energy of a row is the sum over the entries of the row, as for ``Normal``.
    """
    index = pd.Index([1, 2, 5])
    columns = pd.Index(["a", "b"])

    if scalar_dist:
        d = IID(Normal(mu=0, sigma=1), index=index, columns=columns)
        mu = np.zeros((3, 2))
    else:
        d = IID(Normal(mu=[[0, 1]], sigma=1), index=index, columns=columns)
        mu = np.array([[0, 1]] * 3)

    d_arr = Normal(mu=mu, sigma=1, index=index, columns=columns)

    np.testing.assert_allclose(d.energy().values, d_arr.energy().values)

    x = np.arange(6, dtype=float).reshape(3, 2) / 2
    np.testing.assert_allclose(d.energy(x).values, d_arr.energy(x).values)
