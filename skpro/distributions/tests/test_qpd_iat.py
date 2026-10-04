# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for scalar subsetting via iat of quantile parametrized distributions."""

__author__ = ["VividhDesign"]

import numpy as np
import pandas as pd
import pytest

from skpro.distributions import HistogramQPD, QPD_Empirical
from skpro.tests.test_switch import run_test_for_class


def _quantiles():
    idx = pd.MultiIndex.from_product(
        [[0.1, 0.5, 0.9], [0, 1, 2]], names=["quantile", "sample"]
    )
    return pd.DataFrame(
        [[0, 1], [2, 3], [4, 5], [1, 2], [4, 5], [7, 8], [2, 3], [6, 7], [10, 11]],
        index=idx,
        columns=["a", "b"],
    )


def _check_iat_matches_array(d):
    """Check that each iat entry agrees with the array distribution."""
    for row in range(d.shape[0]):
        for col in range(d.shape[1]):
            s = d.iat[row, col]
            assert s.ndim == 0

            np.testing.assert_allclose(s.mean(), d.mean().iloc[row, col])
            np.testing.assert_allclose(s.var(), d.var().iloc[row, col])

            x = d.ppf(0.4).iloc[row, col]
            x_arr = np.full(d.shape, x)
            np.testing.assert_allclose(s.cdf(x), d.cdf(x_arr).iloc[row, col])
            np.testing.assert_allclose(s.ppf(0.7), d.ppf(0.7).iloc[row, col])


@pytest.mark.skipif(
    not run_test_for_class(QPD_Empirical),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_qpd_empirical_iat():
    """Test that iat of QPD_Empirical returns the scalar entry."""
    d = QPD_Empirical(_quantiles())
    assert isinstance(d.iat[1, 1], QPD_Empirical)
    _check_iat_matches_array(d)


@pytest.mark.skipif(
    not run_test_for_class(HistogramQPD),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("tails", ["uniform", "none"])
def test_histogram_qpd_iat(tails):
    """Test that iat of HistogramQPD returns the scalar entry."""
    d = HistogramQPD(_quantiles(), tails=tails)
    _check_iat_matches_array(d)
