# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the Histogram distribution."""

import numpy as np

from skpro.distributions.histogram import Histogram


def test_ppf_out_of_range_probabilities_return_nan():
    """Out-of-range probabilities return NaN for scalar and array histograms."""
    scalar = Histogram(bins=(0, 1, 2), bin_mass=[0.5, 0.5])
    array = Histogram(**Histogram.get_test_params()[0])

    for probability in (-0.1, 1.1):
        assert np.isnan(scalar.ppf(probability))
        ppf = array.ppf(np.full(array.shape, probability))
        assert np.isnan(ppf.to_numpy()).all()
