# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for IID broadcasting a single-column distribution across columns."""

__author__ = ["Chetansahney"]

import numpy as np
import pandas as pd
import pytest

from skpro.distributions import IID, Normal
from skpro.tests.test_switch import run_test_for_class

INDEX = pd.Index([1, 2, 5])
COLUMNS = pd.Index(["a", "b"])
X = pd.DataFrame([[0.5, -1.0], [2.0, 0.0], [1.0, 4.0]], index=INDEX, columns=COLUMNS)
P = pd.DataFrame([[0.1, 0.5], [0.3, 0.7], [0.9, 0.2]], index=INDEX, columns=COLUMNS)


@pytest.mark.skipif(
    not run_test_for_class(IID),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_iid_of_single_entry_distribution():
    """IID of a 1 x 1 distribution broadcasts its entry to all entries."""
    iid = IID(Normal(mu=[[1.0]], sigma=2.0), index=INDEX, columns=COLUMNS)
    expected = Normal(mu=1.0, sigma=2.0, index=INDEX, columns=COLUMNS)

    for method in ["pdf", "log_pdf", "cdf"]:
        pd.testing.assert_frame_equal(
            getattr(iid, method)(X), getattr(expected, method)(X)
        )
    pd.testing.assert_frame_equal(iid.ppf(P), expected.ppf(P))


@pytest.mark.skipif(
    not run_test_for_class(IID),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_iid_of_single_entry_distribution_sample():
    """Sampling broadcasts a 1 x 1 distribution without ``_ppf`` to all entries."""
    from skpro.distributions import Mixture

    mixture = Mixture([Normal(mu=[[0.0]], sigma=1.0), Normal(mu=[[10.0]], sigma=1.0)])
    iid = IID(mixture, index=INDEX, columns=COLUMNS)

    sample = iid.sample()
    assert sample.shape == (3, 2)
    assert sample.index.equals(INDEX)
    assert sample.columns.equals(COLUMNS)

    samples = iid.sample(5)
    assert samples.shape == (15, 2)
    assert samples.index.equals(pd.MultiIndex.from_product([range(5), INDEX]))
    assert np.isfinite(samples.values).all()


@pytest.mark.skipif(
    not run_test_for_class(IID),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("col", ["a", "b"])
def test_iid_of_row_distribution_column_subset(col):
    """A column of an IID of a row distribution can be selected and evaluated."""
    row = Normal(mu=[1.0, 2.0], sigma=[1.0, 3.0], columns=pd.Index(["c", "d"]))
    iid = IID(row, index=INDEX, columns=COLUMNS)
    expected = Normal(
        mu=[[1.0, 2.0]] * 3, sigma=[[1.0, 3.0]] * 3, index=INDEX, columns=COLUMNS
    )

    x, p = X[[col]], P[[col]]
    j = COLUMNS.get_loc(col)
    for subset, subset_expected in [
        (iid.loc[:, [col]], expected.loc[:, [col]]),
        (iid.iloc[:, [j]], expected.iloc[:, [j]]),
    ]:
        for method in ["pdf", "log_pdf", "cdf", "energy"]:
            pd.testing.assert_frame_equal(
                getattr(subset, method)(x), getattr(subset_expected, method)(x)
            )
        pd.testing.assert_frame_equal(subset.ppf(p), subset_expected.ppf(p))

    # evaluating at a frame with only some of the columns selects them
    pd.testing.assert_frame_equal(iid.cdf(x), expected.cdf(x))


@pytest.mark.skipif(
    not run_test_for_class(IID),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_iid_of_row_distribution_raw_values_metrics():
    """Per-column metrics of an IID of a row distribution can be computed."""
    from skpro.metrics import CRPS, LogLoss

    row = Normal(mu=[1.0, 2.0], sigma=[1.0, 3.0], columns=pd.Index(["c", "d"]))
    iid = IID(row, index=INDEX, columns=COLUMNS)
    expected = Normal(
        mu=[[1.0, 2.0]] * 3, sigma=[[1.0, 3.0]] * 3, index=INDEX, columns=COLUMNS
    )

    for metric in [CRPS(multioutput="raw_values"), LogLoss(multioutput="raw_values")]:
        np.testing.assert_allclose(
            np.asarray(metric(X, iid)), np.asarray(metric(X, expected))
        )
