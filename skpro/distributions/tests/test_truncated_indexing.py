"""Tests for indexing on TruncatedDistribution and LeftTruncated."""

import numpy as np
import pandas as pd
import pytest

from skpro.distributions import LeftTruncated, Normal, Poisson, TruncatedDistribution


@pytest.fixture
def index_and_columns():
    idx = pd.Index(["r0", "r1"])
    cols = pd.Index(["c0", "c1"])
    return idx, cols


def test_left_truncated_iat_and_scalar_iloc(index_and_columns):
    """Test LeftTruncated iat and scalar iloc behavior."""
    idx, cols = index_and_columns
    lt_2d = LeftTruncated(
        Poisson(mu=[[1.0, 2.0], [3.0, 4.0]], index=idx, columns=cols), lower=0
    )

    # 1. Scalar indexing via iat and iloc
    for r in range(2):
        for c in range(2):
            cell_iat = lt_2d.iat[r, c]
            cell_iloc = lt_2d.iloc[r, c]

            assert isinstance(cell_iat, LeftTruncated)
            assert isinstance(cell_iloc, LeftTruncated)
            assert cell_iat.ndim == 0
            assert cell_iloc.ndim == 0

            # PMF values must match full DataFrame PMF
            expected_pmf = lt_2d.pmf(2).iloc[r, c]
            np.testing.assert_allclose(cell_iat.pmf(2), expected_pmf)
            np.testing.assert_allclose(cell_iloc.pmf(2), expected_pmf)

    # 2. Row and column slice indexing
    row_sub = lt_2d.iloc[0]
    assert isinstance(row_sub, LeftTruncated)
    assert row_sub.shape == (1, 2)
    assert list(row_sub.index) == ["r0"]
    assert list(row_sub.columns) == ["c0", "c1"]

    col_sub = lt_2d.iloc[:, 1]
    assert isinstance(col_sub, LeftTruncated)
    assert col_sub.shape == (2, 1)
    assert list(col_sub.index) == ["r0", "r1"]
    assert list(col_sub.columns) == ["c1"]

    # 3. Scalar LeftTruncated
    lt_scalar = LeftTruncated(Poisson(mu=2.0), lower=0)
    assert lt_scalar.iat[0, 0].ndim == 0
    assert lt_scalar.iloc[0, 0].ndim == 0
    with pytest.raises(IndexError):
        lt_scalar.iat[1, 0]


def test_truncated_distribution_iat_and_scalar_iloc(index_and_columns):
    """Test TruncatedDistribution iat and scalar iloc behavior."""
    idx, cols = index_and_columns
    tr_2d = TruncatedDistribution(
        Normal([[0.0, 1.0], [2.0, 3.0]], 1.0, index=idx, columns=cols),
        lower=-1.0,
        upper=1.0,
    )

    # 1. Scalar indexing via iat and iloc
    for r in range(2):
        for c in range(2):
            cell_iat = tr_2d.iat[r, c]
            cell_iloc = tr_2d.iloc[r, c]

            assert isinstance(cell_iat, TruncatedDistribution)
            assert isinstance(cell_iloc, TruncatedDistribution)
            assert cell_iat.ndim == 0
            assert cell_iloc.ndim == 0

            # CDF values must match full DataFrame CDF
            expected_cdf = tr_2d.cdf(0.5).iloc[r, c]
            np.testing.assert_allclose(cell_iat.cdf(0.5), expected_cdf)
            np.testing.assert_allclose(cell_iloc.cdf(0.5), expected_cdf)

    # 2. Row and column slice indexing
    row_sub = tr_2d.iloc[0]
    assert isinstance(row_sub, TruncatedDistribution)
    assert row_sub.shape == (1, 2)
    assert list(row_sub.index) == ["r0"]
    assert list(row_sub.columns) == ["c0", "c1"]

    col_sub = tr_2d.iloc[:, 1]
    assert isinstance(col_sub, TruncatedDistribution)
    assert col_sub.shape == (2, 1)
    assert list(col_sub.index) == ["r0", "r1"]
    assert list(col_sub.columns) == ["c1"]

    # 3. Scalar TruncatedDistribution
    tr_scalar = TruncatedDistribution(Normal(0.0, 1.0), lower=-1.0, upper=1.0)
    assert tr_scalar.iat[0, 0].ndim == 0
    assert tr_scalar.iloc[0, 0].ndim == 0
    with pytest.raises(IndexError):
        tr_scalar.iat[0, 1]
