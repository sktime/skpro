"""Non-suite tests for probability distribution objects."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

__author__ = ["fkiraly"]

import pandas as pd
import pytest


def test_proba_example():
    """Test one subsetting case for BaseDistribution."""
    from skpro.distributions.normal import Normal

    n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)

    assert n.shape == (3, 2)

    one_row = n.loc[[1]]
    assert isinstance(one_row, Normal)
    assert one_row.shape == (1, 2)


@pytest.mark.parametrize("subsetter", ["loc", "iloc"])
def test_proba_subsetters_loc_iloc(subsetter):
    """Test one subsetting case for BaseDistribution."""
    from skpro.distributions.normal import Normal

    n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)

    assert n.shape == (3, 2)

    # should result in 2D array distribution (1, 1)
    nss = getattr(n, subsetter)[1, [1]]
    assert isinstance(nss, Normal)
    assert nss.shape == (1, 1)
    assert nss.mu.shape == (1, 1)

    nss = getattr(n, subsetter)[[1], 1]
    assert isinstance(nss, Normal)
    assert nss.shape == (1, 1)
    assert nss.mu.shape == (1, 1)

    # should result in scalar distribution
    nss = getattr(n, subsetter)[1, 1]
    assert isinstance(nss, Normal)
    assert nss.shape == ()

    nss = getattr(n, subsetter)[1, 1]
    assert isinstance(nss, Normal)
    assert nss.shape == ()


def test_proba_subsetters_at_iat():
    """Test one subsetting case for BaseDistribution."""
    from skpro.distributions.normal import Normal

    n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)

    # should result in scalar distribution
    nss = n.iat[1, 1]
    assert isinstance(nss, Normal)
    assert nss.shape == ()
    assert nss == n.iloc[1, 1]

    nss = n.at[1, 1]
    assert isinstance(nss, Normal)
    assert nss.shape == ()
    assert nss == n.loc[1, 1]


def test_proba_index_coercion():
    """Test index coercion for BaseDistribution."""
    from skpro.distributions.normal import Normal

    n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1, columns = ["foo", "bar"])

    assert n.shape == (3, 2)
    assert isinstance(n.index, pd.Index)
    assert isinstance(n.columns, pd.Index)
    assert n.index.equals(pd.RangeIndex(3))
    assert n.columns.equals(pd.Index(["foo", "bar"]))

    n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1, index = ["2", 1, 0])

    assert n.shape == (3, 2)
    assert isinstance(n.index, pd.Index)
    assert isinstance(n.columns, pd.Index)
    assert n.index.equals(pd.Index(["2", 1, 0]))
    assert n.columns.equals(pd.RangeIndex(2))

    # this should coerce to a 2D array of shape (1, 3)
    n = Normal(0, 1, columns=[1, 2, 3])

    assert n.shape == (1, 3)
    assert isinstance(n.index, pd.Index)
    assert isinstance(n.columns, pd.Index)
    assert n.index.equals(pd.RangeIndex(1))
    assert n.columns.equals(pd.Index([1, 2, 3]))
