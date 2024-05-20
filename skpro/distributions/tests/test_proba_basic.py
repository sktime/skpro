"""Non-suite tests for probability distribution objects."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
import pytest

from skpro.utils.validation._dependencies import _check_soft_dependencies


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

    n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1, columns=["foo", "bar"])

    assert n.shape == (3, 2)
    assert isinstance(n.index, pd.Index)
    assert isinstance(n.columns, pd.Index)
    assert n.index.equals(pd.RangeIndex(3))
    assert n.columns.equals(pd.Index(["foo", "bar"]))

    n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1, index=["2", 1, 0])

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


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip if matplotlib is not available",
)
@pytest.mark.parametrize("fun", ["pdf", "ppf", "cdf"])
def test_proba_plotting(fun):
    """Test that plotting functions do not crash and return ax as expected."""
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from skpro.distributions.normal import Normal

    # default case, 2D distribution with n_columns>1
    n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)
    fig, ax = n.plot(fun=fun)
    assert isinstance(fig, Figure)
    assert isinstance(ax, np.ndarray)
    assert ax.shape == n.shape
    assert all([isinstance(a, Axes) for a in ax.flatten()])
    assert all([a.get_figure() == fig for a in ax.flatten()])

    # 1D case requires special treatment of axes
    n = Normal(mu=[[1], [2], [3]], sigma=1)
    fig, ax = n.plot(fun=fun)
    assert isinstance(fig, Figure)
    assert isinstance(ax, type(ax))
    assert ax.shape == (n.shape[0],)
    assert all([isinstance(a, Axes) for a in ax.flatten()])
    assert all([a.get_figure() == fig for a in ax.flatten()])

    # scalar case
    n = Normal(mu=1, sigma=1)
    ax = n.plot(fun=fun)
    assert isinstance(ax, Axes)


def test_to_df_parametric():
    """Tests coercion to DataFrame via get_params_df and to_df."""
    from skpro.distributions.normal import Normal

    cols = ["foo", "bar"]

    # default case, 2D distribution with n_columns>1
    n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1, columns=cols)

    param_names = n.get_param_names()
    params_df = n.get_params_df()
    for k, v in params_df.items():
        assert k in param_names
        assert isinstance(v, pd.DataFrame)
        assert (v.index == n.index).all()
        assert (v.columns == n.columns).all()

    all_params_df = n.to_df()
    assert isinstance(all_params_df, pd.DataFrame)
    assert (all_params_df.index == n.index).all()
    assert isinstance(all_params_df.columns, pd.MultiIndex)

    level0_vals = all_params_df.columns.get_level_values(0).unique()
    level1_vals = all_params_df.columns.get_level_values(1).unique()

    assert (level0_vals == n.columns).all()
    for ix in level1_vals:
        assert ix in param_names
        assert ix not in ["index", "columns"]

    # scalar case
    n = Normal(mu=2, sigma=3)

    param_names = n.get_param_names()
    params_df = n.get_params_df()

    for k, v in params_df.items():
        assert k in param_names
        assert isinstance(v, pd.DataFrame)
        assert (v.index == pd.RangeIndex(1)).all()
        assert (v.columns == pd.RangeIndex(1)).all()

    all_params_df = n.to_df()
    assert isinstance(all_params_df, pd.DataFrame)
    assert (all_params_df.index == pd.RangeIndex(1)).all()
    assert not isinstance(all_params_df.columns, pd.MultiIndex)

    for ix in all_params_df.columns:
        assert ix in param_names
        assert ix not in ["index", "columns"]


def test_head_tail():
    """Test head and tail utility functions."""
    from skpro.distributions.normal import Normal

    cols = ["foo", "bar"]

    # default case, 2D distribution with n_columns>1
    n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1, columns=cols)

    nh = n.head(2)
    assert isinstance(nh, Normal)
    assert nh.shape == (2, 2)
    assert (nh.columns == n.columns).all()
    assert (nh.index == n.index[:2]).all()

    nh2 = n.head()
    assert isinstance(nh2, Normal)
    assert nh2.shape == (3, 2)
    assert (nh2.columns == n.columns).all()
    assert (nh2.index == n.index).all()

    nt = n.tail(2)
    assert isinstance(nt, Normal)
    assert nt.shape == (2, 2)
    assert (nt.columns == n.columns).all()
    assert (nt.index == n.index[1:]).all()

    nt2 = n.tail()
    assert isinstance(nt2, Normal)
    assert nt2.shape == (3, 2)
    assert (nt2.columns == n.columns).all()
    assert (nt2.index == n.index).all()

    # scalar case
    n = Normal(mu=2, sigma=3)

    nh = n.head()
    assert nh.ndim == 0

    nt = n.tail(42)
    assert nt.ndim == 0
