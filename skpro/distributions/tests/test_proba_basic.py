"""Non-suite tests for probability distribution objects."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# adapted from sktime

__author__ = ["fkiraly"]

import numpy as np
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


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip if matplotlib is not available",
)
@pytest.mark.parametrize("fun", ["pdf", "ppf", "cdf"])
def test_proba_plotting(fun):
    """Test that plotting functions do not crash and return ax as expected."""
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

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
