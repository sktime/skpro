# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the Categorical distribution.

The generic distribution suite only checks internal consistency (ppf inverts
cdf, log_pmf = log pmf, output shapes). It cannot detect masses attached to the
wrong support point, a wrong mean, or a wrong energy value. These tests pin the
formulae to hand-computed values for a small, fixed distribution.
"""

__author__ = ["Atishyy27"]

import numpy as np
import pandas as pd
import pytest

from skpro.distributions import Categorical

# reference distribution: P(X=0)=0.2, P(X=1)=0.5, P(X=2)=0.3
P = [0.2, 0.5, 0.3]


def test_moments_default_support():
    """Mean and variance over the default integer codes 0, 1, 2."""
    d = Categorical(p=P)

    # mean = 0*0.2 + 1*0.5 + 2*0.3 = 1.1
    assert np.isclose(d.mean(), 1.1)
    # E[X^2] = 0.5 + 1.2 = 1.7, var = 1.7 - 1.1^2 = 0.49
    assert np.isclose(d.var(), 0.49)


def test_pmf_cdf_ppf_pinned():
    """pmf, cdf and ppf must match the hand-computed values."""
    d = Categorical(p=P)

    assert np.isclose(d.pmf(1), 0.5)
    assert np.isclose(d.pmf(1.5), 0.0)  # off support
    assert np.isclose(d.log_pmf(1), np.log(0.5))

    assert np.isclose(d.cdf(-1), 0.0)
    assert np.isclose(d.cdf(1), 0.7)
    assert np.isclose(d.cdf(1.5), 0.7)  # step function between support points
    assert np.isclose(d.cdf(2), 1.0)

    # ppf is the smallest support point with cdf >= q
    assert d.ppf(0.0) == 0
    assert d.ppf(0.2) == 0
    assert d.ppf(0.7) == 1
    assert d.ppf(0.71) == 2
    assert d.ppf(1.0) == 2
    assert np.isnan(d.ppf(1.5))


def test_explicit_and_unsorted_support():
    """Explicit support values are used, in any given order."""
    d = Categorical(p=[0.1, 0.6, 0.3], values=[10, 20, 30])
    # mean = 1 + 12 + 9 = 22
    assert np.isclose(d.mean(), 22.0)
    assert np.isclose(d.cdf(20), 0.7)
    assert d.ppf(0.7) == 20

    # same distribution as P on {0, 1, 2}, given in a shuffled order
    d_unsorted = Categorical(p=[0.3, 0.2, 0.5], values=[2, 0, 1])
    assert np.isclose(d_unsorted.mean(), 1.1)
    assert np.isclose(d_unsorted.cdf(1), 0.7)
    assert np.isclose(d_unsorted.pmf(2), 0.3)


def test_energy_closed_form():
    """Energy must match the closed forms sum p_i p_j |v_i - v_j|, sum p_i |v_i - x|."""
    d = Categorical(p=P)

    # 2 * (0.2*0.5*1 + 0.2*0.3*2 + 0.5*0.3*1) = 2 * 0.37 = 0.74
    assert np.isclose(d.energy(), 0.74)
    # E|X - 1| = 0.2*1 + 0.5*0 + 0.3*1 = 0.5
    assert np.isclose(d.energy(1), 0.5)
    # E|X - x| for x below the support equals mean - x
    assert np.isclose(d.energy(-2), 1.1 + 2)


def test_array_valued_entrywise():
    """Array-valued case applies masses and support entry-wise."""
    d = Categorical(
        p=[[P, [0.1, 0.9]], [[1.0], [0.25, 0.75]]],
        values=[[[0, 1, 2], [0, 1]], [[7], [-1, 1]]],
        index=pd.Index(["r0", "r1"]),
        columns=pd.Index(["a", "b"]),
    )

    assert d.shape == (2, 2)

    mean = d.mean()
    assert isinstance(mean, pd.DataFrame)
    assert np.isclose(mean.loc["r0", "a"], 1.1)
    assert np.isclose(mean.loc["r0", "b"], 0.9)
    assert np.isclose(mean.loc["r1", "a"], 7.0)
    assert np.isclose(mean.loc["r1", "b"], 0.5)

    # ppf inverts cdf exactly on support points
    x = pd.DataFrame([[1, 1], [7, -1]], index=d.index, columns=d.columns)
    x_back = d.ppf(d.cdf(x))
    assert np.allclose(x.values, x_back.values)

    # subsetting keeps the class and the entry-wise parameters
    sub = d.iloc[[1], [0]]
    assert isinstance(sub, Categorical)
    assert sub.shape == (1, 1)
    assert np.isclose(sub.mean().values[0, 0], 7.0)


def test_shared_support_broadcast():
    """A 1D ``values`` is used as the common support of every entry."""
    d = Categorical(
        p=[[P, [0.1, 0.1, 0.8]]],
        values=[10, 20, 30],
    )
    assert d.shape == (1, 2)
    assert np.isclose(d.mean().values[0, 0], 21.0)  # 2 + 10 + 9
    assert np.isclose(d.mean().values[0, 1], 27.0)  # 1 + 2 + 24


@pytest.mark.parametrize(
    "p, values, match",
    [
        ([0.2, 0.5, 0.4], None, "sum to 1"),
        ([0.5, -0.1, 0.6], None, "non-negative"),
        ([0.5, 0.5], [1, 2, 3], "same length"),
        ([0.5, 0.5], [1, 1], "unique"),
    ],
)
def test_invalid_params_raise(p, values, match):
    """Invalid masses or support raise ValueError with a clear message."""
    with pytest.raises(ValueError, match=match):
        Categorical(p=p, values=values)
