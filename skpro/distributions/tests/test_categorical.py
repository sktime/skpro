# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Categorical-specific tests not covered by generic distribution checks."""

__author__ = ["Atishyy27"]

import numpy as np
import pandas as pd
import pytest

from skpro.distributions import Categorical
from skpro.tests.test_switch import run_test_for_class

# reference distribution: P(X=0)=0.2, P(X=1)=0.5, P(X=2)=0.3
P = [0.2, 0.5, 0.3]

skip_unless_changed = pytest.mark.skipif(
    not run_test_for_class(Categorical),
    reason="run test only if tested object has changed",
)


@skip_unless_changed
def test_moments_default_support():
    """Mean and variance over the default integer codes 0, 1, 2."""
    d = Categorical(p=P)

    # mean = 0*0.2 + 1*0.5 + 2*0.3 = 1.1
    assert np.isclose(d.mean(), 1.1)
    # E[X^2] = 0.5 + 1.2 = 1.7, var = 1.7 - 1.1^2 = 0.49
    assert np.isclose(d.var(), 0.49)


@skip_unless_changed
def test_pmf_cdf_ppf_pinned():
    """pmf, log_pmf, cdf and ppf match hand-computed values."""
    d = Categorical(p=P)

    assert np.isclose(d.pmf(1), 0.5)
    assert d.pmf(1.5) == 0.0  # off support
    assert np.isclose(d.log_pmf(1), np.log(0.5))
    assert np.isneginf(d.log_pmf(1.5))

    assert np.isclose(d.cdf(-1), 0.0)
    assert np.isclose(d.cdf(1), 0.7)
    assert np.isclose(d.cdf(1.5), 0.7)  # step function between support points
    assert d.cdf(2) == 1.0

    # ppf is the smallest support point with cdf >= q
    assert d.ppf(0.0) == 0
    assert d.ppf(0.2) == 0
    assert d.ppf(0.7) == 1
    assert d.ppf(0.71) == 2
    assert d.ppf(1.0) == 2


@skip_unless_changed
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


@skip_unless_changed
def test_accepts_numpy_and_pandas_inputs():
    """p and values can be numpy arrays, 0-d arrays, or pandas Series."""
    d_np = Categorical(p=np.array(P), values=np.array([0, 1, 2]))
    d_0d = Categorical(p=[np.array(0.2), np.array(0.5), np.array(0.3)])
    d_series = Categorical(p=pd.Series(P, index=[5, 6, 7]))
    for d in [d_np, d_0d, d_series]:
        assert np.isclose(d.mean(), 1.1)

    d_nested = Categorical(p=pd.Series([[P], [[1.0]]], index=["r1", "r2"]))
    assert d_nested.shape == (2, 1)
    assert np.isclose(d_nested.mean().values[0, 0], 1.1)


@skip_unless_changed
def test_masses_summing_exactly_are_kept():
    """Masses that already sum to 1 are returned by pmf exactly as passed."""
    d = Categorical(p=[0.2, 0.7, 0.1])
    assert d.pmf(0) == 0.2
    assert d.pmf(1) == 0.7
    assert d.pmf(2) == 0.1


@skip_unless_changed
def test_energy_closed_form():
    """Energy matches sum_ij p_i p_j |v_i - v_j| and sum_i p_i |v_i - x|."""
    d = Categorical(p=P)

    # 2 * (0.2*0.5*1 + 0.2*0.3*2 + 0.5*0.3*1) = 2 * 0.37 = 0.74
    assert np.isclose(d.energy(), 0.74)
    # E|X - 1| = 0.2*1 + 0.5*0 + 0.3*1 = 0.5
    assert np.isclose(d.energy(1), 0.5)
    # E|X - x| for x below the support equals mean - x
    assert np.isclose(d.energy(-2), 1.1 + 2)


@skip_unless_changed
def test_energy_matches_pairwise_sum():
    """Self-energy equals the pairwise sum on random inputs."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        k = int(rng.integers(1, 10))
        p = rng.dirichlet(np.ones(k))
        v = rng.normal(size=k) * 100
        if len(np.unique(v)) < k:
            continue
        d = Categorical(p=list(p), values=list(v))
        pairwise = p @ np.abs(v[:, None] - v[None, :]) @ p
        assert np.isclose(d.energy(), pairwise)


@skip_unless_changed
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

    x = pd.DataFrame([[1, 1], [7, -1]], index=d.index, columns=d.columns)
    x_back = d.ppf(d.cdf(x))
    assert np.allclose(x.values, x_back.values)

    sub = d.iloc[[1], [0]]
    assert isinstance(sub, Categorical)
    assert sub.shape == (1, 1)
    assert np.isclose(sub.mean().values[0, 0], 7.0)


@skip_unless_changed
def test_array_valued_default_codes():
    """Omitted values default to integer codes per entry, for each entry length."""
    d = Categorical(p=[[[0.5, 0.5], [0.2, 0.3, 0.5]]])
    mean = d.mean()
    assert np.isclose(mean.iloc[0, 0], 0.5)  # codes 0, 1
    assert np.isclose(mean.iloc[0, 1], 1.3)  # codes 0, 1, 2


@skip_unless_changed
def test_shared_support():
    """A 1D values is the common support of every entry, also after subsetting."""
    d = Categorical(
        p=[[P, [0.1, 0.1, 0.8]], [[1 / 3, 1 / 3, 1 / 3], [0.6, 0.3, 0.1]]],
        values=[10, 20, 30],
    )
    assert d.shape == (2, 2)
    assert np.isclose(d.mean().values[0, 0], 21.0)  # 2 + 10 + 9
    assert np.isclose(d.mean().values[0, 1], 27.0)  # 1 + 2 + 24

    sub = d.iloc[[1], [1]]
    assert isinstance(sub, Categorical)
    assert np.isclose(sub.mean().values[0, 0], 15.0)  # 6 + 6 + 3
    x = pd.DataFrame([[20]], index=sub.index, columns=sub.columns)
    assert np.isclose(sub.cdf(x).values[0, 0], 0.9)


@skip_unless_changed
@pytest.mark.parametrize(
    "p, values, match",
    [
        ([0.2, 0.5, 0.4], None, "sum to 1"),
        ([0.5, -0.1, 0.6], None, "non-negative"),
        ([0.5, 0.5], [1, 2, 3], "same length"),
        ([0.5, 0.5], [1, 1], "unique"),
        ([], None, "at least one"),
        ([0.5, 0.5], [0, np.inf], "finite"),
        ([0.5, 0.5], [0, np.nan], "finite"),
        ([0.5, np.nan, 0.5], None, "finite"),
        ([[[0.5, 0.5], [1.2, -0.2]]], None, r"p\[0\]\[1\] must be non-negative"),
        ([[[1.0], [1.0]], [[1.0]]], None, "same number of entries"),
        ([[0.5, 0.5], [0.3, 0.7]], None, "nested"),
        ([[[1.0]], 1.0], None, "nested"),
        ([[[[0.3], [0.7]]]], None, "1D"),
        ([[[0.5, 0.5], [1.0]]], [0, 1], "same length"),
        ([[[0.5, 0.5]]], [[[0, 1]], [[0, 1]]], "same nesting"),
        ([[[0.5, 0.5]], [[1.0]]], [[[0, 1]], 5], "same nesting"),
        (0.5, None, "1D"),
        ([[None]], None, "1D"),
    ],
)
def test_invalid_params_raise(p, values, match):
    """Invalid masses or support raise ValueError with a clear message."""
    with pytest.raises(ValueError, match=match):
        Categorical(p=p, values=values)


@skip_unless_changed
def test_scalar_p_rejects_index_and_columns():
    """index and columns are only accepted for the array-valued case."""
    with pytest.raises(ValueError, match="array-valued"):
        Categorical(p=P, index=pd.Index([0]), columns=pd.Index(["a"]))


@skip_unless_changed
def test_masses_within_tolerance_are_rescaled():
    """Masses summing to 1 within tolerance are rescaled, so all methods agree."""
    d = Categorical(p=[0.5, 0.500009], values=[0, 1])
    assert np.isclose(d.pmf(0) + d.pmf(1), 1.0, rtol=0, atol=1e-15)
    assert np.isclose(d.cdf(0), d.pmf(0), rtol=0, atol=1e-15)
    assert d.cdf(1) == 1.0
    assert np.isclose(d.mean(), d.pmf(1), rtol=0, atol=1e-15)


@skip_unless_changed
def test_pmf_exact_at_large_magnitude():
    """pmf keeps distinct support points apart at large magnitudes."""
    d = Categorical(p=[0.5, 0.5], values=[100000, 100001])
    assert d.pmf(100000) == 0.5
    assert d.pmf(100000.5) == 0.0

    d_big = Categorical(p=[0.5, 0.5], values=[1e12, 1e12 + 1])
    assert d_big.pmf(1e12) == 0.5
    assert d_big.pmf(1e12 + 1) == 0.5


@skip_unless_changed
def test_var_accurate_at_large_magnitude():
    """Variance stays accurate for support values far from zero."""
    for base in [1e8, 1e10, 1e12]:
        d = Categorical(p=[0.5, 0.5], values=[base, base + 1])
        assert np.isclose(d.var(), 0.25)

    # unequal masses, where E[X] is not halfway between the support points
    d = Categorical(p=[0.3, 0.7], values=[1e15, 1e15 + 1])
    assert np.isclose(d.var(), 0.21)


@skip_unless_changed
def test_cdf_bounded_and_ppf_inverts_cdf():
    """cdf never exceeds 1, and ppf inverts cdf on support points with mass."""
    rng = np.random.default_rng(42)
    for _ in range(300):
        k = int(rng.integers(2, 8))
        p = rng.dirichlet(np.ones(k))
        v = np.sort(rng.normal(size=k) * 10)
        if p.min() < 1e-6 or len(np.unique(v)) < k:
            continue
        d = Categorical(p=list(p), values=list(v))
        for v_i in v:
            c = d.cdf(v_i)
            assert c <= 1.0
            assert d.ppf(c) == v_i


@skip_unless_changed
def test_ppf_outside_unit_interval_and_nan():
    """ppf returns nan for probabilities outside [0, 1], including nan."""
    d = Categorical(p=P)
    assert np.isnan(d.ppf(np.nan))
    assert np.isnan(d.ppf(-0.1))
    assert np.isnan(d.ppf(1.1))


@skip_unless_changed
def test_zero_mass_support_point():
    """A zero-mass support point has zero pmf and is skipped by ppf."""
    d = Categorical(p=[0.5, 0.0, 0.5])
    assert d.pmf(1) == 0.0
    assert np.isneginf(d.log_pmf(1))
    # ppf returns the smallest support point with cdf >= q, which skips 1
    assert d.ppf(d.cdf(1)) == 0
    assert d.ppf(d.cdf(2)) == 2


@skip_unless_changed
def test_single_point():
    """A one-point distribution is a point mass."""
    d = Categorical(p=[1.0], values=[42])
    assert d.mean() == 42
    assert d.var() == 0
    assert d.pmf(42) == 1.0
    assert d.cdf(41) == 0.0
    assert d.cdf(42) == 1.0
    assert d.ppf(0.0) == 42
    assert d.ppf(1.0) == 42
    assert d.energy() == 0.0


@skip_unless_changed
def test_sample_on_support_with_matching_frequencies():
    """Samples lie on the support, with frequencies close to the masses."""
    np.random.seed(0)
    d = Categorical(p=[0.1, 0.6, 0.3], values=[10, 20, 30])
    spl = d.sample(20000)
    assert set(np.unique(spl.values)) <= {10.0, 20.0, 30.0}
    freq = spl.iloc[:, 0].value_counts(normalize=True)
    assert np.allclose([freq[10.0], freq[20.0], freq[30.0]], [0.1, 0.6, 0.3], atol=0.02)

    d_arr = Categorical(p=[[[0.5, 0.5], [1.0]]], values=[[[0, 1], [7]]])
    spl_arr = d_arr.sample(500)
    assert set(np.unique(spl_arr.iloc[:, 0])) <= {0.0, 1.0}
    assert (spl_arr.iloc[:, 1] == 7.0).all()


@skip_unless_changed
def test_sample_boundary_draw_skips_zero_mass(monkeypatch):
    """A uniform draw at the boundary never lands on a zero-mass point."""
    monkeypatch.setattr(np.random, "uniform", lambda size=None: np.zeros(size))
    d = Categorical(p=[0.0, 0.0, 1.0], values=[10, 20, 30])
    assert (d.sample(3).values == 30.0).all()
