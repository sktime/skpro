"""Tests for Empirical distributions."""

import numpy as np
import pandas as pd
import pytest

from skpro.distributions.empirical import Empirical
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_empirical_iat_index():
    """Test that the index is correctly set after iat call."""
    spl_idx = pd.MultiIndex.from_product([[0, 1], [0, 1, 2]], names=["sample", "time"])
    spl = pd.DataFrame(
        [[0, 1], [2, 3], [10, 11], [6, 7], [8, 9], [4, 5]],
        index=spl_idx,
        columns=["a", "b"],
    )
    emp = Empirical(spl, columns=["a", "b"])

    emp_iat = emp.iat[0, 0]
    assert emp_iat.shape == ()

    assert not isinstance(emp_iat.spl.index, pd.MultiIndex)
    assert (emp_iat.spl.index == [0, 1]).all()


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_empirical_skip_init_sorted():
    """Test that skip_init_sorted parameter works correctly."""
    spl_idx = pd.MultiIndex.from_product(
        [[0, 1, 2], [0, 1, 2]], names=["sample", "time"]
    )
    np.random.seed(42)
    spl = pd.DataFrame(
        np.random.randn(9, 2),
        index=spl_idx,
        columns=["a", "b"],
    )

    emp_sorted = Empirical(spl, skip_init_sorted=False)
    emp_lazy = Empirical(spl, skip_init_sorted=True)

    pd.testing.assert_index_equal(emp_sorted.index, emp_lazy.index)
    pd.testing.assert_index_equal(emp_sorted.columns, emp_lazy.columns)
    pd.testing.assert_frame_equal(emp_sorted.mean(), emp_lazy.mean())
    pd.testing.assert_frame_equal(emp_sorted.var(), emp_lazy.var())
    x = pd.DataFrame([[1, 1], [0, 0], [2, 2]], index=[0, 1, 2], columns=["a", "b"])
    pd.testing.assert_frame_equal(emp_sorted.cdf(x), emp_lazy.cdf(x))
    p = pd.DataFrame(
        [[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]], index=[0, 1, 2], columns=["a", "b"]
    )
    pd.testing.assert_frame_equal(emp_sorted.ppf(p), emp_lazy.ppf(p))
    spl_scalar = pd.Series([1, 2, 3, 4, 3])
    emp_sorted_scalar = Empirical(spl_scalar, skip_init_sorted=False)
    emp_lazy_scalar = Empirical(spl_scalar, skip_init_sorted=True)

    assert emp_sorted_scalar.mean() == emp_lazy_scalar.mean()
    assert emp_sorted_scalar.var() == emp_lazy_scalar.var()
    assert emp_sorted_scalar.cdf(2.5) == emp_lazy_scalar.cdf(2.5)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_empirical_time_indep():
    """Test that time_indep controls independence across time steps (columns).

    When time_indep=True, each time step independently draws a sample index.
    When time_indep=False, all time steps share the same sample index.
    Both modes should produce samples with the correct shape.
    """
    spl_idx = pd.MultiIndex.from_product(
        [[0, 1, 2, 3, 4], [0, 1, 2]], names=["sample", "time"]
    )
    np.random.seed(42)
    spl = pd.DataFrame(
        np.random.randn(15, 2),
        index=spl_idx,
        columns=["a", "b"],
    )

    emp_ti = Empirical(spl, time_indep=True)
    emp_notI = Empirical(spl, time_indep=False)

    n_samples = 10
    # Both should produce a DataFrame with correct multi-index shape
    result_ti = emp_ti.sample(n_samples)
    result_notI = emp_notI.sample(n_samples)

    # Shape: (n_samples * n_instances, n_cols)
    assert result_ti.shape == (n_samples * 3, 2)
    assert result_notI.shape == (n_samples * 3, 2)

    # Values must come from the original sample pool
    all_spl_vals = set(spl.values.flatten().round(8))
    for val in result_ti.values.flatten():
        assert round(val, 8) in all_spl_vals
    for val in result_notI.values.flatten():
        assert round(val, 8) in all_spl_vals


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_empirical_row_indep():
    """Test that row_indep controls independence across rows (instances).

    Tests all 4 combinations of time_indep x row_indep and checks
    that all return samples of the correct shape and with values drawn
    from the original sample pool.
    """
    spl_idx = pd.MultiIndex.from_product(
        [[0, 1, 2, 3], [0, 1, 2]], names=["sample", "time"]
    )
    np.random.seed(0)
    spl = pd.DataFrame(
        np.arange(24, dtype=float).reshape(12, 2),
        index=spl_idx,
        columns=["a", "b"],
    )

    n_samples = 5
    all_spl_vals = set(spl.values.flatten())

    for time_indep in [True, False]:
        for row_indep in [True, False]:
            emp = Empirical(spl, time_indep=time_indep, row_indep=row_indep)
            result = emp.sample(n_samples)
            # shape: (n_samples * n_instances, n_cols) = (5*3, 2) = (15, 2)
            assert result.shape == (n_samples * 3, 2), (
                f"Wrong shape for time_indep={time_indep}, row_indep={row_indep}: "
                f"{result.shape}"
            )
            for val in result.values.flatten():
                assert val in all_spl_vals, (
                    f"Value {val} not in sample pool for "
                    f"time_indep={time_indep}, row_indep={row_indep}"
                )

    # Single sample (n_samples_was_none=True path)
    emp = Empirical(spl, time_indep=True, row_indep=True)
    single = emp.sample()
    assert single.shape == (3, 2)

    emp2 = Empirical(spl, time_indep=False, row_indep=True)
    single2 = emp2.sample()
    assert single2.shape == (3, 2)
