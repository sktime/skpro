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
def test_empirical_energy_all_nan_samples():
    """Test that energy returns NaN for all-NaN sample distributions.

    Regression test for bug where np.sum on a pandas DataFrame used skipna=True,
    silently converting NaN energy contributions to 0.0 instead of NaN.
    """
    spl_idx = pd.MultiIndex.from_product([[0, 1, 2], ["A"]], names=["sample", "loc"])
    y_true = pd.DataFrame({"qty": [10.0]}, index=pd.Index(["A"], name="loc"))

    # Distribution with all-NaN samples
    spl_nan = pd.DataFrame({"qty": [np.nan, np.nan, np.nan]}, index=spl_idx)
    dist_nan = Empirical(spl_nan)

    energy_x = dist_nan.energy(y_true).values
    energy_self = dist_nan.energy().values

    assert np.isnan(energy_x).all(), (
        f"energy(y_true) should be NaN for all-NaN samples, got {energy_x}"
    )
    assert np.isnan(energy_self).all(), (
        f"energy() should be NaN for all-NaN samples, got {energy_self}"
    )


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_empirical_energy_mixed_nan_samples():
    """Test energy with mixed NaN and non-NaN rows.

    Rows with some NaN samples should give NaN energy;
    rows with valid samples should give finite energy.
    """
    spl_idx = pd.MultiIndex.from_product(
        [[0, 1, 2], ["A", "B"]], names=["sample", "loc"]
    )
    # Row "A" has some NaN samples, row "B" has valid samples
    spl = pd.DataFrame(
        {
            "qty": [
                1.0,
                1.0,
                np.nan,
                2.0,
                3.0,
                3.0,
            ]
        },
        index=spl_idx,
    )
    dist = Empirical(spl)
    y_true = pd.DataFrame({"qty": [10.0, 2.0]}, index=pd.Index(["A", "B"], name="loc"))

    energy_x = dist.energy(y_true)
    energy_self = dist.energy()

    # Row "A" (some NaN) should give NaN
    assert np.isnan(energy_x.loc["A", "energy"]), (
        "energy(y_true) for all-NaN row should be NaN"
    )
    assert np.isnan(energy_self.loc["A", "energy"]), (
        "energy() for all-NaN row should be NaN"
    )

    # Row "B" (valid) should give finite energy
    assert np.isfinite(energy_x.loc["B", "energy"]), (
        "energy(y_true) for valid row should be finite"
    )
    assert np.isfinite(energy_self.loc["B", "energy"]), (
        "energy() for valid row should be finite"
    )
