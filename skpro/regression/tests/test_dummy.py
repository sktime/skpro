"""Tests for DummyProbaRegressor."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest

from skpro.regression.dummy import DummyProbaRegressor


@pytest.mark.parametrize("strategy", ["empirical", "normal"])
def test_dummy_predict_var_returns_variance_not_std(strategy):
    """predict_var must return the variance of y, not its standard deviation.

    Regression test for bug #975: ``_predict_var`` filled predictions with
    ``self._sigma`` (the standard deviation) instead of the variance, producing
    a unit-mismatched result (sigma instead of sigma**2).
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"feature": rng.normal(size=50)})
    y = pd.DataFrame({"target": rng.normal(loc=3.0, scale=7.0, size=50)})

    reg = DummyProbaRegressor(strategy=strategy)
    reg.fit(X, y)

    X_test = pd.DataFrame({"feature": rng.normal(size=10)})
    pred_var = reg.predict_var(X_test)

    expected_var = np.var(y.values)
    std_dev = np.std(y.values)

    # every row carries the same constant variance prediction
    assert np.allclose(pred_var.values, expected_var)
    # guard against the regression: variance must not equal the std dev
    assert not np.allclose(pred_var.values, std_dev)


@pytest.mark.parametrize("strategy", ["empirical", "normal"])
def test_dummy_predict_var_shape_and_index(strategy):
    """predict_var output must align with X index and y columns."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"feature": rng.normal(size=30)})
    y = pd.DataFrame({"target": rng.normal(size=30)})

    reg = DummyProbaRegressor(strategy=strategy)
    reg.fit(X, y)

    X_test = pd.DataFrame({"feature": rng.normal(size=5)}, index=[10, 11, 12, 13, 14])
    pred_var = reg.predict_var(X_test)

    assert pred_var.shape == (5, 1)
    assert list(pred_var.index) == [10, 11, 12, 13, 14]
    assert list(pred_var.columns) == ["target"]
