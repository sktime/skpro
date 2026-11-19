"""Tests for fallback probabilistic prediction methods in BaseProbaRegressor."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import warnings

import numpy as np
import pandas as pd
import pytest

from skpro.regression.base import BaseProbaRegressor


class _ZeroVarRegressor(BaseProbaRegressor):
    """Mock regressor returning zero variance for all instances."""

    def _fit(self, X, y):
        return self

    def _predict(self, X):
        cols = self._y_metadata["feature_names"]
        return pd.DataFrame(np.ones(len(X)), index=X.index, columns=cols)

    def _predict_var(self, X):
        cols = self._y_metadata["feature_names"]
        return pd.DataFrame(np.zeros(len(X)), index=X.index, columns=cols)


class _MixedVarRegressor(BaseProbaRegressor):
    """Mock regressor returning mixed zero/nonzero variance."""

    def _fit(self, X, y):
        return self

    def _predict(self, X):
        cols = self._y_metadata["feature_names"]
        return pd.DataFrame(np.ones(len(X)), index=X.index, columns=cols)

    def _predict_var(self, X):
        cols = self._y_metadata["feature_names"]
        # first row has zero variance, rest have nonzero
        var_vals = np.ones(len(X))
        var_vals[0] = 0.0
        return pd.DataFrame(var_vals, index=X.index, columns=cols)


@pytest.fixture
def Xy():
    """Return simple X, y pair for testing."""
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = pd.DataFrame({"y": [0.0, 0.0, 0.0]})
    return X, y


def test_predict_proba_zero_var_returns_delta(Xy):
    """Zero variance predictions should return a Delta distribution."""
    from skpro.distributions.delta import Delta

    X, y = Xy
    model = _ZeroVarRegressor().fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dist = model.predict_proba(X)

    assert isinstance(dist, Delta)
    pred_mean = model.predict(X)
    np.testing.assert_array_equal(dist.mean().values, pred_mean.values)


def test_predict_proba_mixed_var_returns_normal(Xy):
    """Mixed variance predictions should return a Normal distribution."""
    from skpro.distributions.normal import Normal

    X, y = Xy
    model = _MixedVarRegressor().fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dist = model.predict_proba(X)

    assert isinstance(dist, Normal)

    # pdf and log_pdf should be finite for all rows
    x_test = pd.DataFrame({"y": [1.0, 1.0, 1.0]}, index=X.index)
    pdf_vals = dist.pdf(x_test)
    log_pdf_vals = dist.log_pdf(x_test)

    assert np.isfinite(pdf_vals.values).all()
    assert np.isfinite(log_pdf_vals.values).all()
