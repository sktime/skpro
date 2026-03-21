"""Tests for fallback probabilistic prediction methods in BaseProbaRegressor."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import warnings

import numpy as np
import pandas as pd
import pytest

from skpro.regression.base import BaseProbaRegressor


class MockZeroVarRegressor(BaseProbaRegressor):
    """Mock regressor that predicts 0 variance deterministically."""

    def _fit(self, X, y):
        """Fit mock."""
        return self

    def _predict(self, X):
        """Predict exactly 0 for all instances."""
        return pd.DataFrame(np.zeros(len(X)), index=X.index, columns=self._y_metadata["feature_names"])

    def _predict_var(self, X):
        """Predict strict 0 variance for all instances."""
        return pd.DataFrame(np.zeros(len(X)), index=X.index, columns=self._y_metadata["feature_names"])

def test_predict_proba_zero_variance_fallback():
    """Test that zero-variance edge case in _predict_proba doesn't cause NaNs."""
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 0.0, 0.0], name="y")

    model = MockZeroVarRegressor().fit(X, y)
    
    # predict_var evaluates to strictly 0
    # _predict_proba should fallback to predict_var and use the Normal constructor
    dist = model.predict_proba(X)
    
    # We test with evaluate pdf and log_pdf over the exact predicted mean (0.0)
    # Without np.clip, sigma=0 causes divide by 0 and thus a RuntimeWarning + NaN
    x_test = pd.DataFrame({"y": [0.0, 0.0, 0.0]})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pdf_vals = dist.pdf(x_test)
        log_pdf_vals = dist.log_pdf(x_test)

    # It shouldn't output NaNs
    assert not np.isnan(pdf_vals.values).any()
    assert not np.isnan(log_pdf_vals.values).any()

    # The PDF evaluates to a massive spike due to epsilon standard deviation
    assert (pdf_vals.values > 1e6).all()

if __name__ == "__main__":
    test_predict_proba_zero_variance_fallback()
    print("Test passed successfully: Zero variance clipped and stability maintained.")
