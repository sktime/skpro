"""Tests for transformed target regressor edge cases."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from skpro.distributions.normal import Normal
from skpro.regression.base import BaseProbaRegressor
from skpro.regression.compose import TransformedTargetRegressor
from skpro.regression.dummy import DummyProbaRegressor


class _ScaleTransformer(BaseEstimator, TransformerMixin):
    """Simple DataFrame-preserving transformer for target scaling."""

    def __init__(self, factor=10.0):
        self.factor = factor

    def fit(self, X, y=None):
        """Fit and return self."""
        return self

    def transform(self, X):
        """Scale values by the configured factor."""
        return X * self.factor

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X=X, y=y).transform(X=X)

    def inverse_transform(self, X):
        """Undo scaling."""
        return X / self.factor


class _UpdateRecordingRegressor(BaseProbaRegressor):
    """Regressor stub that records the last observed targets."""

    _tags = {"capability:update": True}

    def _fit(self, X, y, C=None):
        self.fit_y_ = y.copy()
        self.update_y_ = None
        return self

    def _update(self, X, y, C=None):
        self.update_y_ = y.copy()
        return self

    def _predict(self, X):
        mean_val = float(self.fit_y_.iloc[:, 0].mean())
        return pd.DataFrame({"target": np.repeat(mean_val, len(X))}, index=X.index)

    def _predict_proba(self, X):
        mean_val = float(self.fit_y_.iloc[:, 0].mean())
        mu = np.repeat(mean_val, len(X)).reshape(-1, 1)
        sigma = np.ones((len(X), 1))
        return Normal(mu=mu, sigma=sigma, index=X.index, columns=["target"])


def test_ttr_without_transformer_predicts_and_returns_distribution():
    """Constructor default ``transformer=None`` should work across predict APIs."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y = pd.DataFrame({"target": [1.0, 3.0, 5.0]})

    reg = TransformedTargetRegressor(
        regressor=DummyProbaRegressor(strategy="normal"),
        transformer=None,
    )
    reg.fit(X, y)

    y_pred = reg.predict(X)
    y_pred_proba = reg.predict_proba(X)

    assert list(y_pred.columns) == ["target"]
    assert y_pred.index.equals(X.index)
    assert y_pred_proba.columns.equals(pd.Index(["target"]))
    assert y_pred_proba.index.equals(X.index)


def test_ttr_update_applies_transformer_before_delegating():
    """Update should transform ``y`` via ``transform``, not call the transformer."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0]})
    y_new = pd.DataFrame({"target": [4.0, 5.0]})
    X_new = pd.DataFrame({"x": [10.0, 11.0]})

    reg = TransformedTargetRegressor(
        regressor=_UpdateRecordingRegressor(),
        transformer=_ScaleTransformer(factor=10.0),
    )
    reg.fit(X, y)
    reg.update(X_new, y_new)

    expected = y_new * 10.0
    pd.testing.assert_frame_equal(reg.regressor_.update_y_, expected)
