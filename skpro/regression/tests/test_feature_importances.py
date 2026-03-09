import numpy as np
import pandas as pd
import pytest

from skpro.regression.base import BaseProbaRegressor


class _TestFIRegressor(BaseProbaRegressor):
    _tags = {
        "capability:feature_importance": True,
        "capability:multioutput": False,
    }

    def _fit(self, X, y):
        self._y_varname = y.columns[0]
        self.feature_importances_ = np.arange(X.shape[1], dtype=float)
        return self

    def _predict(self, X):
        return pd.DataFrame(
            np.zeros(len(X), dtype=float),
            index=X.index,
            columns=[self._y_varname],
        )


class _TestNoFIRegressor(BaseProbaRegressor):
    def _fit(self, X, y):
        self._y_varname = y.columns[0]
        return self

    def _predict(self, X):
        return pd.DataFrame(
            np.zeros(len(X), dtype=float),
            index=X.index,
            columns=[self._y_varname],
        )


class _TestBrokenFIRegressor(BaseProbaRegressor):
    _tags = {"capability:feature_importance": True}

    def _fit(self, X, y):
        self._y_varname = y.columns[0]
        return self

    def _predict(self, X):
        return pd.DataFrame(
            np.zeros(len(X), dtype=float),
            index=X.index,
            columns=[self._y_varname],
        )


def _make_xy(n=10):
    X = pd.DataFrame({"a": np.arange(n), "b": np.arange(n) + 1.0})
    y = pd.DataFrame({"y": np.arange(n) * 0.0})
    return X, y


def test_feature_importances_returns_series_with_feature_names():
    X, y = _make_xy()
    est = _TestFIRegressor().fit(X, y)
    imp = est.feature_importances()
    assert isinstance(imp, pd.Series)
    assert list(imp.index) == list(X.columns)
    assert len(imp) == X.shape[1]


def test_feature_importances_unsupported_raises():
    X, y = _make_xy()
    est = _TestNoFIRegressor().fit(X, y)
    with pytest.raises(NotImplementedError):
        est.feature_importances()


def test_feature_importances_tag_true_but_missing_attr_raises():
    X, y = _make_xy()
    est = _TestBrokenFIRegressor().fit(X, y)
    with pytest.raises(AttributeError):
        est.feature_importances()

