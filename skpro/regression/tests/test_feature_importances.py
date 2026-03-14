import numpy as np
import pandas as pd
import pytest

from skpro.regression.base import BaseProbaRegressor


class _TestFIRegressor(BaseProbaRegressor):
    """Uses attribute feature_importances_ via its own hook."""

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

    def _feature_importances(self):
        return pd.Series(
            np.asarray(self.feature_importances_).ravel(),
            index=self.feature_names_in_,
            name="feature_importance",
        )


class _TestFIOverrideRegressor(BaseProbaRegressor):
    """Implements _feature_importances() returning pd.Series in standard format."""

    _tags = {
        "capability:feature_importance": True,
        "capability:multioutput": False,
    }

    def _fit(self, X, y):
        self._y_varname = y.columns[0]
        return self

    def _predict(self, X):
        return pd.DataFrame(
            np.zeros(len(X), dtype=float),
            index=X.index,
            columns=[self._y_varname],
        )

    def _feature_importances(self):
        return pd.Series(
            [1.0, 2.0],
            index=self.feature_names_in_,
            name="feature_importance",
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
    """When using feature_importances_ attribute, base converts to Series with names."""
    X, y = _make_xy()
    est = _TestFIRegressor().fit(X, y)
    imp = est.feature_importances()
    assert isinstance(imp, pd.Series)
    assert list(imp.index) == list(X.columns)
    assert len(imp) == X.shape[1]
    assert imp.name == "feature_importance"


def test_feature_importances_override_returned_as_is():
    """When _feature_importances() is implemented, its return is used as-is."""
    X, y = _make_xy()
    est = _TestFIOverrideRegressor().fit(X, y)
    imp = est.feature_importances()
    assert isinstance(imp, pd.Series)
    assert list(imp.index) == list(X.columns)
    assert imp.name == "feature_importance"
    assert list(imp) == [1.0, 2.0]


def test_feature_importances_unsupported_raises():
    X, y = _make_xy()
    est = _TestNoFIRegressor().fit(X, y)
    with pytest.raises(NotImplementedError):
        est.feature_importances()


def test_feature_importances_tag_true_but_missing_attr_raises():
    X, y = _make_xy()
    est = _TestBrokenFIRegressor().fit(X, y)
    with pytest.raises(NotImplementedError):
        est.feature_importances()

