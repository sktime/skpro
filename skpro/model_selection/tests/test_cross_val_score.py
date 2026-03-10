"""Tests for cross_val_score utility."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Ahmed"]

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, ShuffleSplit

from skpro.metrics import CRPS, EmpiricalCoverage, LogLoss, PinballLoss
from skpro.model_selection._cross_val import cross_val_score
from skpro.regression.base import BaseProbaRegressor
from skpro.regression.residual import ResidualDouble

CVs = [
    KFold(n_splits=3),
    ShuffleSplit(n_splits=3, test_size=0.5, random_state=42),
]

SINGLE_METRICS = [CRPS, LogLoss, PinballLoss, EmpiricalCoverage]


@pytest.fixture
def diabetes_data():
    """Load test data."""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    return X, y


@pytest.fixture
def estimator():
    """Create a test estimator."""
    return ResidualDouble(LinearRegression(), min_scale=1)


@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("scoring_cls", SINGLE_METRICS)
def test_single_metric_returns_1d_array(diabetes_data, estimator, cv, scoring_cls):
    """Single metric must return a 1-D np.ndarray whose length equals n_splits."""
    X, y = diabetes_data
    scoring = scoring_cls()

    scores = cross_val_score(estimator, X, y, scoring=scoring, cv=cv)

    n_splits = cv.get_n_splits(X)
    assert isinstance(scores, np.ndarray), "Expected np.ndarray for single metric"
    assert scores.ndim == 1, f"Expected 1-D array, got ndim={scores.ndim}"
    assert len(scores) == n_splits, (
        f"Expected array of length {n_splits}, got {len(scores)}"
    )
    # every entry should be a finite float (no NaN / inf for a healthy estimator)
    assert np.all(np.isfinite(scores)), "All scores should be finite"


MULTI_METRIC_COMBOS = [
    [CRPS(), LogLoss()],
    [CRPS(), PinballLoss()],
    [CRPS(), LogLoss(), EmpiricalCoverage()],
]


@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("metrics", MULTI_METRIC_COMBOS)
def test_multi_metric_returns_dataframe_with_expected_columns(
    diabetes_data, estimator, cv, metrics
):
    """List of metrics must return a DataFrame with one named column per metric."""
    X, y = diabetes_data

    scores = cross_val_score(estimator, X, y, scoring=metrics, cv=cv)

    n_splits = cv.get_n_splits(X)
    assert isinstance(scores, pd.DataFrame), "Expected pd.DataFrame for multi-metric"
    assert scores.shape[0] == n_splits, (
        f"Expected {n_splits} rows (one per fold), got {scores.shape[0]}"
    )
    # Column names should be the metric .name attributes (without "test_" prefix)
    expected_columns = [m.name for m in metrics]
    assert list(scores.columns) == expected_columns, (
        f"Expected columns {expected_columns}, got {list(scores.columns)}"
    )
    # all entries should be finite for a healthy estimator
    assert np.all(np.isfinite(scores.to_numpy())), "All scores should be finite"


class _FailingProbaRegressor(BaseProbaRegressor):
    """Regressor that always raises on fit, for error_score tests."""

    _tags = {"capability:survival": False}

    def __init__(self):
        super().__init__()

    def _fit(self, X, y, C=None):
        raise RuntimeError("deliberate fit failure for testing")

    def _predict(self, X):
        raise NotImplementedError  # pragma: no cover

    def _predict_proba(self, X):
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {}


def test_error_score_raise_propagates_exception(diabetes_data):
    """error_score='raise' must propagate the fit exception."""
    X, y = diabetes_data
    cv = KFold(n_splits=3)
    scoring = CRPS()
    bad_est = _FailingProbaRegressor()

    with pytest.raises(RuntimeError, match="deliberate fit failure"):
        cross_val_score(bad_est, X, y, scoring=scoring, cv=cv, error_score="raise")


@pytest.mark.parametrize("error_val", [np.nan, 0.0, -999.0])
def test_error_score_numeric_fills_scores(diabetes_data, error_val):
    """Numeric error_score must fill all fold scores with the given value."""
    X, y = diabetes_data
    cv = KFold(n_splits=3)
    scoring = CRPS()
    bad_est = _FailingProbaRegressor()

    scores = cross_val_score(
        bad_est, X, y, scoring=scoring, cv=cv, error_score=error_val
    )
    n_splits = cv.get_n_splits(X)

    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 1
    assert len(scores) == n_splits

    if np.isnan(error_val):
        assert np.all(np.isnan(scores)), "All fold scores should be NaN"
    else:
        np.testing.assert_array_equal(
            scores,
            np.full(n_splits, error_val),
            err_msg=f"All fold scores should equal {error_val}",
        )


def test_error_score_default_is_nan(diabetes_data):
    """Default error_score (np.nan) must fill all fold scores with NaN."""
    X, y = diabetes_data
    cv = KFold(n_splits=3)
    scoring = CRPS()
    bad_est = _FailingProbaRegressor()

    # don't pass error_score at all – rely on the default
    scores = cross_val_score(bad_est, X, y, scoring=scoring, cv=cv)

    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 1
    assert len(scores) == cv.get_n_splits(X)
    assert np.all(np.isnan(scores)), "Default error_score must yield NaN"
