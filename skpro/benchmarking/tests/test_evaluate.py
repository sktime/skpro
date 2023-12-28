"""Tests for evaluate utility."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# based on the sktime tests of the same name

__author__ = ["fkiraly"]


import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, ShuffleSplit

from skpro.benchmarking.evaluate import evaluate
from skpro.metrics import CRPS, EmpiricalCoverage, LogLoss, PinballLoss
from skpro.regression.residual import ResidualDouble
from skpro.utils.validation._dependencies import _check_soft_dependencies


def _check_evaluate_output(out, cv, y, scoring):
    assert isinstance(out, pd.DataFrame)

    if hasattr(scoring, "get_tag"):
        scitype = scoring.get_tag("scitype:y_pred", raise_error=False)
    else:
        scitype = None

    # Check column names.
    assert set(out.columns) == {
        "fit_time",
        "len_y_train",
        f"{scitype}_time",
        f"test_{scoring.name}",
    }

    # Check number of rows against number of splits.
    n_splits = cv.get_n_splits(y)
    assert out.shape[0] == n_splits

    # Check that all timings are positive.
    assert np.all(out.filter(like="_time") >= 0)

    # Check training set sizes
    assert np.all(out["len_y_train"] > 0)


def _get_pred_method(scoring):
    """Get the prediction method for a given scoring function."""
    pred_type = {
        "pred_quantiles": "predict_quantiles",
        "pred_interval": "predict_interval",
        "pred_proba": "predict_proba",
        None: "predict",
    }

    if hasattr(scoring, "get_tag"):
        scitype = scoring.get_tag("scitype:y_pred", raise_error=False)
    else:
        scitype = None

    return pred_type[scitype]


CVs = [
    KFold(n_splits=3),
    ShuffleSplit(n_splits=3, test_size=0.5, random_state=42),
]

METRICS = [CRPS, EmpiricalCoverage, LogLoss, PinballLoss]


@pytest.mark.parametrize("cv", CVs)
@pytest.mark.parametrize("scoring", METRICS)
@pytest.mark.parametrize("backend", [None, "dask", "loky", "threading"])
def test_evaluate_common_configs(cv, scoring, backend):
    """Test evaluate common configs."""
    # skip test for dask backend if dask is not installed
    if backend == "dask" and not _check_soft_dependencies("dask", severity="none"):
        return None

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    estimator = ResidualDouble(LinearRegression(), min_scale=1)

    scoring = scoring()

    out = evaluate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        backend=backend,
    )
    _check_evaluate_output(out, cv, y, scoring)

    # check scoring
    actual = out.loc[:, f"test_{scoring.name}"]

    n_splits = cv.get_n_splits(X)
    expected = np.empty(n_splits)

    for i, (train, test) in enumerate(cv.split(y)):
        X_train, y_train = X.iloc[train], y.iloc[train]
        X_test, y_test = X.iloc[test], y.iloc[test]
        est = estimator.clone()
        est.fit(X_train, y_train)

        pred_method = _get_pred_method(scoring)
        y_pred = getattr(est, pred_method)(X_test)
        expected[i] = scoring(y_test, y_pred, y_train=y_train)

    np.testing.assert_array_equal(actual, expected)
