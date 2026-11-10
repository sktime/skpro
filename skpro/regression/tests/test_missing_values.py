"""Tests for input checks against the capability:missing tag of regressors."""

import numpy as np
import pandas as pd
import pytest

from skpro.regression.dummy import DummyProbaRegressor
from skpro.tests.test_switch import run_test_module_changed

PREDICT_METHODS = [
    "predict",
    "predict_proba",
    "predict_interval",
    "predict_quantiles",
    "predict_var",
]


class _DummyNoMissing(DummyProbaRegressor):
    """Dummy regressor that declares it does not support missing values."""

    _tags = {"capability:missing": False}


def _get_data():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X.iloc[:50]
    y = pd.DataFrame(y.iloc[:50])
    return X, y


def _with_nan(df):
    df = df.copy()
    df.iloc[0, 0] = np.nan
    return df


@pytest.mark.skipif(
    not run_test_module_changed("skpro.regression"),
    reason="run test only if regression module has changed",
)
@pytest.mark.parametrize("method", PREDICT_METHODS)
def test_predict_raises_on_nan_if_missing_not_supported(method):
    """Predict methods raise informative error on NaN in X if tag is False."""
    X, y = _get_data()
    reg = _DummyNoMissing().fit(X, y)

    with pytest.raises(ValueError, match="does not support missing values"):
        getattr(reg, method)(_with_nan(X))


@pytest.mark.skipif(
    not run_test_module_changed("skpro.regression"),
    reason="run test only if regression module has changed",
)
@pytest.mark.parametrize("nan_in", ["X", "y"])
def test_fit_raises_on_nan_if_missing_not_supported(nan_in):
    """Fit raises informative error on NaN in X or y if tag is False."""
    X, y = _get_data()
    if nan_in == "X":
        X = _with_nan(X)
    else:
        y = _with_nan(y)

    msg = f"{nan_in} passed to _DummyNoMissing contains missing values"
    with pytest.raises(ValueError, match=msg):
        _DummyNoMissing().fit(X, y)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.regression"),
    reason="run test only if regression module has changed",
)
def test_no_error_on_nan_if_missing_supported():
    """No error is raised by the base class if capability:missing is True."""
    X, y = _get_data()
    reg = DummyProbaRegressor()
    assert reg.get_tag("capability:missing")

    reg.fit(X, y)
    reg.predict_proba(_with_nan(X))


@pytest.mark.skipif(
    not run_test_module_changed("skpro.regression"),
    reason="run test only if regression module has changed",
)
def test_no_error_on_clean_data_if_missing_not_supported():
    """No error is raised if the tag is False but data has no NaN."""
    X, y = _get_data()
    reg = _DummyNoMissing().fit(X, y)
    reg.predict_proba(X)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.regression"),
    reason="run test only if regression module has changed",
)
def test_pipeline_with_imputer_accepts_nan():
    """Imputer in Pipeline before a regressor without NaN support works."""
    from sklearn.impute import SimpleImputer

    from skpro.regression.compose import Pipeline

    X, y = _get_data()
    pipe = Pipeline([("imputer", SimpleImputer()), ("reg", _DummyNoMissing())])
    assert pipe.get_tag("capability:missing")

    pipe.fit(_with_nan(X), y)
    pipe.predict_proba(_with_nan(X))
