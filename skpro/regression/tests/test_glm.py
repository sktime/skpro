"""Tests Generalized Linear Model regressor."""

import pandas as pd
import pytest

from skpro.regression.linear import GLMRegressor
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(GLMRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_glm_simple_use():
    """Test simple use of GLM regressor."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:200]
    y = y.iloc[:200]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    glm_reg = GLMRegressor()
    glm_reg.fit(X_train, y_train)
    y_pred = glm_reg.predict(X_test)
    y_pred_proba = glm_reg.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
    assert y_pred_proba.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(GLMRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_glm_with_offset_exposure():
    """Test GLM with offset_var and exposure_var parameters."""
    import numpy as np
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:200]
    y = y.iloc[:200]
    X["off"] = np.ones(X.shape[0]) * 2.1
    X["exp"] = np.arange(1, X.shape[0] + 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    glm_reg = GLMRegressor(
        family="Normal", link="Log", offset_var="off", exposure_var=-1
    )
    glm_reg.fit(X_train, y_train)
    y_pred = glm_reg.predict(X_test)
    y_pred_proba = glm_reg.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
    assert y_pred_proba.shape == y_test.shape
